import torch
import torch.nn as nn
import math



def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class PatchEmbed(nn.Module):
    """
    Split image into patches and embed into token vectors.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                            kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class OverLappingPatchEmbed(nn.Module):
    """
    Split image into patches and embed into token vectors.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                            kernel_size=patch_size, stride=2)
        self.num_patches = (img_size // 2) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Multi-head Self-Attention.
    """
    def __init__(self, dim, num_heads=8, attn_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, N, C = x.shape # batch size, num patches, embedding dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

    def return_attn(self, x):
        B, N, C = x.shape # batch size, num patches, embedding dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2,0,3,1,4)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # return (B, H, N, N) attention for this layer
        return x, attn


class RelativePositionAttention(nn.Module):
    """
    Multi-head Self-Attention with learnable relative 2D positional biases.
    """
    def __init__(self, dim, num_heads=8, grid_size=None, attn_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_dropout)

        # relative positional bias table
        assert grid_size is not None, "grid_size required for relative biases"
        self.grid_size = grid_size
        num_rel = (2*grid_size - 1)
        # table: (num_rel^2, num_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel * num_rel, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        # compute relative position index for queries and keys
        coords = torch.arange(grid_size)
        coords = torch.stack(torch.meshgrid(coords, coords), dim=0)  # [2, G, G]
        coords_flat = coords.flatten(1)                              # [2, N]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]      # [2, N, N]
        rel = rel + (grid_size - 1)                                  # shift to [0, 2G-2]
        rel_index = rel[0] * (2*grid_size - 1) + rel[1]              # [N, N]
        self.register_buffer('relative_position_index', rel_index)

    def forward(self, x):
        # x: [B, N+1, C]  first token is cls
        B, N1, C = x.shape
        N = N1 - 1  # number of patches
        # QKV
        qkv = self.qkv(x).reshape(B, N1, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: [B, H, N+1, head_dim]
        # raw scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N+1,N+1]
        # add relative bias only for patch-patch (exclude cls)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)  # [1,H,N,N]
        # pad cls row/col with zeros
        zero = torch.zeros((1, self.num_heads, 1, N), device=attn.device)
        bias = torch.cat([zero, bias], dim=2)             # [1,H,1,N] + [1,H,N,N] -> [1,H,N+1,N]
        zero2 = torch.zeros((1, self.num_heads, N+1, 1), device=attn.device)
        bias = torch.cat([zero2, bias], dim=3)            # [1,H,N+1,1] + [1,H,N+1,N] -> [1,H,N+1,N+1]
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # aggregate
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        return self.proj_drop(x)



class MLP(nn.Module):
    """
    Feed-forward network with configurable hidden layers.
    """
    def __init__(self, in_features, hidden_features, dropout=0.):
        super().__init__()
        layers = []
        prev_dim = in_features
        # hidden_features: list of ints
        for hf in hidden_features:
            layers.append(nn.Linear(prev_dim, hf))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hf
        # final projection back to in_features
        layers.append(nn.Linear(prev_dim, in_features))
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.sig(x)
        return x


class Block(nn.Module):
    """
    Transformer Encoder Block.
    """
    def __init__(self, dim, num_heads, grid_size=None, mlp_ratio=4., mlp_layers=None,
                 dropout=0., attn_dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, attn_dropout)
        if grid_size is not None:
            self.attn = RelativePositionAttention(dim, num_heads, grid_size, attn_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # determine MLP hidden dims
        if mlp_layers:
            hidden_dims = mlp_layers
        else:
            hidden_dims = [int(dim * mlp_ratio)]
        self.mlp = MLP(dim, hidden_dims, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """
    Vision Transformer supporting multi-target regression and configurable MLP depth.
    """
    def __init__(self, mconfig):
        super().__init__()
        self.patch_embed = PatchEmbed(mconfig.img_size, mconfig.patch_size, 
                                      mconfig.in_chans, mconfig.embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = None

        # Learnable [CLS] token: prepended to patch embeddings to aggregate global information
        self.cls_token = nn.Parameter(torch.zeros(1, 1, mconfig.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, mconfig.embed_dim))
        self.embed_drop = nn.Dropout(mconfig.dropout)

        self.blocks = nn.ModuleList([
            Block(mconfig.embed_dim, mconfig.num_heads, grid_size,
                  mconfig.mlp_ratio, mconfig.mlp_layers,
                  mconfig.dropout, mconfig.attn_dropout)
            for _ in range(mconfig.num_blocks)
        ])
        self.norm = nn.LayerNorm(mconfig.embed_dim, eps=1e-6)
        self.head = nn.Linear(mconfig.embed_dim, mconfig.output_dim)

        self.apply(_init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        ptc_emb = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, ptc_emb), dim=1)
        x = self.embed_drop(x + self.pos_embed)
        #print("x_pos:", x.size())
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

        
        B = x.size(0)
        x = self.patch_embed(x)             # [B, N, C]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)      # [B, N+1, C]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])



class RelativePositionVisionTransformer(nn.Module):
    """
    ViT for regression with relative positional biases.
    """
    def __init__(self, mconfig):
        super().__init__()
        overlapping = False
        if overlapping:
            self.patch_embed = OverLappingPatchEmbed(mconfig.img_size, mconfig.patch_size, 
                                        mconfig.in_chans, mconfig.embed_dim)
        else:
            self.patch_embed = PatchEmbed(mconfig.img_size, mconfig.patch_size, 
                                        mconfig.in_chans, mconfig.embed_dim)
        num_patches = self.patch_embed.num_patches
        #grid_size = int(math.sqrt(num_patches))
        grid_size = None
        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, mconfig.embed_dim))
        # absolute positional embeddings for [CLS] + patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, mconfig.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)    
        self.pos_drop = nn.Dropout(mconfig.dropout)
        # transformer blocks
        self.blocks = nn.ModuleList([
            Block(mconfig.embed_dim, mconfig.num_heads, grid_size,
                  mlp_ratio=mconfig.mlp_ratio, mlp_layers=mconfig.mlp_layers,
                  dropout=mconfig.dropout, attn_dropout=mconfig.attn_dropout)
            for _ in range(mconfig.num_blocks)
        ])
        self.norm = nn.LayerNorm(mconfig.embed_dim, eps=1e-6)
        # regression head
        #self.head = nn.Linear(mconfig.embed_dim, mconfig.output_dim)
        self.local_mpl = nn.Sequential(
                nn.Linear(mconfig.embed_dim, mconfig.embed_dim//2),
                nn.GELU(),
                nn.Dropout(mconfig.dropout),
            )
        self.head = nn.Linear(mconfig.embed_dim+mconfig.embed_dim//2, mconfig.output_dim)
        # init
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x_emb = self.patch_embed(x)             # [B, N, C]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x_emb), dim=1)      # [B, N+1, C]
        x = self.pos_drop(x+self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)


        x = x[:,0]
        #x_emb = x_emb.flatten(1) 
        x_emb = x_emb.mean(dim=1)    # [B,C]
        x_emb = self.local_mpl(x_emb)
        combined = torch.cat([x, x_emb], dim=1)
        return self.head(combined)

        #return self.head(x[:, 0])






class VisionTransformerWithTinyCNN(nn.Module):
    """
    ViT with a small CNN processing the raw input for locality fusion.
    """
    def __init__(self, mconfig):
        super().__init__()
        # patch embedding and transformer body
        self.patch_embed = PatchEmbed(mconfig.img_size, mconfig.patch_size, 
                                      mconfig.in_chans, mconfig.embed_dim)
        num_patches = self.patch_embed.num_patches
        #grid_size = int(math.sqrt(num_patches))
        grid_size = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, mconfig.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, mconfig.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(mconfig.dropout)
        self.blocks = nn.ModuleList([
            Block(mconfig.embed_dim, mconfig.num_heads, grid_size,
                  mlp_ratio=mconfig.mlp_ratio, mlp_layers=mconfig.mlp_layers,
                  dropout=mconfig.dropout, attn_dropout=mconfig.attn_dropout)
            for _ in range(mconfig.num_blocks)
        ])
        self.norm = nn.LayerNorm(mconfig.embed_dim, eps=1e-6)
        # small CNN on raw input to capture locality
        self.input_cnn = TinyCNN(mconfig.embed_dim, mconfig.in_chans, standalone=False) 
        # fusion head: combine CLS token output + CNN features
        self.head = nn.Sequential(
            nn.Linear(mconfig.embed_dim*5, mconfig.embed_dim),
            nn.GELU(),
            nn.Dropout(mconfig.dropout),
            nn.Linear(mconfig.embed_dim, mconfig.output_dim)
        )
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        # CNN branch on raw input
        cnn_feat = self.input_cnn(x).view(B, -1)  # [B, embed_dim//2]
        # transformer branch
        tokens = self.patch_embed(x)              # [B, N, C]
        cls = self.cls_token.expand(B, -1, -1)    # [B,1,C]
        x = torch.cat((cls, tokens), dim=1)       # [B,N+1,C]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]                         # [B, C]
        # fuse and predict
        fused = torch.cat([cls_out, cnn_feat], dim=1)  # [B, C + C/2]
        return self.head(fused)


class TinyCNN(nn.Module):
    def __init__(self, embed_dim, in_chans=1, standalone=False, output_dim=1):
        super().__init__()
        self.standalone = standalone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//16, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim//16),
            nn.GELU(),
            nn.Conv2d(embed_dim//16, embed_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim*4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        if standalone:
            self.fc  = nn.LazyLinear(output_dim)
            self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        if self.standalone:
            #x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.sig(x)
        return x



class VisionTransformerWithCNNInput(nn.Module):
    """
    ViT with a small CNN processing the raw input for locality fusion.
    """
    def __init__(self, mconfig):
        super().__init__()
        # patch embedding and transformer body
        self.patch_embed = PatchEmbed(mconfig.img_size, mconfig.patch_size, 
                                      mconfig.in_chans, mconfig.embed_dim)
        num_patches = self.patch_embed.num_patches
        #grid_size = int(math.sqrt(num_patches))
        grid_size = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, mconfig.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, mconfig.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(mconfig.dropout)
        self.blocks = nn.ModuleList([
            Block(mconfig.embed_dim, mconfig.num_heads, grid_size,
                  mlp_ratio=mconfig.mlp_ratio, mlp_layers=mconfig.mlp_layers,
                  dropout=mconfig.dropout, attn_dropout=mconfig.attn_dropout)
            for _ in range(mconfig.num_blocks)
        ])
        self.norm = nn.LayerNorm(mconfig.embed_dim, eps=1e-6)
        # small CNN on raw input to capture locality
        self.input_cnn = TinyCNN(mconfig.embed_dim, mconfig.in_chans, standalone=False) 
        # fusion head: combine CLS token output + CNN features
        self.head = nn.Sequential(
            nn.Linear(mconfig.embed_dim*5, mconfig.embed_dim),
            nn.GELU(),
            nn.Dropout(mconfig.dropout),
            nn.Linear(mconfig.embed_dim, mconfig.output_dim)
        )
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        # CNN branch on raw input
        cnn_feat = self.input_cnn(x).view(B, -1)  # [B, embed_dim//2]
        # transformer branch
        tokens = self.patch_embed(x)              # [B, N, C]
        cls = self.cls_token.expand(B, -1, -1)    # [B,1,C]
        x = torch.cat((cls, tokens), dim=1)       # [B,N+1,C]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]                         # [B, C]
        # fuse and predict
        fused = torch.cat([cls_out, cnn_feat], dim=1)  # [B, C + C/2]
        return self.head(fused)


class TinyCNN(nn.Module):
    def __init__(self, embed_dim, in_chans=1, standalone=False, output_dim=1):
        super().__init__()
        self.standalone = standalone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//16, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim//16),
            nn.GELU(),
            nn.Conv2d(embed_dim//16, embed_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim*4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        if standalone:
            self.fc  = nn.LazyLinear(output_dim)
            self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        if self.standalone:
            #x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.sig(x)
        return x



        
