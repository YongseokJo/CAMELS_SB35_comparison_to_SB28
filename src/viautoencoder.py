import torch
import torch.nn as nn
import math
from transformer import *

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTAutoencoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=256, depth=4, heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, embed_dim))

        self.encoder = nn.Sequential(*[TransformerBlock(embed_dim, heads) for _ in range(depth)])

        self.decoder = nn.Sequential(*[TransformerBlock(embed_dim, heads) for _ in range(depth)])

        self.to_patch = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_chans = in_chans

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        latent = self.encoder(patches)
        decoded = self.decoder(latent)
        patch_pixels = self.to_patch(decoded)  # (B, N, patch*patch*C)
        H = W = self.img_size // self.patch_size
        x_rec = patch_pixels.view(B, H, W, self.in_chans, self.patch_size, self.patch_size)
        x_rec = x_rec.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_rec = x_rec.view(B, self.in_chans, self.img_size, self.img_size)
        return x_rec

        
class MaskedViTAutoencoder(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=256,
                 encoder_depth=6,
                 decoder_depth=4,
                 num_heads=8,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size

        # ------------------------------------------------------------------
        # 1) patch embedding + positional embeddings
        # ------------------------------------------------------------------
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.enc_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.dec_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # ------------------------------------------------------------------
        # 2) mask token for decoder
        # ------------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ------------------------------------------------------------------
        # 3) encoder: standard ViT blocks
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, heads=num_heads)
            for _ in range(encoder_depth)
        ])

        # ------------------------------------------------------------------
        # 4) decoder: standard ViT blocks
        # ------------------------------------------------------------------
        self.decoder = nn.Sequential(*[
            TransformerBlock(embed_dim, heads=num_heads)
            for _ in range(decoder_depth)
        ])

        # ------------------------------------------------------------------
        # 5) final projection to pixel-space
        # ------------------------------------------------------------------
        self.to_pixels = nn.Linear(embed_dim, patch_size*patch_size*in_chans)


    def forward(self, x):
        B = x.size(0)

        # ---- patchify + flatten ----
        patches = self.patch_embed(x)                       # (B, D, H/P, W/P)
        patches = patches.flatten(2).transpose(1,2)         # (B, N, D)
        patches = patches + self.enc_pos_embed              # add encoder positions

        # ---- 1) generate per-sample random mask ----
        N = self.num_patches
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)           # noise in [0,1]
        ids_shuffle = torch.argsort(noise, dim=1)           # ascend: small = keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(patches, 1, 
                              ids_keep.unsqueeze(-1).repeat(1,1,patches.size(-1)))

        # ---- 2) encode visible patches ----
        encoded = self.encoder(x_keep)                      # (B, L, D)

        # ---- 3) prepare decoder input ----
        # a) unshuffle to original ordering, filling masked slots with mask token
        mask_tokens = self.mask_token.expand(B, N - len_keep, -1)
        x_dec = torch.cat([encoded, mask_tokens], dim=1)    # (B, N, D) but wrong order
        x_dec = torch.gather(x_dec, 1,
                             ids_restore.unsqueeze(-1).repeat(1,1,encoded.size(-1)))
        x_dec = x_dec + self.dec_pos_embed

        # ---- 4) decode ----
        decoded = self.decoder(x_dec)                       # (B, N, D)

        # ---- 5) project to pixels ----
        pixels = self.to_pixels(decoded)                    # (B, N, P*P*C)
        patches_rec = pixels.view(B, N, self.patch_size, self.patch_size, x.shape[1])
        # reshape back to image
        H = W = int(self.num_patches**0.5)
        patches_rec = patches_rec.permute(0,4,1,2,3).contiguous()
        patches_rec = patches_rec.view(B, x.shape[1],
                                       H*self.patch_size,
                                       W*self.patch_size)

        return patches_rec, ids_keep, ids_restore