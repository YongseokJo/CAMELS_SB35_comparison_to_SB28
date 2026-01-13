import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ScaledTanh(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1)


# Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                           # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 14x14 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class CustomCNN(nn.Module):
    def __init__(self,
                input_dim = 256,
                input_channels=1,
                conv_layers=[(16, 3), (32, 3)],  # list of (out_channels, kernel_size)
                hidden_dims=[128],
                output_dim=10,
                activation=nn.ReLU,
                output_positive=False):
        super(CustomCNN, self).__init__()
        

        self.output_positive = output_positive
        self.activation_fn = activation()

        # Convolutional layers
        layers = []
        in_channels = input_channels
        for out_channels, kernel_size in conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(self.activation_fn)
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        # Dummy input to compute output shape after conv layers
        dummy_input = torch.zeros(1, input_channels, input_dim, input_dim)  # assuming 28x28 input like MNIST
        with torch.no_grad():
            conv_out = self.conv(dummy_input)
        conv_out_flat_dim = conv_out.view(1, -1).size(1)

        # Fully connected layers
        fc_layers = []
        prev_dim = conv_out_flat_dim
        for hidden_dim in hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(self.activation_fn)
            prev_dim = hidden_dim
        fc_layers.append(nn.Linear(prev_dim, output_dim))

        if output_positive:
            fc_layers.append(ScaledTanh())
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        """

        
        
import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self,
                 input_dim=256,
                 input_channels=1,
                 conv_layers=[(16, 3), (32, 3)],  # list of (out_channels, kernel_size)
                 hidden_dims=[128],
                 output_dim=10,
                 activation=nn.ReLU,
                 output_positive=False):
        super(CustomCNN, self).__init__()

        self.output_positive = output_positive
        act = activation  # class, not instance

        # Convolutional layers with BatchNorm2d
        conv_modules = []
        in_channels = input_channels
        for out_channels, kernel_size in conv_layers:
            conv_modules += [
                nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                act()
                #nn.MaxPool2d(2)
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_modules)

        # Dummy input to compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_dim, input_dim)
            conv_out = self.conv(dummy)
        conv_out_flat_dim = conv_out.view(1, -1).size(1)

        # Fully connected layers with BatchNorm1d
        fc_modules = []
        prev_dim = conv_out_flat_dim
        for hidden_dim in hidden_dims:
            fc_modules += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                act(),
                nn.Dropout()
            ]
            prev_dim = hidden_dim

        # Final output layer
        fc_modules.append(nn.Linear(prev_dim, output_dim))
        if output_positive:
            #fc_modules.append(ScaledTanh())
            fc_modules.append(nn.Sigmoid())

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class ConventionalCNN(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape,H=1,stride=1, output_positive=True, input_channels=1):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(ConventionalCNN,self).__init__()
        self.output_positive = output_positive
        self.input_channels = input_channels
        
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.input_dim = self.input_shape[0]
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=int(2*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(2*H), out_channels=int(2*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(2*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(2*H), out_channels=int(2*H),
                                kernel_size=2,stride=2,padding=0),
                        nn.BatchNorm2d(int(2*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(2*H), out_channels=int(4*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(4*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(4*H), out_channels=int(4*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(4*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(4*H), out_channels=int(4*H),
                                kernel_size=2,stride=2,padding=0),
                        nn.BatchNorm2d(int(4*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(4*H), out_channels=int(8*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(8*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(8*H), out_channels=int(8*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(8*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(8*H), out_channels=int(8*H),
                                kernel_size=2,stride=2,padding=0),
                        nn.BatchNorm2d(int(8*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(8*H), out_channels=int(16*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(16*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(16*H), out_channels=int(16*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(16*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(16*H), out_channels=int(16*H),
                                kernel_size=2,stride=2,padding=0),
                        nn.BatchNorm2d(int(16*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(16*H), out_channels=int(32*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(32*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(32*H), out_channels=int(32*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(32*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(32*H), out_channels=int(32*H),
                                kernel_size=2,stride=2,padding=0),
                        nn.BatchNorm2d(int(32*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(32*H), out_channels=int(64*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(64*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(64*H), out_channels=int(64*H),
                                kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(int(64*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(64*H), out_channels=int(64*H),
                                kernel_size=2,stride=2,padding=0),
                        nn.BatchNorm2d(int(64*H)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=int(64*H), out_channels=int(128*H),
                                #kernel_size=1,stride=2,padding=0),
                                kernel_size=4,stride=2,padding=0),
                        nn.BatchNorm2d(int(128*H)),
                        nn.LeakyReLU(),
                        )

        """
        # Dummy input to compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.input_dim, self.input_dim)
            conv_out = self.conv(dummy)
        conv_out_flat_dim = conv_out.view(1, -1).size(1)

        self.fcl = nn.Sequential(nn.Flatten(),
                            nn.Dropout(),
                            nn.Linear(conv_out_flat_dim,int(conv_out_flat_dim/4)),
                            nn.LeakyReLU(),
                            nn.Dropout(),
                            nn.Linear(int(conv_out_flat_dim/4),output_shape)
                        )
        """
        if self.input_dim == 512:
            self.conv7 = nn.Sequential(nn.Conv2d(in_channels=int(128*H), out_channels=int(256*H),
                                            kernel_size=3,stride=2,padding=1),
                                nn.BatchNorm2d(int(256*H)),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=int(256*H), out_channels=int(256*H),
                                            kernel_size=3,stride=2,padding=1),
                                nn.BatchNorm2d(int(256*H)),
                                nn.LeakyReLU())
            self.fcl = nn.Sequential(nn.Flatten(),
                                nn.Dropout(),
                                nn.Linear(int(256*H),int(64*H)),
                                nn.LeakyReLU(),
                                nn.Dropout(),
                                nn.Linear(int(64*H),output_shape)
                            )
        if self.input_dim == 256:
            self.fcl = nn.Sequential(nn.Flatten(),
                                nn.Dropout(),
                                nn.Linear(int(128*H),int(64*H)),
                                nn.LeakyReLU(),
                                nn.Dropout(),
                                nn.Linear(int(64*H),output_shape)
                            )
        if self.output_positive:
            self.fcl_scale = nn.Sigmoid() 
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        
    def forward(self, x):
        x = self.conv(x)
        if self.input_dim == 512:
            x = self.conv7(x)
        x = self.fcl(x)
        if self.output_positive:
            x = self.fcl_scale(x)
        return x
    


#import timm



class FlexibleCoaT(nn.Module):
    def __init__(self, img_size=256, in_chans=1, output_dim=2):
        super().__init__()

        # Load base CoaT model
        self.backbone = coat_lite_small(pretrained=False)

        # 🔧 Change first stage's patch_embed to accept 1 input channel
        self.backbone.patch_embeds[0].proj = nn.Conv2d(
            in_chans,
            self.backbone.embed_dims[0],
            kernel_size=7,
            stride=4,
            padding=3
        )

        # 🔨 Replace classifier head with scaled tanh regression head
        final_dim = self.backbone.embed_dims[-1]
        self.backbone.head = nn.Sequential(
            nn.Linear(final_dim, output_dim),
            ScaledTanh()
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.backbone.head(x)
        return x




class FlexibleCNN(nn.Module):
    def __init__(self, H=16, input_channels=1, input_dim=25, output_dim=2, output_positive=True):
        super().__init__()
        self.H = H
        self.input_channels = input_channels

        self.conv = nn.Sequential(
            *self._make_block(input_channels, 2 * H),
            *self._make_block(2 * H, 2 * H, downsample=True),
            *self._make_block(2 * H, 4 * H),
            *self._make_block(4 * H, 4 * H, downsample=True),
            *self._make_block(4 * H, 8 * H),
            *self._make_block(8 * H, 8 * H, downsample=True),
            *self._make_block(8 * H, 16 * H),
            *self._make_block(16 * H, 16 * H, downsample=True),
            *self._make_block(16 * H, 32 * H),
            *self._make_block(32 * H, 32 * H, downsample=True),
            *self._make_block(32 * H, 64 * H),
            *self._make_block(64 * H, 64 * H, downsample=True),
            nn.Conv2d(64 * H, 128 * H, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
            #nn.Conv2d(64 * H, 128 * H, kernel_size=1, stride=1, padding=0),
            #nn.LeakyReLU()
        )

        # Dummy input to compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_dim, input_dim)
            conv_out = self.conv(dummy)
        conv_out_flat_dim = conv_out.view(1, -1).size(1)

        fcl = [nn.Flatten(),
                    nn.Dropout(),
                    nn.Linear(conv_out_flat_dim,int(conv_out_flat_dim/4)),
                    nn.LeakyReLU(),
                    nn.Dropout(),
                    nn.Linear(int(conv_out_flat_dim/4),output_dim)]
                        
        if output_positive:
            fcl.append(nn.Sigmoid())

        self.fcl = nn.Sequential(*fcl)


    def _make_block(self, in_channels, out_channels, downsample=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        ]
        if downsample:
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            ]
        return layers

    def forward(self, x):
        x = self.conv(x)
        return self.fcl(x)
