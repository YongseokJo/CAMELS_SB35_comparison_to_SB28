import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../src/")
from structures import *
from dataloader import *
from validator import *
#import losses
#importlib.reload(losses)
#from losses import *
#import trainer
#importlib.reload(trainer)
#from trainer import *
#dtype = torch.float32
#dtype = torch.float
#torch.cuda.is_bf16_supported()
#dtype = torch.bfloat16
dtype = torch.float
torch.set_default_dtype(dtype)
torch.autograd.set_detect_anomaly(True)
# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import deepspeed
import random
sys.path.append("/mnt/home/yjo10/ceph/myutils")
import plt_utils as pu
import copy
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torchvision import transforms
from torchvision.transforms import ToPILImage


## Dataload
import dataloader
batch_size=64
data, labels,minmax = loadCAMELS(field="Mtot",box="SB35",normalization=True)
labels = labels[:,:2]
output_dim = labels.shape[1]



data = torch.tensor(data,dtype=dtype)
labels = torch.tensor(labels,dtype=dtype)
train_set, val_set = split_expanded_dataset(data, labels, chunk_size=30, val_ratio=0.2)

#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

angles = [0, 90, 180, 270]
fixed_rotations = [
    transforms.RandomRotation((angle, angle)) for angle in angles
]

# 1. Your grayscale augmentation pipeline
train_tfms = transforms.Compose([
    transforms.ToPILImage(), 
    #transforms.RandomResizedCrop(256),# scale=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.2),
    #transforms.RandomChoice(fixed_rotations),

    transforms.RandomRotation(degrees=(0, 360)),
    #transforms.ColorJitter(brightness=0.4, contrast=0.4),
    #ToPILImage(),                 # convert raw tensor → PIL
    transforms.ToTensor(),        # PIL → normalized tensor
    #transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    #RotateBySet([0, 90, 180, 270]),  # whatever angles you like
    #transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# 2. A simpler “val” pipeline (no random transforms)
val_tfms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# 3. Wrap your existing train_set / val_set
train_aug = AugmentedDataset(train_set, train_tfms)
val_aug   = AugmentedDataset(val_set,   val_tfms)

# 4. Create loaders as before
train_loader = DataLoader(train_aug, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_aug,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# 5. Quick sanity check on one batch
imgs, labs = next(iter(train_loader))
print(imgs.shape, labs.shape)  # e.g. torch.Size([batch,1,224,224]), torch.Size([batch])
#imgs, labs = next(iter(val_loader))
#print(imgs.shape, labs.shape)  # e.g. torch.Size([batch,1,224,224]), torch.Size([batch])



## Training
lr = 1e-3
num_epochs = 200
save_dir = "../data/models/"
save_prefix = "SB35_adv"

ds_config = {
    "train_batch_size": batch_size,
    "gradient_accumulation_steps": 1,
    "fp32": {
        "enabled": True  # Enables mixed precision training
    },
    "zero_optimization": {
        "stage": 3  # Enable ZeRO Stage 1 for memory optimization
    }
}

"""
model = CustomCNN(
                input_dim = 512,
                input_channels=1,
                conv_layers=[(4,3), (8,3), (16, 3)],#, (32, 3)],  # list of (out_channels, kernel_size)
                hidden_dims=[128,32],
                output_dim=output_dim,
                output_positive=True,
                activation=nn.LeakyReLU).to(device)
                """

model = ConventionalCNN(input_shape=(512,512), output_shape=output_dim, H=8)

model = model.to(device)

class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(LogMSELoss, self).__init__()
        self.eps = eps  # to prevent log(0)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mean = torch.mean((pred-target)**2,dim=0)
        lmse = torch.sum(torch.log(mean))
        return lmse
        """
        pred_log = torch.log(pred)
        target_log = torch.log(target)
        return torch.mean((pred_log - target_log) ** 2)
        """

#criterion = nn.MSELoss()
criterion = LogMSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
scheduler = CosineAnnealingWarmRestarts(
    optimizer=optimizer,
    T_0=10,     # first cycle length (in epochs)
    T_mult=2,  # cycle length multiplier
    eta_min=1e-7
)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,
    config_params=ds_config
)

os.makedirs(save_dir, exist_ok=True)

best_model_wts = copy.deepcopy(model_engine.module.state_dict())
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    model_engine.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model_engine(inputs)
        loss = criterion(outputs, targets)

        model_engine.backward(loss, retain_graph=True)
        model_engine.step()
        #loss.backward()
        #optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    model_engine.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model_engine(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    scheduler.step(epoch_val_loss)

    # Save model for this epoch
    #model_path = os.path.join(save_dir, f"{save_prefix}_epoch{epoch}.pt")
    #if model_engine.global_rank == 0:
    #    torch.save(model_engine.module.state_dict(), model_path)

    print(f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Save best model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        if model_engine.global_rank == 0:
            best_model_wts = copy.deepcopy(model_engine.module.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, f"{save_prefix}_best.pt"))

    # Plot learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plot/{save_prefix}_learning_curve.png")
    plt.close()
    #plt.show()

    validate_multi_output_regression(model_engine, val_loader, device=device, max_plots=6)
    plt.savefig(f"plot/{save_prefix}_prediction.png")
    

print("Training complete. Best Val Loss: {:.4f}".format(best_val_loss))


