import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
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
dtype = torch.float
torch.set_default_dtype(dtype)
torch.autograd.set_detect_anomaly(True)
# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import os
os.environ["PATH"] += os.pathsep + "~/pyenv/torch/bin/"

#import ninja
#import deepspeed
import random
sys.path.append("/mnt/home/yjo10/ceph/myutils")
import plt_utils as pu
import copy



save_dir = "../data/models/"
#save_prefix = "SB28_conv_no_linear_om"
#save_prefix = "SB28_conv_no_linear_sig_no_monopole_2"
#save_prefix = "SB28_conv_no_linear_all"
#save_prefix = "SB28_conv_no_linear_sn1_Mgas"
#save_prefix = "SB28_conv_no_linear_sn1_Mstar"
#save_prefix = "SB28_conv_no_linear_5cosmo"
#save_prefix = "SB28_conv_no_linear_ns"
#save_prefix = "SB28_conv_no_linear_H"
#save_prefix = "SB28_conv_no_linear_sig_2"
#save_prefix = "SB28_full_conv_no_linear_ob"
save_prefix = "SB28_full_conv_no_linear_om_2"

## Dataload
batch_size=64
#data, labels,minmax = loadCAMELS(field="Mstar",box="SB28",normalization=True,
data, labels, minmax = loadCAMELS(field="Mtot",box="SB28_full",normalization=True,
                                 individual=False,
                                 linear=False, buffer=0.3)
print(data.shape)
## ob (6) H (7) ns (8)
#labels = labels[:,1:2]
#labels = labels[:,2:3]
#labels = labels[:,2:3]
#labels = labels[:,[0,1,6,7,8]]
#labels = labels[:,1:2]
labels = labels[:,0:1]
output_dim = labels.shape[1]


DeepSpeed = False

data = torch.tensor(data,dtype=dtype).to(device)
labels = torch.tensor(labels,dtype=dtype).to(device)
train_set, val_set = split_expanded_dataset(data, labels, chunk_size=15, val_ratio=0.2)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)


## Training
num_epochs = 500


if DeepSpeed:
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
                input_dim = 256,
                input_channels=1,
                #conv_layers=[(4,3), (8,3), (16, 3), (32, 3)],  # list of (out_channels, kernel_size)
                #conv_layers=[(2,3),(4,3), (8,3), (16, 3), (32, 3),(64,3)],  # list of (out_channels, kernel_size)
                conv_layers=[(2,3), (8,3), (32, 3), (64, 3)],  # list of (out_channels, kernel_size)
                hidden_dims=[128, 32],
                output_dim=output_dim,
                output_positive=True,
                activation=nn.LeakyReLU).to(device)
                """



#model = ConventionalCNN(input_shape=(256,256), output_shape=output_dim, H=32, output_positive=True)
model = ConventionalCNN(input_shape=(256,256), output_shape=output_dim, H=16, output_positive=True)






# Load the saved state dict
#model.load_state_dict(torch.load("../data/models/SB28_epoch100.pt"))

model = model.to(device)
#optimizer = optim.AdamW(model.parameters(), lr=lr)
#criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
scheduler = CosineAnnealingWarmRestarts(
    optimizer=optimizer,
    T_0=10,     # first cycle length (in epochs)
    T_mult=2,  # cycle length multiplier
    eta_min=1e-9
)

if DeepSpeed:
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config_params=ds_config
    )




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

criterion = nn.MSELoss()
#criterion = LogMSELoss()


os.makedirs(save_dir, exist_ok=True)

if DeepSpeed:
    best_model_wts = copy.deepcopy(model_engine.module.state_dict())
else:
    best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    if DeepSpeed:
        model_engine.train()
    else:
        model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if DeepSpeed:
            outputs = model_engine(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        if DeepSpeed:
            model_engine.backward(loss, retain_graph=True)
            model_engine.step()
        else:
            loss.backward(loss, retain_graph=True)
            optimizer.step()

        #loss.backward()
        #optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    if DeepSpeed:
        model_engine.eval()
    else:
        model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if DeepSpeed:
                outputs = model_engine(inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    scheduler.step(epoch_val_loss)
    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']}")


    print(f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Save best model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        if DeepSpeed:
            best_model_wts = copy.deepcopy(model_engine.module.state_dict())
            if model_engine.global_rank == 0:
                torch.save(best_model_wts, os.path.join(save_dir, f"{save_prefix}_best.pt"))
        else:
            best_model_wts = copy.deepcopy(model.state_dict())
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

    if DeepSpeed:
        validate_multi_output_regression(model_engine, val_loader, device=device, max_plots=6)
    else:
        validate_multi_output_regression(model, val_loader, device=device, max_plots=6)
    plt.savefig(f"plot/{save_prefix}_prediction.png")

print("Training complete. Best Val Loss: {:.4f}".format(best_val_loss))
