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
dtype = torch.float
torch.set_default_dtype(dtype)
torch.autograd.set_detect_anomaly(True)
# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import os
os.environ["PATH"] += os.pathsep + "~/pyenv/torch/bin/"

import ninja
import deepspeed
import random
sys.path.append("/mnt/home/yjo10/ceph/myutils")
import plt_utils as pu
import copy
# Cosine‑annealing with warmup
import timm
from timm.scheduler import create_scheduler
from types import SimpleNamespace
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage


def setup():
    dist.init_process_group(backend='nccl')

def cleanup():
    dist.destroy_process_group()

def main(num_workers=8):
    setup()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()


    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    ## Dataload
    batch_size=32
    data, labels,minmax = loadCAMELS(field="Mtot",box="SB28",normalization=True)
    labels = labels[:,0:2]
    output_dim = labels.shape[1]
    #data = data[:,:224,:224]



    # 1. Your grayscale augmentation pipeline
    train_tfms = transforms.Compose([
        transforms.ToPILImage(), 
        #transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(degrees=(-30, 30)),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4),
        #ToPILImage(),                 # convert raw tensor → PIL
        transforms.ToTensor(),        # PIL → normalized tensor
        #transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        #transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    # 2. A simpler “val” pipeline (no random transforms)
    val_tfms = transforms.Compose([
        #ToPILImage(),
        transforms.ToPILImage(), 
        #transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    # 3. Load data
    data = torch.tensor(data,dtype=dtype)
    labels = torch.tensor(labels,dtype=dtype)
    train_set, val_set = split_expanded_dataset(data, labels, chunk_size=15, val_ratio=0.2)

    # 4. Wrap your existing train_set / val_set
    train_aug = AugmentedDataset(train_set, train_tfms)
    val_aug   = AugmentedDataset(val_set,   val_tfms)

    # 5. Create loaders as before
    #train_loader = DataLoader(train_aug, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    #val_loader   = DataLoader(val_aug,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    # 5. Setup samplers (important for DDP)
    train_sampler = DistributedSampler(train_aug, shuffle=True)
    val_sampler   = DistributedSampler(val_aug, shuffle=False)


    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_aug,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # shuffle only if not distributed
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True,
        persistent_workers=True
    )

    ## Training
    epochs = 400
    save_dir = "../data/models/"
    save_prefix = "SB28_CoaT_resize_new"

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



    # Load the saved state dict
    #model.load_state_dict(torch.load("../data/models/SB28_epoch100.pt"))


    # create CoaT with 1 input channel, 1 output
    model = timm.create_model(
        'coatnet_3_224',
        #'coat_lite_small',      # or any other 'coat_*' variant
        pretrained=False,
        in_chans=1,             # single‐channel input
        #num_classes=2           # single regression output
    )
    out_features = 1000 #model.head.out_features
    # OPTIONAL: wrap the output in a tanh to bound it
    model = nn.Sequential(
        model,
        nn.Linear(out_features, 256),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.Linear(256, 32),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.Linear(32, output_dim),
        nn.Sigmoid()
    )




    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    #optimizer = optim.AdamW(model.parameters(), lr=lr)

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



    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)

    #epochs = 100
    # build a simple args‐like object
    args = SimpleNamespace(
        sched='cosine',      # schedule type
        lr=1e-3,             # base learning rate
        epochs=epochs,          # total epochs
        warmup_lr=1e-4,      # learning rate to start warmup from
        warmup_epochs=5,     # how many epochs to warm up
        min_lr=1e-7          # (optional) final min LR
    )
    scheduler, _ = create_scheduler(
        args,
        optimizer=optimizer
        )

    """
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=10,     # first cycle length (in epochs)
        T_mult=2,  # cycle length multiplier
        eta_min=1e-7
    )
    """

    """
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    """

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config_params=ds_config
    )


    os.makedirs(save_dir, exist_ok=True)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []




    for epoch in range(1, epochs + 1):
        #train_loader.sampler.set_epoch(epoch)  # very important!
        train_sampler.set_epoch(epoch)  # very important!
        #model_engine.train()
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs) #model_engine(inputs)
            loss = criterion(outputs, targets)

            #model_engine.backward(loss, retain_graph=True)
            #model_engine.step()

            #model.backward(loss, retain_graph=True)
            #model.step()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        #model_engine.eval()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                #inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs) #model_engine(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        #scheduler.step(epoch_val_loss)
        scheduler.step(epoch)
        #scheduler.step()

        # Save model for this epoch
        #model_path = os.path.join(save_dir, f"{save_prefix}_epoch{epoch}.pt")
        #if model_engine.global_rank == 0:
        #    torch.save(model_engine.module.state_dict(), model_path)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.module.state_dict())
            if model.global_rank == 0:
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

        validate_multi_output_regression(model, val_loader, device=device, max_plots=6)
        plt.savefig(f"plot/{save_prefix}_prediction.png")

    print("Training complete. Best Val Loss: {:.4f}".format(best_val_loss))
    cleanup()


    
if __name__ == "__main__":
    num_workers = 12
    main(num_workers)
