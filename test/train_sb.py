import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader import (
    AugmentedDataset,
    ExpandedInstanceDataset,
    loadCAMELS,
    split_expanded_dataset_from_json,
)
from src.structures import ConventionalCNN
from src.validator import validate_multi_output_regression


@dataclass
class TrainConfig:
    box: str
    field: str
    sim: str
    individual: bool
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    chunk_size: int
    image_size: int
    label_slice: Optional[Tuple[int, int]]
    label_indices: Optional[List[int]]
    h: int
    output_positive: bool
    loss: str
    scheduler: str
    onecycle_pct_start: float
    onecycle_div_factor: float
    onecycle_final_div_factor: float
    save_dir: Path
    plot_dir: Path
    save_prefix: str
    device: str
    augment: str
    sb35_mode: str
    crop: Optional[str]
    split_mode: str
    split_seed: int
    val_ratio: float
    test_ratio: float


class LogMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = torch.mean((pred - target) ** 2, dim=0)
        return torch.sum(torch.log(mean + self.eps))


def _maybe_float(x: str) -> Optional[float]:
    if x is None:
        return None
    x = str(x).strip()
    if x.lower() in {"none", "null", ""}:
        return None
    return float(x)


def _parse_label_slice(text: str) -> Tuple[int, int]:
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("label slice must look like 'start:end'")
    return int(parts[0]), int(parts[1])


def _parse_label_indices(text: str) -> List[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip() != ""]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid label indices: {e}")


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _infer_chunk_and_size(box: str, sb35_mode: str) -> Tuple[int, int]:
    if box in {"SB28", "SB28_full"}:
        return 15, 256

    if box == "SB35":
        if sb35_mode == "full30":
            return 30, 512
        if sb35_mode == "half15":
            return 15, 512
        if sb35_mode == "cutout15":
            return 15, 256
        raise ValueError(f"Unknown SB35 mode: {sb35_mode}")

    raise ValueError(f"Unsupported box: {box}")


def _apply_sb35_windowing(data: np.ndarray, step: int = 10, length: int = 5) -> np.ndarray:
    # Mirrors the SB35_half / SB35_cutout scripts: pick 5 frames every 10 frames.
    arr = np.arange(data.shape[0])
    starts = np.arange(0, len(arr) - length + 1, step)
    idx = (starts[:, None] + np.arange(length)[None, :]).reshape(-1)
    return data[idx]


def _apply_crop(data: np.ndarray, crop: str) -> np.ndarray:
    # data is [N, H, W]
    h = data.shape[1]
    w = data.shape[2]
    hh = h // 2
    ww = w // 2
    if crop == "tl":
        return data[:, :hh, :ww]
    if crop == "tr":
        return data[:, :hh, ww:]
    if crop == "bl":
        return data[:, hh:, :ww]
    if crop == "br":
        return data[:, hh:, ww:]
    raise ValueError(f"Unknown crop: {crop}")


def _random_split_expanded_dataset(
    data: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Random train/val/test split (shuffled) with specified ratios."""
    from torch.utils.data import Subset

    n_chunks = int(labels.shape[0])
    if int(data.shape[0]) != n_chunks * chunk_size:
        raise ValueError(
            f"data.shape[0]={data.shape[0]} != labels.shape[0]*chunk_size={n_chunks * chunk_size}"
        )

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_chunks, generator=rng)

    n_val = int(n_chunks * val_ratio)
    n_test = int(n_chunks * test_ratio)
    n_train = n_chunks - n_val - n_test

    train_chunks = perm[:n_train]
    val_chunks = perm[n_train : n_train + n_val]
    test_chunks = perm[n_train + n_val :]

    def expand(chunks: torch.Tensor) -> torch.Tensor:
        return (chunks[:, None] * chunk_size + torch.arange(chunk_size)[None, :]).reshape(-1)

    train_idx = expand(train_chunks).tolist()
    val_idx = expand(val_chunks).tolist()
    test_idx = expand(test_chunks).tolist()

    full_ds = ExpandedInstanceDataset(data, labels, chunk_size)
    return Subset(full_ds, train_idx), Subset(full_ds, val_idx), Subset(full_ds, test_idx)


def _maybe_build_transforms(augment: str, image_size: int):
    if augment == "none":
        return None, None

    try:
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError(
            f"augment='{augment}' requires torchvision, but import failed: {e}"
        )

    if augment == "advanced":
        train_tfms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomRotation(degrees=(0, 360)),
                transforms.ToTensor(),
            ]
        )
        val_tfms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        return train_tfms, val_tfms

    raise ValueError(f"Unknown augment: {augment}")


def _build_model(model_name: str, image_size: int, output_dim: int, h: int, output_positive: bool):
    if model_name == "conv":
        return ConventionalCNN(
            input_shape=(image_size, image_size),
            output_shape=output_dim,
            H=h,
            output_positive=output_positive,
        )

    if model_name == "coatnet_3_224":
        try:
            import timm
        except Exception as e:
            raise RuntimeError(f"model='{model_name}' requires timm: {e}")

        base = timm.create_model(
            "coatnet_3_224",
            pretrained=False,
            in_chans=1,
        )
        out_features = 1000
        head = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(32, output_dim),
            nn.Sigmoid() if output_positive else nn.Identity(),
        )
        return nn.Sequential(base, head)

    raise ValueError(f"Unknown model: {model_name}")


def train_one(cfg: TrainConfig, model_name: str) -> None:
    torch.set_default_dtype(torch.float)
    device = torch.device(cfg.device)

    data_np, labels_np, _minmax = loadCAMELS(
        field=cfg.field,
        box=cfg.box,
        normalization=True,
        individual=cfg.individual,
        linear=False,
        buffer=0.3,
        sim=cfg.sim,
    )

    # SB35 special variants
    if cfg.box == "SB35" and cfg.sb35_mode in {"half15", "cutout15"}:
        data_np = _apply_sb35_windowing(data_np)
        if cfg.sb35_mode == "cutout15":
            if cfg.crop is None:
                # Default matches the most common branch in SB35_cutout.py
                cfg.crop = "tl"
            data_np = _apply_crop(data_np, cfg.crop)

    # Labels selection
    labels = labels_np
    if cfg.label_indices is not None and cfg.label_slice is not None:
        raise ValueError("Use only one of --label-indices or --label-slice")
    if cfg.label_indices is not None:
        labels = labels[:, cfg.label_indices]
    elif cfg.label_slice is not None:
        a, b = cfg.label_slice
        labels = labels[:, a:b]

    output_dim = int(labels.shape[1])

    data = torch.tensor(data_np, dtype=torch.float, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float, device=device)

    if cfg.split_mode == "json":
        train_set, val_set, test_set = split_expanded_dataset_from_json(
            data,
            labels_t,
            chunk_size=cfg.chunk_size,
        )
    elif cfg.split_mode == "random":
        train_set, val_set, test_set = _random_split_expanded_dataset(
            data,
            labels_t,
            chunk_size=cfg.chunk_size,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            seed=cfg.split_seed,
        )
    else:
        raise ValueError(f"Unknown split_mode: {cfg.split_mode}")

    train_tfms, val_tfms = _maybe_build_transforms(cfg.augment, cfg.image_size)
    if train_tfms is not None and val_tfms is not None:
        train_ds = AugmentedDataset(train_set, train_tfms)
        val_ds = AugmentedDataset(val_set, val_tfms)
        test_ds = AugmentedDataset(test_set, val_tfms)
    else:
        train_ds, val_ds, test_ds = train_set, val_set, test_set

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = _build_model(model_name, cfg.image_size, output_dim, cfg.h, cfg.output_positive).to(device)

    if cfg.loss == "mse":
        criterion = nn.MSELoss()
    elif cfg.loss == "logmse":
        criterion = LogMSELoss()
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = None
    scheduler_step_on = "none"

    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        scheduler_step_on = "val"
    elif cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-9,
        )
        scheduler_step_on = "epoch"
    elif cfg.scheduler == "onecycle":
        # OneCycleLR is often a very strong default for CNNs. It should be stepped
        # *every batch*.
        steps_per_epoch = max(1, int(np.ceil(len(train_loader.dataset) / cfg.batch_size)))
        max_lr = cfg.lr
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.onecycle_pct_start,
            div_factor=cfg.onecycle_div_factor,
            final_div_factor=cfg.onecycle_final_div_factor,
        )
        scheduler_step_on = "batch"
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    _ensure_dirs(cfg.save_dir, cfg.plot_dir)

    best_val_loss = float("inf")
    best_state = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if scheduler is not None and scheduler_step_on == "batch":
                scheduler.step()
            running += loss.item() * inputs.size(0)

        epoch_train_loss = running / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running += loss.item() * inputs.size(0)

        epoch_val_loss = val_running / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if scheduler is not None:
            if scheduler_step_on == "val":
                scheduler.step(epoch_val_loss)
            elif scheduler_step_on == "epoch":
                scheduler.step(epoch)

        print(
            f"Epoch {epoch}/{cfg.epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, cfg.save_dir / f"{cfg.save_prefix}_best.pt")

        # learning curve
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epoch + 1), train_losses, label="Train Loss")
        plt.plot(range(1, epoch + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / f"{cfg.save_prefix}_learning_curve.png")
        plt.close()

        validate_multi_output_regression(model, val_loader, device=device, max_plots=6)
        plt.savefig(cfg.plot_dir / f"{cfg.save_prefix}_prediction.png")
        plt.close()

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

    # Optional test evaluation using best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, _ = validate_multi_output_regression(model, test_loader, device=device, max_plots=6)
    plt.savefig(cfg.plot_dir / f"{cfg.save_prefix}_test_prediction.png")
    plt.close()
    print(f"Test MSE: {test_mse:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Unified SB28/SB35 trainer (no DeepSpeed)")
    p.add_argument("--box", required=True, choices=["SB28", "SB28_full", "SB35"])
    p.add_argument("--field", default="Mtot")
    p.add_argument("--sim", default="IllustrisTNG")
    p.add_argument("--individual", action="store_true", default=False, help="Use per-sample normalization in loadCAMELS (default: False)")

    p.add_argument("--save-prefix", required=True)
    p.add_argument("--save-dir", default=str(REPO_ROOT / "data" / "models"))
    p.add_argument("--plot-dir", default=str(REPO_ROOT / "test" / "plot"))

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-1)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--label-slice", type=_parse_label_slice, default=None, help="e.g. 6:7")
    p.add_argument("--label-indices", type=_parse_label_indices, default=None, help="e.g. 0,1,6")

    p.add_argument("--model", default="conv", choices=["conv", "coatnet_3_224"])
    p.add_argument("--h", type=int, default=16, help="Conv width multiplier for ConventionalCNN")
    p.add_argument("--output-positive", action="store_true", default=True)
    p.add_argument("--no-output-positive", dest="output_positive", action="store_false")

    p.add_argument("--loss", choices=["mse", "logmse"], default="mse")
    p.add_argument("--scheduler", choices=["cosine", "plateau", "onecycle"], default="cosine")

    # OneCycle tuning knobs (only used when --scheduler onecycle)
    p.add_argument("--onecycle-pct-start", type=float, default=0.1)
    p.add_argument("--onecycle-div-factor", type=float, default=25.0)
    p.add_argument("--onecycle-final-div-factor", type=float, default=1e4)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--augment", choices=["none", "advanced"], default="none")

    p.add_argument(
        "--sb35-mode",
        choices=["full30", "half15", "cutout15"],
        default="full30",
        help="SB35 variants from old scripts",
    )
    p.add_argument("--crop", choices=["tl", "tr", "bl", "br"], default=None)

    # Split options
    p.add_argument(
        "--split-mode",
        choices=["json", "random"],
        default="json",
        help="'json' uses deterministic splits from data/splits/; 'random' shuffles with seed",
    )
    p.add_argument("--split-seed", type=int, default=42, help="Random seed for split_mode=random")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio for split_mode=random")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio for split_mode=random")

    args = p.parse_args()

    chunk_size, image_size = _infer_chunk_and_size(args.box, args.sb35_mode)

    cfg = TrainConfig(
        box=args.box,
        field=args.field,
        sim=args.sim,
        individual=args.individual,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        chunk_size=chunk_size,
        image_size=image_size,
        label_slice=args.label_slice,
        label_indices=args.label_indices,
        h=args.h,
        output_positive=args.output_positive,
        loss=args.loss,
        scheduler=args.scheduler,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
        save_dir=Path(args.save_dir),
        plot_dir=Path(args.plot_dir),
        save_prefix=args.save_prefix,
        device=args.device,
        augment=args.augment,
        sb35_mode=args.sb35_mode,
        crop=args.crop,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_one(cfg, model_name=args.model)


if __name__ == "__main__":
    main()
