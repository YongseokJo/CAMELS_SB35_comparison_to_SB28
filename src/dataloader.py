import numpy as np
import pandas as pd
import torch
import os
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from torch.utils.data import Dataset, DataLoader, Subset, random_split, TensorDataset
#from torchvision.transforms import ToPILImage

try:
    import random
except ImportError:  # pragma: no cover
    random = None

try:
    from torchvision.transforms import functional as F
except Exception:  # pragma: no cover
    F = None


def _default_splits_dir() -> Path:
    # repo_root/src/dataloader.py -> repo_root/data/splits
    return Path(__file__).resolve().parents[1] / "data" / "splits"


def load_splits_json(
    n_samples: int,
    splits_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, List[int]]:
    """Load deterministic train/val/test indices for a given dataset size.

    Expects files like:
      - data/splits/splits_1024.json
      - data/splits/splits_2048.json
    """
    splits_dir_path = Path(splits_dir) if splits_dir is not None else _default_splits_dir()
    split_path = splits_dir_path / f"splits_{n_samples}.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Cannot find split file for n_samples={n_samples}: {split_path}. "
            f"Available: {sorted(p.name for p in splits_dir_path.glob('splits_*.json'))}"
        )

    with split_path.open("r") as f:
        splits = json.load(f)

    for key in ("train", "val", "test"):
        if key not in splits:
            raise ValueError(f"Split file {split_path} missing required key '{key}'")

    meta = splits.get("metadata", {})
    if isinstance(meta, dict) and "n_samples" in meta and int(meta["n_samples"]) != int(n_samples):
        raise ValueError(
            f"Split metadata mismatch in {split_path}: metadata.n_samples={meta['n_samples']} != {n_samples}"
        )

    def _validate(name: str, idxs: Sequence[int]) -> List[int]:
        out = [int(i) for i in idxs]
        if len(out) == 0:
            raise ValueError(f"Split '{name}' in {split_path} is empty")
        if min(out) < 0 or max(out) >= n_samples:
            raise ValueError(
                f"Split '{name}' indices out of range for n_samples={n_samples} in {split_path}: "
                f"min={min(out)} max={max(out)}"
            )
        return out

    train = _validate("train", splits["train"])
    val = _validate("val", splits["val"])
    test = _validate("test", splits["test"])

    # Basic overlap check
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError(f"Splits in {split_path} overlap (train/val/test must be disjoint)")

    return {"train": train, "val": val, "test": test}


def split_dataset_from_splits(
    dataset: Dataset,
    splits: Dict[str, Sequence[int]],
) -> Tuple[Subset, Subset, Subset]:
    train_set = Subset(dataset, list(map(int, splits["train"])))
    val_set = Subset(dataset, list(map(int, splits["val"])))
    test_set = Subset(dataset, list(map(int, splits["test"])))
    return train_set, val_set, test_set


def _expand_chunk_indices(chunk_indices: Sequence[int], chunk_size: int) -> torch.LongTensor:
    idx = torch.as_tensor(list(map(int, chunk_indices)), dtype=torch.long)
    # Each chunk index corresponds to a contiguous block of frames.
    starts = idx * chunk_size
    offsets = torch.arange(chunk_size, dtype=torch.long)
    return (starts[:, None] + offsets[None, :]).reshape(-1)


def split_expanded_dataset_from_json(
    data: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 15,
    splits_dir: Optional[Union[str, Path]] = None,
    monopole: Optional[torch.Tensor] = None,
):
    """Deterministic train/val/test split for ExpandedInstanceDataset.

    - Uses labels.shape[0] (number of cosmology/astro parameter sets) to select split file.
    - Interprets split indices as *chunk indices* (0..N-1), then expands to frame indices.
    """
    n_chunks = int(labels.shape[0])
    if int(data.shape[0]) != n_chunks * int(chunk_size):
        raise ValueError(
            f"Expanded dataset expects data.shape[0] == labels.shape[0] * chunk_size; "
            f"got data.shape[0]={data.shape[0]} labels.shape[0]={labels.shape[0]} chunk_size={chunk_size}"
        )

    splits = load_splits_json(n_chunks, splits_dir=splits_dir)

    train_indices = _expand_chunk_indices(splits["train"], chunk_size)
    val_indices = _expand_chunk_indices(splits["val"], chunk_size)
    test_indices = _expand_chunk_indices(splits["test"], chunk_size)

    full_dataset = ExpandedInstanceDataset(data, labels, chunk_size)
    train_set = Subset(full_dataset, train_indices.tolist())
    val_set = Subset(full_dataset, val_indices.tolist())
    test_set = Subset(full_dataset, test_indices.tolist())

    if monopole is not None:
        return (
            train_set,
            val_set,
            test_set,
            monopole[train_indices],
            monopole[val_indices],
            monopole[test_indices],
        )

    return train_set, val_set, test_set



def loadCAMELS(field="Mtot", box="SB28", normalization=True, 
                individual=False, linear=False, buffer=0.3, sim="IllustrisTNG", use_mask=False):
    if box not in ["SB28", "SB35", "LH", "1P", "CV", "AREPO-SIMBA", "SB28_full"]:
        print("\'box\' should be either LH, SB28 for 25 Mpc/h or SB35 for 50 Mpc/h.")
        raise

    if box in {"LH", "1P", "CV"}:
        base_path = f"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/{sim}/"
        path = base_path+f"Maps_{field}_{sim}_{box}_z=0.00.npy"
    if box == "AREPO-SIMBA":
        #base_path = f"/mnt/ceph/users/fgarcia/data_products/simba_test_latest"
        #path = base_path+f"Images_{field}_simba_test_bh_fgas_1_032_z=0.05.npy"
        base_path = "/mnt/ceph/users/fgarcia/data_products/simba_test_latest_temp_maps_z0"
        path = base_path + "gas_temperature_033.npy"

    if box == "SB28_full":
        base_path = f"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/{sim}/"
        path = base_path+f"Maps_{field}_{sim}_SB28_z=0.00.npy"

    if box == "SB28":
        base_path = f"/mnt/ceph/users/camels/Results/images_{sim}_{box}/"
        path = base_path+f"Images_{field}_{sim}_{box}_z=0.00.npy"
    if box == "SB35":
        base_path = f"/mnt/ceph/users/camels/Results/images_IllustrisTNG_{box}/"
        path = base_path+f"Maps_{field}_IllustrisTNG_{box}_z=0.00.npy"
    if box == "AREPO-SIMBA":
        base_path = f"/mnt/ceph/users/fgarcia/data_products/simba_test_latest_temp_maps"
        path = base_path+f"Maps_{field}_IllustrisTNG_{box}_z=0.00.npy"
    if not os.path.isfile(path):
        print(f"Cannot find {path}!")
        raise

    data = np.load(path)
    if box in {"LH", "1P", "CV", "SB28_full"}:
        if box == "SB28_full":
            path = base_path+f"params_SB28_{sim}.txt"
        else:
            path = base_path+f"params_{box}_{sim}.txt"
        with open(path) as f:
            num_cols = len(f.readline().strip().split())
        params = np.loadtxt(path, usecols=range(0,num_cols))
    if box == "LH":
        path = f"/mnt/ceph/users/camels/Parameters/{sim}/CosmoAstroSeed_{sim}_L25n256_LH.txt"
        with open(path) as f:
            num_cols = len(f.readline().strip().split())
        params = np.loadtxt(path, usecols=range(1,num_cols))
        #print(params)
    if box == "SB28":
        path = base_path+f"CosmoAstroSeed_{box}.txt"
        with open(path) as f:
            num_cols = len(f.readline().strip().split())
        params = np.loadtxt(path, usecols=range(1,num_cols))
    if box == "SB35":
        path = base_path + "params_SB35_IllustrisTNG.txt"
        params = np.loadtxt(path)

    #df = pd.read_csv("/mnt/ceph/users/camels/Sims/IllustrisTNG/SB28/SB28_param_minmax.csv")
    #minmax = np.c_[df['MinVal'].to_numpy(),df['MaxVal'].to_numpy()] ## (28,2) min, max

    if normalization:    

        data_norm = np.empty_like(data, dtype=data.dtype)
        if use_mask:
            mask = (data != 0)
            data = data[mask]

        if linear:
            data = np.log10(data)
            data_norm[mask] = (data-data.max())/(data.max()-data.min())
            data_norm[~mask] = data_norm[mask].min()
        elif individual:
            sums = data.sum(axis=(1, 2), keepdims=True) # shape = (N,1,1)
            data = data/sums
            data = np.log10(data)
            data_norm[mask] = (data-data.mean())/data.std()
            data_norm[~mask] = data_norm[mask].min()

            """
            # old method
            means = data.mean(axis=(1, 2), keepdims=True)     # shape = (N,1,1)
            stds = data.std(axis=(1, 2), ddof=0, keepdims=True)  # shape = (N,1,1)
            data = (data-means)/stds
            """
        else:
            data = np.log10(data)
            #data = (data-data.mean())/data.std()
            if use_mask:
                data_norm[mask] = (data-data.mean())/data.std()
                data_norm[~mask] = data_norm[mask].min()
            else:
                data_norm = (data-data.mean())/data.std()
        data = data_norm

        _min = params.min(axis=0)
        _max = params.max(axis=0)
        _min -= (_max-_min)*(1-buffer)
        _max += (_max-_min)*buffer
        params = (params-_min)/(_max-_min)
        #params = (params-minmax[:,0])/(minmax[:,1]-minmax[:,0])
    else:
        _min = params.min(axis=0)
        _max = params.max(axis=0)
        _min -= (_max-_min)*(1-buffer)
        _max += (_max-_min)*buffer
    return data, params, np.c_[_min,_max]

    

class ExpandedInstanceDataset(Dataset):
    def __init__(self, data, labels, chunk_size=15):
        assert data.shape[0] % chunk_size == 0
        N = data.shape[0] // chunk_size
        assert labels.shape[0] == N

        self.chunk_size = chunk_size
        self.data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])  
        self.labels = labels.repeat_interleave(chunk_size, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def split_expanded_dataset(data, labels, chunk_size=15, val_ratio=0.2, shuffle=True, monopole=None):
    # Backwards-compatible random train/val split (no test set).
    N = labels.shape[0]  # number of chunks
    num_val = int(N * val_ratio)
    num_train = N - num_val

    # Chunk indices: [0, 1, ..., N-1]
    torch.manual_seed(42)  # reproducibility
    if shuffle:
        all_indices = torch.randperm(N)
    else:
        all_indices = torch.arange(N)
    train_chunks = all_indices[:num_train]
    val_chunks = all_indices[num_train:]

    # Convert chunk indices to frame indices
    def expand_indices(chunks):
        return torch.cat([torch.arange(c * chunk_size, (c + 1) * chunk_size) for c in chunks])

    train_indices = expand_indices(train_chunks)
    val_indices = expand_indices(val_chunks)

    if monopole is not None:
        monopole_train = monopole[train_indices]
        monopole_valid = monopole[val_indices]

        full_dataset = ExpandedInstanceDataset(data, labels, chunk_size)
        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)

        return train_set, val_set, monopole_train, monopole_valid
    else:
        full_dataset = ExpandedInstanceDataset(data, labels, chunk_size)
        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)

        return train_set, val_set


class ChunkedImageDataset(Dataset):
    def __init__(self, data, labels=None, chunk_size=15):
        assert len(data) % chunk_size == 0, "Total data must be divisible by chunk size"
        self.chunk_size = chunk_size
        self.num_chunks = len(data) // chunk_size
        # data is expected to be [N_frames, H, W] (or [N_frames, 1, H, W])
        if data.ndim == 3:
            h, w = int(data.shape[1]), int(data.shape[2])
            self.data = data.view(self.num_chunks, chunk_size, h, w)
        elif data.ndim == 4:
            c, h, w = int(data.shape[1]), int(data.shape[2]), int(data.shape[3])
            self.data = data.view(self.num_chunks, chunk_size, c, h, w)
        else:
            raise ValueError(f"Unsupported data shape for ChunkedImageDataset: {tuple(data.shape)}")

        if labels is not None:
            assert len(labels) == self.num_chunks, "Labels must match number of chunks"
            self.labels = labels
        else:
            self.labels = None

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]
    
def create_chunked_loaders(data, labels=None, val_ratio=0.2, batch_size=4, chunk_size=15):
    dataset = ChunkedImageDataset(data, labels, chunk_size=chunk_size)
    
    N = len(dataset)
    N_val = int(val_ratio * N)
    N_train = N - N_val

    print(N, N_train, N_val)
    torch.manual_seed(42)
    train_set, val_set = random_split(dataset, [N_train, N_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



        
"""
class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base      = base_dataset
        self.transform = transform
        self.to_pil    = ToPILImage()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]   # x might be a torch.Tensor *or* already a PIL.Image

        if isinstance(x, torch.Tensor):
            # squeeze out the channel dim if it’s 1×H×W
            if x.ndim == 3 and x.size(0) == 1:
                x = x.squeeze(0)
            img = self.to_pil(x)     # now we know x was a Tensor
        elif isinstance(x, Image.Image):
            img = x                 # already PIL, don’t convert again
        else:
            raise TypeError(f"Unsupported image type: {type(x)}")

        img = self.transform(img)   # your pipeline of PIL→PIL→…→Tensor
        return img, y
        """

        
class AugmentedDataset(Dataset):
    def __init__(self, base_ds, transform):
        self.base     = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]       # x: Tensor [1,H,W] or [H,W]
        img = self.transform(x)     # apply the Compose(...) above
        return img, y

        
class RotateBySet:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        if random is None or F is None:
            raise RuntimeError("RotateBySet requires 'random' and torchvision to be installed")
        angle = random.choice(self.angles)
        return F.rotate(img, angle)