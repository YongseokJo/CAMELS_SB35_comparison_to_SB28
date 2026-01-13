# Comparison25_50

Code + notebooks for learning cosmology/astrophysics parameters from CAMELS 2D maps, with a focus on comparisons across box sizes (SB28 ~25 Mpc/h, SB35 ~50 Mpc/h) and cross-evaluation between datasets.

The core models live in `src/` (e.g. a regression Vision Transformer in `src/transformer.py`). Most experiments and figures are produced from notebooks in `notebooks/` and scripts in `test/`.

## Repository layout

- `src/`
  - Model definitions (e.g. ViT-style regression), losses, validation utilities.
  - Data utilities including deterministic splits and CAMELS map loading.
- `data/`
  - `data/splits/`: deterministic train/val/test split indices (`splits_1024.json`, `splits_2048.json`).
  - `data/models/`: local model checkpoints (ignored by git).
- `notebooks/`: exploration, training, cross-evaluation and plotting.
- `test/`: runnable experiment scripts and Slurm helpers.
- `plot/`: generated figures/tables (ignored by git).

## Data access notes

The CAMELS loader in `src/dataloader.py` currently references paths on a shared filesystem (e.g. `/mnt/home/...` and `/mnt/ceph/...`). If you are running elsewhere, you’ll need to update those base paths to match your environment.

Deterministic dataset splits are loaded from `data/splits/` via `load_splits_json()` / `split_expanded_dataset_from_json()`.

## Environment

This repo assumes a Python + PyTorch stack.

Typical requirements:
- Python 3.10+
- `torch`, `torchvision`
- `numpy`, `pandas`, `matplotlib`

If you want, I can add a `requirements.txt` (or `environment.yml`) once you confirm your preferred install method.

## Running

### Notebooks

Open any notebook in `notebooks/` and run with a kernel that has the dependencies above.

### Scripts

The `test/` folder contains experiment scripts and Slurm submission helpers.

Example:

- Run a script locally:
  - `python test/SB28.py`

(Exact arguments/paths vary by script; see the top of each file.)

## Outputs and version control

This repo generates many large artifacts (plots, logs, caches, model checkpoints). The `.gitignore` is set up to keep source code, notebooks, and split definitions tracked, while excluding:
- virtual environments (`.venv/`)
- python caches (`__pycache__/`, `*.pyc`)
- logs and Slurm outputs (`test/log/`, `*.out`, `*.err`)
- generated plots (`plot/`, `test/plot/`)
- model checkpoints (`*.pt`, `data/models/`)
- notebook-generated binaries (`notebooks/*.png`, `notebooks/*.pkl`, etc.)
