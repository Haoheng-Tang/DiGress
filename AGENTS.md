# AGENTS.md

## Purpose
This repository trains and samples graph diffusion models (DiGress) on molecular datasets (`qm9`, `guacamol`, `moses`) and non-molecular graph datasets (`sbm`, `comm20`, `planar`, `inpatient`).

This file is the working guide for future coding agents in this repo.

## Quick Start
- Create/activate environment and install deps from `requirements.txt` plus repo extras (`pip install -e .`).
- Compile ORCA before non-molecular motif/orbit metrics:
  - `cd src/analysis/orca`
  - `g++ -O2 -std=c++11 -o orca orca.cpp`
- Run from repo root with:
  - `python src/main.py +experiment=debug.yaml`

## Entrypoint And Config
- Entrypoint is `src/main.py` (not root `main.py`).
- Hydra config root: `configs/config.yaml`.
- Default composition:
  - `general=general_default`
  - `model=discrete`
  - `train=train_default`
  - `dataset=qm9`
- Hydra run directory is `outputs/YYYY-MM-DD/HH-MM-SS-<general.name>`.
- Runtime artifacts are written under the run dir (`graphs/`, `chains/`, `checkpoints/` if enabled).

## Architecture Map
- `src/main.py`: dataset/task dispatch, model construction, Trainer setup, resume/test behavior.
- `src/diffusion_model_discrete.py`: main discrete diffusion training/sampling flow.
- `src/diffusion_model.py`: continuous diffusion alternative.
- `src/models/transformer_model.py`: graph transformer over node/edge/global features.
- `src/datasets/`: dataset loaders + dataset infos + datamodules.
- `src/diffusion/`: schedules, transitions, sampling utilities, extra features.
- `src/metrics/`: train/val/test losses and metric accumulators.
- `src/analysis/`: sampling metrics, visualization, RDKit utilities, ORCA integration.

## Data And Tensor Invariants
- Dense graph tensors are handled through `utils.PlaceHolder(X, E, y)` and `mask(node_mask)`.
- Node features `X`: shape `(bs, n, dx)`; edge features `E`: `(bs, n, n, de)`; global `y`: `(bs, dy)`.
- Edge tensors must stay symmetric; diagonal is zeroed in multiple places.
- For discrete edge features, channel `0` is "no-edge" (`utils.encode_no_edge`).
- Many datasets use empty graph-level targets `y` with shape `(1, 0)`.

## Dataset Notes
- Molecular: `qm9`, `guacamol`, `moses` use RDKit parsing and molecular metrics.
- Non-molecular: `sbm`, `comm20`, `planar`, `inpatient` use SPECTRE-style metrics.
- Inpatient loader (`src/datasets/inpatient_dataset.py`) now supports larger datasets and splits dynamically.
- Accepted raw filenames include:
  - `penn_inpatient_pavilion_subgraph_<n>.json`
  - `subgraph_<n>.json`
  - any `.json` with a numeric suffix (fallback)
- Split policy is deterministic and configurable through `configs/dataset/inpatient.yaml`:
  - `train_split_ratio` (default `0.8`)
  - `val_split_ratio` (default `0.1`)
  - `split_seed` (default `0`)
- At least 3 raw graphs are required to produce non-empty train/val/test splits.

## Logging And Checkpoint Behavior
- W&B is manually initialized via `utils.setup_wandb`; Lightning logger is disabled (`logger=[]`).
- Checkpoints are written when `train.save_model=True`.
- `general.test_only` expects an absolute checkpoint path.
- `general.resume` is interpreted relative to outputs root in resume-adaptive path logic.

## Editing Guidelines For Agents
- Keep dataset split/load logic and `DatasetInfos` synchronized (node/edge type counts, max nodes, atom metadata).
- If adding a new dataset:
  - add loader/info in `src/datasets/`
  - add config in `configs/dataset/`
  - wire dataset branch in `src/main.py`
- If modifying model input channels, also update/verify:
  - `dataset_infos.compute_input_output_dims(...)`
  - extra feature modules in `src/diffusion/extra_features*.py`
- Preserve symmetry/masking assumptions in any new sampling or loss code.

## Known Gotchas
- README commands mention `python main.py`; in this repo use `python src/main.py`.
- `graph_tool` and `torch_geometric` compatibility can be environment-sensitive.
- `TrainLossDiscrete.log_epoch_metrics` references `self.train_y_loss` (likely typo); it is usually masked by `y` having zero dims.
- `src/datasets/spectre_dataset.py` appends each `data` object twice in `process()`; verify intent before changing.

## Practical Smoke Checks
- Fast sanity run:
  - `python src/main.py +experiment=debug.yaml`
- Non-molecular metric path sanity (after ORCA compile):
  - `python src/main.py dataset=planar +experiment=debug.yaml`
