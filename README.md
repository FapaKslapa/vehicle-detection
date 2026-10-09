# vehicle-detection

Deep learning-based vehicle detection and traffic counting from drone-captured aerial images (TRANSSET).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
just setup
```

`just sync` runs `uv sync` and installs the PyTorch build matching the machine with `uv pip install --torch-backend`:

| Hardware | Backend | Wheels |
|---|---|---|
| Apple Silicon (MPS) | none | PyPI |
| NVIDIA, CUDA 13 driver | `cu130` | download.pytorch.org/whl/cu130 |
| NVIDIA, older driver | `cu126` | download.pytorch.org/whl/cu126 |
| AMD (ROCm, Linux) | `rocm7.2` | download.pytorch.org/whl/rocm7.2 |
| Intel GPU (XPU) | `xpu` | download.pytorch.org/whl/xpu |
| CPU only | `cpu` | download.pytorch.org/whl/cpu |

Force a backend with `TORCH_BACKEND=cpu just sync`. Always run later commands with
`uv run --no-sync` (the `just` recipes already do) so the backend is not replaced.
At runtime the best device is picked automatically (CUDA/ROCm > XPU > MPS > CPU),
see `vehicle_detection.device`.

## Environment and data

Create a `.env` with the variables read by `vehicle_detection.settings` (never commit it) and validate it:

```bash
just env-check
just data-pull
```

Datasets live in the S3 bucket via DVC, split manifests in `splits/` are tracked by git,
weights go to Weights & Biases Artifacts. Run `just` to list all commands.

## Workflow

- `main`: releases only, versioned with release-please.
- `development`: day-to-day work. Commit directly or use short-lived branches from it.
- Conventional Commits (`feat:`, `fix:`, `chore:`, ...).

## Layout

```
src/vehicle_detection/   data, training, inference, counting, evaluation, reporting, settings
configs/                 one config per experiment
splits/                  versioned train/val/test manifests (tracked by git)
notebooks/               EDA and analysis (outputs stripped)
data/                    DVC-managed datasets (not in git, only .dvc pointers)
tests/
```
