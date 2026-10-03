# vehicle-detection

Deep learning-based vehicle detection and traffic counting from drone-captured aerial images (TRANSSET).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv pip install --torch-backend=auto torch torchvision   # picks CUDA / ROCm / XPU / CPU for this machine
uv run pre-commit install
```

On macOS the default build already uses Apple MPS, so the second step is optional.
Run later commands with `uv run --no-sync ...` to keep the backend-specific torch build.

At runtime the best available device is picked automatically
(CUDA/ROCm > XPU > MPS > CPU), see `vehicle_detection.device`.

## Workflow

- `main`: releases only, versioned with release-please.
- `development`: day-to-day work. Commit directly or use short-lived branches from it.
- Conventional Commits (`feat:`, `fix:`, `chore:`, ...).

## Layout

```
src/vehicle_detection/   data, training, inference, counting, evaluation, reporting
configs/                 one config per experiment
notebooks/               EDA and analysis (outputs stripped)
tests/
```
