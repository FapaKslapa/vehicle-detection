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

## Tiling and sliced inference

Tiles are generated per split from the session manifest (`splits/v1.csv`, columns `image_id,session_id,location,split`), so every tile inherits the split of its source image and no session can appear in two splits.

```bash
just tiles path/to/instances.json path/to/images
just tiles path/to/instances.json path/to/images tiling=e3_640 overwrite=true
```

Tiles land in `data/tiles/<split>/` together with `annotations_coco.json` and are ignored by git. Tiling parameters live in `configs/tiling/` and are validated before any work starts.

Evaluation always runs on full images. `SlicedPredictor` (`vehicle_detection.inference.sliced`) tiles an image, merges the per tile detections and returns boxes in full image coordinates; `to_coco_results` exports them in the common format. `AdaptivePredictor` runs a coarse pass first and applies the sliced predictor only to the regions that contain vehicles. `measure` (`vehicle_detection.inference.benchmark`) reports seconds per image and peak CUDA memory.

## Experiment workflow (E3 and E4)

The real TRANSSET COCO file stores `file_name` without extension and category ids 1 to 8, so it is normalised first. Frames are shuffled crops of 4K videos, so sessions are not recoverable from file order.

```bash
just normalise-coco data/raw/transset/TRANSSET_v2/label_coco/coco.json data/raw/transset/TRANSSET_v2/images
just provisional-manifest data/processed/coco.json data/raw/transset/TRANSSET_v2/images
just tiles data/processed/coco.json data/raw/transset/TRANSSET_v2/images manifest_path=splits/provisional.csv tiling.image_extension=.jpg
just yolo-export
just train-tiles data/yolo/data.yaml weights=yolo26n.pt epochs=50 imgsz=512
just predict data/processed/coco.json data/raw/transset/TRANSSET_v2/images runs/E3b/weights/best.pt mode=sliced split=val
just compare data/processed/coco.json data/raw/transset/TRANSSET_v2/images runs/E3b/weights/best.pt split=val
```

`splits/provisional.csv` groups visually similar frames into sessions and splits by group. It is a smoke-test split, ignored by git, and must be replaced by the shared `splits/v1.csv` before any number goes into the report. `vehicle_detection.experiments.metrics` computes COCO AP with the small, medium and large breakdown plus counting MAE and RMSE; it is provisional and is meant to be replaced by the shared evaluation module with the same inputs.

`just compare` evaluates `full`, `sliced` and `adaptive` variants of the same weights on one split and writes `comparison.csv` and `comparison.md` with detection metrics, counting error, seconds per image and peak memory. Predictions are kept down to a confidence of 0.01 (`configs/postprocess/eval.yaml`) so that AP is computed on the full precision-recall curve, and the counting threshold grid must stay above that floor. The counting threshold is tuned on the split it is given, so tune on `val` and pass `threshold=<value>` for `test`; tuning on `test` is refused. Every run writes `run.json` or `provenance.json` with the full resolved config and the manifest hash, and reports built on a `provisional*` manifest are marked as such. Variants can override weights, mode and tiling in `configs/compare.yaml`, which is how E3b weights are compared with the baseline.

## Faster R-CNN multiscale training

`configs/train/faster_rcnn.yaml` configures random short-edge sizes of 480, 576,
672, 768, and 800 pixels, with a 1333-pixel long-edge cap. Build the model with
the config's model settings to enable this during training:

```python
from vehicle_detection.training.faster_rcnn import load_config, setup_faster_rcnn

config = load_config()
model = setup_faster_rcnn(
    num_classes=len(config["data"]["classes"]), seed=0, **config["model"]
)
model.train()
# Inside your training loop, pass original float images in [0, 1] and targets:
# losses = model(images, targets)
# loss = sum(losses.values())
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
```

The model's transform samples a size per image on every training forward pass,
preserves aspect ratio, and scales boxes with the image. Targets must contain
pixel-coordinate `xyxy` boxes and labels starting at 1. Calling `model.eval()`
uses the last configured size (800), subject to the same long-edge cap.
Pass an integer `min_size` for single-scale training. Direct setup calls retain
the original 800/1333 defaults; multiscale settings are enabled by the config.

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

### Run Faster R-CNN multiscale training

Use normalized COCO annotations and a session-separated train/val manifest:

```bash
just train-faster-rcnn data/processed/coco.json data/raw/transset/TRANSSET_v2/images data.manifest_path=splits/v1.csv training.epochs=50
```

The equivalent command (also usable directly in PowerShell) is:

```bash
uv run --no-sync python -m vehicle_detection.training.faster_rcnn_train data.coco_path=data/processed/coco.json data.images_dir=data/raw/transset/TRANSSET_v2/images
```

This workflow is only for Faster R-CNN. It loads RGB float images at original
resolution and converts COCO boxes to pixel xyxy coordinates. The model transform
randomly selects the configured short edge for each training image and scales its
boxes; validation uses the last short edge with the same long-edge cap. Category
labels are derived from sorted COCO category IDs, with zero reserved for background;
the COCO categories are authoritative rather than the descriptive classes list.

Each epoch trains with SGD and evaluates full-image COCO AP on val only. The test
split is never used. Outputs are best.pt (highest validation AP in this run), last.pt,
checkpoint.pt (model and optimizer for resume), metrics.jsonl, and run.json
(resolved settings, categories, and manifest hash). best.pt and last.pt are plain
state dictionaries compatible with the existing torchvision inference adapter.
That adapter uses its default 800/1333 resize settings; custom model resizing must
also be configured when constructing a model for inference. Existing output
directories are refused to protect previous runs.

For an offline CPU smoke run, use model.weights=null training.device=cpu
training.epochs=1 training.batch_size=1 'model.min_size=[64,96]'
model.max_size=128 training.output_dir=runs/frcnn-smoke. Default pretrained weights
require a download on their first use. Resume into a new output directory:

```bash
uv run --no-sync python -m vehicle_detection.training.faster_rcnn_train data.coco_path=data/processed/coco.json data.images_dir=data/raw/transset/TRANSSET_v2/images training.resume=runs/faster-rcnn-multiscale/checkpoint.pt training.output_dir=runs/frcnn-resumed training.epochs=100
```

Keep the original model settings and dataset when resuming; epochs is the total
number of epochs including those already completed. The resumed directory selects
its own best.pt among its new epochs. Provisional manifests are marked in run.json
and must not be used for final reported results.
