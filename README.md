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
just tiles data/processed/coco.json data/raw/transset/TRANSSET_v2/images manifest_path=splits/v1.csv tiling.image_extension=.jpg
just yolo-export
just train-tiles data/yolo/data.yaml weights=yolo26n.pt epochs=50 imgsz=512
just predict data/processed/coco.json data/raw/transset/TRANSSET_v2/images runs/E3b/weights/best.pt mode=sliced split=val
just compare data/processed/coco.json data/raw/transset/TRANSSET_v2/images runs/E3b/weights/best.pt split=val
```

`splits/v1.csv` is the frozen scene-based split (21 scenes; 3,099 / 842 / 763 images for train / val / test), built with `just scene-split`, and it is the only split behind the numbers in the report. `splits/provisional.csv` is a smoke-test split and must not be used for reported results. `vehicle_detection.evaluation.metrics` computes COCO AP with the small, medium and large breakdown, precision, recall and F1 at IoU 0.5, and counting MAE and RMSE; `compare`, the bootstrap and the evaluation tools below (`just analyse`, `just dedupe`) all use it. It used to live in `vehicle_detection.experiments.metrics`, which still works as an alias.

`just compare` evaluates `full`, `sliced` and `adaptive` variants of the same weights on one split and writes `comparison.csv` and `comparison.md` with detection metrics, counting error, seconds per image and peak memory. Predictions are kept down to a confidence of 0.01 (`configs/postprocess/eval.yaml`) so that AP is computed on the full precision-recall curve, and the counting threshold grid must stay above that floor. The counting threshold is tuned on the split it is given, so tune on `val` and pass `threshold=<value>` for `test`; tuning on `test` is refused. Every run writes `run.json` or `provenance.json` with the full resolved config and the manifest hash, and reports built on a `provisional*` manifest are marked as such. Variants can override weights, mode and tiling in `configs/compare.yaml`, which is how E3b weights are compared with the baseline.

## Evaluation, counting and error analysis

`just analyse <run-dir>` takes the output folder of `just compare` and explains the numbers in
`comparison.csv`. It reuses the run's split, image subset and per-variant thresholds (from
`provenance.json` and `comparison.csv`), so every figure lines up with what `compare` reported.

```bash
just compare data/processed/coco.json data/raw/transset/TRANSSET_v2/images data/weights/E3b-yolo26n-512.pt split=val
just analyse runs/compare
```

Everything lands in `runs/compare/analysis/`; start with `summary.md`.

- **Finding vs naming** (`localisation.csv`): AP, precision, recall and F1 with and without the
  class. A large gap means vehicles are found but given the wrong class.
- **Error types** (`outcomes.csv`, `error_types.png`): every prediction above the threshold is
  `correct`, `duplicate`, `cross_class_duplicate` (a second box of another class on a vehicle
  that is already found, which inflates counts because NMS runs per class), `classification`,
  `localization`, `both` or `background`; every true vehicle is `detected`, `wrong_class`,
  `poorly_localized` or `missed`. Definitions follow TIDE and are documented in
  `vehicle_detection.evaluation.errors`; `correct` equals the true positives of
  `evaluation.metrics`.
- **Confusion matrix** (`<variant>/confusion.png`): which class each found vehicle was given.
- **Recall breakdowns** (`recall_breakdown.csv`, `recall_by_*.png`): by class, size, whether
  the vehicle is cut by the image edge, size and edge together (are the small misses the
  cut-off vehicles?), vehicles per image, location and scene. Size bins are the dataset's
  quartiles of sqrt(area) (60, 77, 100 px), because only 2% of TRANSSET boxes are COCO-small;
  change them with `--size-edges`.
- **Car types merged** (`merged.csv`, `merged_counting.csv`, `<variant>/confusion_merged.png`):
  class accuracy, AP and per-class count error with car, hatchback, sedan and suv scored as
  one "passenger car" class, and the share of wrong-class errors that are only car-type
  mix-ups. Choose another group with `--merge-group "name=a,b,..."`, or `--merge-group none`.
- **Counting** (`counting.csv`, `<variant>/counts_per_image.csv`, `<variant>/counts.png`):
  per-image true and predicted counts, total and per class, with MAE, RMSE, signed bias and
  MAE relative to the mean count. `vehicle_detection.counting.counts.predicted_counts` gives the
  same per-image counts for unseen images without labels.
- **Gallery** (`<variant>/gallery/`): the worst images with every box tagged by its outcome.

`just analyse runs/compare --help` lists the options (`--variants`, `--gallery 0`, ...).

### Class-agnostic duplicate removal

Detector NMS and SAHI's tile merging run per class, so one vehicle can survive as both "car"
and "sedan" and be counted twice (`cross_class_duplicate` above). `just dedupe <run-dir>`
applies class-agnostic NMS to the saved predictions, without running the model, and writes
`<run-dir>_agnostic/` with the original variants plus `<variant>-agn50` and `-agn70`. The
counting threshold of the new variants is re-tuned on val; on test pass
`--thresholds-from <val run>_agnostic/comparison.csv`. The new folder works with
`just analyse` and `experiments.bootstrap` like any `compare` run:

```bash
just dedupe runs/val_all
just analyse runs/val_all_agnostic
uv run --no-sync python -m vehicle_detection.experiments.bootstrap run_dir=runs/val_all_agnostic coco_path=data/processed/coco.json reference=E1-full-1024 split=val out_dir=runs/val_all_agnostic/bootstrap
```

### Counting vehicles in new images

`just count <images-folder> <weights>` is the counting output of the system for images
without labels. It runs the detector on every `.jpg`, `.jpeg` and `.png` in the folder,
removes boxes that overlap a stronger box of any class (IoU 0.5), keeps boxes at or above the
counting threshold, and writes to `runs/count/`:

- `counts.csv`: one row per image, with the total and one column per class
- `detections.json`: the counted boxes, with the image list and categories
- `run.json`: the resolved config, a hash of the weights, timing and the totals

```bash
just count path/to/new/images data/weights/E1-control-yolo26n-1024.pt
just count path/to/new/images data/weights/E1-control-yolo26n-1024.pt out_dir=runs/count_demo
```

The defaults in `configs/count.yaml` are the setting chosen on val for E1-full-1024 (threshold
0.35 after class-agnostic NMS at IoU 0.5). Another model or mode needs its own threshold:
take it from `just dedupe` on that model's validation run and pass `threshold=<value>`, plus
`mode=`, `kind=` and `input_size=` as for `just predict`. `dedupe_iou=null` counts the raw
boxes. Class names come from `data/processed/coco.json` (`categories_from=`).

## Faster R-CNN multiscale training

`configs/train/faster_rcnn.yaml` configures random short-edge sizes of 480, 576,
672, 768, and 800 pixels, with a 1333-pixel long-edge cap. Build the model with
the config's model settings to enable this during training:

```python
from vehicle_detection.training.faster_rcnn import load_config, setup_faster_rcnn

config = load_config()
model = setup_faster_rcnn(num_classes=len(config["data"]["classes"]), seed=0, **config["model"])
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

## Scene based split

TRANSSET frames are shuffled crops of a few dozen hovering videos, some of them rotated, so neither file order nor a per image split is safe. `vehicle_detection.data.scene_split` embeds every image with DINOv2 (also at 11 rotations, so rotated copies stay with their source), groups images into scenes with Louvain on a rotation invariant k nearest neighbour graph, labels two locations by clustering the scenes, and assigns whole scenes to train, val and test with a seeded annealing search that balances images, locations and class counts.

```bash
just scene-split data/processed/coco.json data/raw/transset/TRANSSET_v2/images splits/v1_candidate.csv --neighbors 12 --resolution 1.0 --max-group-share 0.1 --seed 7
```

It writes `splits/v1_candidate.csv` and `splits/v1_candidate.report.json`. The report lists group sizes, location and class shares per split and the share of near duplicate image pairs that end up in different splits at similarity 0.85, 0.9, 0.95 and 0.97, next to the 46.5 percent a random 70/15/15 split would give. A candidate is only a proposal: check the scene mosaics by eye before freezing it as `splits/v1.csv`.

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
checkpoint.pt (model and optimizer for resume), metrics.jsonl, train.log (progress,
timing and peak GPU memory, readable during training), and run.json (resolved settings,
categories, and manifest hash). Set training.lr_steps to divide the learning rate by ten
at those zero-based epochs; the default empty list keeps it constant. best.pt and last.pt are plain
state dictionaries compatible with the existing torchvision inference adapter.
That adapter resizes to 800/1333 unless input_size is set, so pass the training size
(for example kind=torchvision input_size=1024). `configs/compare_frcnn.yaml` compares
both Faster R-CNN runs with the E1 control at 1024 pixels on val. Existing output
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
