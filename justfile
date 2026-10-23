set dotenv-load := true

default:
    @just --list

setup: sync
    uv run --no-sync pre-commit install

sync:
    #!/usr/bin/env bash
    set -eu
    backend=$(scripts/detect_backend.sh)
    echo "torch backend: ${backend}"
    if [ "${backend}" = "none" ]; then
        uv sync
    else
        uv sync --no-install-package torch --no-install-package torchvision
        uv pip install --torch-backend="${backend}" torch torchvision
    fi
    uv run --no-sync python -m vehicle_detection.device

lint:
    uv run --no-sync ruff check .
    uv run --no-sync ruff format --check .

fmt:
    uv run --no-sync ruff check --fix .
    uv run --no-sync ruff format .

test:
    uv run --no-sync pytest

check: lint test

env-check:
    uv run --no-sync python -m vehicle_detection.settings

dvc-setup: env-check
    uv run --no-sync dvc remote modify --local garage url "s3://${S3_BUCKET}"
    uv run --no-sync dvc remote modify --local garage endpointurl "${S3_ENDPOINT_URL}"
    uv run --no-sync dvc remote modify --local garage region "${S3_REGION:-garage}"
    uv run --no-sync dvc remote modify --local garage access_key_id "${AWS_ACCESS_KEY_ID}"
    uv run --no-sync dvc remote modify --local garage secret_access_key "${AWS_SECRET_ACCESS_KEY}"

data-pull: dvc-setup
    uv run --no-sync dvc pull data/raw.dvc data/processed.dvc

data-push: dvc-setup
    uv run --no-sync dvc push

tiles coco images *overrides:
    uv run --no-sync python -m vehicle_detection.data.build_tiles coco_path={{coco}} images_dir={{images}} {{overrides}}

provisional-manifest coco images out="splits/provisional.csv" *flags:
    uv run --no-sync python -m vehicle_detection.data.provisional --coco {{coco}} --images {{images}} --out {{out}} {{flags}}

normalise-coco coco images out="data/processed/coco.json":
    uv run --no-sync python -m vehicle_detection.data.normalise --coco {{coco}} --images {{images}} --out {{out}}

yolo-export tiles="data/tiles" out="data/yolo" *flags:
    uv run --no-sync python -m vehicle_detection.data.yolo_export --tiles {{tiles}} --out {{out}} {{flags}}

train-tiles data *overrides:
    uv run --no-sync python -m vehicle_detection.data.yolo_export --check --tiles "${TILES_DIR:-data/tiles}" --out "$(dirname {{data}})"
    uv run --no-sync python -m vehicle_detection.training.tile_train data={{data}} {{overrides}}

predict coco images weights *overrides:
    uv run --no-sync python -m vehicle_detection.inference.run coco_path={{coco}} images_dir={{images}} weights={{weights}} {{overrides}}

compare coco images weights *overrides:
    uv run --no-sync python -m vehicle_detection.experiments.compare coco_path={{coco}} images_dir={{images}} weights={{weights}} {{overrides}}

compare-merge coco images weights *overrides:
    uv run --no-sync python -m vehicle_detection.experiments.compare --config-name compare_merge coco_path={{coco}} images_dir={{images}} weights={{weights}} {{overrides}}

scene-split coco images out="splits/v1_candidate.csv" *flags:
    uv run --no-sync python -m vehicle_detection.data.scene_split --coco {{coco}} --images {{images}} --out {{out}} --cache data/processed/scene_embeddings.npz {{flags}}

analyse run_dir *flags:
    uv run --no-sync python -m vehicle_detection.evaluation.analyse --run-dir {{run_dir}} {{flags}}

dedupe run_dir *flags:
    uv run --no-sync python -m vehicle_detection.evaluation.dedupe --run-dir {{run_dir}} {{flags}}
