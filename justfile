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
    uv run --no-sync dvc pull

data-push: dvc-setup
    uv run --no-sync dvc push
