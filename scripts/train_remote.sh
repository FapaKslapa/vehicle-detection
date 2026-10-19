#!/usr/bin/env bash
set -euo pipefail

epochs="${EPOCHS:-50}"
imgsz="${IMGSZ:-512}"
weights="${WEIGHTS:-yolo26n.pt}"
name="${RUN_NAME:-E3b-yolo26n-512}"
manifest="${MANIFEST:-splits/v1_candidate.csv}"
tile_ext="${TILE_EXT:-.png}"
slice_size="${SLICE_SIZE:-512}"
overlap="${OVERLAP:-0.2}"
export TILES_DIR="${TILES_DIR:-data/tiles}"
tiles_dir="${TILES_DIR}"
yolo_dir="${YOLO_DIR:-data/yolo}"
seed="${SEED:-0}"

for tool in uv just; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "manca $tool nel PATH" >&2
        exit 1
    fi
done

if [ ! -f .env ]; then
    echo "manca .env con le credenziali S3 (vedi README)" >&2
    exit 1
fi

just sync
just data-pull
just tiles data/processed/coco.json data/raw/transset/TRANSSET_v2/images \
    "manifest_path=${manifest}" "tiling.image_extension=${tile_ext}" \
    "tiling.slice_size=${slice_size}" "tiling.overlap_ratio=${overlap}" "out_dir=${tiles_dir}"
just yolo-export "${tiles_dir}" "${yolo_dir}"
just train-tiles "${yolo_dir}/data.yaml" \
    "weights=${weights}" "epochs=${epochs}" "imgsz=${imgsz}" "seed=${seed}" "name=${name}"

mkdir -p data/weights
cp "runs/${name}/weights/best.pt" "data/weights/${name}.pt"
uv run --no-sync dvc add data/weights
uv run --no-sync dvc push
echo "pesi in data/weights/${name}.pt, committa data/weights.dvc"
