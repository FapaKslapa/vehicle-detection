#!/usr/bin/env bash
set -eu

valid="cpu cu126 cu130 rocm7.2 xpu none"

if [ -n "${TORCH_BACKEND:-}" ]; then
    case " $valid " in
        *" $TORCH_BACKEND "*) echo "$TORCH_BACKEND"; exit 0 ;;
        *) echo "TORCH_BACKEND non valido: $TORCH_BACKEND (ammessi: $valid)" >&2; exit 1 ;;
    esac
fi

if [ "$(uname -s)" = "Darwin" ]; then
    echo none
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    major=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -1)
    if [ "${major:-0}" -ge 13 ]; then echo cu130; else echo cu126; fi
elif [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
    echo rocm7.2
elif command -v xpu-smi >/dev/null 2>&1 || command -v sycl-ls >/dev/null 2>&1; then
    echo xpu
else
    echo cpu
fi
