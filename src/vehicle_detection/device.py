from __future__ import annotations

import torch


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    device = device or best_device()
    if device.type == "cuda":
        backend = "ROCm" if torch.version.hip else "CUDA"
        return f"{backend}: {torch.cuda.get_device_name(device)}"
    if device.type == "xpu":
        return f"XPU: {torch.xpu.get_device_name(device)}"
    return device.type


if __name__ == "__main__":
    print(f"device: {describe_device()}")
