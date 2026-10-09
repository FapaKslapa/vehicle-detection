from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")
BYTES_PER_MB = 2**20
P95 = 95


@dataclass(frozen=True, slots=True)
class CostReport:
    images: int
    mean_seconds: float
    p95_seconds: float
    peak_memory_mb: float | None


def measure(
    predict: Callable[[T], object],
    images: Sequence[T],
    device: torch.device,
    warmup: int = 1,
) -> CostReport:
    _, report = run_measured(predict, _identity, images, device, warmup)
    return report


def run_measured(
    predict: Callable[[T], R],
    loader: Callable[[S], T],
    items: Sequence[S],
    device: torch.device,
    warmup: int = 1,
) -> tuple[list[R], CostReport]:
    if not items:
        raise ValueError("serve almeno un'immagine per misurare il costo")
    if warmup < 0:
        raise ValueError(f"warmup non può essere negativo: {warmup}")
    for item in items[:warmup]:
        predict(loader(item))
        _synchronize(device)
    _reset_peak_memory(device)
    results: list[R] = []
    durations = []
    for item in items:
        loaded = loader(item)
        start = time.perf_counter()
        results.append(predict(loaded))
        _synchronize(device)
        durations.append(time.perf_counter() - start)
    report = CostReport(
        images=len(items),
        mean_seconds=statistics.fmean(durations),
        p95_seconds=float(np.percentile(durations, P95)),
        peak_memory_mb=_peak_memory_mb(device),
    )
    return results, report


def _identity(value: T) -> T:
    return value


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / BYTES_PER_MB
    return None
