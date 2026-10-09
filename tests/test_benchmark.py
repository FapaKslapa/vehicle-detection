import pytest
import torch

from vehicle_detection.inference import benchmark
from vehicle_detection.inference.benchmark import CostReport, measure


def fake_clock(monkeypatch, values):
    ticks = iter(values)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))


def test_mean_and_p95_exclude_the_warmup(monkeypatch):
    calls = []
    fake_clock(monkeypatch, [0.0, 1.0, 1.0, 3.0, 3.0, 6.0])
    report = measure(calls.append, ["a", "b", "c"], torch.device("cpu"), warmup=1)
    assert calls == ["a", "a", "b", "c"]
    assert report.images == 3
    assert report.mean_seconds == pytest.approx(2.0)
    assert report.p95_seconds == pytest.approx(2.9)
    assert report.peak_memory_mb is None


def test_zero_warmup(monkeypatch):
    calls = []
    fake_clock(monkeypatch, [0.0, 2.0])
    report = measure(calls.append, ["a"], torch.device("cpu"), warmup=0)
    assert calls == ["a"]
    assert report.mean_seconds == pytest.approx(2.0)


def test_cuda_peak_memory_is_reported(monkeypatch):
    fake_clock(monkeypatch, [0.0, 1.0])
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: events.append("sync"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device=None: events.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 5 * 2**20)
    report = measure(lambda image: None, ["a"], torch.device("cuda"), warmup=0)
    assert report.peak_memory_mb == pytest.approx(5.0)
    assert events.index("reset") < events.index("sync")


def test_requires_images():
    with pytest.raises(ValueError, match="almeno un'immagine"):
        measure(lambda image: None, [], torch.device("cpu"))


def test_rejects_negative_warmup():
    with pytest.raises(ValueError, match="warmup"):
        measure(lambda image: None, ["a"], torch.device("cpu"), warmup=-1)


def test_prediction_errors_propagate():
    def failing(image):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        measure(failing, ["a"], torch.device("cpu"), warmup=0)


def test_report_is_immutable():
    report = CostReport(1, 1.0, 1.0, None)
    with pytest.raises(AttributeError):
        report.images = 2


def test_run_measured_returns_results_in_order(monkeypatch):
    fake_clock(monkeypatch, [0.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    results, report = benchmark.run_measured(
        lambda image: image.upper(), str.strip, [" a", " b"], torch.device("cpu"), warmup=1
    )
    assert results == ["A", "B"]
    assert report.images == 2


def test_run_measured_excludes_loading_time():
    import time

    def slow_loader(item):
        time.sleep(0.05)
        return item

    _, report = benchmark.run_measured(
        lambda image: image, slow_loader, ["a", "b"], torch.device("cpu"), warmup=0
    )
    assert report.mean_seconds < 0.02


def test_run_measured_loads_each_item_once_plus_warmup():
    loaded = []

    def loader(item):
        loaded.append(item)
        return item

    benchmark.run_measured(lambda image: image, loader, ["a", "b", "c"], torch.device("cpu"))
    assert loaded == ["a", "a", "b", "c"]


def test_run_measured_rejects_empty_items():
    with pytest.raises(ValueError, match="almeno un'immagine"):
        benchmark.run_measured(lambda x: x, lambda x: x, [], torch.device("cpu"))
