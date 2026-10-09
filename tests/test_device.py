import pytest
import torch

from vehicle_detection.device import best_device, describe_device


def set_backends(monkeypatch, *, cuda=False, xpu=False, mps=False, hip=None):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(
        torch,
        "xpu",
        type(
            "Xpu",
            (),
            {
                "is_available": staticmethod(lambda: xpu),
                "get_device_name": staticmethod(lambda device=None: "Fake XPU"),
            },
        ),
        raising=False,
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_best_device_is_usable():
    device = best_device()
    assert device.type in {"cuda", "xpu", "mps", "cpu"}
    assert describe_device(device)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"cuda": True, "xpu": True, "mps": True}, "cuda"),
        ({"xpu": True, "mps": True}, "xpu"),
        ({"mps": True}, "mps"),
        ({}, "cpu"),
    ],
)
def test_priority_order(monkeypatch, flags, expected):
    set_backends(monkeypatch, **flags)
    assert best_device().type == expected


def test_describe_nvidia(monkeypatch):
    set_backends(monkeypatch, cuda=True)
    assert describe_device() == "CUDA: Fake GPU"


def test_describe_amd_rocm(monkeypatch):
    set_backends(monkeypatch, cuda=True, hip="7.2")
    assert describe_device() == "ROCm: Fake GPU"


def test_describe_xpu(monkeypatch):
    set_backends(monkeypatch, xpu=True)
    assert describe_device() == "XPU: Fake XPU"


@pytest.mark.parametrize("kind", ["mps", "cpu"])
def test_describe_plain(monkeypatch, kind):
    set_backends(monkeypatch, mps=kind == "mps")
    assert describe_device() == kind
