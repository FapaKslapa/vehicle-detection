from vehicle_detection.device import best_device, describe_device


def test_best_device_is_usable():
    device = best_device()
    assert device.type in {"cuda", "xpu", "mps", "cpu"}
    assert describe_device(device)
