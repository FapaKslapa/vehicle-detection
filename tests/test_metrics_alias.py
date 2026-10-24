from vehicle_detection.evaluation import metrics as current
from vehicle_detection.experiments import metrics as alias


def public_names(module):
    names = set()
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        defined_here = getattr(value, "__module__", None) == module.__name__
        constant = name.isupper() and not isinstance(value, type) and not callable(value)
        if defined_here or constant:
            names.add(name)
    return names


def test_the_old_path_exposes_every_public_name_of_the_new_module():
    assert set(alias.__all__) == public_names(current)


def test_the_old_path_gives_the_very_same_objects():
    for name in alias.__all__:
        assert getattr(alias, name) is getattr(current, name)
