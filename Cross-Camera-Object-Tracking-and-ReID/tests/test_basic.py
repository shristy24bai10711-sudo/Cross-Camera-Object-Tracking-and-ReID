import os
from src.utils import load_config
from src.cross_camera_matcher import CrossCameraMatcher
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG = os.path.join(ROOT, "configs", "config.yaml")


def test_config_loads():
    cfg = load_config(CONFIG)
    for key in ("detector", "tracker", "reid", "io", "runtime"):
        assert key in cfg


def test_config_values_sane():
    cfg = load_config(CONFIG)
    assert 0 < cfg["detector"]["confidence_threshold"] < 1
    assert cfg["tracker"]["max_age"] > 0
    assert 0 < cfg["reid"]["match_threshold"] < 1


def test_matcher_keeps_classes_separate():
    matcher = CrossCameraMatcher(0.8)
    person = np.ones(2048, dtype=np.float32)
    person /= np.linalg.norm(person)
    vehicle = np.ones(2048, dtype=np.float32)
    vehicle[0] = -1
    vehicle /= np.linalg.norm(vehicle)
    gid1, _ = matcher.match_or_register(person, class_id=0)
    gid2, _ = matcher.match_or_register(vehicle, class_id=2)
    assert gid1 != gid2


def test_no_gui_calls_in_src():
    banned = ("imshow", "namedWindow", "waitKey")
    for fname in os.listdir(os.path.join(ROOT, "src")):
        if fname.endswith(".py"):
            text = open(os.path.join(ROOT, "src", fname), encoding="utf-8").read()
            assert not any(term in text for term in banned)
