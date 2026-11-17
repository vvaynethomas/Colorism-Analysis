import importlib.util
from pathlib import Path

import cv2
import numpy as np

from colorism import faces

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'make_walkthrough.py'


def _load_script():
    spec = importlib.util.spec_from_file_location('make_walkthrough', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_walkthrough_on_synthetic_image(tmp_path, monkeypatch):
    # A grey "face" square on a dark background; the Haar detector is stood in for.
    image = np.full((240, 240, 3), 20, dtype=np.uint8)
    image[60:160, 70:170] = 140
    image[100:110, 90:150] = 90  # a darker band inside the face
    path = tmp_path / 'synthetic.png'
    cv2.imwrite(str(path), image)
    monkeypatch.setattr(faces, 'detect_faces', lambda img: [(70, 60, 100, 100)])

    out = tmp_path / 'figures' / 'walkthrough.png'
    steps = _load_script().main([str(path), '--out', str(out)])

    assert out.exists() and out.stat().st_size > 0
    assert steps['crop'].shape == (60, 60, 3)
    assert steps['mask'].sum() == 60 * 60
    assert steps['crop_box'] == (90, 80, 60, 60)
    expected = faces.get_greyscale_stats(steps['crop'])
    assert [steps['stats'][k] for k in faces.STAT_COLUMNS] == expected
    assert steps['stats']['median'] == 140 and steps['stats']['minimum'] == 90
