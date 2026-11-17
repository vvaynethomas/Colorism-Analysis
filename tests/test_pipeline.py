import cv2
import numpy as np
import pandas as pd

import colorism
from colorism import faces
from colorism.__main__ import main


def _write_jpg(path, value, size=200):
    cv2.imwrite(str(path), np.full((size, size, 3), value, dtype=np.uint8))


def test_crop_by_scale_keeps_center():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert colorism.crop_by_scale(img, 0.6).shape == (60, 120, 3)


def test_greyscale_stats_of_uniform_crop():
    crop = np.full((50, 50, 3), 120, dtype=np.uint8)
    mean, low, high, median, mode = colorism.get_greyscale_stats(crop)
    assert (mean, low, high, median, mode) == (120, 120, 120, 120, 120)


def test_greyscale_stats_rejects_extremes_and_missing():
    assert np.isnan(colorism.get_greyscale_stats(np.full((10, 10, 3), 250, np.uint8))).all()
    assert np.isnan(colorism.get_greyscale_stats(np.nan)).all()


def test_list_images_and_extract_face_rejections(tmp_path):
    _write_jpg(tmp_path / 'small.jpg', 100, size=50)
    _write_jpg(tmp_path / 'blank.jpg', 100)
    (tmp_path / 'notes.txt').write_text('x')
    paths = colorism.list_images(str(tmp_path))
    assert [p.split('/')[-1] for p in paths] == ['blank.jpg', 'small.jpg']
    assert all(np.isnan(colorism.extract_face(p)) for p in paths)  # too small / no face


def test_cli_end_to_end_with_groups(tmp_path, monkeypatch):
    images = tmp_path / 'images'
    images.mkdir()
    rows = []
    for i, (group, value) in enumerate([('a', 80), ('a', 90), ('b', 150), ('b', 160), ('b', 170)]):
        _write_jpg(images / f'{i}.jpg', value)
        rows.append({'file': f'val/{i}.jpg', 'race': group})
    pd.DataFrame(rows).to_csv(tmp_path / 'labels.csv', index=False)
    # Stand in for the Haar detector: the "face" is the centre of the uniform image.
    monkeypatch.setattr(faces, 'extract_face', lambda p: cv2.imread(p)[50:150, 50:150])

    out = tmp_path / 'out'
    main([str(images), '--labels', str(tmp_path / 'labels.csv'), '--group', 'race',
          '--out', str(out)])

    stats_df = pd.read_csv(out / 'image_stats.csv')
    assert len(stats_df) == 5 and stats_df['mean'].notna().all()
    by_group = pd.read_csv(out / 'by_race.csv', index_col=0)
    assert by_group.loc['a', '50%'] == 85 and by_group.loc['b', '50%'] == 160
    for name in ['summary.csv', 'distributions.png', 'by_race.png']:
        assert (out / name).exists()
