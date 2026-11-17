"""Face detection, cropping and grayscale summary statistics for a folder of JPEG portraits.

Pipeline (see ``batch_process``):

1. ``list_images`` finds ``.jpg`` files.
2. ``extract_face`` rejects images under 120 px on either side, finds frontal faces with
   OpenCV's Haar cascade on a histogram-equalised grayscale copy, and keeps the central 60%
   of the face box (a random face is chosen when several are found).
3. ``get_greyscale_stats`` converts the crop to grayscale and summarises its nonzero pixels.
   Crops whose mean or mode falls outside 30-230 are treated as detection failures.
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

STAT_COLUMNS = ['mean', 'minimum', 'maximum', 'median', 'mode']
MIN_SIDE_PX = 120
FACE_SCALE = 0.6
VALID_RANGE = (30, 230)

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def list_images(source):
    """Return .jpg paths from a file path, a directory, or a list of either."""
    items = [source] if isinstance(source, str) else list(source)
    image_list = []
    for item in items:
        if item.endswith('.jpg'):
            image_list.append(item)
        elif os.path.isdir(item):
            image_list.extend(os.path.join(item, name) for name in sorted(os.listdir(item))
                              if name.endswith('.jpg'))
    return image_list


def crop_by_scale(img, scale=1.0):
    """Keep the central ``scale`` fraction of an image in each dimension."""
    height, width = img.shape[:2]
    center_x, center_y = width / 2, height / 2
    half_w, half_h = width * scale / 2, height * scale / 2
    return img[int(center_y - half_h):int(center_y + half_h),
               int(center_x - half_w):int(center_x + half_w)]


def extract_face(image_path, store_locally=False, output_directory='', group=False):
    """Return the cropped BGR face from ``image_path``, or NaN when none can be extracted.

    With ``store_locally`` every detected face crop is written to ``output_directory``
    (default: a folder named after the image). ``group=True`` returns the first face
    instead of a random one when several are detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f'could not read {image_path}, storing NAN')
        return np.nan
    if image.shape[0] < MIN_SIDE_PX or image.shape[1] < MIN_SIDE_PX:
        print(f'resolution for {image_path} is too low: {image.shape}')
        return np.nan

    greyscale = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    detected_faces = _FACE_CASCADE.detectMultiScale(greyscale, 1.1, 4)
    face_images = [crop_by_scale(image[y:y + h, x:x + w], scale=FACE_SCALE)
                   for (x, y, w, h) in detected_faces]
    if not face_images:
        print(f'no face detected in {image_path}, storing NAN')
        return np.nan

    if store_locally:
        stem = Path(image_path).stem
        out_dir = Path(output_directory or stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, face in enumerate(face_images):
            cv2.imwrite(str(out_dir / f'{stem}_{i}.jpg'), face)

    if len(face_images) == 1 or group:
        return face_images[0]
    return random.choice(face_images)


def get_greyscale_stats(image):
    """Return [mean, min, max, median, mode] of the nonzero grayscale pixels of a BGR crop.

    Returns five NaNs when no image is given or when the mean or mode falls outside
    ``VALID_RANGE`` (almost always a mis-detected face).
    """
    missing = [np.nan] * len(STAT_COLUMNS)
    if not isinstance(image, np.ndarray):
        print('no image passed, storing NAN')
        return missing
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = greyscale[np.nonzero(greyscale)]
    cur_mean = pixels.mean()
    cur_mode = stats.mode(pixels, axis=None, keepdims=False).mode
    low, high = VALID_RANGE
    if not (low <= cur_mean <= high and low <= cur_mode <= high):
        print('face detection error, storing NAN')
        return missing
    return [cur_mean, pixels.min(), pixels.max(), np.median(pixels), cur_mode]


def batch_process(directory=''):
    """Run the pipeline on every .jpg in ``directory``.

    Returns one row per image: ``name`` (the file stem with ``%20`` decoded to spaces),
    ``file`` (the file name), the cropped ``face_image`` array, and the five grayscale
    statistics. Rows where extraction failed hold NaN statistics.
    """
    im_path_list = list_images(directory)
    face_list = [extract_face(path) for path in im_path_list]
    stat_lists = [get_greyscale_stats(face) for face in face_list]
    names_list = [Path(path).stem.replace('%20', ' ') for path in im_path_list]
    path_df = pd.DataFrame({'name': names_list, 'file': [Path(p).name for p in im_path_list],
                            'face_image': face_list})
    stats_df = pd.DataFrame(stat_lists, columns=STAT_COLUMNS)
    return pd.concat([path_df, stats_df], axis=1)
