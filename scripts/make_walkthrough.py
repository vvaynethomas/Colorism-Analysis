"""Draw the pipeline step by step on one image: ``figures/walkthrough.png``.

    python scripts/make_walkthrough.py                 # scikit-image's astronaut sample (NASA, public domain)
    python scripts/make_walkthrough.py photo.jpg --out figures/my_walkthrough.png

Every panel comes from the ``colorism`` package's own functions, so the figure shows
exactly what ``python -m colorism`` does to each image.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorism import faces  # noqa: E402

INK, MUTED, BAR = '#1f2933', '#616e7c', '#4a6fa5'
LINES = {'mean': ('#c44e52', '-'), 'median': ('#1f2933', '--'), 'mode': ('#2f855a', ':')}


def load_demo_image():
    """scikit-image's ``astronaut`` sample as a BGR array."""
    from skimage import data
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


def run_steps(image):
    """Run the package's steps on a BGR image and return every intermediate."""
    boxes = faces.detect_faces(image)
    if not boxes:
        raise ValueError('no face detected')
    box = boxes[0]
    crop = faces.crop_face(image, box)
    pixels = faces.face_pixels(crop)
    stats = dict(zip(faces.STAT_COLUMNS, faces.get_greyscale_stats(crop)))

    # Where the counted pixels sit in the original image.
    x, y, w, h = box
    cw, ch = crop.shape[1], crop.shape[0]
    cx = x + int(w / 2 - w * faces.FACE_SCALE / 2)
    cy = y + int(h / 2 - h * faces.FACE_SCALE / 2)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[cy:cy + ch, cx:cx + cw] = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 0
    return {'boxes': boxes, 'box': box, 'crop_box': (cx, cy, cw, ch), 'crop': crop,
            'mask': mask, 'pixels': pixels, 'stats': stats}


def plot_walkthrough(image, steps, path, caption='demo image'):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    (a1, a2, a3), (a4, a5, a6) = axes

    a1.imshow(rgb)
    a1.set_title('1. Original image')

    a2.imshow(rgb)
    for bx, by, bw, bh in steps['boxes']:
        a2.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec='#f6ad55', lw=2))
    cx, cy, cw, ch = steps['crop_box']
    a2.add_patch(Rectangle((cx, cy), cw, ch, fill=False, ec='white', lw=1.5, ls='--'))
    a2.set_title('2. Haar cascade face box (solid)\nand central 60% kept (dashed)')

    a3.imshow(cv2.cvtColor(steps['crop'], cv2.COLOR_BGR2RGB))
    a3.set_title(f'3. Face crop ({cw} x {ch} px)')

    dimmed = np.where(steps['mask'][..., None], rgb, (rgb * 0.15).astype(np.uint8))
    a4.imshow(dimmed)
    a4.set_title(f'4. Pixel mask: {steps["mask"].sum():,} nonzero pixels\nof the crop are counted')

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shown = np.where(steps['mask'], grey, 255)
    a5.imshow(shown[cy:cy + ch, cx:cx + cw], cmap='gray', vmin=0, vmax=255)
    a5.set_title('5. Grayscale values of masked pixels')

    for ax in (a1, a2, a3, a4, a5):
        ax.set_xticks([])
        ax.set_yticks([])

    a6.hist(steps['pixels'], bins=np.arange(0, 257, 4), color=BAR, edgecolor='white', lw=0.5)
    ymax = a6.get_ylim()[1]
    for i, name in enumerate(['mean', 'median', 'mode']):
        value = steps['stats'][name]
        if np.isnan(value):
            continue
        color, style = LINES[name]
        a6.axvline(value, color=color, ls=style, lw=2)
        a6.text(4, ymax * (0.92 - 0.09 * i), f'{name} {value:.1f}', color=INK, fontsize=10,
                bbox={'fc': 'white', 'ec': color, 'lw': 1.5, 'boxstyle': 'round,pad=0.25'})
    a6.set_xlim(0, 255)
    a6.set_xlabel('grayscale value (0 = black, 255 = white)', color=MUTED)
    a6.set_ylabel('pixels', color=MUTED)
    a6.set_title('6. Intensity histogram and statistics')
    for side in ('top', 'right'):
        a6.spines[side].set_visible(False)

    fig.suptitle(f'colorism pipeline, step by step ({caption}); numbers are the pipeline output '
                 'for this example', color=INK)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('image', nargs='?', help='a portrait (default: scikit-image astronaut)')
    parser.add_argument('--out', default='figures/walkthrough.png')
    parser.add_argument('--caption', help='label for the image in the figure title')
    args = parser.parse_args(argv)

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            parser.error(f'could not read {args.image}')
        caption = args.caption or Path(args.image).name
    else:
        image = load_demo_image()
        caption = args.caption or 'NASA public-domain sample image'
    steps = run_steps(image)
    plot_walkthrough(image, steps, args.out, caption=caption)
    summary = ', '.join(f'{k} {v:.1f}' for k, v in steps['stats'].items())
    print(f'{summary}\nwrote {args.out}')
    return steps


if __name__ == '__main__':
    main()
