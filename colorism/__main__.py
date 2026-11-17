"""Command line entry point: ``python -m colorism IMAGES_DIR [--labels CSV --group COL]``."""
import argparse
from pathlib import Path

import pandas as pd

from .faces import batch_process
from .report import group_summary, plot_distributions, plot_group_comparison, summarise


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m colorism', description=__doc__)
    parser.add_argument('images', help='folder of .jpg portraits')
    parser.add_argument('--out', default='outputs', help='output folder (default: outputs)')
    parser.add_argument('--labels', help='optional CSV with one row per image file')
    parser.add_argument('--file-col', default='file',
                        help='column in --labels holding the image file name (default: file)')
    parser.add_argument('--group', help='column in --labels to compare groups by, e.g. race')
    parser.add_argument('--title', default='Grayscale intensity of face crops')
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats_df = batch_process(args.images).drop(columns='face_image')
    found = stats_df['mean'].notna().sum()
    print(f'{len(stats_df)} images, {found} with a usable face crop')

    if args.labels:
        labels = pd.read_csv(args.labels)
        labels['file'] = labels[args.file_col].map(lambda f: Path(str(f)).name)
        stats_df = stats_df.merge(labels.drop(columns=[c for c in ['name'] if c in labels]),
                                  on='file', how='left')

    stats_df.to_csv(out / 'image_stats.csv', index=False)
    summary = summarise(stats_df)
    summary.to_csv(out / 'summary.csv')
    print(summary.round(1).to_string())
    plot_distributions(stats_df, out / 'distributions.png', title=args.title)

    if args.group:
        table = group_summary(stats_df, args.group)
        table.to_csv(out / f'by_{args.group}.csv')
        print(table.round(1).to_string())
        plot_group_comparison(stats_df, args.group, out / f'by_{args.group}.png')
    print(f'wrote results to {out}/')


if __name__ == '__main__':
    main()
