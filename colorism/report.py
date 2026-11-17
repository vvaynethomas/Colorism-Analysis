"""Aggregate summaries and figures for a table of per-image grayscale statistics."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .faces import STAT_COLUMNS  # noqa: E402

PLOTTED = ['mean', 'median', 'mode']


def summarise(stats_df):
    """Descriptive statistics of the five grayscale measures over all valid images."""
    return stats_df[STAT_COLUMNS].dropna().describe()


def group_summary(stats_df, group_col, measure='mean'):
    """Count, mean, sd and quartiles of ``measure`` for each value of ``group_col``,
    sorted from darkest to lightest median."""
    valid = stats_df.dropna(subset=[measure, group_col])
    table = valid.groupby(group_col)[measure].describe()
    return table.sort_values('50%')


def plot_distributions(stats_df, path, title='Grayscale intensity of face crops'):
    """Histogram of mean, median and mode intensity across images, one panel each."""
    valid = stats_df[PLOTTED].dropna()
    fig, axes = plt.subplots(1, len(PLOTTED), figsize=(12, 3.6), sharey=True)
    bins = np.arange(0, 256, 10)
    for ax, col in zip(axes, PLOTTED):
        values = valid[col]
        ax.hist(values, bins=bins, color='#4a6fa5', edgecolor='white')
        ax.axvline(values.median(), color='#c44e52', linewidth=1.5)
        ax.set_title(f'{col}  (median {values.median():.0f})')
        ax.set_xlim(0, 255)
        ax.set_xlabel('grayscale value (0 = black, 255 = white)')
    axes[0].set_ylabel('images')
    fig.suptitle(f'{title}, n = {len(valid)}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_comparison(stats_df, group_col, path, measure='mean'):
    """Box plot of ``measure`` per group, ordered by median."""
    valid = stats_df.dropna(subset=[measure, group_col])
    order = valid.groupby(group_col)[measure].median().sort_values().index
    data = [valid.loc[valid[group_col] == g, measure] for g in order]
    labels = [f'{g}\n(n={len(d)})' for g, d in zip(order, data)]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(order)), 4.2))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel(f'{measure} grayscale value of face crop')
    ax.set_ylim(0, 255)
    ax.set_title(f'{measure} face-crop intensity by {group_col}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
