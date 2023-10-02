import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, bootstrap

COLOR_PALETTE = ["#F5793A", "#A95AA1", "#85C0F9", "#0F2080", "#7D6C7A"]


def get_statistics(result):
    mean = np.mean(result, axis=1)
    low = np.percentile(result, 2.5, axis=1)
    high = np.percentile(result, 97.5, axis=1)
    return mean, low, high


def plot_with_statistics(x, mean, low, high, color, legend, linestyle="-", alpha=0.3):
    plt.fill_between(x, low, high, edgecolor=color, facecolor=color, alpha=alpha)
    plt.plot(x, mean, color=color, linestyle=linestyle, label=legend)


def spearman_with_bootstrap_ci(x, y, n_resamples=2_000, ci=0.95):
    return bootstrap(
        (
            x,
            y,
        ),
        lambda x, y, axis=0: spearmanr(x, y, axis=axis)[0],
        n_resamples=n_resamples,
        confidence_level=ci,
        paired=True,
        vectorized=False,
    )
