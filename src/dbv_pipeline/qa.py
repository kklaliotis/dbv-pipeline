"""Shared figure helpers for the QA outputs of the stages.

One visual system for every figure: a fixed categorical colour order (colour follows the entity, never its
rank), thin marks, recessive axes and grid, text in ink colours, a legend whenever there are two or more series.
matplotlib is imported inside the functions so that importing this module costs nothing.
"""

import numpy as np

# Categorical slots, in fixed order (validated palette, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#898781"        # a parent / reference population drawn behind the series
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def style():
    """Apply the shared matplotlib style. Call once at the start of a stage's QA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK_2, "text.color": INK, "axes.titlecolor": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": NEUTRAL, "ytick.color": NEUTRAL, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9,
    })
    return plt


def save(fig, directory, name):
    """Save a figure as PNG in a QA directory and close it. Returns the path."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    path = Path(directory) / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def step_hist(ax, values, bins, color, label, density=False, lw=2.0, fill=False):
    """Histogram drawn as a thin step line (optionally lightly filled). Returns the counts."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    h, edges = np.histogram(values, bins=bins, density=density)
    ax.stairs(h, edges, color=color, lw=lw, label=label, fill=False)
    if fill:
        ax.stairs(h, edges, color=color, alpha=0.15, fill=True, lw=0)
    return h, edges


def funnel(ax, labels, counts, color=SERIES[0]):
    """Horizontal bars for a selection funnel, each labelled with its count and its fraction of the first."""
    counts = np.asarray(counts, float)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, height=0.62, color=color)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.22)
    for yi, c in zip(y, counts):
        ax.text(c + counts.max() * 0.012, yi, f"{int(c):,}  ({100 * c / counts[0]:.0f}%)",
                va="center", ha="left", color=INK_2, fontsize=9)
    ax.set_xlabel("objects")


def binned_fraction(x, selected, bins, min_count=20):
    """Fraction of `selected` per bin of `x`, with binomial errors. Bins with too few objects are NaN."""
    x = np.asarray(x, float)
    selected = np.asarray(selected, bool)
    idx = np.digitize(x, bins) - 1
    n = np.array([(idx == k).sum() for k in range(len(bins) - 1)], float)
    s = np.array([(selected & (idx == k)).sum() for k in range(len(bins) - 1)], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n >= min_count, s / n, np.nan)
        err = np.where(n >= min_count, np.sqrt(f * (1 - f) / n), np.nan)
    return 0.5 * (bins[1:] + bins[:-1]), f, err, n


def rgb_montage(stamps, titles, ncols, suptitle, pixel_scale, stretch=None, q=8.0):
    """Montage of colour stamps.

    stamps: list of (3, H, W) arrays ordered red, green, blue channel (e.g. i, r, g bands), or None for a
    stamp that could not be read. A common asinh stretch is used so that brightness is comparable between
    panels; `stretch` defaults to a robust multiple of the sky noise.
    """
    from astropy.visualization import make_lupton_rgb
    plt = style()
    good = [s for s in stamps if s is not None]
    if stretch is None:
        sky = np.concatenate([s[1][:6].ravel() for s in good] + [s[1][-6:].ravel() for s in good])
        stretch = 12.0 * 1.4826 * np.median(np.abs(sky - np.median(sky)))
    nrows = int(np.ceil(len(stamps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.75 * ncols, 1.95 * nrows + 0.5), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, s, t in zip(axes.ravel(), stamps, titles):
        if s is None:
            ax.text(0.5, 0.5, "not read", ha="center", va="center", transform=ax.transAxes, color=NEUTRAL)
        else:
            ax.imshow(make_lupton_rgb(s[0], s[1], s[2], stretch=stretch, Q=q), origin="lower",
                      interpolation="nearest")
            c = (s.shape[-1] - 1) / 2.0
            ax.plot([c - 6, c - 3], [c, c], color="white", lw=0.8, alpha=0.8)
            ax.plot([c + 3, c + 6], [c, c], color="white", lw=0.8, alpha=0.8)
        ax.set_title(t, fontsize=7.5, fontweight="normal", color=INK_2, loc="center", pad=2)
    n = good[0].shape[-1] if good else 0
    fig.suptitle(f"{suptitle}\n{n} x {n} px = {n * pixel_scale:.1f} arcsec, colour = i r g, common stretch",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig

def make_cutout_montage(stamps, ncols, pixel_scale, suptitle, empty=False):
    """Montage of cutouts for a selection catalogue.

    catalog: DataFrame with the selection catalogue, including the columns "id" and "selected".
    stamps: list of (3, H, W) arrays ordered red, green, blue channel (e.g. i, r, g bands), or None for a
    stamp that could not be read. A common asinh stretch is used so that brightness is comparable between
    panels; `stretch` defaults to a robust multiple of the sky noise.
    """
    if empty:
        titles = [f"empty" for _ in stamps]
    else:
        titles = [f"selected" for _ in stamps]

    return rgb_montage(stamps, titles, ncols, suptitle, pixel_scale)

