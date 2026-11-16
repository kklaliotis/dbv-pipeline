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
    import sys
    import matplotlib
    if "ipykernel" not in sys.modules:      # in a notebook keep the inline backend, so that figures are shown
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

def make_cutout_qa(dataset_dir, bands, pixel_scale, n=24, ncols=6, seed=0, empty=False):
    """Colour montage (i, r, g) of `n` random stamps of a DebvaderDataset directory.

    Returns the figure, or None when the dataset has no stamps or lacks one of the g, r, i bands.
    """
    from pathlib import Path
    dataset_dir = Path(dataset_dir)
    rgb = [b for b in ("i", "r", "g") if b in bands]
    if len(rgb) != 3:
        return None
    images = np.load(dataset_dir / "images.npy", mmap_mode="r")    # (N, H, W, C)
    if len(images) == 0:
        return None
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(images), size=min(n, len(images)), replace=False))
    channels = [list(bands).index(b) for b in rgb]
    stamps = [np.moveaxis(np.asarray(images[k][..., channels], float), -1, 0) for k in pick]
    kind = "empty-sky" if empty else "isolated"
    return make_cutout_montage(stamps, ncols, pixel_scale, f"{len(stamps)} random {kind} stamps of {len(images):,}",
                               empty=empty)


def make_psf_qa(fwhm, ellipticity, checks, bands):
    """
    Make QA plots for the PSF FWHM distributions. We want:
     - FWHM histogram per band with the configured and suggested G_int marked
     - PSF ellipticity histogram per band

    Parameters
    ----------
    fwhm : numpy.ndarray, shape (N, C)
        The measured FWHM values for each object (N) and band (C), arcsec.
    ellipticity : numpy.ndarray, shape (N, C)
        The PSF ellipticity of each object and band.
    checks : dict of dicts {band: check}
        The checks for each band: check["fwhm"] is the configured G_int FWHM, check["suggested_fwhm"] the
        suggested one.
    bands : list of str

    Returns
    -------
    fig_fwhm : matplotlib.figure.Figure
        One panel per band: FWHM histogram, configured (dashed) and suggested (dotted) G_int.
    fig_ellip : matplotlib.figure.Figure
        The PSF ellipticity histogram, one line per band.
    """
    plt = style()
    C = len(bands)
    ncols = min(C, 3)
    nrows = int(np.ceil(C / ncols))
    fig_fwhm, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.ravel()[C:]:
        ax.set_axis_off()
    lo = min(np.nanmin(fwhm), min(c["fwhm"] for c in checks.values())) * 0.9
    hi = np.nanmax(fwhm) * 1.05
    bins = np.linspace(lo, hi, 40)
    for c, (ax, band) in enumerate(zip(axes.ravel(), bands)):
        color = SERIES[c % len(SERIES)]
        step_hist(ax, fwhm[:, c], bins, color, f"{band} PSF", fill=True)
        ax.axvline(checks[band]["fwhm"], color=INK, ls="--", lw=1.2, label="configured G_int")
        ax.axvline(checks[band]["suggested_fwhm"], color=NEUTRAL, ls=":", lw=1.5, label="suggested G_int")
        ax.set_title(f"{band}" + ("" if checks[band]["ok"] else "  (FAILED)"))
        ax.set_xlabel("PSF FWHM [arcsec]")
        if c % ncols == 0:
            ax.set_ylabel("objects")
    axes.ravel()[0].legend(loc="upper right", fontsize=7)
    fig_fwhm.suptitle("PSF FWHM per band against the internal PSF", x=0.01, ha="left", fontweight="bold")
    fig_fwhm.tight_layout()

    fig_ellip, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(0.05, float(np.nanpercentile(ellipticity, 99.5)) * 1.1), 40)
    for c, band in enumerate(bands):
        step_hist(ax, ellipticity[:, c], bins, SERIES[c % len(SERIES)], band, lw=1.6)
    ax.set_xlabel("PSF ellipticity |e| (Gaussian-weighted moments)")
    ax.set_ylabel("objects")
    ax.set_title("PSF ellipticity per band")
    ax.legend(ncols=2)

    return fig_fwhm, fig_ellip


def make_noise_qa(whitened, f_b, f_mean, f_mad, bands):
    """
    Make QA plots for the noise measurement. We want:
    - Histogram of whitened pixels per band, compared against a unit Gaussian
    - Chi-squared per pixel of each stamp, before and after applying the noise factor f_b
    - Noise factor (mean and MAD) per band

    Parameters
    ----------
    whitened : np.ndarray
        The whitened pixels, shape (N, H, W, C), with masked pixels set to NaN.
    f_b : np.ndarray
        The noise factor used downstream, per band.
    f_mean, f_mad : np.ndarray
        The plain and robust estimators, per band.
    bands : list of str

    Returns
    -------
    fig_whitened : matplotlib.figure.Figure
        The figure object for the whitened pixels histogram.
    fig_chi2 : matplotlib.figure.Figure
        The figure object for the per-stamp chi-squared histograms.
    fig_noise_factor : matplotlib.figure.Figure
        The figure object for the noise factors per band.
    """
    from scipy.stats import norm

    plt = style()
    C = whitened.shape[3]
    ncols = min(C, 3)
    nrows = int(np.ceil(C / ncols))

    # Histogram of whitened pixels per band (density, log scale to show the tails), against a unit Gaussian
    fig_whitened, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False)
    bins = np.linspace(-6, 6, 97)
    x = 0.5 * (bins[1:] + bins[:-1])
    for c, ax in enumerate(axes.ravel()):
        if c >= C:
            ax.set_axis_off()
            continue
        step_hist(ax, whitened[..., c], bins, SERIES[c % len(SERIES)], f"{bands[c]}", density=True, fill=True)
        ax.plot(x, norm.pdf(x), color=INK, lw=1.0, ls="--", label="N(0, 1)")
        ax.plot(x, norm.pdf(x, scale=np.sqrt(f_b[c])), color=NEUTRAL, lw=1.0, ls=":",
                label=f"N(0, f_b = {f_b[c]:.3f})")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel("image / sqrt(variance)")
        ax.legend(loc="upper right", fontsize=7)
    fig_whitened.suptitle("Whitened empty-sky pixels", x=0.01, ha="left", fontweight="bold")
    fig_whitened.tight_layout()

    # Chi-squared per pixel of each stamp, before and after applying the noise factor f_b
    with np.errstate(invalid="ignore"):
        chi2 = np.nanmean(whitened**2, axis=(1, 2))             # (N, C)
    fig_chi2, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False)
    finite = chi2[np.isfinite(chi2)]
    hi = float(np.percentile(finite, 99.5)) * 1.1 if finite.size else 2.0
    bins = np.linspace(0, max(hi, 1.5), 50)
    for c, ax in enumerate(axes.ravel()):
        if c >= C:
            ax.set_axis_off()
            continue
        step_hist(ax, chi2[:, c], bins, NEUTRAL, "before f_b")
        step_hist(ax, chi2[:, c] / f_b[c], bins, SERIES[c % len(SERIES)], "after f_b", fill=True)
        ax.axvline(1.0, color=INK, lw=1.0, ls="--")
        ax.set_title(bands[c])
        ax.set_xlabel("chi2 / pixel of a stamp")
        ax.legend(loc="upper right", fontsize=7)
    fig_chi2.suptitle("Empty-sky chi2 per pixel, model = 0", x=0.01, ha="left", fontweight="bold")
    fig_chi2.tight_layout()

    # Noise factor per band, both estimators
    fig_noise_factor, ax = plt.subplots(figsize=(6, 3.6))
    xb = np.arange(C)
    ax.plot(xb, f_mean, "o", color=SERIES[0], label="mean of squares")
    ax.plot(xb, f_mad, "s", color=SERIES[1], label="robust (MAD)")
    ax.axhline(1.0, color=NEUTRAL, lw=1.0, ls="--")
    ax.set_xticks(xb, bands)
    ax.set_xlabel("band")
    ax.set_ylabel("noise factor f_b")
    ax.set_title("Noise factor per band")
    ax.legend()

    return fig_whitened, fig_chi2, fig_noise_factor
