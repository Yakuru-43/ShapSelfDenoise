"""
figstyle.py
===========
Shared, publication-oriented matplotlib style and small helpers used by every
figure in `make_figures.py`. Keeping this in one place means all figures share
the same fonts, colours and sizes, so they look like they belong to the same
paper.

Nothing here touches the model or the data; it is pure presentation.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe: never needs a display / X server
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Colours and labels. ONE colour per concept, used consistently everywhere so a
# reader who learns "blue = our method" on Figure 1 keeps that mapping later.
# --------------------------------------------------------------------------- #
COLORS = {
    "shap":   "#2c6fbb",   # ShapSelfDenoise (our method)
    "self":   "#e07b39",   # original SelfDenoise (random ensemble)
    "none":   "#9aa0a6",   # undefended baseline
    "attack": "#c0392b",   # "under attack" bars
    "lime":   "#7d4fb0",   # LIME-masking ablation variant
    "random": "#3f9b5d",   # random-masking ablation variant
}

LABELS = {
    "shap":   "ShapSelfDenoise (ours)",
    "self":   "SelfDenoise",
    "none":   "No defence",
    "attack": "Under attack",
    "lime":   "LIME masking",
    "random": "Random masking",
}

# Marker / linestyle defaults for the two main methods.
STYLE = {
    "shap": dict(color=COLORS["shap"], marker="*", markersize=13, linewidth=2),
    "self": dict(color=COLORS["self"], marker="o", markersize=6,  linewidth=2),
}


def set_style():
    """Apply the global rcParams. Call once at the start of make_figures."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        # DejaVu Serif ships with matplotlib, so this never falls back to a
        # missing-font warning, yet still gives the LaTeX-ish serif look.
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })


def save(fig, fig_dir, name):
    """Write a figure as BOTH a vector PDF (for LaTeX) and a PNG (for preview)."""
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  [saved] {os.path.join(fig_dir, name)}.pdf (+ .png)")


def dataset_title(name):
    """Pretty dataset names for axis / panel titles."""
    return {"sst2": "SST-2", "agnews": "AG News"}.get(name, name)


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=8, dy=0.01):
    """Print the numeric value on top of each bar (nice for slide-ready plots)."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)