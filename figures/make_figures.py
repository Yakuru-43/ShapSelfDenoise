#!/usr/bin/env python3
"""
make_figures.py
===============
Generates the figures proposed for the ShapSelfDenoise paper revision. Each
figure is a self-contained function; every function degrades gracefully if its
input data is missing (it prints what to run and moves on), so you can generate
whatever you have data for at any moment.

DATA SOURCES (where each figure reads from)
-------------------------------------------
  * out/time_comparison/<ds>/per_input.csv        <- comparet_time.py
  * out/figures_data/ensemble_sweep.csv           <- collect_figure_data.py --task ensemble
  * out/figures_data/selection_ablation.csv       <- collect_figure_data.py --task selection
  * out/figures_data/shap_samples_sweep.csv       <- collect_figure_data.py --task shap_samples
  * out/figures_data/shap_examples.json           <- collect_figure_data.py --task examples
  * out/attack/<ds>/<method>/SHAP_Defence/<prec>/results.txt   <- existing `--mode attack` runs
  * out/certify_<ds>_mask_rate_<r>_sample_size_<n>/results.log <- existing `--mode certify` runs

FIGURES (keys you can pass to --figures)
----------------------------------------
  schematic        method schematic, our pipeline vs the original (NO data needed)
  pareto           accuracy-vs-time Pareto frontier (sweeps the original's ensemble size)
  time_vs_ensemble inference time vs ensemble size
  acc_vs_ensemble  accuracy vs ensemble size
  time_bar         mean inference time, our method vs original (grouped by dataset)
  time_dist        per-example inference-time distribution (box plot)
  robustness       clean / under-attack / with-defence accuracy bars
  maskrate         accuracy vs mask rate (from the certify sweep)
  ablation         selection strategy at equal budget: SHAP vs LIME vs random
  shap_samples     accuracy & time vs the number of SHAP samples

USAGE
-----
  python figures/make_figures.py                       # everything available
  python figures/make_figures.py --figures schematic   # just the schematic (always works)
  python figures/make_figures.py --figures pareto,time_vs_ensemble,acc_vs_ensemble
  python figures/make_figures.py --datasets sst2        # restrict to one dataset
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# Make `figstyle` importable no matter the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Small data readers (each returns None / empty when the data is not there yet)
# --------------------------------------------------------------------------- #
def _csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _err(std_series):
    """Error-bar values: turn a std Series into a clean array (NaN -> 0)."""
    return np.nan_to_num(np.asarray(std_series, dtype=float), nan=0.0)


def load_time_by_dataset(time_dir, datasets):
    out = {}
    for ds in datasets:
        df = _csv(os.path.join(time_dir, ds, "per_input.csv"))
        if df is not None and len(df):
            out[ds] = df
    return out


def parse_attack_results(attack_root):
    """Parse every out/attack/<ds>/<method>/SHAP_Defen[cs]e/<prec>/results.txt.

    Each file has three lines of the form 'Label : value'. The repo is
    inconsistent about British vs American spelling ('defence'/'defense') in
    BOTH the folder name and the line label, and old 'SHAP_Defense' folders can
    coexist with new 'SHAP_Defence' ones for the same run, so we read both and
    keep one row per (dataset, method, precision) -- preferring 'SHAP_Defence',
    which is what the current (bug-fixed) code writes."""
    rows = []
    pattern = os.path.join(attack_root, "*", "*", "SHAP_Defen[cs]e", "*", "results.txt")
    for path in sorted(glob.glob(pattern)):
        parts = path.split(os.sep)
        try:
            i = parts.index("attack")
            dataset, method, defence_folder, precision = (
                parts[i + 1], parts[i + 2], parts[i + 3], parts[i + 4])
        except (ValueError, IndexError):
            continue
        vals = {"original": None, "under_attack": None, "with_defence": None}
        with open(path) as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip().lower()
                try:
                    num = float(val.strip())
                except ValueError:
                    continue
                if "under attack" in key:
                    vals["under_attack"] = num
                elif "defen" in key:            # defence OR defense
                    vals["with_defence"] = num
                elif "original" in key:
                    vals["original"] = num
        rows.append(dict(dataset=dataset, method=method, precision=precision,
                         spelling=defence_folder, **vals))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Prefer the British 'SHAP_Defence' spelling when both exist for a run.
    df["pref"] = (df["spelling"] == "SHAP_Defence").astype(int)
    df = (df.sort_values("pref")
            .drop_duplicates(["dataset", "method", "precision"], keep="last")
            .drop(columns=["pref", "spelling"])
            .sort_values(["dataset", "method"]).reset_index(drop=True))
    return df


def parse_certify(out_root, dataset):
    """Read accuracy at each mask rate from the certify_* result folders."""
    rows = []
    pattern = os.path.join(out_root, f"certify_{dataset}_mask_rate_*_sample_size_*", "results.log")
    for path in glob.glob(pattern):
        text = open(path).read()
        mr = re.search(r"Mask Rate:\s*([0-9.]+)", text)
        acc = re.search(r"Accuracy:\s*([0-9.]+)", text)
        if mr and acc:
            rows.append((round(float(mr.group(1)), 1), float(acc.group(1))))
    if not rows:
        return None
    return (pd.DataFrame(rows, columns=["mask_rate", "accuracy"])
            .drop_duplicates("mask_rate").sort_values("mask_rate").reset_index(drop=True))


def present_datasets(df, datasets):
    return [d for d in datasets if (df["dataset"] == d).any()]


def _skip(name, msg):
    print(f"  [skip] {name}: {msg}")


# --------------------------------------------------------------------------- #
# Figure 1 - Accuracy vs time Pareto frontier  (suggestion #1, the headline)
# --------------------------------------------------------------------------- #
def fig_pareto(ensemble, datasets, fig_dir):
    if ensemble is None:
        return _skip("pareto", "missing out/figures_data/ensemble_sweep.csv "
                               "(run: collect_figure_data.py --task ensemble)")
    dss = present_datasets(ensemble, datasets)
    fig, axes = plt.subplots(1, len(dss), figsize=(5.0 * len(dss), 4.0), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        sub = ensemble[ensemble.dataset == ds]
        sd = (sub[sub.method == "SelfDenoise"]
              .groupby("num_copies")
              .agg(t=("mean_time_per_example", "mean"), tsd=("mean_time_per_example", "std"),
                   a=("accuracy", "mean"), asd=("accuracy", "std"))
              .reset_index().sort_values("num_copies"))
        if len(sd):
            ax.errorbar(sd.t, sd.a, xerr=_err(sd.tsd), yerr=_err(sd.asd),
                        label=fs.LABELS["self"], capsize=2, **fs.STYLE["self"])
            for _, r in sd.iterrows():
                ax.annotate(f"{int(r.num_copies)}", (r.t, r.a),
                            textcoords="offset points", xytext=(4, 5), fontsize=7,
                            color=fs.COLORS["self"])
        sh = sub[sub.method == "ShapSelfDenoise"]
        if len(sh):
            ax.errorbar([sh.mean_time_per_example.mean()], [sh.accuracy.mean()],
                        xerr=[_err([sh.mean_time_per_example.std()])[0]],
                        yerr=[_err([sh.accuracy.std()])[0]],
                        label=fs.LABELS["shap"], capsize=3, linestyle="none", **fs.STYLE["shap"])
        ax.text(0.03, 0.97, "better", transform=ax.transAxes, fontsize=8,
                ha="left", va="top", style="italic", color="#555")
        ax.annotate("", xy=(0.02, 0.99), xytext=(0.12, 0.89), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#999", lw=1))
        ax.set_xlabel("mean inference time / example (s)")
        ax.set_ylabel("accuracy")
        ax.set_title(fs.dataset_title(ds))
        ax.legend(loc="lower right")
    fig.suptitle("Accuracy vs inference time (numbers = SelfDenoise ensemble size)", fontsize=11)
    fs.save(fig, fig_dir, "fig_pareto_accuracy_vs_time")


# --------------------------------------------------------------------------- #
# Figures 2 & 4 - vs ensemble size (shared machinery)            (#2 and #4)
# --------------------------------------------------------------------------- #
def _vs_ensemble(ensemble, datasets, fig_dir, ycol, ylabel, name, suptitle):
    if ensemble is None:
        return _skip(name, "missing ensemble_sweep.csv "
                           "(run: collect_figure_data.py --task ensemble)")
    dss = present_datasets(ensemble, datasets)
    fig, axes = plt.subplots(1, len(dss), figsize=(5.0 * len(dss), 4.0), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        sub = ensemble[ensemble.dataset == ds]
        sd = (sub[sub.method == "SelfDenoise"]
              .groupby("num_copies")
              .agg(m=(ycol, "mean"), s=(ycol, "std")).reset_index().sort_values("num_copies"))
        if len(sd):
            ax.plot(sd.num_copies, sd.m, label=fs.LABELS["self"], **fs.STYLE["self"])
            ax.fill_between(sd.num_copies, sd.m - _err(sd.s), sd.m + _err(sd.s),
                            color=fs.COLORS["self"], alpha=0.15)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sd.num_copies)
            ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        sh = sub[sub.method == "ShapSelfDenoise"]
        if len(sh):
            m, s = sh[ycol].mean(), _err([sh[ycol].std()])[0]
            ax.axhline(m, ls="--", color=fs.COLORS["shap"], lw=2, label=fs.LABELS["shap"])
            ax.axhspan(m - s, m + s, color=fs.COLORS["shap"], alpha=0.12)
        ax.set_xlabel("ensemble size (number of masked copies)")
        ax.set_ylabel(ylabel)
        ax.set_title(fs.dataset_title(ds))
        ax.legend(loc="best")
    fig.suptitle(suptitle, fontsize=11)
    fs.save(fig, fig_dir, name)


def fig_time_vs_ensemble(ensemble, datasets, fig_dir):
    _vs_ensemble(ensemble, datasets, fig_dir, "mean_time_per_example",
                 "mean inference time / example (s)", "fig_time_vs_ensemble",
                 "Inference time grows with the ensemble; ShapSelfDenoise is constant")


def fig_acc_vs_ensemble(ensemble, datasets, fig_dir):
    _vs_ensemble(ensemble, datasets, fig_dir, "accuracy",
                 "accuracy", "fig_accuracy_vs_ensemble",
                 "SelfDenoise needs a large ensemble to match a single SHAP-guided copy")


# --------------------------------------------------------------------------- #
# Figure 3 - mean inference time bar chart                             (#3)
# --------------------------------------------------------------------------- #
def fig_time_bar(time_by_ds, fig_dir):
    if not time_by_ds:
        return _skip("time_bar", "no out/time_comparison/<ds>/per_input.csv "
                                 "(run: comparet_time.py)")
    dss = list(time_by_ds)
    x = np.arange(len(dss))
    w = 0.38
    shap_m = [time_by_ds[d].shap_time.mean() for d in dss]
    shap_s = [time_by_ds[d].shap_time.std() for d in dss]
    self_m = [time_by_ds[d].selfdenoise_time.mean() for d in dss]
    self_s = [time_by_ds[d].selfdenoise_time.std() for d in dss]
    fig, ax = plt.subplots(figsize=(1.6 + 1.8 * len(dss), 4.0))
    b1 = ax.bar(x - w / 2, shap_m, w, yerr=_err(shap_s), capsize=3,
                color=fs.COLORS["shap"], label=fs.LABELS["shap"])
    b2 = ax.bar(x + w / 2, self_m, w, yerr=_err(self_s), capsize=3,
                color=fs.COLORS["self"], label=fs.LABELS["self"])
    fs.annotate_bars(ax, b1, fmt="{:.2f}", dy=max(self_m) * 0.01)
    fs.annotate_bars(ax, b2, fmt="{:.2f}", dy=max(self_m) * 0.01)
    ax.set_xticks(x, [fs.dataset_title(d) for d in dss])
    ax.set_ylabel("mean inference time / example (s)")
    ax.set_title("Per-example inference time")
    ax.legend()
    fs.save(fig, fig_dir, "fig_time_bar")


# --------------------------------------------------------------------------- #
# Figure 3b - per-example time distribution (box plot)                (#3, rigor)
# --------------------------------------------------------------------------- #
def fig_time_dist(time_by_ds, fig_dir):
    if not time_by_ds:
        return _skip("time_dist", "no per_input.csv (run: comparet_time.py)")
    dss = list(time_by_ds)
    data, positions, colors, centers = [], [], [], []
    for i, d in enumerate(dss):
        base = i * 2.0
        data += [time_by_ds[d].shap_time.values, time_by_ds[d].selfdenoise_time.values]
        positions += [base + 0.65, base + 1.35]
        colors += [fs.COLORS["shap"], fs.COLORS["self"]]
        centers.append(base + 1.0)
    fig, ax = plt.subplots(figsize=(1.6 + 1.8 * len(dss), 4.0))
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    medianprops=dict(color="black"), showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(centers, [fs.dataset_title(d) for d in dss])
    ax.set_ylabel("inference time / example (s)")
    ax.set_title("Inference-time distribution per example")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="none",
                          markerfacecolor=fs.COLORS[k], markeredgecolor="none",
                          markersize=9, label=fs.LABELS[k]) for k in ("shap", "self")]
    ax.legend(handles=handles, loc="upper left")
    fs.save(fig, fig_dir, "fig_time_distribution")


# --------------------------------------------------------------------------- #
# Figure 5 - robustness bars                                          (#5)
# --------------------------------------------------------------------------- #
def fig_robustness(attack_df, datasets, fig_dir):
    if attack_df is None:
        return _skip("robustness", "no out/attack/.../SHAP_Defence/.../results.txt "
                                   "(run: `--mode attack` for each dataset/method)")
    df = attack_df[attack_df.dataset.isin(datasets)].reset_index(drop=True)
    if not len(df):
        return _skip("robustness", "attack results found, but none for the chosen datasets")
    groups = [f"{fs.dataset_title(r.dataset)}\n{r.method}" for _, r in df.iterrows()]
    x = np.arange(len(groups))
    w = 0.26
    series = [("original", "none", "Clean (no attack)"),
              ("under_attack", "attack", "Under attack"),
              ("with_defence", "shap", "With SHAP defence")]
    fig, ax = plt.subplots(figsize=(2.0 + 1.7 * len(groups), 4.2))
    for j, (col, ckey, lab) in enumerate(series):
        bars = ax.bar(x + (j - 1) * w, df[col].values, w,
                      color=fs.COLORS[ckey], label=lab)
        fs.annotate_bars(ax, bars, fmt="{:.2f}")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Robustness: clean vs under attack vs defended")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fs.save(fig, fig_dir, "fig_robustness_bars")


# --------------------------------------------------------------------------- #
# Figure 7 - accuracy vs mask rate (certify sweep)                    (#7)
# --------------------------------------------------------------------------- #
def fig_maskrate(certify_by_ds, fig_dir):
    if not certify_by_ds:
        return _skip("maskrate", "no certify_* result folders "
                                 "(run: scripts/certify/<ds>/certify.sh)")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    palette = [fs.COLORS["shap"], fs.COLORS["self"], fs.COLORS["lime"]]
    for (ds, df), c in zip(certify_by_ds.items(), palette):
        ax.plot(df.mask_rate, df.accuracy, marker="o", color=c,
                linewidth=2, label=fs.dataset_title(ds))
    ax.set_xlabel("mask rate")
    ax.set_ylabel("accuracy (certify sweep)")
    ax.set_title("Accuracy vs mask rate")
    ax.legend()
    fs.save(fig, fig_dir, "fig_accuracy_vs_maskrate")


# --------------------------------------------------------------------------- #
# Figure 9 - selection-strategy ablation at equal budget              (#9)
# --------------------------------------------------------------------------- #
def fig_ablation(sel_df, datasets, fig_dir):
    if sel_df is None:
        return _skip("ablation", "missing selection_ablation.csv "
                                 "(run: collect_figure_data.py --task selection)")
    dss = present_datasets(sel_df, datasets)
    order = ["random", "lime", "shap"]   # ours last so it reads as the punchline
    x = np.arange(len(dss))
    w = 0.26
    fig, ax = plt.subplots(figsize=(2.0 + 1.8 * len(dss), 4.0))
    for j, strat in enumerate(order):
        means, stds = [], []
        for ds in dss:
            cell = sel_df[(sel_df.dataset == ds) & (sel_df.strategy == strat)]
            means.append(cell.accuracy.mean() if len(cell) else np.nan)
            stds.append(cell.accuracy.std() if len(cell) else 0.0)
        bars = ax.bar(x + (j - 1) * w, means, w, yerr=_err(pd.Series(stds)), capsize=3,
                      color=fs.COLORS[strat],
                      label={"shap": "SHAP (ours)", "lime": "LIME", "random": "Random"}[strat])
        fs.annotate_bars(ax, bars, fmt="{:.2f}")
    ax.set_xticks(x, [fs.dataset_title(d) for d in dss])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Word-selection strategy at equal budget (one masked copy)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fs.save(fig, fig_dir, "fig_selection_ablation")


# --------------------------------------------------------------------------- #
# Figure 10 - accuracy & time vs number of SHAP samples               (#10)
# --------------------------------------------------------------------------- #
def fig_shap_samples(ss_df, datasets, fig_dir):
    if ss_df is None:
        return _skip("shap_samples", "missing shap_samples_sweep.csv "
                                     "(run: collect_figure_data.py --task shap_samples)")
    dss = present_datasets(ss_df, datasets)
    fig, axes = plt.subplots(1, len(dss), figsize=(5.0 * len(dss), 4.0), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        g = (ss_df[ss_df.dataset == ds].groupby("n_samples")
             .agg(a=("accuracy", "mean"), t=("mean_time_per_example", "mean"))
             .reset_index().sort_values("n_samples"))
        ax.plot(g.n_samples, g.a, marker="o", color=fs.COLORS["shap"], linewidth=2,
                label="accuracy")
        ax.set_xlabel("SHAP samples (captum n_samples)")
        ax.set_ylabel("accuracy", color=fs.COLORS["shap"])
        ax.tick_params(axis="y", labelcolor=fs.COLORS["shap"])
        ax.set_title(fs.dataset_title(ds))
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.plot(g.n_samples, g.t, marker="s", ls="--", color=fs.COLORS["self"],
                 linewidth=2, label="time")
        ax2.set_ylabel("mean time / example (s)", color=fs.COLORS["self"])
        ax2.tick_params(axis="y", labelcolor=fs.COLORS["self"])
    fig.suptitle("Accuracy and cost vs the number of SHAP samples", fontsize=11)
    fs.save(fig, fig_dir, "fig_shap_samples_tradeoff")


# --------------------------------------------------------------------------- #
# Figure 11 - method schematic (NO data required)                     (#11)
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, color, fc=None, fontsize=9):
    fc = fc or "white"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.6, edgecolor=color, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, color="#222")
    return (x, y + h / 2), (x + w, y + h / 2)  # (left-anchor, right-anchor)


def _arrow(ax, p0, p1, color="#555"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.5, color=color, zorder=1,
                                 shrinkA=2, shrinkB=2))


def fig_schematic(fig_dir):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    w, h = 1.75, 0.85

    # ---- Top row: ShapSelfDenoise (single SHAP-guided path) ---------------- #
    ax.text(0.05, 5.55, "ShapSelfDenoise (ours)", fontsize=12, fontweight="bold",
            color=fs.COLORS["shap"])
    y = 4.45
    xs = [0.1, 2.15, 4.2, 6.25, 8.3, 10.35]
    labels = ["Input\nsentence", "SHAP\nimportance", "Mask\ntop-p%",
              "Denoise\n(LLM)", "Classify", "Prediction"]
    anchors = []
    for x, lab in zip(xs, labels):
        ww = w if lab != "Prediction" else 1.55
        anchors.append(_box(ax, x, y, ww, h, lab, fs.COLORS["shap"],
                            fc="#eaf1fb" if lab in ("SHAP\nimportance", "Mask\ntop-p%") else "white"))
    for i in range(len(anchors) - 1):
        _arrow(ax, anchors[i][1], anchors[i + 1][0], color=fs.COLORS["shap"])
    ax.text(6.25 + w / 2, y - 0.35, "one copy", ha="center", fontsize=8,
            style="italic", color=fs.COLORS["shap"])

    # ---- Bottom row: original SelfDenoise (random ensemble + vote) --------- #
    ax.text(0.05, 2.75, "SelfDenoise (original)", fontsize=12, fontweight="bold",
            color=fs.COLORS["self"])
    y = 1.55
    a_in = _box(ax, 0.1, y, w, h, "Input\nsentence", fs.COLORS["self"])
    # stacked "N random masks"
    for k, dx in enumerate((0.16, 0.08, 0.0)):
        xx, yy = 2.15 + dx, y + 0.18 - 0.09 * k
        ax.add_patch(FancyBboxPatch((xx, yy), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                    linewidth=1.4, edgecolor=fs.COLORS["self"],
                                    facecolor="#fdeee2", zorder=2))
    ax.text(2.15 + w / 2 + 0.05, y + h / 2 + 0.05, "N random\nmasks", ha="center",
            va="center", fontsize=9, zorder=3)
    a_mask_r = (2.15 + w + 0.16, y + h / 2)
    a_den = _box(ax, 4.2, y, w, h, "Denoise\n(LLM) \u00d7N", fs.COLORS["self"])
    a_cls = _box(ax, 6.25, y, w, h, "Classify\n\u00d7N", fs.COLORS["self"])
    a_vote = _box(ax, 8.3, y, w, h, "Majority\nvote", fs.COLORS["self"], fc="#fdeee2")
    a_pred = _box(ax, 10.35, y, 1.55, h, "Prediction", fs.COLORS["self"])
    _arrow(ax, a_in[1], (2.15, y + h / 2), color=fs.COLORS["self"])
    for p0, p1 in [(a_mask_r, a_den[0]), (a_den[1], a_cls[0]),
                   (a_cls[1], a_vote[0]), (a_vote[1], a_pred[0])]:
        _arrow(ax, p0, p1, color=fs.COLORS["self"])
    ax.text(6.25 + w / 2, y - 0.35, "N copies (default 100)", ha="center", fontsize=8,
            style="italic", color=fs.COLORS["self"])

    fig.suptitle("One SHAP-guided masked copy vs an ensemble of random masks",
                 fontsize=12)
    fs.save(fig, fig_dir, "fig_method_schematic")


# --------------------------------------------------------------------------- #
# Figure 12 - SHAP heatmap worked example                             (#12)
# --------------------------------------------------------------------------- #
def fig_heatmap(examples, fig_dir, ncol=8):
    if not examples:
        return _skip("heatmap", "missing shap_examples.json "
                                "(run: collect_figure_data.py --task examples)")
    cmap = plt.get_cmap("Reds")
    made = 0
    for k, ex in enumerate(examples):
        words = ex.get("words") or ex.get("text", "").split()
        vals = np.asarray(ex.get("shap_values", []), dtype=float)
        if not len(words) or len(vals) != len(words):
            continue
        masked = set(ex.get("masked_indices", []))
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = plt.Normalize(vmin, vmax if vmax > vmin else vmin + 1e-9)
        nrow = int(np.ceil(len(words) / ncol))
        fig, ax = plt.subplots(figsize=(1.1 * ncol, 0.9 * nrow + 1.4))
        ax.set_xlim(0, ncol)
        ax.set_ylim(-nrow, 1)
        ax.axis("off")
        for i, word in enumerate(words):
            col, row = i % ncol, i // ncol
            x, y = col, -row
            ax.add_patch(Rectangle((x + 0.03, y - 0.78), 0.94, 0.74,
                                   facecolor=cmap(norm(vals[i])),
                                   edgecolor=(fs.COLORS["shap"] if i in masked else "#dddddd"),
                                   linewidth=(2.4 if i in masked else 0.6), zorder=2))
            txt = "[mask]" if i in masked else word
            ax.text(x + 0.5, y - 0.41, txt, ha="center", va="center", fontsize=8,
                    color=("white" if norm(vals[i]) > 0.6 else "#222"),
                    fontweight=("bold" if i in masked else "normal"), zorder=3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, orientation="horizontal")
        cbar.set_label("SHAP importance", fontsize=9)
        pb, pa = ex.get("pred_before", "?"), ex.get("pred_after", "?")
        lab = ex.get("label", "?")
        ax.set_title(f"{fs.dataset_title(ex.get('dataset', ''))}: "
                     f"prediction {pb} \u2192 {pa}   (true label {lab})", fontsize=10)
        fs.save(fig, fig_dir, f"fig_shap_heatmap_{ex.get('dataset', 'ex')}_{k}")
        made += 1
    if not made:
        _skip("heatmap", "shap_examples.json had no usable examples "
                         "(need aligned 'words' and 'shap_values')")


# --------------------------------------------------------------------------- #
# Registry + CLI
# --------------------------------------------------------------------------- #
def build_registry(ctx):
    return {
        "schematic":        lambda: fig_schematic(ctx["fig_dir"]),
        "pareto":           lambda: fig_pareto(ctx["ensemble"], ctx["datasets"], ctx["fig_dir"]),
        "time_vs_ensemble": lambda: fig_time_vs_ensemble(ctx["ensemble"], ctx["datasets"], ctx["fig_dir"]),
        "acc_vs_ensemble":  lambda: fig_acc_vs_ensemble(ctx["ensemble"], ctx["datasets"], ctx["fig_dir"]),
        "time_bar":         lambda: fig_time_bar(ctx["time_by_ds"], ctx["fig_dir"]),
        "time_dist":        lambda: fig_time_dist(ctx["time_by_ds"], ctx["fig_dir"]),
        "robustness":       lambda: fig_robustness(ctx["attack"], ctx["datasets"], ctx["fig_dir"]),
        "maskrate":         lambda: fig_maskrate(ctx["certify_by_ds"], ctx["fig_dir"]),
        "ablation":         lambda: fig_ablation(ctx["selection"], ctx["datasets"], ctx["fig_dir"]),
        "shap_samples":     lambda: fig_shap_samples(ctx["shap_samples"], ctx["datasets"], ctx["fig_dir"]),
        "heatmap":          lambda: fig_heatmap(ctx["examples"], ctx["fig_dir"]),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures", default="all",
                   help="Comma-separated subset, or 'all'. Keys: " +
                        "schematic,pareto,time_vs_ensemble,acc_vs_ensemble,time_bar,"
                        "time_dist,robustness,maskrate,ablation,shap_samples,heatmap")
    p.add_argument("--datasets", default="sst2,agnews",
                   help="Comma-separated datasets to include.")
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "out", "figures_data"),
                   help="Where the sweep CSVs / examples json live.")
    p.add_argument("--time-dir", default=os.path.join(REPO_ROOT, "out", "time_comparison"),
                   help="Where comparet_time.py wrote per_input.csv.")
    p.add_argument("--attack-root", default=os.path.join(REPO_ROOT, "out", "attack"),
                   help="Root of the attack-mode outputs.")
    p.add_argument("--out-root", default=os.path.join(REPO_ROOT, "out"),
                   help="Root holding the certify_* folders.")
    p.add_argument("--fig-dir", default=os.path.join(REPO_ROOT, "figures", "output"),
                   help="Where to write the generated figures.")
    return p.parse_args()


def main():
    args = parse_args()
    fs.set_style()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    ctx = {
        "datasets": datasets,
        "fig_dir": args.fig_dir,
        "ensemble": _csv(os.path.join(args.data_dir, "ensemble_sweep.csv")),
        "selection": _csv(os.path.join(args.data_dir, "selection_ablation.csv")),
        "shap_samples": _csv(os.path.join(args.data_dir, "shap_samples_sweep.csv")),
        "time_by_ds": load_time_by_dataset(args.time_dir, datasets),
        "attack": parse_attack_results(args.attack_root),
        "certify_by_ds": {d: parse_certify(args.out_root, d)
                          for d in datasets if parse_certify(args.out_root, d) is not None},
        "examples": (json.load(open(os.path.join(args.data_dir, "shap_examples.json")))
                     if os.path.exists(os.path.join(args.data_dir, "shap_examples.json")) else None),
    }

    registry = build_registry(ctx)
    keys = list(registry) if args.figures == "all" else \
        [k.strip() for k in args.figures.split(",") if k.strip()]

    print(f"Figures -> {args.fig_dir}")
    for key in keys:
        if key not in registry:
            print(f"  [warn] unknown figure '{key}' (skipped)")
            continue
        try:
            registry[key]()
        except Exception as e:  # one broken figure must not kill the rest
            print(f"  [error] {key}: {type(e).__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()