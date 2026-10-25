# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fengrui Zhang
# Adapted from report_figures.ipynb; see README.md.
"""Plot saved results for the original five-rate, three-speed scan layout."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from abp import msd_theory

def make_report_figures(data_root, destination):
    data_root, destination = Path(data_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data_root / "summary.json").read_text())
    drs = [.25, .5, 1., 2., 4.]
    colors = ["#285f9c", "#35898b", "#55a868", "#c98524", "#a65781"]
    tag = lambda x: f"{x:g}".replace(".", "p")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.labelsize": 14, "axes.titlesize": 14,
        "legend.fontsize": 11, "axes.grid": True,
        "grid.alpha": .17, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.8,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for dr, color, row in zip(drs, colors, summary["correlation"]):
        d = np.load(data_root / "data" / f"correlation_DR_{tag(dr)}.npz", allow_pickle=False)
        t = d["time"]
        axes[0].plot(t[::6], d["mean"][::6], ".", color=color,
                     ms=4, label=fr"$D_R={dr:g}$")
        axes[0].plot(t, np.exp(-t/row["tau_fit"]), color=color)
    axes[0].set(xlim=(0, 12), ylim=(-.03, 1.03),
                xlabel="Lag $t$", ylabel="$C(t)$",
                title="(a) Direction autocorrelation")
    axes[0].legend(loc="upper right", fontsize=9.5)
    rows = summary["correlation"]
    x = np.array(drs)
    y = np.array([r["tau_fit"] for r in rows])
    lower = np.array([r["tau_ci_low"] for r in rows])
    upper = np.array([r["tau_ci_high"] for r in rows])
    midpoint = (lower + upper) / 2
    axes[1].plot(x, 1/x, "k--", label=r"$1/D_R$")
    # A percentile interval need not contain the fitted point, especially in
    # a small example. Draw its actual bounds without clipping either end.
    axes[1].errorbar(x, midpoint,
                     yerr=np.vstack([midpoint - lower, upper - midpoint]),
                     fmt="none", capsize=4, color=colors[0])
    axes[1].plot(x, y, "o", color=colors[0], label="Fit and 95% CI")
    axes[1].set(xscale="log", yscale="log", xlabel="$D_R$",
                ylabel=r"$\tau_R$", title="(b) Persistence time")
    axes[1].set_xticks(drs, labels=[f"{v:g}" for v in drs])
    axes[1].set_yticks(drs, labels=[f"{v:g}" for v in drs])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].legend()
    fig.tight_layout(pad=.8, w_pad=1.2)
    fig.savefig(destination / "report_correlation.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for dr, color in zip(drs, colors):
        d = np.load(data_root / "data" / f"msd_v_1_DR_{tag(dr)}.npz", allow_pickle=False)
        t = d["time"][1:]
        axes[0, 0].loglog(t, d["mean"][1:], color=color,
                         label=fr"$D_R={dr:g}$")
        axes[0, 0].loglog(t, msd_theory(t, 1, dr), "k:", lw=.9)
    for v, color in zip([.5, 1., 2.], colors):
        d = np.load(data_root / "data" / f"msd_v_{tag(v)}_DR_1.npz", allow_pickle=False)
        t = d["time"][1:]
        axes[0, 1].loglog(t, d["mean"][1:], color=color,
                         label=fr"$v={v:g}$")
        axes[0, 1].loglog(t, msd_theory(t, v, 1), "k:", lw=.9)
    axes[0, 0].set(title="(a) Rotation scan at $v=1$",
                   xlabel="$t$", ylabel="MSD")
    axes[0, 1].set(title="(b) Speed scan at $D_R=1$",
                   xlabel="$t$", ylabel="MSD")
    for ax in axes[0]:
        ax.legend(fontsize=9.5, ncol=2)
    for ax, key, fixed in [(axes[1, 0], "DR", "v"),
                           (axes[1, 1], "v", "DR")]:
        rows = sorted([r for r in summary["diffusion"]
                       if r[fixed] == 1], key=lambda r: r[key])
        x = np.array([r[key] for r in rows])
        y = np.array([r["D_fit"] for r in rows])
        error = np.array([
            [r["D_fit"]-r["D_ci_low"] for r in rows],
            [r["D_ci_high"]-r["D_fit"] for r in rows]
        ])
        ax.errorbar(x, y, yerr=error, fmt="o", capsize=4,
                    color=colors[0], label="Simulation")
        ax.plot(x, 1/(2*x) if key == "DR" else x*x/2,
                "k--", label="Theory")
        ax.set(xscale="log", yscale="log",
               xlabel="$D_R$" if key == "DR" else "$v$",
               ylabel=r"$D_{\mathrm{eff}}$")
        ax.set_xticks(x, labels=[f"{v:g}" for v in x])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([.125, .5, 2], labels=["0.125", "0.5", "2"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(fontsize=9.5)
    axes[1, 0].set_title("(c) Diffusion versus rotation")
    axes[1, 1].set_title("(d) Diffusion versus speed")
    fig.tight_layout(pad=.8, h_pad=1.2, w_pad=1.2)
    fig.savefig(destination / "report_msd.png")
    plt.close(fig)
