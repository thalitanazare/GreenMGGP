"""Additional result figures (presentation only; reads the saved runs, reruns nothing).

* ``fig_fronts.pdf``      -- returned sets of the representative runs (rho = 1) in the (cost, error) plane,
                             identification (top) and free-run validation (bottom), tau-selected model ringed;
* ``fig_ab.pdf``          -- selected models in the (a, b) plane of additions and multiplications, with the
                             feasible lattice and lines of constant cost for rho = 1 and rho_K;
* ``fig_regressors.pdf``  -- frequency of each regressor in the selected model and in the returned set (E2, E3).

The representative run of each method is the one declared in the study: median validation NRMSE of the
selected model (ties by seed), the same rule for every method.

Run:  python figures_results.py
"""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "Resultados")
FIG = os.path.join(ROOT, "27Collab-Draft", "figures")
TAU, ETA = 0.05, 1e-6
RHO_K = 729 / 64
CONFIGS = ["SO", "MO-lex", "Green"]
COL = {"SO": "#555555", "MO-lex": "#E69F00", "Green": "#009E73"}
MK = {"SO": "s", "MO-lex": "D", "Green": "o"}
SEARCH = {"E1": dict(G=5, h=2), "E2": dict(G=5, h=2), "E3": dict(G=8, h=3)}
TITLE = {"E1": "E1: Tutorial (noise-free)", "E2": "E2: Tutorial (noisy)", "E3": "E3: Bouc–Wen hysteresis"}
TRUE_TERMS = {"y(k-1)", "u(k)", "u(k)*y(k-1)"}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(cost):
    runs = json.load(open(os.path.join(RES, cost, "runs.json")))
    df = pd.read_csv(os.path.join(RES, cost, "per_run_metrics.csv"))
    return runs, df


def select_tau(models):
    jmin = min(m["J"] for m in models)
    ok = [m for m in models if m["J"] <= (1 + TAU) * jmin + ETA]
    return min(ok, key=lambda m: (m["C"], m["J"]))


def median_run(runs, df, key, cfg):
    sub = df[(df.example == key) & (df.config == cfg)].sort_values(["sel_val", "seed"]).reset_index(drop=True)
    seed = int(sub.loc[len(sub) // 2, "seed"])
    return next(r for r in runs if r["example"] == key and r["config"] == cfg and r["seed"] == seed)


def tex_term(t):
    """'u(k-1)*y(k-1)' -> math text with powers."""
    c = Counter(t.split("*"))
    parts = []
    for f, p in c.items():
        f = f.replace("(k-", "(k\\!-\\!")
        parts.append(f + (f"^{p}" if p > 1 else ""))
    return "$" + "\\,".join(parts) + "$"


def style(ax):
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# returned sets of representative runs
# ---------------------------------------------------------------------------
def fig_fronts(runs, df):
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.5), sharex="col")
    for j, key in enumerate(["E1", "E2", "E3"]):
        seeds = []
        for cfg in CONFIGS:
            r = median_run(runs, df, key, cfg)
            seeds.append(f"{cfg} {r['seed']}")
            ms = sorted(r["delivered"], key=lambda m: (m["C"], m["J"]))
            sel = select_tau(r["delivered"])
            for i, err in enumerate(["J", "val"]):
                ax = axs[i, j]
                C = np.array([m["C"] for m in ms]); E = np.maximum([m[err] for m in ms], 1e-16)
                if len(ms) > 1:
                    ax.step(C, E, where="post", color=COL[cfg], lw=0.9, alpha=0.85, zorder=2)
                ax.scatter(C, E, s=16, marker=MK[cfg], color=COL[cfg], edgecolor="white", lw=0.5, zorder=3)
                ax.scatter([sel["C"]], [max(sel[err], 1e-16)], s=70, facecolor="none", edgecolor=COL[cfg],
                           lw=1.0, zorder=4)
        for i in range(2):
            axs[i, j].set_yscale("log"); style(axs[i, j])
        axs[0, j].set_title(TITLE[key])
        axs[1, j].set_xlabel(r"arithmetic cost $C_1$ (operations per sample)")
        print(key, "representative seeds:", ", ".join(seeds))
    axs[0, 0].set_ylabel("identification NRMSE"); axs[1, 0].set_ylabel("validation NRMSE")
    handles = [Line2D([], [], marker=MK[c], color=COL[c], mec="white", ms=5, lw=0.9, label=c) for c in CONFIGS]
    handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", ms=8, label=r"selected $\mathcal{M}_\tau$"))
    plt.tight_layout(h_pad=0.6, rect=(0, 0.06, 1, 1))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.8)
    fig.savefig(os.path.join(FIG, "fig_fronts.pdf")); plt.close(fig)


# ---------------------------------------------------------------------------
# (a, b) plane
# ---------------------------------------------------------------------------
def fig_ab(df1, dfK):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.45))
    groups = [("SO", df1, "SO", dict(marker="s", fc=COL["SO"], ec=COL["SO"]), r"SO"),
              ("Green", df1, "Green", dict(marker="o", fc=COL["Green"], ec=COL["Green"]), r"Green, $\rho=1$"),
              ("Green", dfK, "GreenK", dict(marker="o", fc="white", ec=COL["Green"]), r"Green, $\rho_K$")]
    offs = {"SO": (0.0, 0.0), "Green": (-0.17, 0.0), "GreenK": (0.17, 0.0)}
    for ax, key in zip(axs, ["E1", "E2", "E3"]):
        G, D = SEARCH[key]["G"], 2 ** SEARCH[key]["h"]
        lat = [(a, b) for a in range(1, G + 1) for b in range(a, a * D + 1)]
        amax = max(df1[df1.example == key].sel_p1.max(), dfK[dfK.example == key].sel_p1.max()) + 1.5
        bmax = max(df1[df1.example == key].sel_sumd.max(), dfK[dfK.example == key].sel_sumd.max()) + 3
        lat = [(a, b) for a, b in lat if a <= amax and b <= bmax]
        ax.scatter(*zip(*lat), s=3, color="0.8", zorder=1, lw=0)
        modes = {}
        for cfg, d, tag, mk, lab in groups:
            sub = d[(d.example == key) & (d.config == cfg)]
            cnt = Counter(zip(sub.sel_p1, sub.sel_sumd))
            modes[tag] = cnt.most_common(1)[0][0]
            for (a, b), n in cnt.items():
                ax.scatter(a + offs[tag][0], b, s=8 + 2.6 * n, marker=mk["marker"], facecolor=mk["fc"],
                           edgecolor=mk["ec"], lw=0.9, zorder=3, alpha=0.9)
        # lines of constant cost through the most frequent Green selections
        aa = np.linspace(0.5, amax, 50)
        a1, b1 = modes["Green"]; aK, bK = modes["GreenK"]
        ax.plot(aa, (a1 + b1) - aa, color=COL["Green"], lw=0.8, ls="-", alpha=0.7, zorder=2)
        ax.plot(aa, bK + (aK - aa) / RHO_K, color=COL["Green"], lw=0.8, ls="--", alpha=0.7, zorder=2)
        ax.set_xlim(0.5, amax); ax.set_ylim(0.5, bmax)
        ax.set_xlabel(r"additions $a$ (distinct terms)")
        ax.set_title(TITLE[key]); style(ax)
    axs[0].set_ylabel(r"multiplications $b$")
    handles = [Line2D([], [], marker="s", ls="", color=COL["SO"], ms=5, label="SO"),
               Line2D([], [], marker="o", ls="", color=COL["Green"], ms=5, label=r"Green, $\rho=1$"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=COL["Green"], ms=5, label=r"Green, $\rho_K$"),
               Line2D([], [], color=COL["Green"], lw=0.8, label=r"constant $C_1$"),
               Line2D([], [], color=COL["Green"], lw=0.8, ls="--", label=r"constant $C_{\rho_K}$"),
               Line2D([], [], marker="o", ls="", color="0.8", ms=3, label=r"feasible $(a,b)$")]
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.3, columnspacing=1.4)
    fig.savefig(os.path.join(FIG, "fig_ab.pdf")); plt.close(fig)


# ---------------------------------------------------------------------------
# regressor frequencies
# ---------------------------------------------------------------------------
def fig_regressors(runs, df, n_rows=10):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.0))
    cols = [(cfg, kind) for kind in ("selected", "set") for cfg in CONFIGS]
    for ax, key in zip(axs, ["E2", "E3"]):
        freq = {c: Counter() for c in cols}
        n = {cfg: 0 for cfg in CONFIGS}
        for r in runs:
            if r["example"] != key:
                continue
            cfg = r["config"]; n[cfg] += 1
            for t in set(select_tau(r["delivered"])["terms"]):
                freq[(cfg, "selected")][t] += 1
            for t in {t for m in r["delivered"] for t in m["terms"]}:
                freq[(cfg, "set")][t] += 1
        allt = Counter()
        for c in cols:
            for t, v in freq[c].items():
                allt[t] = max(allt[t], v / n[c[0]])
        rows = [t for t, _ in allt.most_common(n_rows)]
        if key == "E2":
            rows = sorted(TRUE_TERMS) + [t for t in rows if t not in TRUE_TERMS][: n_rows - 3]
        M = np.array([[freq[c][t] / n[c[0]] for c in cols] for t in rows])
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.8,
                        color="white" if v > 0.6 else "0.25")
        ax.set_xticks(range(len(cols)), [c[0] for c in cols], rotation=0)
        ax.set_yticks(range(len(rows)), [tex_term(t) for t in rows])
        if key == "E2":
            for lab in ax.get_yticklabels()[:3]:
                lab.set_fontweight("bold")
            ax.axhline(2.5, color="0.2", lw=0.8)
        ax.axvline(2.5, color="white", lw=2.0)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.text(1, -0.7, r"selected model $\mathcal{M}_\tau$", ha="center", va="bottom", fontsize=7)
        ax.text(4, -0.7, "returned set", ha="center", va="bottom", fontsize=7)
        ax.set_title(TITLE[key], pad=16)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("fraction of runs (%)"); cb.set_ticks([0, 0.5, 1], labels=["0", "50", "100"])
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIG, "fig_regressors.pdf")); plt.close(fig)


def main():
    runs1, df1 = load("ops")
    _, dfK = load("karatsuba")
    fig_fronts(runs1, df1)
    fig_ab(df1, dfK)
    fig_regressors(runs1, df1)
    print("saved fig_fronts.pdf, fig_ab.pdf, fig_regressors.pdf in", FIG)


if __name__ == "__main__":
    main()
