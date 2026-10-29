"""Figures for the stage-two report, from runs/main_lr=0.003_steps=15000."""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = "runs/main_lr=0.003_steps=15000"
OUT = Path("figures")
VARIANTS = ["single_token", "frozen", "linear", "softmax"]
LABELS = {"single_token": "single token", "frozen": "frozen attention",
          "linear": "linear attention", "softmax": "softmax (control)"}
COLORS = {"single_token": "#2a6fdb", "frozen": "#1b9e77", "linear": "#d95f02", "softmax": "#7f7f7f"}


def load():
    d = collections.defaultdict(list)
    for f in glob.glob(f"{RUNS}/K*.json"):
        r = json.load(open(f))
        d[(r["config"]["variant"], r["config"]["K"])].append(r)
    return d


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    Ks = sorted({k for _, k in d})
    d_mlp = next(iter(d.values()))[0]["config"]["d_mlp"]

    # 1. K_eff vs K, against the naive one-neuron-per-AND baseline
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(Ks, [min(k, d_mlp) for k in Ks], "k--", lw=1.2, label=f"naive baseline: one neuron per AND (d_mlp={d_mlp})")
    ax.axhline(d_mlp, color="k", lw=0.6, alpha=0.4)
    for v in VARIANTS:
        m = [np.mean([r["k_eff"] for r in d[(v, k)]]) for k in Ks]
        s = [np.std([r["k_eff"] for r in d[(v, k)]]) for k in Ks]
        ax.errorbar(Ks, m, yerr=s, marker="o", capsize=3, color=COLORS[v], label=LABELS[v])
    ax.fill_between(Ks, d_mlp, max(Ks) / 4, color="#2a6fdb", alpha=0.06)
    ax.text(Ks[0], d_mlp * 1.5, "CiS region: more ANDs than neurons", fontsize=8, color="#2a6fdb")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks, [str(k) for k in Ks])
    ax.set_ylim(0, d_mlp * 1.8)
    ax.set_xlabel("number of AND targets K")
    ax.set_ylabel("ANDs actually computed (K_eff)")
    ax.set_title("No trained MLP computes more ANDs than it has neurons\n(m=64 features, d_model=32, d_mlp=32, 3 seeds)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(OUT / "k_eff_vs_K.png", dpi=160)

    # 2. Constraint check: is the pre-MLP residual additive in features? Did attention compute anything?
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    gap = [np.mean([r["attention_additivity_gap"] for k in Ks for r in d[(v, k)]]) for v in VARIANTS]
    axes[0].bar([LABELS[v] for v in VARIANTS], gap, color=[COLORS[v] for v in VARIANTS])
    axes[0].set_ylabel("non-additivity of pre-MLP residual")
    axes[0].set_title("Attention is exactly linear in features\nexcept with softmax", fontsize=9.5)
    probe = [[r["nl_probe_mean"] for r in d[(v, 16)]] for v in VARIANTS]
    intact = [[r["nl_mean"] for r in d[(v, 16)]] for v in VARIANTS]
    x = np.arange(len(VARIANTS))
    axes[1].bar(x - 0.2, [np.mean(a) for a in intact], 0.4, label="full model", color=[COLORS[v] for v in VARIANTS])
    axes[1].bar(x + 0.2, [np.mean(a) for a in probe], 0.4, label="best linear probe of pre-MLP residual",
                color=[COLORS[v] for v in VARIANTS], alpha=0.35, hatch="//")
    axes[1].axhline(0.05, color="k", lw=0.8, ls=":")
    axes[1].set_xticks(x, [LABELS[v] for v in VARIANTS])
    axes[1].set_ylabel("nonlinear score (0 = linear, 1 = perfect AND)")
    axes[1].set_title("K=16: the nonlinear work happens after attention", fontsize=9.5)
    axes[1].legend(fontsize=7)
    for a in axes:
        a.tick_params(axis="x", labelsize=7.5, rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "constraint_check.png", dpi=160)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
