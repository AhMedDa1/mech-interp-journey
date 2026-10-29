"""Hand-built CiS solution for the single-token task: can this architecture compute
more ANDs than it has MLP neurons at all?

Construction (in the spirit of Hanni et al.'s U-AND): each target pair (a, b) is
assigned to `copies` neurons. A neuron reads the union S of the features in its
assigned pairs with weight 1 and bias -1, so act = ReLU(sum_{i in S} x_i - 1) is
non-zero only when two or more of its features are on. Each neuron therefore serves
several ANDs at once (computation in superposition), and sparsity keeps the
interference small. The readout is the best linear map from [acts, x] to the
targets (least squares), i.e. the residual stream carries x unchanged
(d_model >= m) and only the MLP is nonlinear.

If K_eff of this construction exceeds d_mlp, CiS solutions exist in the
architecture, and a trained model that stays below d_mlp is failing to find them.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cis_constrained import Config, make_pairs, nl_score, r2_per_output, sample_x, targets


def build_mlp(pairs: np.ndarray, m: int, d_mlp: int, copies: int, seed: int):
    rng = np.random.default_rng(seed)
    W = np.zeros((d_mlp, m))
    slots = np.concatenate([rng.permutation(d_mlp) for _ in range(copies * len(pairs) // d_mlp + 1)])
    s = 0
    for a, b in pairs:
        used = set()
        for _ in range(copies):
            while slots[s] in used:  # different neurons for the copies of one pair
                s += 1
            n = slots[s]
            used.add(n)
            W[n, a] = W[n, b] = 1.0
            s += 1
    return W, -np.ones(d_mlp)


def lstsq_r2(F_fit, y_fit, F_te, y_te, lam=1e-4):
    F_fit = np.hstack([F_fit, np.ones((len(F_fit), 1))])
    F_te = np.hstack([F_te, np.ones((len(F_te), 1))])
    Wout = np.linalg.solve(F_fit.T @ F_fit + lam * np.eye(F_fit.shape[1]), F_fit.T @ y_fit)
    return r2_per_output(F_te @ Wout, y_te)


def evaluate(cfg: Config, copies: int) -> dict:
    pairs = make_pairs(cfg.m, cfg.K, cfg.seed)
    g = torch.Generator().manual_seed(1_000_000 + cfg.seed)
    x_fit, x_te = sample_x(cfg.n_eval, cfg, g).numpy(), sample_x(cfg.n_eval, cfg, g).numpy()
    y_fit, y_te = targets(torch.tensor(x_fit), pairs).numpy(), targets(torch.tensor(x_te), pairs).numpy()
    W, b = build_mlp(pairs.numpy(), cfg.m, cfg.d_mlp, copies, cfg.seed)
    act = lambda x: np.maximum(x @ W.T + b, 0)
    r2lin = lstsq_r2(x_fit, y_fit, x_te, y_te)
    r2 = lstsq_r2(np.hstack([act(x_fit), x_fit]), y_fit, np.hstack([act(x_te), x_te]), y_te)
    nl = nl_score(r2, r2lin)
    return {"K": cfg.K, "d_mlp": cfg.d_mlp, "p": cfg.p, "m": cfg.m, "copies": copies,
            "features_per_neuron": float((W > 0).sum(1).mean()),
            "nl_mean": float(nl.mean()), "k_eff": float(np.clip(nl, 0, 1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--d_mlp", type=int, default=32)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.1])
    ap.add_argument("--Ks", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--copies", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = []
    for p in args.ps:
        for K in args.Ks:
            best = None
            for c in args.copies:
                r = evaluate(Config(m=args.m, p=p, K=K, d_mlp=args.d_mlp, n_eval=100_000), c)
                rows.append(r)
                best = r if best is None or r["k_eff"] > best["k_eff"] else best
            print(f"p={p:<5} K={K:<4} best copies={best['copies']} feats/neuron={best['features_per_neuron']:.1f} "
                  f"NL={best['nl_mean']:.3f} K_eff={best['k_eff']:.1f} / d_mlp={args.d_mlp}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f)


if __name__ == "__main__":
    main()
