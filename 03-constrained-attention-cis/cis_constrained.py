"""Constrained-attention toy for computation in superposition (CiS).

Stage one showed that softmax attention absorbs XOR-like routing, so ablating the
MLP barely hurts. Here we take the nonlinearity out of attention in three ways and
ask whether the MLP then computes many pairwise ANDs in superposition.

Task: m sparse boolean features x in {0,1}^m (each on with prob p). The model must
output K pairwise products y_k = x_a * x_b (AND) through a *linear* readout,
trained with MSE. No LayerNorm anywhere (LayerNorm is also a nonlinearity).

Variants (how the features reach the MLP):
  single_token : all features packed into one embedding vector; attention over one
                 token is just a linear map.
  frozen       : one token per active feature + PAD, read at a CLS token whose
                 attention pattern is fixed and uniform -> attention is linear in x.
  linear       : same tokens, attention scores q.k without softmax. The query comes
                 from the fixed CLS token, so each score depends on one token only
                 and the attention output is additive over features.
  softmax      : same tokens, ordinary softmax attention (the control; softmax
                 normalisation couples features, so attention can do nonlinear work).

Four-criterion test (see README):
  C1  MLP necessary: nonlinear score with the MLP mean-ablated <= 0.05
  C2  attention did no nonlinear work: ridge probe on pre-MLP residual <= 0.05
  C3  more functions than neurons: K_eff = sum_k clip(NL_k, 0, 1) > d_mlp
  C4  not a noise / seed artefact: C3 holds with noise-free labels on every seed

Nonlinear score per output: NL_k = (R2_k - R2lin_k) / (1 - R2lin_k), where R2lin_k
is the best linear fit from the true features x. NL=0 means no better than linear,
NL=1 means the AND is computed perfectly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("single_token", "frozen", "linear", "softmax")


@dataclasses.dataclass
class Config:
    variant: str = "single_token"
    m: int = 64             # number of sparse boolean features
    p: float = 0.06         # probability each feature is on
    max_active: int = 10    # cap on active features (sequence length for token variants)
    K: int = 64             # number of AND targets (feature pairs)
    d_model: int = 32
    d_mlp: int = 32
    n_heads: int = 4
    steps: int = 6000
    batch: int = 2048
    lr: float = 3e-3
    weight_decay: float = 0.0
    seed: int = 0
    n_eval: int = 100_000
    device: str = "cpu"


# ---------------------------------------------------------------- data

def make_pairs(m: int, K: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(10_000 + seed)
    all_pairs = torch.combinations(torch.arange(m), r=2)
    assert K <= len(all_pairs), f"K={K} > number of pairs {len(all_pairs)}"
    return all_pairs[torch.randperm(len(all_pairs), generator=g)[:K]]


def sample_x(n: int, cfg: Config, g: torch.Generator) -> torch.Tensor:
    x = (torch.rand(n, cfg.m, generator=g) < cfg.p).float()
    # cap the number of active features at max_active (drop a random excess)
    over = x.sum(1) > cfg.max_active
    if over.any():
        rows = x[over]
        noise = torch.rand(rows.shape, generator=g) * rows
        keep = noise.topk(cfg.max_active, dim=1).indices
        capped = torch.zeros_like(rows).scatter_(1, keep, 1.0)
        x[over] = capped
    return x


def targets(x: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    return x[:, pairs[:, 0]] * x[:, pairs[:, 1]]


def to_tokens(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    """[CLS, active feature ids..., PAD...] with length 1 + max_active."""
    cls_id, pad_id = cfg.m, cfg.m + 1
    n = x.shape[0]
    # sort so active features come first; ids of inactive slots replaced by PAD
    order = torch.argsort(-x - torch.arange(cfg.m, device=x.device) * 1e-6, dim=1)[:, : cfg.max_active]
    active = torch.gather(x, 1, order) > 0
    ids = torch.where(active, order, torch.full_like(order, pad_id))
    return torch.cat([torch.full((n, 1), cls_id, dtype=torch.long, device=x.device), ids], dim=1)


@torch.no_grad()
def additivity_gap(model: "ConstrainedToy", cfg: Config, n: int = 2000) -> float:
    """Check whether the pre-MLP residual is affine in x: for disjoint feature sets
    A, B, resid(A+B) - resid(A) - resid(B) + resid(0) should be 0. Returns the size of
    that gap relative to the typical size of a single-feature effect."""
    g = torch.Generator().manual_seed(123)
    dev = next(model.parameters()).device
    xa = torch.zeros(n, cfg.m)
    xb = torch.zeros(n, cfg.m)
    for i in range(n):
        idx = torch.randperm(cfg.m, generator=g)[:4]
        xa[i, idx[:2]] = 1
        xb[i, idx[2:]] = 1
    f = lambda x: model.attend(x.to(dev))[0] + model.attend(x.to(dev))[1]
    zero = f(torch.zeros(1, cfg.m))
    gap = f(xa + xb) - f(xa) - f(xb) + zero
    scale = (f(xa) - zero).norm(dim=-1).mean()
    return float(gap.norm(dim=-1).mean() / scale)


# ---------------------------------------------------------------- model

class ConstrainedToy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        if cfg.variant == "single_token":
            self.embed = nn.Linear(cfg.m, d)
        else:
            self.embed = nn.Embedding(cfg.m + 2, d)  # features, CLS, PAD
            self.d_head = d // cfg.n_heads
            self.W_Q = nn.Linear(d, d, bias=False)
            self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)
        self.mlp_in = nn.Linear(d, cfg.d_mlp)
        self.mlp_out = nn.Linear(cfg.d_mlp, d)
        self.unembed = nn.Linear(d, cfg.K)
        self.mlp_mean: torch.Tensor | None = None  # set for mean-ablation

    def attend(self, x: torch.Tensor):
        cfg = self.cfg
        if cfg.variant == "single_token":
            r = self.embed(x)
            return r, self.W_O(self.W_V(r)), None  # attention over one token = linear map
        toks = to_tokens(x, cfg).to(x.device)
        e = self.embed(toks)                      # (B, T, d)
        r = e[:, 0]                               # CLS residual
        B, T, d = e.shape
        h = cfg.n_heads
        v = self.W_V(e).view(B, T, h, self.d_head)
        if cfg.variant == "frozen":
            w = torch.full((B, h, T), 1.0 / T, device=x.device)
        else:
            q = self.W_Q(r).view(B, h, self.d_head)
            k = self.W_K(e).view(B, T, h, self.d_head)
            s = torch.einsum("bhd,bthd->bht", q, k) / math.sqrt(self.d_head)
            w = s / T if cfg.variant == "linear" else s.softmax(-1)
        z = torch.einsum("bht,bthd->bhd", w, v).reshape(B, d)
        return r, self.W_O(z), w

    def forward(self, x: torch.Tensor, ablate_mlp: bool = False, cache: bool = False):
        r, attn_out, w = self.attend(x)
        resid_mid = r + attn_out
        act = F.relu(self.mlp_in(resid_mid))
        mlp_out = self.mlp_out(act)
        if ablate_mlp:
            mlp_out = self.mlp_mean.expand_as(mlp_out)
        y_hat = self.unembed(resid_mid + mlp_out)
        if cache:
            return y_hat, {"resid_mid": resid_mid, "act": act, "mlp_out": mlp_out}
        return y_hat


# ---------------------------------------------------------------- metrics

def r2_per_output(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    ss_res = ((y - y_hat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def ridge_fit_predict(Xtr, Ytr, Xte, lam=1e-3):
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return Xte @ W


def nl_score(r2, r2lin):
    return (r2 - r2lin) / np.maximum(1.0 - r2lin, 1e-12)


# ---------------------------------------------------------------- run

def run(cfg: Config, log_every: int = 1000, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    pairs = make_pairs(cfg.m, cfg.K, cfg.seed)
    model = ConstrainedToy(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)

    t0 = time.time()
    curve = []
    for step in range(cfg.steps):
        x = sample_x(cfg.batch, cfg, g)
        y = targets(x, pairs)
        loss = F.mse_loss(model(x.to(dev)), y.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == cfg.steps - 1:
            curve.append((step, loss.item()))
            if verbose:
                print(f"  [{cfg.variant} K={cfg.K} d_mlp={cfg.d_mlp} s={cfg.seed}] "
                      f"step {step:5d} loss {loss.item():.6f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- evaluation on fresh data (separate fit set for the linear baselines)
    g_eval = torch.Generator().manual_seed(1_000_000 + cfg.seed)
    x_fit, x_te = sample_x(cfg.n_eval, cfg, g_eval), sample_x(cfg.n_eval, cfg, g_eval)
    y_fit, y_te = targets(x_fit, pairs).numpy(), targets(x_te, pairs).numpy()

    @torch.no_grad()
    def forward_all(x, **kw):
        outs, caches = [], []
        for i in range(0, len(x), 8192):
            o, c = model(x[i:i + 8192].to(dev), cache=True, **kw)
            outs.append(o.cpu())
            caches.append({k: v.cpu() for k, v in c.items()})
        cat = {k: torch.cat([c[k] for c in caches]).numpy() for k in caches[0]}
        return torch.cat(outs).numpy(), cat

    model.eval()
    _, c_fit = forward_all(x_fit)
    y_hat, c_te = forward_all(x_te)
    model.mlp_mean = torch.tensor(c_fit["mlp_out"].mean(0), device=dev)
    y_hat_abl, _ = forward_all(x_te, ablate_mlp=True)

    r2lin = r2_per_output(ridge_fit_predict(x_fit.numpy(), y_fit, x_te.numpy()), y_te)
    r2 = r2_per_output(y_hat, y_te)
    r2_abl = r2_per_output(y_hat_abl, y_te)
    r2_probe = r2_per_output(ridge_fit_predict(c_fit["resid_mid"], y_fit, c_te["resid_mid"]), y_te)

    nl = nl_score(r2, r2lin)
    nl_abl = nl_score(r2_abl, r2lin)
    nl_probe = nl_score(r2_probe, r2lin)
    k_eff = float(np.clip(nl, 0, 1).sum())

    # how many AND outputs each MLP neuron participates in (polysemantic computation):
    # a neuron "serves" output k if |W_U W_out|_{k,n} is >= 25% of output k's max weight
    with torch.no_grad():
        eff = (model.unembed.weight @ model.mlp_out.weight).abs().cpu().numpy()  # (K, d_mlp)
    alive = (c_te["act"] > 0).mean(0) > 1e-3
    serve = eff >= 0.25 * eff.max(1, keepdims=True)
    serve[:, ~alive] = False

    res = {
        "config": dataclasses.asdict(cfg),
        "train_seconds": time.time() - t0,
        "final_train_loss": curve[-1][1],
        "curve": curve,
        "r2_mean": float(r2.mean()),
        "r2lin_mean": float(r2lin.mean()),
        "nl_mean": float(nl.mean()),
        "nl_ablated_mean": float(nl_abl.mean()),
        "nl_probe_mean": float(nl_probe.mean()),
        "k_eff": k_eff,
        "naive_baseline_k_eff": float(min(cfg.K, cfg.d_mlp)),
        "attention_additivity_gap": additivity_gap(model, cfg),
        "alive_neurons": int(alive.sum()),
        "outputs_per_neuron_median": float(np.median(serve[:, alive].sum(0))) if alive.any() else 0.0,
        "C1_mlp_necessary": bool(nl_abl.mean() <= 0.05),
        "C2_attention_linear": bool(nl_probe.mean() <= 0.05),
        "C3_more_functions_than_neurons": bool(k_eff > cfg.d_mlp),
        "nl_per_output": nl.tolist(),
    }
    if verbose:
        print(f"  -> NL={res['nl_mean']:.3f}  NL_ablated={res['nl_ablated_mean']:.3f}  "
              f"NL_probe={res['nl_probe_mean']:.3f}  K_eff={k_eff:.1f} (d_mlp={cfg.d_mlp})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    out = args.__dict__.pop("out")
    cfg = Config(**args.__dict__)
    res = run(cfg)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
