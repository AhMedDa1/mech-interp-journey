"""Run a named grid of configs in parallel (one CPU thread per run)."""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402

from cis_constrained import Config, run  # noqa: E402

BASE = dict(lr=1e-2, steps=10_000)

SWEEPS = {
    # where does the single-token MLP beat the naive baseline? (fast scan)
    "regime": dict(variant=["single_token"], p=[0.02, 0.04, 0.06, 0.1],
                   d_model=[32, 64], K=[32, 64, 128], d_mlp=[32]),
    # the main comparison: four variants x number of AND targets, 3 seeds
    "main": dict(variant=["single_token", "frozen", "linear", "softmax"],
                 K=[16, 32, 64, 128, 256], seed=[0, 1, 2]),
}


def _one(kw: dict, out_dir: str) -> str:
    torch.set_num_threads(1)
    cfg = Config(**{**BASE, **kw})
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    path = Path(out_dir) / f"{name}.json"
    if path.exists():
        return f"skip {name}"
    res = run(cfg, verbose=False)
    path.write_text(json.dumps(res))
    return (f"{name}: NL={res['nl_mean']:.3f} abl={res['nl_ablated_mean']:.2f} "
            f"probe={res['nl_probe_mean']:.3f} K_eff={res['k_eff']:.1f}/{cfg.d_mlp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=list(SWEEPS))
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--set", nargs="*", default=[], help="override base, e.g. m=128 d_model=64")
    args = ap.parse_args()
    for s in args.set:
        k, v = s.split("=")
        typ = type(Config.__dataclass_fields__[k].default)
        BASE[k] = typ(v)
    grid = SWEEPS[args.sweep]
    keys = list(grid)
    jobs = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    tag = args.sweep + ("_" + "_".join(args.set) if args.set else "")
    out_dir = Path("runs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "base.json").write_text(json.dumps(BASE))
    print(f"{len(jobs)} runs -> {out_dir}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_one, j, str(out_dir)) for j in jobs]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
