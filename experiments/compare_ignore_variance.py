#!/usr/bin/env python
"""Ablation: `ignore_variance` on/off × N chains ∈ {10, 500}, focus on M-term.

동일 ensemble 에 대해 VP-SGLD 4 번 돌려서 step-wise M_mean, drift/noise norm
추이를 plot. 저장: 그래프 PNG + raw diagnostic npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tabebm.vp_sgld import vp_sgld_from_ensemble


def run_one(
    class_dir: Path, *, n_samples: int, ignore_variance: bool,
    n_steps: int, beta: float, eta: float, tau: float, sigma_start: float,
    seed: int, gpu: int,
) -> tuple[np.ndarray, list[str]]:
    """Return (diag_matrix shape (T, n_keys), diag_cols)."""
    _, diags = vp_sgld_from_ensemble(
        class_dir,
        n_samples=n_samples, n_steps=n_steps,
        beta=beta, eta=eta, tau=tau, sigma_start=sigma_start,
        auto_beta=True, ignore_variance=ignore_variance,
        seed=seed, gpu=gpu,
        return_diagnostics=True,
    )
    keys = [k for k in diags[0].keys()
            if isinstance(diags[0][k], (int, float))]
    arr = np.array([[d[k] for k in keys] for d in diags], dtype=np.float64)
    return arr, keys


def plot_ablation(results: dict, out_dir: Path, ensemble_name: str, class_c: int):
    """results: {(ignore, N): (arr (T, n_keys), keys)}"""
    (any_arr, any_keys) = next(iter(results.values()))
    T = any_arr.shape[0]
    steps = np.arange(T)
    idx = {k: i for i, k in enumerate(any_keys)}

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)

    # 레이아웃 상수
    colors = {True: "tab:orange", False: "tab:blue"}
    labels = {True: "ignore_variance=True  (M=I)",
               False: "ignore_variance=False (VP-SGLD)"}
    Ns = sorted({N for (_, N) in results.keys()})
    assert len(Ns) == 2, f"expected 2 N values, got {Ns}"
    n_low, n_high = Ns

    metrics = [
        ("M_mean",          "M  (preconditioner mean)"),
        ("drift_norm",      "‖η·M·μ‖  (drift term norm)"),
        ("drift_over_noise","drift / noise ratio (log)"),
    ]

    for row, (key, yl) in enumerate(metrics):
        col_ix = idx[key]
        for col_i, N in enumerate(Ns):
            ax = axes[row, col_i]
            for ig in [False, True]:
                arr, _ = results[(ig, N)]
                y = arr[:, col_ix]
                ax.plot(steps, y, color=colors[ig], label=labels[ig],
                        linewidth=1.8, marker='o', markersize=3.5)
            ax.set_ylabel(yl, fontsize=10)
            if row == 0:
                ax.set_title(f"N = {N} chains", fontsize=11, fontweight='bold')
            if row == len(metrics) - 1:
                ax.set_xlabel("SGLD step", fontsize=10)
            if key == "drift_over_noise":
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.axhline(1.0 if key == "M_mean" else 0.0,
                        color="k", ls=":", lw=0.6, alpha=0.5)
            if row == 0 and col_i == 0:
                ax.legend(fontsize=8, loc='best', framealpha=0.9)

    fig.suptitle(
        f"VP-SGLD ablation  ·  {ensemble_name}  ·  class {class_c}\n"
        f"ignore_variance=OFF (파랑)  vs  ON (주황)    —    N={n_low}  vs  N={n_high}",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    out_path = out_dir / f"ablation_M_comparison_c{class_c}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  saved: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-root", type=Path,
                    default=Path("experiments/ebms/20260415_210238_Subsample-Distance_EBM"))
    ap.add_argument("--classes", type=int, nargs="+", default=[0])
    ap.add_argument("--n-low", type=int, default=10)
    ap.add_argument("--n-high", type=int, default=500)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--sigma-start", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        args.out = args.ensemble_root / "comparisons" / f"{ts}_ignore_variance_ablation"
    args.out.mkdir(parents=True, exist_ok=True)

    # record run config
    (args.out / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    print(f"[ablation] ensemble={args.ensemble_root.name}  "
          f"N∈{{{args.n_low},{args.n_high}}}  T={args.n_steps}")
    print(f"  out_dir: {args.out}")

    for c in args.classes:
        class_dir = args.ensemble_root / f"c{c}"
        print(f"\n=== class {c} ===")
        results = {}
        t0 = time.time()
        for N in [args.n_low, args.n_high]:
            for ig in [False, True]:
                tag = f"N={N:<3d}  ignore_variance={ig}"
                t1 = time.time()
                arr, keys = run_one(
                    class_dir, n_samples=N, ignore_variance=ig,
                    n_steps=args.n_steps,
                    beta=args.beta, eta=args.eta, tau=args.tau,
                    sigma_start=args.sigma_start,
                    seed=args.seed, gpu=args.gpu,
                )
                results[(ig, N)] = (arr, keys)
                print(f"  {tag}  ({time.time()-t1:.1f}s)")
        print(f"class {c}: 4 runs in {time.time()-t0:.1f}s total")

        # save raw per-step diag for all 4 runs
        np.savez_compressed(
            args.out / f"diag_raw_c{c}.npz",
            diag_cols=np.array(keys, dtype=object),
            **{f"ignore{int(ig)}_N{N}": arr
                 for (ig, N), (arr, _) in results.items()},
        )

        plot_ablation(results, args.out, args.ensemble_root.name, c)

    print(f"\nDone → {args.out}")


if __name__ == "__main__":
    main()
