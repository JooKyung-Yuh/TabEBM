#!/usr/bin/env python
"""nb 02.2 의 sweep 을 ProcessPool 로 진짜 병렬 실행 (GIL 우회).

같은 출력 구조 (`sessions/<ts>_<tag>/sweeps/<folder>/diag_raw_c{c}.npz` +
`run_config.json` + `session_config.json`) 로 저장하므로 nb 02.2 의 후속 분석
셀들 (M-vs-N, heatmap, overlay, summary) 이 그대로 동작.

GIL 의존하는 ThreadPool 대비 ~8x 빠름. 58 task 기준 127 분 → 15 분.

사용 예:
    python experiments/run_sweep_processpool.py \\
      --ensemble-root experiments/ebms/20260415_210238_Subsample-Distance_EBM \\
      --classes 0 1 \\
      --n-high 500 --n-steps 50 \\
      --beta 1.0 --eta 0.05 --tau 1.0 --sigma-start 0.1 --auto-beta \\
      --gpus 0 1 2 3 \\
      --session-tag all_axes_processpool \\
      --sweep-beta 0.01 1.0 100.0 1e4 1e6 \\
      --sweep-eta 0.01 0.05 0.2 1.0 \\
      --sweep-ignore-variance
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# CUDA spawn 안전
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# ---------------------------------------------------------------------------
# Worker (module-level — picklable)
# ---------------------------------------------------------------------------
def _sgld_worker(args_tuple):
    """별도 프로세스에서 한 SGLD task 실행."""
    import sys
    import time
    import json
    import numpy as np

    sys.path.insert(0, "/home/work/JooKyung/TabEBM/src")
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/experiments")
    from tabebm.vp_sgld import vp_sgld_from_ensemble

    (
        task_idx, ensemble_root, folder, class_c, gpu, cfg, sweeps_dir
    ) = args_tuple

    ensemble_root = str(ensemble_root)
    sweeps_dir = str(sweeps_dir)
    class_dir = f"{ensemble_root}/c{class_c}"

    t0 = time.time()
    _, diags, traj = vp_sgld_from_ensemble(
        class_dir,
        n_samples=cfg["n_high"], n_steps=cfg["n_steps"],
        beta=cfg["beta"], eta=cfg["eta"], tau=cfg["tau"],
        sigma_start=cfg["sigma_start"],
        auto_beta=cfg["auto_beta"],
        ignore_variance=cfg["ignore_variance"],
        seed=cfg["seed"], gpu=gpu,
        return_diagnostics=True, return_trajectory=True,
    )
    keys = [k for k in diags[0].keys() if isinstance(diags[0][k], (int, float))]
    arr = np.array([[d[k] for k in keys] for d in diags], dtype=np.float64)
    traj_np = traj.numpy().astype(np.float32)

    out_path = f"{sweeps_dir}/{folder}/diag_raw_c{class_c}.npz"
    np.savez_compressed(
        out_path,
        diag_cols=np.array(keys, dtype=object),
        diag=arr, trajectory=traj_np,
    )
    idx_M = keys.index("M_mean")
    return dict(
        task_idx=task_idx,
        folder=folder, class_c=class_c, gpu=gpu,
        dt=time.time() - t0,
        M_init=float(arr[0, idx_M]),
        M_final=float(arr[-1, idx_M]),
        pid=os.getpid(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ensemble + 환경
    p.add_argument("--ensemble-root", type=Path, required=True)
    p.add_argument("--classes", type=int, nargs="+", default=[0, 1])
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--session-tag", type=str, default="sweep")
    # FIXED
    p.add_argument("--n-high", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--sigma-start", type=float, default=0.1)
    p.add_argument("--auto-beta", dest="auto_beta", action="store_true", default=False)
    p.add_argument("--no-auto-beta", dest="auto_beta", action="store_false")
    # SWEEPS
    p.add_argument("--sweep-beta", type=float, nargs="+", default=None)
    p.add_argument("--sweep-eta", type=float, nargs="+", default=None)
    p.add_argument("--sweep-tau", type=float, nargs="+", default=None)
    p.add_argument("--sweep-sigma-start", type=float, nargs="+", default=None)
    p.add_argument("--sweep-n-steps", type=int, nargs="+", default=None)
    p.add_argument("--sweep-auto-beta", action="store_true", default=False,
                    help="auto_beta 를 [False, True] 로 sweep")
    p.add_argument("--sweep-ignore-variance", action="store_true", default=False,
                    help="ignore_variance 를 [False, True] 로 sweep")
    # 옵션
    p.add_argument("--skip-baseline-dup", action="store_true", default=True,
                    help="sweep 값이 FIXED baseline 과 동일한 경우 skip (기본 ON)")
    p.add_argument("--no-skip-baseline-dup", dest="skip_baseline_dup",
                    action="store_false")
    p.add_argument("--no-baseline", action="store_true", default=False,
                    help="baseline config 자체 실행 안 함 (sweep value 만 돌림)")
    return p


def _fmt_time(s):
    s = int(s); m, s = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s" if m else f"{s:d}s"


def _cfg_folder_name(overrides):
    if not overrides:
        return "baseline"
    (axis, val), = overrides.items()
    return f"{axis}__{val}"


def main():
    args = _build_parser().parse_args()
    assert (args.ensemble_root / "c0" / "meta.json").exists(), \
        f"no c0/meta.json under {args.ensemble_root}"

    # SWEEPS dict 빌드
    SWEEPS = {}
    for k in ["beta", "eta", "tau", "sigma_start", "n_steps"]:
        v = getattr(args, f"sweep_{k.replace('_', '_')}", None)  # noqa
        if v: SWEEPS[k] = v
    if args.sweep_auto_beta:    SWEEPS["auto_beta"] = [False, True]
    if args.sweep_ignore_variance: SWEEPS["ignore_variance"] = [False, True]

    FIXED = dict(
        n_high=args.n_high, n_steps=args.n_steps,
        beta=args.beta, eta=args.eta, tau=args.tau,
        sigma_start=args.sigma_start, auto_beta=args.auto_beta,
        ignore_variance=False, seed=args.seed,
    )

    # 세션 디렉토리
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.ensemble_root / "comparisons" / "sessions" / \
                   f"{session_ts}_{args.session_tag}"
    sweeps_dir = session_dir / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session_config.json").write_text(json.dumps({
        "ensemble_root": str(args.ensemble_root),
        "ARGS_parsed": vars(args),
        "SWEEPS": SWEEPS, "FIXED": FIXED,
        "session_ts": session_ts, "engine": "ProcessPoolExecutor",
    }, indent=2, default=str))

    # configs 펼치기 (baseline + sweep values)
    all_configs = [] if args.no_baseline else [({}, "baseline")]
    for axis, values in SWEEPS.items():
        baseline_val = FIXED[axis]
        for v in values:
            if args.skip_baseline_dup and v == baseline_val:
                print(f"[skip dedup] {axis}={v} == baseline → 건너뜀", flush=True)
                continue
            all_configs.append(({axis: v}, _cfg_folder_name({axis: v})))

    # folder + run_config.json 미리 생성 (worker 가 npz 만 쓰게)
    for overrides, folder in all_configs:
        cfg = {**FIXED, **overrides}
        out_dir = sweeps_dir / folder
        out_dir.mkdir(exist_ok=True)
        (out_dir / "run_config.json").write_text(json.dumps({
            "overrides": overrides, "cfg": cfg, "folder": folder,
        }, indent=2, default=str))

    # task list — round-robin GPU 배정
    tasks = []
    for ci, (overrides, folder) in enumerate(all_configs):
        cfg = {**FIXED, **overrides}
        for c in args.classes:
            gpu = args.gpus[(ci * len(args.classes) + c) % len(args.gpus)]
            tasks.append((len(tasks), str(args.ensemble_root), folder,
                            c, gpu, cfg, str(sweeps_dir)))

    total = len(tasks)
    print(f"\n{'='*72}")
    print(f"ENSEMBLE_ROOT : {args.ensemble_root}")
    print(f"SESSION_DIR   : {session_dir}")
    print(f"FIXED         : {FIXED}")
    print(f"SWEEPS        : {SWEEPS}")
    print(f"GPUS          : {args.gpus}  ({len(args.gpus)} processes)")
    print(f"tasks         : {total}  ({len(all_configs)} configs × {len(args.classes)} class)")
    print(f"engine        : ProcessPoolExecutor (GIL 우회 진짜 병렬)")
    print(f"{'='*72}\n", flush=True)

    # 실행
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t_start = time.time()
    completed = []
    with ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:
        futures = {ex.submit(_sgld_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            completed.append(r)
            elapsed = time.time() - t_start
            done = len(completed)
            rate = elapsed / done
            eta = rate * (total - done) / max(len(args.gpus), 1)  # 병렬 보정
            pct = done / total * 100
            bar_w = 24
            filled = int(bar_w * done / total)
            bar = "█" * filled + "░" * (bar_w - filled)
            print(f"  [{bar}] {pct:5.1f}%  {done:>2d}/{total}  "
                   f"{r['folder']:<28} c{r['class_c']} gpu={r['gpu']} pid={r['pid']}  "
                   f"({r['dt']:5.1f}s)  M={r['M_init']:.3f}→{r['M_final']:.3f}  "
                   f"| elapsed {_fmt_time(elapsed)}  ETA {_fmt_time(eta)}",
                   flush=True)
    wall = time.time() - t_start
    serial_eq = sum(r["dt"] for r in completed)
    print(f"\n{'='*72}")
    print(f"✓ DONE — wallclock {_fmt_time(wall)}   serial-equivalent {_fmt_time(serial_eq)}   "
          f"speedup {serial_eq/max(wall,0.01):.2f}x")
    print(f"{'='*72}")
    print(f"results: {session_dir}")
    print(f"\nNext steps (nb 02.2 의 분석 셀들 그대로 사용 가능):")
    print(f"  - § 2 (M-vs-N): SWEEPS_DIR = {sweeps_dir}")
    print(f"  - § 3 (heatmap+체인): 동일")
    print(f"  - § 4 (axis overlay): 동일")
    print(f"  - § 5 (summary): 동일")
    print(f"  또는 nb 03 의 SESSION 모드: SESSION_DIR = {session_dir}")


if __name__ == "__main__":
    main()
