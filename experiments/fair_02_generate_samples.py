#!/usr/bin/env python
"""Step 2/3: generate VP-SGLD + TabEBM single synthetic samples.

Usage:
    python experiments/fair_02_generate_samples.py \
        --eval_dir experiments/fair_eval/20260417_210000 \
        --session experiments/ebms/.../sessions/20260417_... \
        --folders baseline beta__10000000 \
        --n_syn 250 --gpus 0 1 2 3 --procs_per_gpu 5

VP config 지정 방법:
  1) --session + --folders  : sweep session 에서 로드 (folders 생략 시 전부)
  2) --vp_configs           : inline JSON

Output:
    {eval_dir}/
      sample_config.json
      samples/
        split_0/vp_baseline/c0.npy, c1.npy
        split_0/tabebm_single/c0.npy, c1.npy
        ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eval_dir', required=True)
    ap.add_argument('--session', default=None,
                    help='sweep session dir (sweeps/ 하위에 run_config.json 있는 구조)')
    ap.add_argument('--folders', nargs='+', default=None,
                    help='session 내 folder 선택 (생략 시 전부)')
    ap.add_argument('--vp_configs', default=None,
                    help='inline JSON list of VP config dicts')
    ap.add_argument('--n_syn', type=int, default=250,
                    help='N_SYN_PER_CLASS (default 250)')
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--procs_per_gpu', type=int, default=5)
    ap.add_argument('--no_single', action='store_true',
                    help='TabEBM single 생성 건너뛰기')
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    config = json.loads((eval_dir / 'config.json').read_text())

    data = np.load(eval_dir / 'data.npz')
    X_all, y_all = data['X_all'], data['y_all']
    sp = np.load(eval_dir / 'splits.npz')
    splits = [(sp[f'tr_{i}'], sp[f'te_{i}'])
              for i in range(config['n_splits'])]
    classes = config['classes']

    # --- VP configs ---
    vp_configs = []
    if args.session:
        sweeps = Path(args.session).resolve() / 'sweeps'
        folders = args.folders or sorted(
            p.name for p in sweeps.iterdir()
            if p.is_dir() and (p / 'run_config.json').exists())
        for fn in folders:
            rc = json.loads((sweeps / fn / 'run_config.json').read_text())
            cfg = rc['cfg']
            vp_configs.append(dict(
                name=f'vp_{fn}',
                beta=cfg['beta'], eta=cfg['eta'], tau=cfg['tau'],
                sigma_start=cfg['sigma_start'], n_steps=cfg['n_steps'],
                auto_beta=cfg['auto_beta'],
                ignore_variance=cfg.get('ignore_variance', False),
            ))
    elif args.vp_configs:
        vp_configs = json.loads(args.vp_configs)
    else:
        raise SystemExit('--session 또는 --vp_configs 중 하나 필요')

    sample_config = {
        'vp_configs': vp_configs,
        'n_syn_per_class': args.n_syn,
        'gpus': args.gpus,
        'procs_per_gpu': args.procs_per_gpu,
        'session': str(args.session) if args.session else None,
        'folders': args.folders,
        'include_single': not args.no_single,
    }
    (eval_dir / 'sample_config.json').write_text(json.dumps(sample_config, indent=2))

    # --- build tasks ---
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from fair_eval_worker import run_one_sgld_task

    ens_dir = str(eval_dir / 'ensembles')
    samples_dir = eval_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)

    sgld_tasks = []
    task_id = 0
    n_gpu = len(args.gpus)

    for split_i, (tr, _) in enumerate(splits):
        for ci, cfg in enumerate(vp_configs):
            for c in classes:
                gpu = args.gpus[task_id % n_gpu]
                sgld_tasks.append((
                    'vp', split_i, c, {**cfg, '_ci': ci},
                    tr, X_all, y_all,
                    args.n_syn, config['seed'], gpu, ens_dir,
                ))
                task_id += 1
        if not args.no_single:
            for c in classes:
                gpu = args.gpus[task_id % n_gpu]
                sgld_tasks.append((
                    'single', split_i, c, config['seed'],
                    tr, X_all, y_all,
                    args.n_syn, config['seed'], gpu, ens_dir,
                ))
                task_id += 1

    n_total = len(sgld_tasks)
    max_workers = n_gpu * args.procs_per_gpu

    print(f'eval_dir:    {eval_dir}')
    print(f'VP configs:  {len(vp_configs)}')
    for c in vp_configs:
        print(f'  {c["name"]:<30} β={c["beta"]:<8g} η={c["eta"]:<6g} '
              f'τ={c["tau"]:<5g} ig_var={c.get("ignore_variance", False)}')
    print(f'tasks:       {n_total}  ({config["n_splits"]} splits × '
          f'{len(vp_configs)} VP × {len(classes)} cls'
          f'{f" + {len(classes)} single" if not args.no_single else ""})')
    print(f'workers:     {max_workers}  ({n_gpu} GPUs × {args.procs_per_gpu})')
    print()

    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one_sgld_task, t): t for t in sgld_tasks}
        for f in as_completed(futs):
            r = f.result()
            out = samples_dir / f'split_{r["split_i"]}' / r['cfg_name']
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f'c{r["class_c"]}.npy', r['samples'])

            done += 1
            if done % 10 == 0 or done == n_total:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_total - done)
                m, s = divmod(int(eta), 60)
                print(f'  [{done:>3d}/{n_total}]  '
                      f'elapsed {_fmt(elapsed)}  ETA {m}m{s:02d}s', flush=True)

    dt = time.time() - t0
    print(f'\nStep 2 done — {_fmt(dt)}')
    print(f'  {samples_dir}')


def _fmt(s):
    s = int(s)
    m, s = divmod(s, 60)
    return f'{m}m{s:02d}s' if m else f'{s}s'


if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
