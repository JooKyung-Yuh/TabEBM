"""Fair evaluation workers — ProcessPool 용 (picklable)."""
from __future__ import annotations


def compute_member_energy(args_tuple):
    """Heatmap 용: 한 EBM 멤버의 energy 를 PCA grid 위에서 계산."""
    import sys
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/src")
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/experiments")
    import torch
    from pathlib import Path
    from ensemble_ebm import rebuild_ebm

    class_dir, member_k, X_grid_d, gpu = args_tuple
    ebm, _ = rebuild_ebm(Path(class_dir) / f'ebm_{member_k}', gpu=gpu)
    device = f'cuda:{gpu}'
    X_t_full = torch.from_numpy(X_grid_d).float().to(device).unsqueeze(0)
    CHUNK = 16384
    N = X_t_full.shape[1]
    parts = []
    with torch.no_grad():
        for i in range(0, N, CHUNK):
            X_t = X_t_full[:, i:i + CHUNK]
            logits = ebm.model.forward([X_t], return_logits=True)
            if logits.dim() == 3:
                logits = logits.squeeze(0)
            if logits.shape[0] == 2 and logits.shape[1] != 2:
                logits = logits.T
            parts.append(-torch.logsumexp(logits, dim=1))
        energy = torch.cat(parts).cpu().numpy()
    return member_k, energy


def fit_one_split_ensemble(args_tuple):
    """Phase 1: 한 split 의 ensemble fit (빠름, ~3s)."""
    import sys, time, json, os
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/src")
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/experiments")
    import numpy as np
    from pathlib import Path
    from fit_ensemble_v2 import fit_class_ensemble

    (split_i, tr_idx, X_all, y_all, classes,
     ensemble_methods, ensemble_method_params, shared_corners, K,
     seed, split_ens_dir) = args_tuple

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    split_seed = seed + split_i * 100
    ens_dir = Path(split_ens_dir) / f'split_{split_i}'

    if (ens_dir / 'c0' / 'meta.json').exists():
        return dict(split_i=split_i, dt=0, cached=True)

    t0 = time.time()
    for c in classes:
        X_class = X_tr[y_tr == c]
        fit_class_ensemble(
            class_dir=ens_dir / f'c{c}',
            X_class=X_class, X_all=X_tr, y_all=y_tr,
            methods=ensemble_methods, K=K, seed=split_seed,
            method_params=ensemble_method_params,
            dataset='stock', target_class=int(c), n_real=len(X_tr),
            shared_corners=shared_corners,
        )
    return dict(split_i=split_i, dt=time.time()-t0, cached=False)


def run_one_sgld_task(args_tuple):
    """Phase 2: 개별 SGLD task (VP-SGLD 또는 TabEBM single)."""
    import sys, time, os
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/src")
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/experiments")
    import numpy as np
    from pathlib import Path
    from tabebm.vp_sgld import vp_sgld_from_ensemble, clear_ensemble_cache
    from tabebm.TabEBM import TabEBM

    (task_type, split_i, class_c, cfg_or_seed, tr_idx, X_all, y_all,
     n_syn, seed, gpu, split_ens_dir) = args_tuple

    t0 = time.time()
    ens_dir = Path(split_ens_dir) / f'split_{split_i}'
    split_seed = seed + split_i * 100
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]

    if task_type == 'vp':
        cfg = cfg_or_seed
        clear_ensemble_cache()
        traj_checkpoints = cfg.get('_traj_checkpoints')
        need_traj = traj_checkpoints is not None and len(traj_checkpoints) > 0
        result = vp_sgld_from_ensemble(
            ens_dir / f'c{class_c}',
            n_samples=n_syn, n_steps=cfg['n_steps'],
            beta=cfg['beta'], eta=cfg['eta'], tau=cfg['tau'],
            sigma_start=cfg['sigma_start'], auto_beta=cfg['auto_beta'],
            ignore_variance=cfg.get('ignore_variance', False),
            seed=split_seed + cfg.get('_ci', 0), gpu=gpu,
            return_trajectory=need_traj,
        )
        if need_traj:
            samples, traj = result
            samples = samples.numpy()
            traj_np = traj.numpy()
            valid_steps = [s for s in traj_checkpoints if s <= cfg['n_steps']]
            traj_dict = {f'step_{s}': traj_np[s].astype(np.float16) for s in valid_steps}
            traj_dict['steps'] = np.array(valid_steps)
        else:
            samples = result.numpy()
            traj_dict = None
        return dict(task_type='vp', split_i=split_i, class_c=class_c,
                     cfg_name=cfg['name'], samples=samples,
                     traj_dict=traj_dict,
                     dt=time.time()-t0, gpu=gpu, pid=os.getpid())

    elif task_type == 'single':
        te = TabEBM(device=f'cuda:{gpu}')
        seed_offset = 0
        if isinstance(cfg_or_seed, dict):
            cfg = cfg_or_seed
            cfg_name = cfg.get('name', 'tabebm_single')
            seed_offset = int(cfg.get('seed_offset', 0))
            gen_kw = {k: cfg[k] for k in
                      ('sgld_steps', 'sgld_step_size', 'sgld_noise_std',
                       'starting_point_noise_std', 'distance_negative_class')
                      if k in cfg}
        else:
            cfg_name = 'tabebm_single'
            gen_kw = {}
        gen_seed = split_seed + seed_offset

        BATCH = 1000
        if n_syn <= BATCH:
            res = te.generate(X_tr, y_tr, num_samples=n_syn,
                              seed=gen_seed, **gen_kw)
            samples = res[f'class_{int(class_c)}']
        else:
            parts = []
            remaining = n_syn
            batch_i = 0
            while remaining > 0:
                bs = min(BATCH, remaining)
                res = te.generate(X_tr, y_tr, num_samples=bs,
                                  seed=gen_seed + batch_i * 7, **gen_kw)
                parts.append(res[f'class_{int(class_c)}'])
                remaining -= bs
                batch_i += 1
            samples = np.concatenate(parts, axis=0)

        return dict(task_type='single', split_i=split_i, class_c=class_c,
                     cfg_name=cfg_name, samples=samples,
                     dt=time.time()-t0, gpu=gpu, pid=os.getpid())
