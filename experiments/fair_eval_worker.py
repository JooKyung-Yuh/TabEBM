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
     seed, split_ens_dir, dataset, n_real) = args_tuple

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    split_seed = seed + split_i * 100
    ens_dir = Path(split_ens_dir) / f'split_{split_i}'

    # Strong cache check: all classes × all K members present
    def _fully_cached():
        for c in classes:
            cdir = ens_dir / f'c{c}'
            if not (cdir / 'meta.json').exists():
                return False
            try:
                meta = json.loads((cdir / 'meta.json').read_text())
                n_expected = meta.get('n_ebms', K)
            except Exception:
                return False
            for k in range(n_expected):
                if not (cdir / f'ebm_{k}' / 'config.json').exists():
                    return False
                if not (cdir / f'ebm_{k}' / 'surrogate_data.npz').exists():
                    return False
        return True

    if _fully_cached():
        return dict(split_i=split_i, dt=0, cached=True)

    t0 = time.time()
    for c in classes:
        X_class = X_tr[y_tr == c]
        fit_class_ensemble(
            class_dir=ens_dir / f'c{c}',
            X_class=X_class, X_all=X_tr, y_all=y_tr,
            methods=ensemble_methods, K=K, seed=split_seed,
            method_params=ensemble_method_params,
            dataset=dataset, target_class=int(c), n_real=n_real,
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
        if not isinstance(cfg_or_seed, dict):
            raise ValueError(
                "single task_type requires cfg dict with explicit SGLD + "
                "distance_negative_class fields. Legacy non-dict fallback removed "
                "(unseen defaults caused unfair baseline comparisons)."
            )
        te = TabEBM(device=f'cuda:{gpu}')
        cfg = cfg_or_seed
        cfg_name = cfg.get('name', 'tabebm_single')
        seed_offset = int(cfg.get('seed_offset', 0))
        gen_kw = {k: cfg[k] for k in
                  ('sgld_steps', 'sgld_step_size', 'sgld_noise_std',
                   'starting_point_noise_std', 'distance_negative_class')
                  if k in cfg}
        if 'distance_negative_class' not in gen_kw:
            raise ValueError(
                f"cfg {cfg_name!r} missing 'distance_negative_class'. "
                f"Explicit value required for reproducible corner geometry."
            )
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


def eval_split_classifier_task(args_tuple):
    """Per-(split, classifier) eval task.

    One task = 1 classifier × (baseline + all aug configs) for one split.
    Returns one row dict {split, classifier, baseline, <sname>: bacc, ...}.

    Ideal parallel unit: 10 splits × 6 classifiers = 60 tasks over 28 workers.
    Fast classifiers (KNN, LR, RF, XGBoost) finish quickly → freeing workers
    for slow TabPFN/MLP. TabPFN loads once per worker lifetime (ProcessPool
    reuses workers) so subsequent fits are fast.
    """
    import sys, os, time, json
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/src")
    sys.path.insert(0, "/home/work/JooKyung/TabEBM/experiments")

    import numpy as np
    from pathlib import Path
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    import xgboost as xgb
    from tabpfn import TabPFNClassifier

    (split_i, clf_name, seed_i,
     X_tr, y_tr, X_val, y_val, X_te, y_te,
     aug_settings, samples_dir_str, classes,
     n_syn_total, n_syn_per_class, gpu) = args_tuple

    samples_dir = Path(samples_dir_str)

    def build(name, seed):
        if name == 'knn':
            return KNeighborsClassifier(n_jobs=-1)
        if name == 'lr':
            return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=seed)
        if name == 'rf':
            return RandomForestClassifier(n_jobs=-1, random_state=seed)
        if name == 'xgboost':
            return xgb.XGBClassifier(
                n_jobs=-1, eval_metric='logloss', use_label_encoder=False,
                random_state=seed, early_stopping_rounds=10,
            )
        if name == 'mlp':
            return MLPClassifier(max_iter=500, random_state=seed)
        if name == 'tabpfn':
            return TabPFNClassifier(
                n_estimators=1, device=f'cuda:{gpu}', random_state=seed,
                ignore_pretraining_limits=True,
            )
        raise ValueError(f'unknown classifier: {name}')

    def fit_and_score(X_fit, y_fit):
        clf = build(clf_name, seed_i)
        # option B — xgboost uses external val for early stopping
        if clf_name == 'xgboost':
            clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        else:
            clf.fit(X_fit, y_fit)
        return float(balanced_accuracy_score(y_te, clf.predict(X_te)) * 100)

    t0 = time.time()
    row = {
        'n_syn_total': int(n_syn_total),
        'n_syn_per_class': int(n_syn_per_class),
        'split': int(split_i),
        'classifier': clf_name,
        'baseline': fit_and_score(X_tr, y_tr),
    }

    for sname in aug_settings:
        smp_parts = []
        smp_labels = []
        for c in classes:
            p = samples_dir / f'split_{split_i}' / sname / f'c{c}.npy'
            arr = np.load(p)[:n_syn_per_class]
            smp_parts.append(arr)
            smp_labels.append(np.full(len(arr), c))
        X_aug = np.vstack([X_tr] + smp_parts)
        y_aug = np.concatenate([y_tr] + smp_labels)
        row[sname] = fit_and_score(X_aug, y_aug)

    row['_dt'] = round(time.time() - t0, 2)
    row['_pid'] = os.getpid()
    row['_gpu'] = int(gpu)
    return row
