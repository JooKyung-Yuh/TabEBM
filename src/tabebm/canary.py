"""
Canary energies — TabPFN-version drift detection for saved EBM ensembles.

Each saved ensemble member can have a `canary.npz` attached holding a few
fixed evaluation points and the energies the just-fit member produced at
them. On any later rebuild, recomputed energies must match within `atol`
or the saved ensemble is no longer bit-faithful (TabPFN version change,
CUDA/driver change, etc.). Cheap: O(KB) on disk, sub-second to verify.

Use:
    from tabebm.canary import attach_canary, verify_canary
    attach_canary('experiments/ebms/stock_distance_v2/ebm_0', n_canary=16)
    verify_canary('experiments/ebms/stock_distance_v2/ebm_0')   # → 0.0 if fine
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tabebm.TabEBM import TabEBM


def _energy_no_grad(tabebm: TabEBM, x: torch.Tensor) -> torch.Tensor:
    """Forward-only energy at points x of shape (B, d). No autograd."""
    with torch.no_grad():
        x_3d = x.unsqueeze(0)
        logits = tabebm.model.forward([x_3d], return_logits=True)
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        if logits.shape[0] == 2 and logits.shape[1] != 2:
            logits = logits.T
        return -torch.logsumexp(logits, dim=1)


def attach_canary(ebm_dir: str | Path, n_canary: int = 16, gpu: int = 0) -> np.ndarray:
    """Compute canary energies on a freshly rebuilt member, save to canary.npz."""
    try:
        from experiments.ensemble_ebm import rebuild_ebm
    except ModuleNotFoundError:
        from ensemble_ebm import rebuild_ebm

    ebm_dir = Path(ebm_dir)
    surr = np.load(ebm_dir / "surrogate_data.npz")
    canary_X = surr["X_ebm"][: min(n_canary, len(surr["X_ebm"]))].astype(np.float32)

    tabebm, _ = rebuild_ebm(ebm_dir, gpu=gpu)
    canary_E = _energy_no_grad(tabebm, torch.from_numpy(canary_X).to(f"cuda:{gpu}")).cpu().numpy()

    np.savez(ebm_dir / "canary.npz", X=canary_X, E=canary_E)
    return canary_E


def verify_canary(ebm_dir: str | Path, gpu: int = 0, atol: float = 1e-5) -> float | None:
    """Refit member, recompute canary, return max abs diff. Raises on mismatch."""
    try:
        from experiments.ensemble_ebm import rebuild_ebm
    except ModuleNotFoundError:
        from ensemble_ebm import rebuild_ebm

    ebm_dir = Path(ebm_dir)
    canary_path = ebm_dir / "canary.npz"
    if not canary_path.exists():
        return None
    saved = np.load(canary_path)
    tabebm, _ = rebuild_ebm(ebm_dir, gpu=gpu)
    fresh_E = _energy_no_grad(tabebm, torch.from_numpy(saved["X"]).float().to(f"cuda:{gpu}")).cpu().numpy()
    diff = float(np.abs(fresh_E - saved["E"]).max())
    if diff > atol:
        raise ValueError(
            f"Canary mismatch in {ebm_dir.name}: max abs diff {diff:.3e} > atol {atol}. "
            "TabPFN version drift or environment change suspected."
        )
    return diff
