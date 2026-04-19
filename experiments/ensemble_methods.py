"""Randomization methods for TabEBM ensembles.

Each registered method contributes ONE orthogonal dimension of randomness
per member. Active methods combine independently: each member samples each
active method once, and all sampled values are stored in the member's
`config.json` for reproducibility.

Adding a new method:
    @register('MyMethod')
    def sample_my_method(rng, **params) -> dict:
        ...
        return {'method_my_method': {'some_field': value}}

Then consume it in `build_surrogate_data` (or elsewhere) by reading
`cfg['method_my_method']`.
"""
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np


METHODS: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        METHODS[name] = fn
        return fn

    return deco


# ---------------------------------------------------------------------------
# Method samplers — each returns a dict to be merged into the member config.
# ---------------------------------------------------------------------------


@register("Subsample")
def sample_subsample(rng: np.random.Generator, X_class: np.ndarray,
                     ratio: float = 0.25,
                     ratio_range: tuple[float, float] | None = None) -> dict:
    """Subsample D_c for one member.

    Default behavior is a fixed 25% subset per member. Pass `ratio_range`
    to recover the older randomized-fraction behavior.
    """
    if ratio_range is not None:
        ratio = float(rng.uniform(*ratio_range))
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"`ratio` must be in (0, 1], got {ratio}.")
    n = max(2, int(len(X_class) * ratio))
    idx = rng.choice(len(X_class), size=n, replace=False)
    return {"method_subsample": {"ratio": ratio, "positives_idx": idx.tolist()}}


@register("Distance")
def sample_distance(rng: np.random.Generator,
                    mode: str = "random",
                    value: float | None = None,
                    dist_range: tuple[float, float] = (1.0, 30.0),
                    k_idx: int | None = None,
                    k_total: int | None = None) -> dict:
    """Hypercube radius α per member.

    Modes:
      - 'fixed'  : every member uses the same `value` (diagnostic: isolates
                   other randomness sources).
      - 'sweep'  : deterministic linspace over `dist_range` across K members
                   (member k_idx gets `linspace(lo, hi, k_total)[k_idx]`).
      - 'random' : uniform in `dist_range`.
    """
    if mode == "fixed":
        if value is None:
            raise ValueError("mode='fixed' requires `value`.")
        dist = float(value)
    elif mode == "sweep":
        if k_idx is None or k_total is None:
            raise ValueError("mode='sweep' requires k_idx / k_total "
                             "(injected by sample_member_configs).")
        grid = np.linspace(dist_range[0], dist_range[1], max(k_total, 1))
        dist = float(grid[k_idx])
    elif mode == "random":
        dist = float(rng.uniform(*dist_range))
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'fixed' | 'sweep' | 'random'.")
    return {"method_distance": {"neg_distance": dist}}


@register("CornerNoise")
def sample_corner_noise(rng: np.random.Generator,
                        noise_range: tuple[float, float] = (0.1, 2.0)) -> dict:
    """Per-corner Gaussian noise std. Applied to negatives after corner placement."""
    return {"method_corner_noise": {"noise_std": float(rng.uniform(*noise_range))}}


@register("NumFakeCorners")
def sample_num_corners(rng: np.random.Generator, d: int,
                       n_range: tuple[int, int] | None = None) -> dict:
    """Number of corners sampled from the 2^d vertices of the hypercube."""
    if n_range is None:
        n_range = (4, min(2 ** d, 64))
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    return {"method_num_fake_corners": {"n_corners": n}}


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def sample_member_configs(
    methods: list[str],
    K: int,
    seed: int,
    X_class: np.ndarray,
    method_params: dict | None = None,
    shared_corners: bool = True,
) -> list[dict]:
    """Return K per-member config dicts with all active method params resolved.

    `method_params` allows overriding each sampler's kwargs, e.g.:
        method_params = {'Distance': {'dist_range': (2.0, 20.0)}}

    `shared_corners`:
        True  → 모든 K 멤버가 같은 4 corner vertex (시각 비교 편함, Subsample/Distance 로 diversity)
        False → 멤버마다 다른 corner vertex (d>2 일 때 2^d 중 서로 다른 4 개 → 자연 diversity)
    """
    method_params = method_params or {}
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method: {m!r}. Available: {list(METHODS)}")

    rng = np.random.default_rng(seed)
    d = int(X_class.shape[1])

    configs = []
    for k in range(K):
        corner_seed = int(seed) if shared_corners else int(seed + k)
        cfg: dict = {"member_idx": k, "seed": int(seed + k), "corner_seed": corner_seed}
        for m in methods:
            fn = METHODS[m]
            sig = inspect.signature(fn)
            raw_kwargs = dict(method_params.get(m, {}))
            # Inject positional dependencies that specific methods need
            if "X_class" in sig.parameters:
                raw_kwargs.setdefault("X_class", X_class)
            if "d" in sig.parameters:
                raw_kwargs.setdefault("d", d)
            if "k_idx" in sig.parameters:
                raw_kwargs.setdefault("k_idx", k)
            if "k_total" in sig.parameters:
                raw_kwargs.setdefault("k_total", K)
            # Filter to the function's signature so extra keys don't crash
            kwargs = {k2: v for k2, v in raw_kwargs.items() if k2 in sig.parameters}
            cfg.update(fn(rng=rng, **kwargs))
        cfg["_methods"] = list(methods)
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Surrogate data builder from a resolved config
# ---------------------------------------------------------------------------


def _build_corners(d: int, distance: float, n: int | None, rng: np.random.Generator) -> np.ndarray:
    """Generate `n` distinct vertices of the [-distance, +distance]^d hypercube.

    n=None → TabEBM default (4 symmetric vertices).
    n >= 2^d → all vertices.

    For d > 2 we mirror the paper's logic: each accepted vertex `v` adds its
    antipode `-v` immediately, enforcing point symmetry about the origin.
    """
    max_corners = 2 ** d
    if n is None:
        # match TabEBM's default: 2D deterministic square, else 4 diverse points
        n = 4

    if d == 2 and n == 4:
        return distance * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)

    if n >= max_corners:
        # full hypercube vertex set
        signs = np.array(np.meshgrid(*[[-1, 1]] * d)).T.reshape(-1, d)
        # deterministic: first n
        signs = signs[:n]
    else:
        # Paper-style symmetric sampling: every accepted v also deposits -v.
        seen: set[tuple[int, ...]] = set()
        ordered: list[tuple[int, ...]] = []
        while len(seen) < n:
            v = tuple(int(x) for x in rng.choice([-1, 1], size=d))
            if v in seen:
                continue
            neg_v = tuple(-x for x in v)
            seen.add(v); ordered.append(v)
            if neg_v not in seen:
                seen.add(neg_v); ordered.append(neg_v)
        signs = np.array(ordered[:n])
    return distance * signs.astype(np.float64)


def build_surrogate_data(X_class: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """From a resolved member config, produce (X_ebm, y_ebm).

    y convention matches TabEBM.add_surrogate_negative_samples: 0 = real, 1 = surrogate.
    """
    # Per-member rng: subsample indices, corner Gaussian noise.
    rng = np.random.default_rng(cfg["seed"])
    # Shared rng: which vertices of {-1,+1}^d become corners. Same for all K
    # members in a run, so holding Distance/NumFakeCorners fixed yields the
    # same corner SET across members (varies only via α scale / CornerNoise).
    # MT19937 (legacy RandomState) — TabEBM.add_surrogate_negative_samples 와 동일 RNG 사용.
    # 같은 seed → 같은 4 vertex (bit-exact set) 보장.
    corner_rng = np.random.RandomState(cfg.get("corner_seed", cfg["seed"]))
    d = int(X_class.shape[1])

    # positives (Subsample or all)
    sub = cfg.get("method_subsample") or {}
    if sub:
        X_pos = X_class[np.array(sub["positives_idx"])]
    else:
        X_pos = X_class

    # distance (Distance or fixed default)
    dist = float(cfg.get("method_distance", {}).get("neg_distance", 5.0))

    # number of corners (NumFakeCorners or default)
    n_corners = cfg.get("method_num_fake_corners", {}).get("n_corners")

    # corner noise std (CornerNoise or 0)
    noise_std = float(cfg.get("method_corner_noise", {}).get("noise_std", 0.0))

    X_neg = _build_corners(d=d, distance=dist, n=n_corners, rng=corner_rng)
    if noise_std > 0.0:
        X_neg = X_neg + rng.normal(0.0, noise_std, size=X_neg.shape)

    X_ebm = np.vstack([X_pos, X_neg]).astype(np.float64)
    y_ebm = np.concatenate([
        np.zeros(len(X_pos), dtype=np.int64),
        np.ones(len(X_neg), dtype=np.int64),
    ])
    return X_ebm, y_ebm
