from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

try:
    from .reproducibility import DEFAULT_PROJECT_RANDOM_SEED
except ImportError:
    from src.reproducibility import DEFAULT_PROJECT_RANDOM_SEED


DEFAULT_THETA_BOUNDS = (
    (0.00, 0.02),   # x0
    (0.80, 0.99),   # y0
    (80.0, 140.0), # Q_NE
    (140.0, 180.0), # Q_PE
)
DEFAULT_OFFSET_SEARCH_RANGE = (-600, 600)

_DEFAULT_BASE_THETA_MIN_WIDTHS = np.array([1e-6, 1e-6, 1e-4, 1e-4], dtype=float)


def _build_rest_segmentation_preset(
    *,
    max_iters: int,
    search_range: tuple[int, int],
    local_maxiter_init: int,
    local_maxiter_iter: int,
):
    return {
        "V_error_max_threshold": 0.02,
        "rmsev_stop_threshold": 0.005,
        "rmseq_segmentation_stop_threshold": 0.25,
        "post_merge_offset_gap_threshold": 0.3,
        "max_iters": max_iters,
        "search_range": search_range,
        "verbose": False,
        "de_workers": 32,
        "de_maxiter": 30,
        "de_popsize": 16,
        "de_refit_period": 5,
        "local_maxiter_init": local_maxiter_init,
        "local_maxiter_iter": local_maxiter_iter,
        "random_seed": DEFAULT_PROJECT_RANDOM_SEED,
        "min_seg_len": 3,
        "split_selection_mode": "evaluate_split_or_drop",
        "min_split_segment_len": None,
        "drop_point_penalty": None,
        "force_drop_split_on_no_candidate": True,
        "segmentation": True,
    }


def _build_rest_uncertainty_preset(
    *,
    n_iter: int = 32,
    n_jobs: int = 32,
    use_robust_noise_std: bool = False,
    segment_generation_mode: str = "direct_voltage_noise",
    local_bounds_base_frac: float = 0.2,
    multistart_base_frac: float = 0.2,
    objective_rmsev_weight: float = 200,
    objective_prior_weight: float = 0.0,
    objective_stop_threshold: float = 3.0,
    local_maxiter_init: int = 50,
    local_maxiter_iter: int = 50,
    max_fit_loops: int = 2,
    multistart_n: int = 2,
    multistart_workers: int = 1,
    fit_patience: int = 2,
):
    return {
        "n_iter": n_iter,
        "use_robust_noise_std": use_robust_noise_std,
        "segment_generation_mode": str(segment_generation_mode),
        "segmentation": False,
        "random_seed": DEFAULT_PROJECT_RANDOM_SEED,
        "n_jobs": n_jobs,
        "iterative_kwargs": {
            "search_range": DEFAULT_OFFSET_SEARCH_RANGE,
            "skip_global_init": True,
            "objective_mode": "combined",
            "local_bounds_base_frac": float(local_bounds_base_frac),
            "multistart_base_frac": float(multistart_base_frac),
            "objective_rmsev_weight": float(objective_rmsev_weight),
            "objective_prior_weight": float(objective_prior_weight),
            "objective_stop_threshold": float(objective_stop_threshold),
            "local_maxiter_init": int(local_maxiter_init),
            "local_maxiter_iter": int(local_maxiter_iter),
            "max_fit_loops": int(max_fit_loops),
            "multistart_n": int(multistart_n),
            "multistart_workers": int(multistart_workers),
            "fit_patience": int(fit_patience),
        },
    }


FIT_PRESETS = {
    "default": {
        "theta_bounds": DEFAULT_THETA_BOUNDS,
        "rest_segmentation": _build_rest_segmentation_preset(
            max_iters=300,
            search_range=DEFAULT_OFFSET_SEARCH_RANGE,
            local_maxiter_init=150,
            local_maxiter_iter=80,
        ),
        "rest_uncertainty": _build_rest_uncertainty_preset(),
    },
}


def _bounds_arrays(bounds):
    lower = np.array([float(item[0]) for item in bounds], dtype=float)
    upper = np.array([float(item[1]) for item in bounds], dtype=float)
    return lower, upper


def _sanitize_positive_float(value: Any, fallback: float):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(value) or value <= 0:
        return float(fallback)
    return float(value)


def _extract_base_theta(base_result: dict[str, Any] | None):
    if not isinstance(base_result, dict) or "theta" not in base_result:
        return None
    theta = np.asarray(base_result["theta"], dtype=float)
    if theta.shape != (4,) or not np.all(np.isfinite(theta)):
        return None
    return theta


def build_base_theta_half_widths(
    theta,
    base_frac: float,
    *,
    minimum_widths=_DEFAULT_BASE_THETA_MIN_WIDTHS,
):
    theta_arr = np.asarray(theta, dtype=float)
    minimum_widths_arr = np.asarray(minimum_widths, dtype=float)
    base_frac = _sanitize_positive_float(base_frac, fallback=0.10)
    return np.maximum(base_frac * np.abs(theta_arr), minimum_widths_arr)


def build_base_theta_multistart_rel_scales(
    theta,
    bounds,
    base_frac: float,
    *,
    minimum_rel_scale: float = 1e-6,
):
    theta_arr = np.asarray(theta, dtype=float)
    lower, upper = _bounds_arrays(bounds)
    span = upper - lower
    base_frac = _sanitize_positive_float(base_frac, fallback=0.10)
    rel_scales = np.divide(
        base_frac * np.abs(theta_arr),
        span,
        out=np.full_like(theta_arr, float(minimum_rel_scale), dtype=float),
        where=span > 0,
    )
    return np.maximum(rel_scales, float(minimum_rel_scale))


def materialize_rest_uncertainty_iterative_kwargs(
    base_result,
    iterative_kwargs=None,
    *,
    theta_bounds=None,
):
    kwargs = {} if iterative_kwargs is None else deepcopy(iterative_kwargs)
    bounds = tuple(kwargs.get("bounds", theta_bounds or DEFAULT_THETA_BOUNDS))
    kwargs["bounds"] = bounds

    theta_base = _extract_base_theta(base_result)
    if theta_base is not None:
        kwargs.setdefault("warm_start_theta", theta_base.copy())

    local_bounds_base_frac = kwargs.pop("local_bounds_base_frac", None)
    if (
        local_bounds_base_frac is not None
        and "local_bounds_half_widths" not in kwargs
        and theta_base is not None
    ):
        kwargs["local_bounds_half_widths"] = tuple(
            build_base_theta_half_widths(theta_base, local_bounds_base_frac).tolist()
        )

    multistart_base_frac = kwargs.pop("multistart_base_frac", None)
    if (
        multistart_base_frac is not None
        and "multistart_rel_scales" not in kwargs
        and theta_base is not None
    ):
        kwargs["multistart_rel_scales"] = tuple(
            build_base_theta_multistart_rel_scales(theta_base, bounds, multistart_base_frac).tolist()
        )

    objective_mode = str(kwargs.get("objective_mode", "rmseq")).strip().lower()
    if objective_mode == "combined":
        # Legacy scale-based normalization is disabled. Drop any stale keys so
        # materialized configs reflect the actual unnormalized objective.
        kwargs.pop("objective_rmseq_scale", None)
        kwargs.pop("objective_rmsev_scale", None)

        prior_weight = float(kwargs.get("objective_prior_weight", 0.0) or 0.0)
        if prior_weight > 0 and theta_base is not None:
            kwargs.setdefault("objective_prior_theta", theta_base.copy())
            if "objective_prior_scales" not in kwargs:
                if "local_bounds_half_widths" in kwargs:
                    kwargs["objective_prior_scales"] = tuple(kwargs["local_bounds_half_widths"])
                else:
                    lower, upper = _bounds_arrays(bounds)
                    kwargs["objective_prior_scales"] = tuple((upper - lower).tolist())

    return kwargs






def get_scope_presets(data_scope: str):
    if data_scope not in FIT_PRESETS:
        available_scopes = ", ".join(sorted(FIT_PRESETS.keys()))
        raise KeyError(f"Unknown data_scope={data_scope!r}. Available scopes: {available_scopes}")
    return deepcopy(FIT_PRESETS[data_scope])




def get_runtime_fit_configs(data_scope: str):
    presets = get_scope_presets(data_scope)
    theta_bounds = presets["theta_bounds"]

    runtime = {
        "theta_bounds": theta_bounds,
        "rest_segmentation": deepcopy(presets["rest_segmentation"]),
        "rest_uncertainty": deepcopy(presets["rest_uncertainty"]),
    }

    runtime["rest_segmentation"]["bounds"] = theta_bounds
    runtime["rest_uncertainty"]["iterative_kwargs"] = deepcopy(runtime["rest_uncertainty"]["iterative_kwargs"])
    runtime["rest_uncertainty"]["iterative_kwargs"]["bounds"] = theta_bounds

    return runtime


