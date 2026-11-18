
from __future__ import annotations

from dataclasses import dataclass

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

try:
    from .analysis_core import get_OCV_extrapolated, get_OCV_inv_extrapolated
    from .fit_presets import DEFAULT_OFFSET_SEARCH_RANGE, DEFAULT_THETA_BOUNDS
    from .reproducibility import DEFAULT_PROJECT_RANDOM_SEED, derive_seed
except ImportError:
    from src.analysis_core import get_OCV_extrapolated, get_OCV_inv_extrapolated
    from src.fit_presets import DEFAULT_OFFSET_SEARCH_RANGE, DEFAULT_THETA_BOUNDS
    from src.reproducibility import DEFAULT_PROJECT_RANDOM_SEED, derive_seed


try:
    dataclass(slots=True)
except TypeError:
    def slots_dataclass(*args, **kwargs):
        kwargs.pop("slots", None)
        return dataclass(*args, **kwargs)
else:
    slots_dataclass = dataclass














# =========================================================
# Data container
# =========================================================
@slots_dataclass(slots=True)
class SegmentData:
    raw: np.ndarray
    q: np.ndarray
    v: np.ndarray
    dq_true: np.ndarray
    idx: np.ndarray   # original row indices


def _make_segment(seg, idx=None):
    arr = np.asarray(seg, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Each segment must be a 2D array with at least 2 columns: [Q, V].")

    if idx is None:
        idx = np.arange(arr.shape[0], dtype=int)
    else:
        idx = np.asarray(idx, dtype=int)

    if arr.shape[0] != idx.shape[0]:
        raise ValueError("seg and idx must have the same length.")

    q = np.ascontiguousarray(arr[:, 0])
    v = np.ascontiguousarray(arr[:, 1])
    dq_true = np.diff(q) if q.size >= 2 else np.empty(0, dtype=float)

    return SegmentData(
        raw=arr,
        q=q,
        v=v,
        dq_true=dq_true,
        idx=idx,
    )


def _as_segment_list(QV_array_or_segments):
    """
    - single ndarray: original indices = 0, 1, ..., N-1
    - list of ndarrays: indices are assigned consecutively across segments
    """
    if isinstance(QV_array_or_segments, (list, tuple)):
        segments = []
        offset = 0
        for seg in QV_array_or_segments:
            arr = np.asarray(seg, dtype=float)
            idx = np.arange(offset, offset + arr.shape[0], dtype=int)
            segments.append(_make_segment(arr, idx=idx))
            offset += arr.shape[0]
        return segments

    arr = np.asarray(QV_array_or_segments, dtype=float)
    idx = np.arange(arr.shape[0], dtype=int)
    return [_make_segment(arr, idx=idx)]


# =========================================================
# dQ objective
# =========================================================
DQ_WEIGHTED_VOLTAGE_MIN = 2.5
DQ_WEIGHTED_VOLTAGE_MAX = 4.2
DQ_WEIGHTED_VOLTAGE_STEP = 0.1
DQ_WEIGHTED_BIN_EDGES = np.round(
    np.arange(
        DQ_WEIGHTED_VOLTAGE_MIN,
        DQ_WEIGHTED_VOLTAGE_MAX + DQ_WEIGHTED_VOLTAGE_STEP,
        DQ_WEIGHTED_VOLTAGE_STEP,
    ),
    decimals=10,
)


def squared_dQ_error_seg_from_finv(seg, f_inv, return_vmid=False):
    if seg.dq_true.size == 0:
        empty = np.empty(0, dtype=float)
        if return_vmid:
            return empty, empty
        return empty

    q_from_v = np.asarray(f_inv(seg.v), dtype=float)
    dq_model = np.diff(q_from_v)
    diff = seg.dq_true - dq_model
    sqerr = diff * diff

    if return_vmid:
        v_mid = 0.5 * (seg.v[:-1] + seg.v[1:])
        return sqerr, v_mid
    return sqerr


def _weighted_rmse_from_voltage_bins(bin_sqerr_sum, bin_counts):
    non_empty = bin_counts > 0
    if not np.any(non_empty):
        return np.inf

    bin_rmse = np.sqrt(bin_sqerr_sum[non_empty] / bin_counts[non_empty])
    bin_weights = 1.0 / bin_counts[non_empty]
    return float(np.dot(bin_weights, bin_rmse) / np.sum(bin_weights))


def compute_dQ_rmse(theta, segments, return_sqerrs=False, weighted=False):
    x0, y0, Q_NE, Q_PE = theta
    f_inv = get_OCV_inv_extrapolated(x0, y0, Q_NE, Q_PE)

    sse = 0.0
    n_total = 0
    sqerrs_per_seg = [] if return_sqerrs else None

    if weighted:
        n_bins = DQ_WEIGHTED_BIN_EDGES.size - 1
        bin_sqerr_sum = np.zeros(n_bins, dtype=float)
        bin_counts = np.zeros(n_bins, dtype=int)
        seg_bin_indices = [] if return_sqerrs else None

    for seg_like in segments:
        seg = seg_like if isinstance(seg_like, SegmentData) else _make_segment(seg_like)

        if weighted:
            sqerr, v_mid = squared_dQ_error_seg_from_finv(seg, f_inv, return_vmid=True)
            bin_indices = np.full(sqerr.shape, -1, dtype=int)
        else:
            sqerr = squared_dQ_error_seg_from_finv(seg, f_inv)

        if return_sqerrs:
            sqerrs_per_seg.append(sqerr)
            if weighted:
                seg_bin_indices.append(bin_indices)

        if sqerr.size:
            if weighted:
                valid = np.isfinite(sqerr) & np.isfinite(v_mid)
                valid &= (v_mid >= DQ_WEIGHTED_BIN_EDGES[0]) & (v_mid <= DQ_WEIGHTED_BIN_EDGES[-1])
                if np.any(valid):
                    current_bins = np.searchsorted(
                        DQ_WEIGHTED_BIN_EDGES,
                        v_mid[valid],
                        side="right",
                    ) - 1
                    current_bins = np.clip(current_bins, 0, n_bins - 1)
                    if return_sqerrs:
                        seg_bin_indices[-1][valid] = current_bins
                    bin_sqerr_sum += np.bincount(
                        current_bins,
                        weights=sqerr[valid],
                        minlength=n_bins,
                    )
                    bin_counts += np.bincount(current_bins, minlength=n_bins)
            else:
                sse += float(np.sum(sqerr))
                n_total += sqerr.size

    if weighted:
        rmse = _weighted_rmse_from_voltage_bins(bin_sqerr_sum, bin_counts)
        if return_sqerrs:
            weighted_sqerrs_per_seg = []
            for sqerr, bin_indices in zip(sqerrs_per_seg, seg_bin_indices):
                if sqerr.size == 0:
                    weighted_sqerrs_per_seg.append(sqerr)
                    continue

                sqerr_weighted = np.zeros_like(sqerr, dtype=float)
                valid = bin_indices >= 0
                if np.any(valid):
                    sqerr_weighted[valid] = sqerr[valid] / bin_counts[bin_indices[valid]]
                weighted_sqerrs_per_seg.append(sqerr_weighted)
            sqerrs_per_seg = weighted_sqerrs_per_seg
    else:
        rmse = float(np.sqrt(sse / n_total)) if n_total > 0 else np.inf

    if return_sqerrs:
        return rmse, sqerrs_per_seg
    return rmse


def dQ_rmse_objective(theta, segments, weighted=False):
    return compute_dQ_rmse(theta, segments, return_sqerrs=False, weighted=weighted)


def _bounds_arrays(bounds):
    lower = np.array([float(item[0]) for item in bounds], dtype=float)
    upper = np.array([float(item[1]) for item in bounds], dtype=float)
    return lower, upper


def normalize_theta_bounds(bounds):
    lower, upper = _bounds_arrays(bounds)
    return tuple((float(lo), float(hi)) for lo, hi in zip(lower, upper))


def clip_theta_to_bounds(theta, bounds=DEFAULT_THETA_BOUNDS):
    theta_arr = np.asarray(theta, dtype=float)
    lower, upper = _bounds_arrays(bounds)
    if theta_arr.shape != lower.shape:
        raise ValueError(
            f"theta shape {theta_arr.shape} does not match bounds dimensionality {lower.shape}."
        )
    return np.clip(theta_arr, lower, upper)


def build_local_theta_bounds(
    bounds=DEFAULT_THETA_BOUNDS,
    *,
    center_theta,
    half_widths=None,
    rel_scales=None,
):
    if center_theta is None:
        raise ValueError("center_theta is required to build local bounds.")

    base_bounds = normalize_theta_bounds(bounds)
    lower, upper = _bounds_arrays(base_bounds)
    center = clip_theta_to_bounds(center_theta, base_bounds)
    span = upper - lower

    if half_widths is not None:
        widths = np.asarray(half_widths, dtype=float)
    elif rel_scales is not None:
        widths = np.asarray(rel_scales, dtype=float) * span
    else:
        return base_bounds

    if widths.shape != center.shape:
        raise ValueError(
            f"local bound width shape {widths.shape} does not match theta shape {center.shape}."
        )
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Local bound widths must be finite positive values.")

    local_lower = np.maximum(lower, center - widths)
    local_upper = np.minimum(upper, center + widths)
    if np.any(local_upper <= local_lower):
        raise ValueError("Local bounds collapsed; widen the local bound widths.")

    return tuple((float(lo), float(hi)) for lo, hi in zip(local_lower, local_upper))




def build_theta_objective(
    segments,
    *,
    objective_mode="rmseq",
    weighted=False,
    search_range=DEFAULT_OFFSET_SEARCH_RANGE,
    rmsev_weight=0.0,
    prior_theta=None,
    prior_weight=0.0,
    prior_scales=None,
    bounds=DEFAULT_THETA_BOUNDS,
):
    objective_mode_normalized = str(objective_mode).strip().lower()
    if objective_mode_normalized not in {"rmseq", "combined"}:
        raise ValueError(
            f"Unknown objective_mode={objective_mode!r}. Expected 'rmseq' or 'combined'."
        )

    try:
        rmsev_weight = float(rmsev_weight)
    except (TypeError, ValueError):
        rmsev_weight = 0.0
    if not np.isfinite(rmsev_weight):
        rmsev_weight = 0.0

    try:
        prior_weight = float(prior_weight)
    except (TypeError, ValueError):
        prior_weight = 0.0
    if not np.isfinite(prior_weight):
        prior_weight = 0.0

    lower, upper = _bounds_arrays(bounds)
    span = upper - lower

    prior_theta_arr = None
    prior_scales_arr = None
    if prior_theta is not None and prior_weight > 0:
        prior_theta_arr = clip_theta_to_bounds(prior_theta, bounds)
        if prior_scales is None:
            prior_scales_arr = span.copy()
        else:
            prior_scales_arr = np.asarray(prior_scales, dtype=float)
            if prior_scales_arr.shape != prior_theta_arr.shape:
                raise ValueError(
                    f"prior_scales shape {prior_scales_arr.shape} does not match theta shape {prior_theta_arr.shape}."
                )
            invalid = ~np.isfinite(prior_scales_arr) | (prior_scales_arr <= 0)
            if np.any(invalid):
                prior_scales_arr = prior_scales_arr.copy()
                prior_scales_arr[invalid] = span[invalid]
        prior_scales_arr = np.where(prior_scales_arr > 0, prior_scales_arr, span)

    def objective(theta):
        theta_arr = np.asarray(theta, dtype=float)
        rmse_q = compute_dQ_rmse(theta_arr, segments, return_sqerrs=False, weighted=weighted)
        total = float(rmse_q)

        if objective_mode_normalized == "combined":
            if rmsev_weight > 0:
                rmse_v, _ = compute_V_RMSE_and_offsets(theta_arr, segments, search_range)
                total += float(rmsev_weight) * float(rmse_v)

            if prior_theta_arr is not None and prior_weight > 0:
                delta = (theta_arr - prior_theta_arr) / prior_scales_arr
                total += float(prior_weight) * float(np.dot(delta, delta))

        return float(total)

    metadata = {
        "mode": objective_mode_normalized,
        "label": "RMSE_Q" if objective_mode_normalized == "rmseq" else "combined objective",
        "rmsev_weight": float(rmsev_weight),
        "prior_weight": float(prior_weight),
        "prior_theta": None if prior_theta_arr is None else prior_theta_arr.copy(),
        "prior_scales": None if prior_scales_arr is None else prior_scales_arr.copy(),
    }
    return objective, metadata




# =========================================================
# Optimization
# =========================================================
def run_global_optimization(
    segments,
    bounds=DEFAULT_THETA_BOUNDS,
    verbose=True,
    de_workers=1,
    maxiter=100,
    popsize=8,
    tol=1e-4,
    seed=DEFAULT_PROJECT_RANDOM_SEED,
    print_every=10,
    weighted=False,
    objective_fn=None,
    objective_label="RMSE_Q",
):
    if objective_fn is None:
        def objective_fn(theta):
            return dQ_rmse_objective(theta, segments, weighted=weighted)

    de_state = {"gen": 0}

    def de_callback(x, convergence):
        de_state["gen"] += 1
        if verbose and (de_state["gen"] == 1 or de_state["gen"] % print_every == 0):
            fval = objective_fn(x)
            print(
                f"[DE gen {de_state['gen']:4d}] "
                f"{objective_label}={fval:.8f} | "
                f"x0={x[0]:.6f}, y0={x[1]:.6f}, "
                f"Q_NE={x[2]:.4f}, Q_PE={x[3]:.4f}",
                flush=True
            )
        return False

    result_de = differential_evolution(
        objective_fn,
        bounds=bounds,
        strategy="best1bin",
        popsize=popsize,
        tol=tol,
        maxiter=maxiter,
        seed=seed,
        polish=False,
        updating="deferred",
        workers=de_workers,
        callback=de_callback,
        disp=False,
    )

    if verbose:
        x = result_de.x
        print(
            f"[DE done] {objective_label}={result_de.fun:.8f} | "
            f"x0={x[0]:.6f}, y0={x[1]:.6f}, "
            f"Q_NE={x[2]:.4f}, Q_PE={x[3]:.4f}",
            flush=True
        )

    return result_de.x


def run_local_optimization(
    theta0,
    segments,
    bounds=DEFAULT_THETA_BOUNDS,
    verbose=True,
    maxiter=150,
    ftol=1e-8,
    print_every=10,
    weighted=False,
    objective_fn=None,
    objective_label="RMSE_Q",
):
    if objective_fn is None:
        def objective_fn(theta):
            return dQ_rmse_objective(theta, segments, weighted=weighted)

    if verbose:
        theta0_arr = np.asarray(theta0, dtype=float)
        print(
            "[LBFGSB init] "
            f"x0={theta0_arr[0]:.6f}, y0={theta0_arr[1]:.6f}, "
            f"Q_NE={theta0_arr[2]:.4f}, Q_PE={theta0_arr[3]:.4f}",
            flush=True
        )

    local_state = {"it": 0}

    def local_callback(xk):
        local_state["it"] += 1
        if verbose and (local_state["it"] == 1 or local_state["it"] % print_every == 0):
            fval = objective_fn(xk)
            print(
                f"[LBFGSB {local_state['it']:4d}] "
                f"{objective_label}={fval:.8f} | "
                f"x0={xk[0]:.6f}, y0={xk[1]:.6f}, "
                f"Q_NE={xk[2]:.4f}, Q_PE={xk[3]:.4f}",
                flush=True
            )

    result_local = minimize(
        objective_fn,
        x0=theta0,
        method="L-BFGS-B",
        bounds=bounds,
        callback=local_callback,
        options={
            "maxiter": maxiter,
            "ftol": ftol,
        },
    )

    if verbose:
        x = result_local.x
        print(
            f"[LBFGSB done] success={result_local.success}, "
            f"{objective_label}={result_local.fun:.8f} | "
            f"x0={x[0]:.6f}, y0={x[1]:.6f}, "
            f"Q_NE={x[2]:.4f}, Q_PE={x[3]:.4f}",
            flush=True
        )

    return result_local.x


# =========================================================
# V-RMSE / offset
# =========================================================
def _V_mse_offset(q, v, f, offset):
    err = v - f(q + offset)
    return float(np.mean(err * err))


def best_offset_for_segment_with_V_RMSE(seg, f, search_range=DEFAULT_OFFSET_SEARCH_RANGE, xatol=1e-3):
    def objective(off):
        return _V_mse_offset(seg.q, seg.v, f, off)

    res = minimize_scalar(
        objective,
        bounds=search_range,
        method="bounded",
        options={"xatol": xatol},
    )

    best_off = float(res.x)
    best_rmse = float(np.sqrt(res.fun))
    return best_off, best_rmse


def V_error_seg(seg, f, offset):
    return seg.v - f(seg.q + offset)


def compute_V_RMSE_and_offsets(theta, segments, search_range=DEFAULT_OFFSET_SEARCH_RANGE, xatol=1e-3):
    x0, y0, Q_NE, Q_PE = theta
    f = get_OCV_extrapolated(x0, y0, Q_NE, Q_PE)

    sse = 0.0
    n_total = 0
    offsets = []

    for seg in segments:
        if seg.q.size < 2:
            offsets.append(None)
            continue

        off, _ = best_offset_for_segment_with_V_RMSE(seg, f, search_range, xatol=xatol)
        offsets.append(off)

        verr = V_error_seg(seg, f, off)
        sse += float(np.dot(verr, verr))
        n_total += verr.size

    rmse = float(np.sqrt(sse / n_total)) if n_total > 0 else np.inf
    return rmse, offsets


def compute_V_error_max_and_offsets(theta, segments, search_range=DEFAULT_OFFSET_SEARCH_RANGE, xatol=1e-3):
    x0, y0, Q_NE, Q_PE = theta
    f = get_OCV_extrapolated(x0, y0, Q_NE, Q_PE)

    offsets = []
    abs_v_err_max = -np.inf
    has_valid_point = False

    for seg in segments:
        if seg.q.size < 2:
            offsets.append(None)
            continue

        off, _ = best_offset_for_segment_with_V_RMSE(seg, f, search_range, xatol=xatol)
        offsets.append(off)

        verr = V_error_seg(seg, f, off)
        if verr.size > 0:
            has_valid_point = True
            cur_max = float(np.max(np.abs(verr)))
            if cur_max > abs_v_err_max:
                abs_v_err_max = cur_max

    if not has_valid_point:
        abs_v_err_max = np.inf

    return abs_v_err_max, offsets


def find_max_v_error_point(theta, segments, search_range=DEFAULT_OFFSET_SEARCH_RANGE, xatol=1e-3):
    x0, y0, Q_NE, Q_PE = theta
    f = get_OCV_extrapolated(x0, y0, Q_NE, Q_PE)

    best_seg_idx = None
    best_point_idx = None
    best_abs_err = -np.inf
    offsets = []

    for seg_idx, seg in enumerate(segments):
        if seg.q.size < 2:
            offsets.append(None)
            continue

        off, _ = best_offset_for_segment_with_V_RMSE(seg, f, search_range, xatol=xatol)
        offsets.append(off)

        verr = V_error_seg(seg, f, off)
        if verr.size == 0:
            continue

        point_idx = int(np.argmax(np.abs(verr)))
        abs_err = float(np.abs(verr[point_idx]))
        if abs_err > best_abs_err:
            best_abs_err = abs_err
            best_seg_idx = seg_idx
            best_point_idx = point_idx

    return best_seg_idx, best_point_idx, best_abs_err, offsets






def _evaluate_segment_metrics(theta, segments, search_range=DEFAULT_OFFSET_SEARCH_RANGE, weighted=False):
    rmseQ, sqerrs_per_seg = compute_dQ_rmse(
        theta,
        segments,
        return_sqerrs=True,
        weighted=weighted,
    )
    rmseV, offsets = compute_V_RMSE_and_offsets(theta, segments, search_range)
    V_error_max, offsets = compute_V_error_max_and_offsets(theta, segments, search_range)
    return rmseQ, sqerrs_per_seg, rmseV, V_error_max, offsets


def merge_adjacent_segments_by_offset(segments, offsets, offset_gap_threshold=0.3):
    if not segments:
        return [], [], 0

    working = [seg if isinstance(seg, SegmentData) else _make_segment(seg) for seg in segments]
    merged_segments = []
    merged_count = 0
    i = 0

    while i < len(working):
        cur_raw = working[i].raw.copy()
        cur_idx = working[i].idx.copy()
        cur_off = offsets[i] if i < len(offsets) else None
        j = i + 1

        while (
            j < len(working)
            and cur_off is not None
            and j < len(offsets)
            and offsets[j] is not None
            and abs(float(cur_off) - float(offsets[j])) <= offset_gap_threshold
        ):
            cur_raw = np.vstack([cur_raw, working[j].raw])
            cur_idx = np.concatenate([cur_idx, working[j].idx])
            merged_count += 1
            j += 1

        order = np.argsort(cur_idx)
        merged_segments.append(_make_segment(cur_raw[order], idx=cur_idx[order]))
        i = j

    merged_offsets = [None] * len(merged_segments)
    return merged_segments, merged_offsets, merged_count


# =========================================================
# Tiny-segment cleanup
#   - segmentation=True mode uses a simple rule:
#       if a segment length is < min_seg_len, drop it immediately
#   - no merge logic is used
#   - dropped original indices are recorded
# =========================================================
def cleanup_segments(
    segments,
    *,
    min_seg_len=3,
    verbose=True,
    iter_no=None,
    dropped_history=None,
):
    if dropped_history is None:
        dropped_history = []

    if not segments:
        return [], dropped_history

    working = [seg if isinstance(seg, SegmentData) else _make_segment(seg) for seg in segments]
    kept = []

    for i, seg in enumerate(working):
        n = len(seg.raw)
        if n < min_seg_len:
            if verbose:
                print(
                    f"[cleanup] drop tiny segment at idx={i}, n={n} (threshold < {min_seg_len})",
                    flush=True,
                )
            dropped_history.append({
                "iter": iter_no,
                "indices": seg.idx.copy(),
                "length": int(len(seg.idx)),
                "reason": "tiny_segment_immediate_drop",
            })
            continue
        kept.append(seg)

    return kept, dropped_history




def split_segment_force(seg, cut_idx):
    raw = seg.raw
    idx = seg.idx

    left_raw = raw[:cut_idx + 1]
    right_raw = raw[cut_idx + 1:]

    left_idx = idx[:cut_idx + 1]
    right_idx = idx[cut_idx + 1:]

    new_segs = []
    if len(left_raw) > 1:
        new_segs.append(_make_segment(left_raw, idx=left_idx))
    if len(right_raw) > 1:
        new_segs.append(_make_segment(right_raw, idx=right_idx))

    return new_segs


def apply_split_force(segments, seg_idx, cut_idx):
    new_segments = []
    for i, seg in enumerate(segments):
        if i != seg_idx:
            new_segments.append(seg)
        else:
            new_segments.extend(split_segment_force(seg, cut_idx))
    return new_segments


def drop_point_keep_segment(segments, seg_idx, point_idx):
    new_segments = []
    dropped_record = None

    for i, seg in enumerate(segments):
        if i != seg_idx:
            new_segments.append(seg)
            continue

        keep_mask = np.ones(len(seg.raw), dtype=bool)
        keep_mask[point_idx] = False

        dropped_record = {
            "indices": np.asarray([seg.idx[point_idx]], dtype=int),
            "length": 1,
            "reason": "max_v_error_point_drop_keep_segment",
        }

        kept_raw = seg.raw[keep_mask]
        kept_idx = seg.idx[keep_mask]
        if len(kept_raw) > 1:
            new_segments.append(_make_segment(kept_raw, idx=kept_idx))

    return new_segments, dropped_record


def drop_point_and_split_segment(
    segments,
    seg_idx,
    point_idx,
    *,
    reason="forced_drop_point_and_split",
):
    new_segments = []
    dropped_record = None

    for i, seg in enumerate(segments):
        if i != seg_idx:
            new_segments.append(seg)
            continue

        raw = seg.raw
        idx = seg.idx
        if point_idx < 0 or point_idx >= len(raw):
            continue

        dropped_record = {
            "indices": np.asarray([idx[point_idx]], dtype=int),
            "length": 1,
            "reason": reason,
        }

        left_raw = raw[:point_idx]
        left_idx = idx[:point_idx]
        right_raw = raw[point_idx + 1:]
        right_idx = idx[point_idx + 1:]

        if len(left_raw) > 0:
            new_segments.append(_make_segment(left_raw, idx=left_idx))
        if len(right_raw) > 0:
            new_segments.append(_make_segment(right_raw, idx=right_idx))

    return new_segments, dropped_record


def _segment_signature(segments):
    sig = []
    for seg in segments:
        arr = seg.raw
        idx = seg.idx
        n = len(arr)
        q0, v0 = float(arr[0, 0]), float(arr[0, 1])
        q1, v1 = float(arr[-1, 0]), float(arr[-1, 1])
        i0, i1 = int(idx[0]), int(idx[-1])
        sig.append((n, i0, i1, round(q0, 12), round(v0, 12), round(q1, 12), round(v1, 12)))
    return tuple(sig)


def find_best_effective_split(segments, sqerrs_per_seg, verbose=True):
    candidates = []

    for si, e in enumerate(sqerrs_per_seg):
        if e.size == 0:
            continue
        idx = int(np.argmax(e))
        val = float(e[idx])
        candidates.append((val, si, idx))

    if not candidates:
        return None, None, None

    candidates.sort(reverse=True, key=lambda x: x[0])
    old_sig = _segment_signature(segments)

    for rank, (val, si, idx) in enumerate(candidates, start=1):
        new_segments = apply_split_force(segments, si, idx)
        new_sig = _segment_signature(new_segments)

        if new_sig != old_sig:
            if verbose and rank > 1:
                print(
                    f"[split-select] skipped {rank-1} ineffective candidate(s), "
                    f"choose seg={si}, idx={idx}, sqerr={val:.6e}",
                    flush=True
                )
            return si, idx, val

    return None, None, None


def compute_dQ_sse_and_count(theta, segments, weighted=False):
    if weighted:
        rmse = compute_dQ_rmse(theta, segments, weighted=True)
        n_total = int(sum(max(0, len(seg.raw if isinstance(seg, SegmentData) else seg) - 1) for seg in segments))
        return float(rmse * rmse * n_total), n_total

    x0, y0, Q_NE, Q_PE = theta
    f_inv = get_OCV_inv_extrapolated(x0, y0, Q_NE, Q_PE)

    sse = 0.0
    n_total = 0
    for seg_like in segments:
        seg = seg_like if isinstance(seg_like, SegmentData) else _make_segment(seg_like)
        sqerr = squared_dQ_error_seg_from_finv(seg, f_inv)
        if sqerr.size:
            finite = np.isfinite(sqerr)
            if np.any(finite):
                sse += float(np.sum(sqerr[finite]))
                n_total += int(np.sum(finite))

    return sse, n_total


def _flatten_finite_sqerrs(sqerrs_per_seg):
    values = []
    for sqerr in sqerrs_per_seg:
        arr = np.asarray(sqerr, dtype=float)
        if arr.size:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                values.append(finite)
    if not values:
        return np.empty(0, dtype=float)
    return np.concatenate(values)


def _finite_boundary_candidates_from_sqerrs(sqerrs_per_seg):
    boundary_candidates = []
    for seg_idx, sqerr in enumerate(sqerrs_per_seg):
        arr = np.asarray(sqerr, dtype=float)
        for boundary_idx, value in enumerate(arr):
            if np.isfinite(value):
                boundary_candidates.append((float(value), int(seg_idx), int(boundary_idx)))
    boundary_candidates.sort(reverse=True, key=lambda item: item[0])
    return boundary_candidates


def choose_split_or_drop_action(
    theta,
    segments,
    sqerrs_per_seg,
    *,
    weighted=False,
    min_split_segment_len=3,
    drop_point_penalty=None,
    verbose=True,
):
    working = [seg if isinstance(seg, SegmentData) else _make_segment(seg) for seg in segments]
    current_sse, current_n = compute_dQ_sse_and_count(theta, working, weighted=weighted)
    finite_sqerrs = _flatten_finite_sqerrs(sqerrs_per_seg)
    adaptive_drop_penalty = (
        float(np.median(finite_sqerrs))
        if drop_point_penalty is None and finite_sqerrs.size
        else 0.0 if drop_point_penalty is None else float(drop_point_penalty)
    )

    boundary_candidates = _finite_boundary_candidates_from_sqerrs(sqerrs_per_seg)
    if not boundary_candidates:
        return None

    old_sig = _segment_signature(working)

    for rank, (max_sqerr, seg_idx, boundary_idx) in enumerate(boundary_candidates, start=1):
        if seg_idx >= len(working):
            continue
        seg = working[seg_idx]
        n_seg = len(seg.raw)

        candidates = []

        left_len = boundary_idx + 1
        right_len = n_seg - left_len
        if left_len >= min_split_segment_len and right_len >= min_split_segment_len:
            split_segments = apply_split_force(working, seg_idx, boundary_idx)
            if _segment_signature(split_segments) != old_sig:
                sse, n_eval = compute_dQ_sse_and_count(theta, split_segments, weighted=weighted)
                candidates.append({
                    "action": "split_boundary",
                    "segments": split_segments,
                    "dropped_record": None,
                    "point_idx": None,
                    "score": float(sse),
                    "sse": float(sse),
                    "n_eval_boundaries": int(n_eval),
                    "dropped_points": 0,
                })

        for action, point_idx in (
            ("drop_left_point", boundary_idx),
            ("drop_right_point", boundary_idx + 1),
        ):
            if point_idx < 0 or point_idx >= n_seg:
                continue
            if n_seg - 1 < min_split_segment_len:
                continue

            drop_segments, dropped_record = drop_point_keep_segment(working, seg_idx, point_idx)
            if dropped_record is None or _segment_signature(drop_segments) == old_sig:
                continue

            sse, n_eval = compute_dQ_sse_and_count(theta, drop_segments, weighted=weighted)
            dropped_points = int(dropped_record.get("length", 1))
            candidates.append({
                "action": action,
                "segments": drop_segments,
                "dropped_record": dropped_record,
                "point_idx": int(point_idx),
                "score": float(sse + adaptive_drop_penalty * dropped_points),
                "sse": float(sse),
                "n_eval_boundaries": int(n_eval),
                "dropped_points": dropped_points,
            })

        if not candidates:
            continue

        candidates.sort(key=lambda item: item["score"])
        best = candidates[0]
        if best["score"] >= current_sse - 1e-12:
            if verbose:
                print(
                    f"[split-select] boundary rank={rank}, seg={seg_idx}, idx={boundary_idx}, "
                    f"max_sqerr={max_sqerr:.6e}, no split/drop action improved dQ SSE",
                    flush=True,
                )
            return None

        if verbose:
            print(
                f"[split-select] boundary rank={rank}, seg={seg_idx}, idx={boundary_idx}, "
                f"max_sqerr={max_sqerr:.6e}, action={best['action']}, "
                f"score {current_sse:.6e}->{best['score']:.6e}",
                flush=True,
            )

        best.update({
            "segment_idx": int(seg_idx),
            "boundary_idx": int(boundary_idx),
            "max_sqerr": float(max_sqerr),
            "candidate_rank": int(rank),
            "score_before": float(current_sse),
            "sse_before": float(current_sse),
            "n_eval_boundaries_before": int(current_n),
            "drop_point_penalty_used": float(adaptive_drop_penalty),
        })
        return best

    return None


def choose_forced_drop_split_action(
    theta,
    segments,
    sqerrs_per_seg,
    *,
    weighted=False,
    verbose=True,
):
    working = [seg if isinstance(seg, SegmentData) else _make_segment(seg) for seg in segments]
    current_sse, current_n = compute_dQ_sse_and_count(theta, working, weighted=weighted)
    boundary_candidates = _finite_boundary_candidates_from_sqerrs(sqerrs_per_seg)
    if not boundary_candidates:
        return None

    old_sig = _segment_signature(working)

    for rank, (max_sqerr, seg_idx, boundary_idx) in enumerate(boundary_candidates, start=1):
        if seg_idx >= len(working):
            continue

        seg = working[seg_idx]
        n_seg = len(seg.raw)
        candidates = []

        for action, point_idx in (
            ("forced_drop_left_point_and_split", boundary_idx),
            ("forced_drop_right_point_and_split", boundary_idx + 1),
        ):
            if point_idx < 0 or point_idx >= n_seg:
                continue

            forced_segments, dropped_record = drop_point_and_split_segment(
                working,
                seg_idx,
                point_idx,
                reason=action,
            )
            if dropped_record is None or _segment_signature(forced_segments) == old_sig:
                continue

            sse, n_eval = compute_dQ_sse_and_count(theta, forced_segments, weighted=weighted)
            candidates.append({
                "action": action,
                "segments": forced_segments,
                "dropped_record": dropped_record,
                "point_idx": int(point_idx),
                "score": float(sse),
                "sse": float(sse),
                "n_eval_boundaries": int(n_eval),
                "dropped_points": int(dropped_record.get("length", 1)),
            })

        if not candidates:
            continue

        candidates.sort(key=lambda item: item["score"])
        best = candidates[0]
        if verbose:
            print(
                f"[split-select forced] boundary rank={rank}, seg={seg_idx}, idx={boundary_idx}, "
                f"max_sqerr={max_sqerr:.6e}, action={best['action']}, "
                f"score {current_sse:.6e}->{best['score']:.6e}",
                flush=True,
            )

        best.update({
            "segment_idx": int(seg_idx),
            "boundary_idx": int(boundary_idx),
            "max_sqerr": float(max_sqerr),
            "candidate_rank": int(rank),
            "score_before": float(current_sse),
            "sse_before": float(current_sse),
            "n_eval_boundaries_before": int(current_n),
            "drop_point_penalty_used": 0.0,
            "forced": True,
        })
        return best

    return None




def run_multistart_local_optimization(
    theta_center,
    segments,
    bounds=DEFAULT_THETA_BOUNDS,
    verbose=True,
    n_starts=8,
    workers=1,
    maxiter=120,
    ftol=1e-10,
    print_every=10,
    seed=DEFAULT_PROJECT_RANDOM_SEED,
    rel_scales=(0.12, 0.12, 0.10, 0.10),
    weighted=False,
    objective_fn=None,
    objective_label="RMSE_Q",
):
    rng = np.random.default_rng(seed)

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower

    theta_center = np.asarray(theta_center, dtype=float)

    candidates = [theta_center.copy()]

    for _ in range(max(0, n_starts - 1)):
        noise = rng.normal(loc=0.0, scale=np.asarray(rel_scales, dtype=float) * span)
        trial = theta_center + noise
        trial = np.clip(trial, lower, upper)
        candidates.append(trial)

    best_theta = None
    best_obj = np.inf

    if verbose:
        print(
            f"[multi-start] n_starts={len(candidates)}, workers={workers} around current theta",
            flush=True,
        )

    def _run_one_candidate(k, theta0):
        theta_try = run_local_optimization(
            theta0,
            segments,
            bounds=bounds,
            verbose=False,
            maxiter=maxiter,
            ftol=ftol,
            print_every=print_every,
            weighted=weighted,
            objective_fn=objective_fn,
            objective_label=objective_label,
        )
        if objective_fn is None:
            obj_try = dQ_rmse_objective(theta_try, segments, weighted=weighted)
        else:
            obj_try = objective_fn(theta_try)
        return {
            "k": int(k),
            "theta0": np.asarray(theta0, dtype=float),
            "theta_try": np.asarray(theta_try, dtype=float),
            "obj_try": float(obj_try),
        }

    if workers is None:
        workers = 1
    workers = max(1, int(workers))

    if workers == 1 or len(candidates) == 1:
        trial_results = []
        for k, theta0 in enumerate(candidates, start=1):
            if verbose:
                print(
                    f"[multi-start {k:2d}/{len(candidates)}] "
                    f"init=({theta0[0]:.6f}, {theta0[1]:.6f}, {theta0[2]:.4f}, {theta0[3]:.4f})",
                    flush=True
                )

            trial = _run_one_candidate(k, theta0)
            trial_results.append(trial)

            if verbose:
                print(
                    f"[multi-start {k:2d}/{len(candidates)} done] fun={trial['obj_try']:.8f}",
                    flush=True
                )
    else:
        if verbose:
            for k, theta0 in enumerate(candidates, start=1):
                print(
                    f"[multi-start {k:2d}/{len(candidates)} queued] "
                    f"init=({theta0[0]:.6f}, {theta0[1]:.6f}, {theta0[2]:.4f}, {theta0[3]:.4f})",
                    flush=True
                )

        trial_results = Parallel(n_jobs=min(workers, len(candidates)), prefer="threads")(
            delayed(_run_one_candidate)(k, theta0)
            for k, theta0 in enumerate(candidates, start=1)
        )

        trial_results.sort(key=lambda item: item["k"])
        if verbose:
            for trial in trial_results:
                print(
                    f"[multi-start {trial['k']:2d}/{len(candidates)} done] "
                    f"fun={trial['obj_try']:.8f}",
                    flush=True
                )

    for trial in trial_results:
        if trial["obj_try"] < best_obj:
            best_obj = trial["obj_try"]
            best_theta = trial["theta_try"].copy()

    if best_theta is None:
        best_theta = theta_center.copy()
        if objective_fn is None:
            best_obj = dQ_rmse_objective(best_theta, segments, weighted=weighted)
        else:
            best_obj = objective_fn(best_theta)

    if verbose:
        print(
            f"[multi-start best] {objective_label}={best_obj:.8f} | "
            f"x0={best_theta[0]:.6f}, y0={best_theta[1]:.6f}, "
            f"Q_NE={best_theta[2]:.4f}, Q_PE={best_theta[3]:.4f}",
            flush=True
        )

    return best_theta


def iterative_segmentation(
    QV_array,
    bounds=DEFAULT_THETA_BOUNDS,
    V_error_max_threshold=0.03,
    rmsev_stop_threshold=None,
    rmseq_segmentation_stop_threshold=0.5,
    max_iters=25,
    max_fit_loops=10,
    search_range=DEFAULT_OFFSET_SEARCH_RANGE,
    verbose=True,
    de_workers=1,
    de_maxiter=100,
    de_popsize=8,
    de_refit_period=10,
    local_maxiter_init=200,
    local_maxiter_iter=120,
    min_seg_len=3,
    split_selection_mode="evaluate_split_or_drop",
    min_split_segment_len=None,
    drop_point_penalty=None,
    force_drop_split_on_no_candidate=True,
    segmentation=True,
    multistart_n=8,
    multistart_workers=1,
    multistart_seed=None,
    random_seed=DEFAULT_PROJECT_RANDOM_SEED,
    multistart_rel_scales=(0.12, 0.12, 0.10, 0.10),
    multistart_ftol=1e-10,
    fit_patience=4,
    rmseq_tol_abs=1e-5,
    rmseq_tol_rel=1e-4,
    post_merge_offset_gap_threshold=0.3,
    weighted=False,
    warm_start_theta=None,
    skip_global_init=False,
    local_bounds_half_widths=None,
    local_bounds_rel_scales=None,
    objective_mode="rmseq",
    objective_rmsev_weight=0.0,
    objective_prior_weight=0.0,
    objective_prior_theta=None,
    objective_prior_scales=None,
    objective_stop_threshold=None,
):
    segments = _as_segment_list(QV_array)

    if len(segments) == 0:
        raise ValueError("No usable segments.")

    total_input_len = sum(len(seg.raw) for seg in segments)
    dropped_history = []

    cuts_history = []
    rmseV_history = []
    rmseQ_history = []
    V_error_max_history = []
    objective_history = []

    summary_history = []
    cumulative_dropped = 0

    optimization_bounds = normalize_theta_bounds(bounds)
    if (
        (local_bounds_half_widths is not None or local_bounds_rel_scales is not None)
        and warm_start_theta is None
    ):
        raise ValueError("warm_start_theta is required when using local bounds.")

    warm_start_theta_arr = None
    if warm_start_theta is not None:
        warm_start_theta_arr = clip_theta_to_bounds(warm_start_theta, optimization_bounds)
        if local_bounds_half_widths is not None or local_bounds_rel_scales is not None:
            optimization_bounds = build_local_theta_bounds(
                optimization_bounds,
                center_theta=warm_start_theta_arr,
                half_widths=local_bounds_half_widths,
                rel_scales=local_bounds_rel_scales,
            )

    effective_prior_theta = objective_prior_theta
    if effective_prior_theta is None and warm_start_theta_arr is not None and objective_prior_weight > 0:
        effective_prior_theta = warm_start_theta_arr

    def build_current_objective(current_segments):
        return build_theta_objective(
            current_segments,
            objective_mode=objective_mode,
            weighted=weighted,
            search_range=search_range,
            rmsev_weight=objective_rmsev_weight,
            prior_theta=effective_prior_theta,
            prior_weight=objective_prior_weight,
            prior_scales=objective_prior_scales,
            bounds=optimization_bounds,
        )

    objective_fn, objective_meta = build_current_objective(segments)
    objective_label = objective_meta["label"]
    split_selection_mode_normalized = str(split_selection_mode).strip().lower()
    effective_min_split_segment_len = (
        int(min_seg_len) if min_split_segment_len is None else int(min_split_segment_len)
    )
    if effective_min_split_segment_len < 2:
        raise ValueError("min_split_segment_len must be at least 2 when provided.")

    rmsev_stop_active = rmsev_stop_threshold is not None
    if rmsev_stop_active:
        rmsev_stop_threshold = float(rmsev_stop_threshold)
        if not np.isfinite(rmsev_stop_threshold) or rmsev_stop_threshold <= 0:
            raise ValueError("rmsev_stop_threshold must be a positive finite value when provided.")

    def segmentation_stop_reached(current_rmseV, current_V_error_max):
        if not segmentation:
            return False
        if rmsev_stop_active:
            return bool(np.isfinite(current_rmseV) and current_rmseV <= rmsev_stop_threshold)
        return bool(np.isfinite(current_V_error_max) and current_V_error_max <= V_error_max_threshold)

    termination_reason = None
    base_random_seed = None if random_seed is None else int(random_seed)
    if base_random_seed is not None and multistart_seed is None:
        multistart_seed = derive_seed(base_random_seed, "multistart")

    def _stage_seed(*parts):
        if base_random_seed is None:
            return None
        return derive_seed(base_random_seed, *parts)

    if verbose:
        print(f"[start] n_segments={len(segments)}", flush=True)
        print(f"[start] segment lengths = {[len(seg.raw) for seg in segments]}", flush=True)
        print(f"[start] segmentation = {segmentation}", flush=True)
        if segmentation:
            print(f"[start] split_selection_mode = {split_selection_mode_normalized}", flush=True)
            print(f"[start] min_split_segment_len = {effective_min_split_segment_len}", flush=True)
            print(f"[start] force_drop_split_on_no_candidate = {bool(force_drop_split_on_no_candidate)}", flush=True)
        print(f"[start] DE refit period = {de_refit_period}", flush=True)
        print(f"[start] objective = {objective_meta['mode']}", flush=True)
        if base_random_seed is not None:
            print(f"[start] random_seed = {base_random_seed}", flush=True)
        if warm_start_theta_arr is not None:
            print(
                "[start] warm-start theta = "
                f"({warm_start_theta_arr[0]:.6f}, {warm_start_theta_arr[1]:.6f}, "
                f"{warm_start_theta_arr[2]:.4f}, {warm_start_theta_arr[3]:.4f})",
                flush=True,
            )
        if optimization_bounds != normalize_theta_bounds(bounds):
            print(f"[start] local bounds = {optimization_bounds}", flush=True)
        if segmentation:
            if rmsev_stop_active:
                print(
                    "[start] termination: split/cleanup exhaustion or max_iters "
                    f"(stop when RMSE_V <= {rmsev_stop_threshold:.6f}, "
                    f"switch to point-drop mode when RMSE_Q <= {rmseq_segmentation_stop_threshold:.6f})",
                    flush=True,
                )
            else:
                print(
                    "[start] termination: split/cleanup exhaustion or max_iters "
                    f"(stop when max|V_err| <= {V_error_max_threshold:.6f}, "
                    f"switch to point-drop mode when RMSE_Q <= {rmseq_segmentation_stop_threshold:.6f})",
                    flush=True,
                )
        else:
            if objective_mode == "combined" and objective_stop_threshold is not None:
                print(
                    "[start] termination: combined-objective threshold or max_fit_loops "
                    f"(stop when objective <= {float(objective_stop_threshold):.6f})",
                    flush=True,
                )
            else:
                print("[start] termination: RMSE_Q-based improvement criterion", flush=True)

    # -------------------------------------------------
    # initial optimization: global + local
    # -------------------------------------------------
    if skip_global_init:
        if warm_start_theta_arr is None:
            raise ValueError("skip_global_init=True requires warm_start_theta.")
        theta0 = clip_theta_to_bounds(warm_start_theta_arr, optimization_bounds)
        if verbose:
            print(
                "[start] skip_global_init=True -> start from warm-start local optimization",
                flush=True,
            )
    else:
        theta0 = run_global_optimization(
            segments,
            bounds=optimization_bounds,
            verbose=verbose,
            de_workers=de_workers,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-4,
            seed=_stage_seed("global_init"),
            print_every=10,
            weighted=weighted,
            objective_fn=objective_fn,
            objective_label=objective_label,
        )

    theta = run_local_optimization(
        theta0,
        segments,
        bounds=optimization_bounds,
        verbose=verbose,
        maxiter=local_maxiter_init,
        ftol=1e-8,
        print_every=10,
        weighted=weighted,
        objective_fn=objective_fn,
        objective_label=objective_label,
    )

    rmseQ, sqerrs_per_seg, rmseV, V_error_max, offsets = _evaluate_segment_metrics(
        theta,
        segments,
        search_range,
        weighted=weighted,
    )

    rmseV_history.append(rmseV)
    rmseQ_history.append(rmseQ)
    V_error_max_history.append(V_error_max)
    objective_value = float(objective_fn(theta))
    objective_history.append(objective_value)

    summary_history.append({
        "iter": 0,
        "stage": "init",
        "objective_value": objective_value,
        "rmseQ": rmseQ,
        "rmseV": rmseV,
        "V_error_max": V_error_max,
        "n_segments": len(segments),
        "dropped_this_iter": 0,
        "cumulative_dropped": 0,
        "remaining_points": int(sum(len(seg.raw) for seg in segments)),
        "best_so_far": True,
        "rmseQ_improvement": np.nan,
        "objective_improvement": np.nan,
        "no_improve_count": 0,
    })

    if verbose:
        print(
            f"[init] segs={len(segments)}, {objective_label}={objective_value:.6f}, "
            f"RMSE_Q={rmseQ:.6f}, RMSE_V={rmseV:.6f}, "
            f"max|V_err|={V_error_max:.6f}",
            flush=True
        )

    if segmentation and rmsev_stop_active and rmseV <= rmsev_stop_threshold:
        if verbose:
            print(
                f"[stop-init] RMSE_V={rmseV:.6f} <= {rmsev_stop_threshold:.6f}",
                flush=True,
            )
        termination_reason = "init_rmsev_stop_threshold"
        max_iters = 0
    elif segmentation and (not rmsev_stop_active) and V_error_max <= V_error_max_threshold:
        if verbose:
            print(
                f"[stop-init] max|V_err|={V_error_max:.6f} <= {V_error_max_threshold:.6f}",
                flush=True,
            )
        termination_reason = "init_v_error_max_threshold"
        max_iters = 0

    # -------------------------------------------------
    # segmentation OFF: RMSE_Q-driven multi-start fitting
    # -------------------------------------------------
    if not segmentation:
        if verbose:
            if objective_meta["mode"] == "combined":
                print("[mode] segmentation OFF -> combined-objective multi-start local fitting", flush=True)
            else:
                print("[mode] segmentation OFF -> RMSE_Q-driven multi-start local fitting", flush=True)

        best_theta = theta.copy()
        best_rmseQ = float(rmseQ)
        best_rmseV = float(rmseV)
        best_V_error_max = float(V_error_max)
        best_offsets = list(offsets)
        best_sqerrs_per_seg = sqerrs_per_seg
        best_objective = float(objective_value)
        threshold_active = bool(
            objective_meta["mode"] == "combined"
            and objective_stop_threshold is not None
            and np.isfinite(objective_stop_threshold)
        )
        threshold_reached = bool(
            threshold_active and best_objective <= float(objective_stop_threshold)
        )

        no_improve_count = 0

        if threshold_reached and verbose:
            print(
                f"[stop-fit-init] best {objective_label}={best_objective:.6f} "
                f"<= {float(objective_stop_threshold):.6f}",
                flush=True,
            )
        else:
            for fit_loop in range(1, max_fit_loops + 1):
                theta_candidate = run_multistart_local_optimization(
                    theta_center=best_theta,
                    segments=segments,
                    bounds=optimization_bounds,
                    verbose=verbose,
                    n_starts=multistart_n,
                    workers=multistart_workers,
                    maxiter=local_maxiter_iter,
                    ftol=multistart_ftol,
                    print_every=10,
                    seed=None if multistart_seed is None else derive_seed(multistart_seed, "fit_loop", fit_loop),
                    rel_scales=multistart_rel_scales,
                    weighted=weighted,
                    objective_fn=objective_fn,
                    objective_label=objective_label,
                )

                rmseQ_try, sqerrs_per_seg_try = compute_dQ_rmse(
                    theta_candidate,
                    segments,
                    return_sqerrs=True,
                    weighted=weighted,
                )
                rmseV_try, offsets_try = compute_V_RMSE_and_offsets(
                    theta_candidate, segments, search_range
                )
                V_error_max_try, offsets_try = compute_V_error_max_and_offsets(
                    theta_candidate, segments, search_range
                )
                objective_try = float(objective_fn(theta_candidate))

                rmseV_history.append(rmseV_try)
                rmseQ_history.append(rmseQ_try)
                V_error_max_history.append(V_error_max_try)
                objective_history.append(objective_try)

                improve_thresh = max(rmseq_tol_abs, rmseq_tol_rel * max(1.0, abs(best_rmseQ)))
                rmseQ_improvement = best_rmseQ - rmseQ_try
                objective_improvement = best_objective - objective_try

                improved = False
                if objective_meta["mode"] == "combined":
                    improve_thresh = max(rmseq_tol_abs, rmseq_tol_rel * max(1.0, abs(best_objective)))
                    if objective_try < best_objective - improve_thresh:
                        improved = True
                    elif abs(objective_try - best_objective) <= improve_thresh and rmseV_try < best_rmseV:
                        improved = True
                else:
                    if rmseQ_try < best_rmseQ - improve_thresh:
                        improved = True
                    elif abs(rmseQ_try - best_rmseQ) <= improve_thresh and rmseV_try < best_rmseV:
                        improved = True

                if improved:
                    best_theta = theta_candidate.copy()
                    best_rmseQ = float(rmseQ_try)
                    best_rmseV = float(rmseV_try)
                    best_V_error_max = float(V_error_max_try)
                    best_offsets = list(offsets_try)
                    best_sqerrs_per_seg = sqerrs_per_seg_try
                    best_objective = float(objective_try)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                threshold_reached = bool(
                    threshold_active and best_objective <= float(objective_stop_threshold)
                )

                summary_history.append({
                    "iter": fit_loop,
                    "stage": "fit_loop",
                    "objective_value": objective_try,
                    "rmseQ": rmseQ_try,
                    "rmseV": rmseV_try,
                    "V_error_max": V_error_max_try,
                    "n_segments": len(segments),
                    "dropped_this_iter": 0,
                    "cumulative_dropped": cumulative_dropped,
                    "remaining_points": int(sum(len(seg.raw) for seg in segments)),
                    "best_so_far": improved,
                    "rmseQ_improvement": rmseQ_improvement,
                    "objective_improvement": objective_improvement,
                    "no_improve_count": no_improve_count,
                })

                if verbose:
                    threshold_suffix = (
                        f", threshold={float(objective_stop_threshold):.6f}, reached={threshold_reached}"
                        if threshold_active else ""
                    )
                    print(
                        f"[fit loop {fit_loop}] "
                        f"trial {objective_label}={objective_try:.6f}, "
                        f"trial RMSE_Q={rmseQ_try:.6f}, trial RMSE_V={rmseV_try:.6f}, "
                        f"trial max|V_err|={V_error_max_try:.6f} | "
                        f"best {objective_label}={best_objective:.6f}, "
                        f"best RMSE_Q={best_rmseQ:.6f}, best RMSE_V={best_rmseV:.6f}, "
                        f"improved={improved}, no_improve_count={no_improve_count}"
                        f"{threshold_suffix}",
                        flush=True
                    )

                if threshold_reached:
                    if verbose:
                        print(
                            f"[stop-fit] best {objective_label}={best_objective:.6f} "
                            f"<= {float(objective_stop_threshold):.6f}",
                            flush=True
                        )
                    break

                if (not threshold_active) and fit_patience is not None and no_improve_count >= fit_patience:
                    if verbose:
                        print(
                            f"[stop-fit] no RMSE_Q improvement for {fit_patience} consecutive loops",
                            flush=True
                        )
                    break

        theta = best_theta
        rmseQ = best_rmseQ
        rmseV = best_rmseV
        V_error_max = best_V_error_max
        offsets = best_offsets
        sqerrs_per_seg = best_sqerrs_per_seg

        if verbose:
            print(
                f"[final-best] {objective_label}={best_objective:.6f}, "
                f"RMSE_Q={rmseQ:.6f}, RMSE_V={rmseV:.6f}, "
                f"max|V_err|={V_error_max:.6f}",
                flush=True
            )

        kept_indices = np.sort(np.concatenate([seg.idx for seg in segments])).astype(int)
        dropped_indices = np.setdiff1d(np.arange(total_input_len, dtype=int), kept_indices)

        return {
            "theta": theta,
            "segments": [seg.raw for seg in segments],
            "segment_indices": [seg.idx.copy() for seg in segments],
            "rmseV_history": rmseV_history,
            "rmseQ_history": rmseQ_history,
            "objective_history": objective_history,
            "V_error_max_history": V_error_max_history,
            "cuts_history": cuts_history,
            "offsets_final": offsets,
            "dropped_history": dropped_history,
            "dropped_indices": dropped_indices,
            "kept_indices": kept_indices,
            "summary_history": summary_history,
            "rmseQ_weighted": bool(weighted),
            "objective_mode": objective_meta["mode"],
            "objective_config": objective_meta,
            "objective_stop_threshold": (
                None if objective_stop_threshold is None else float(objective_stop_threshold)
            ),
            "objective_stop_reached": bool(threshold_reached),
            "random_seed": base_random_seed,
            "optimization_bounds": optimization_bounds,
            "warm_start_theta": None if warm_start_theta_arr is None else warm_start_theta_arr.copy(),
            "V_error_max_threshold": float(V_error_max_threshold),
            "rmsev_stop_threshold": None if not rmsev_stop_active else float(rmsev_stop_threshold),
            "segmentation_stop_metric": "RMSE_V" if rmsev_stop_active else "V_error_max",
            "split_selection_mode": split_selection_mode_normalized,
            "min_split_segment_len": int(effective_min_split_segment_len),
            "drop_point_penalty": None if drop_point_penalty is None else float(drop_point_penalty),
            "force_drop_split_on_no_candidate": bool(force_drop_split_on_no_candidate),
            "termination_reason": "segmentation_false",
        }

    # -------------------------------------------------
    # -------------------------------------------------
    last_completed_iter = 0
    last_used_de = True
    point_drop_mode = bool(segmentation and rmseQ <= rmseq_segmentation_stop_threshold)

    for it in range(1, max_iters + 1):
        prev_V_error_max = float(V_error_max)
        old_sig = _segment_signature(segments)
        n_drop_before = len(dropped_history)
        action_record = {
            "action": None,
            "action_segment_idx": np.nan,
            "action_boundary_idx": np.nan,
            "action_point_idx": np.nan,
            "action_max_sqerr": np.nan,
            "action_score_before": np.nan,
            "action_score_after": np.nan,
            "action_sse_after": np.nan,
            "action_dropped_points": 0,
            "action_drop_point_penalty_used": np.nan,
            "action_candidate_rank": np.nan,
            "action_forced": False,
        }

        if point_drop_mode:
            worst_seg_idx, worst_idx, worst_err, _ = find_max_v_error_point(
                theta, segments, search_range=search_range
            )

            if worst_seg_idx is None:
                if verbose:
                    print(f"[iter {it}] no max-V-error point candidate -> stop", flush=True)
                termination_reason = "no_max_v_error_point_candidate"
                break

            if verbose:
                print(
                    f"[iter {it}] point-drop(seg={worst_seg_idx}, idx={worst_idx}, "
                    f"max|V_err|={worst_err:.6f})",
                    flush=True
                )

            cuts_history.append((worst_seg_idx, worst_idx))
            segments, dropped_record = drop_point_keep_segment(segments, worst_seg_idx, worst_idx)
            if dropped_record is not None:
                dropped_record["iter"] = it
                dropped_history.append(dropped_record)
                action_record.update({
                    "action": "point_drop_mode",
                    "action_segment_idx": int(worst_seg_idx),
                    "action_point_idx": int(worst_idx),
                    "action_max_sqerr": np.nan,
                    "action_dropped_points": int(dropped_record.get("length", 1)),
                    "action_forced": False,
                })

            segments, dropped_history = cleanup_segments(
                segments,
                min_seg_len=min_seg_len,
                verbose=verbose,
                iter_no=it,
                dropped_history=dropped_history,
            )
        else:
            if split_selection_mode_normalized in {"evaluate_split_or_drop", "split_or_drop"}:
                selected_action = choose_split_or_drop_action(
                    theta,
                    segments,
                    sqerrs_per_seg,
                    weighted=weighted,
                    min_split_segment_len=effective_min_split_segment_len,
                    drop_point_penalty=drop_point_penalty,
                    verbose=verbose,
                )

                if selected_action is None:
                    if segmentation_stop_reached(rmseV, V_error_max):
                        if verbose:
                            print(
                                f"[iter {it}] no beneficial split/drop candidate, "
                                "but segmentation stop condition is already satisfied -> stop",
                                flush=True,
                            )
                        termination_reason = (
                            "rmsev_stop_threshold"
                            if rmsev_stop_active
                            else "v_error_max_threshold"
                        )
                        break

                    if force_drop_split_on_no_candidate:
                        selected_action = choose_forced_drop_split_action(
                            theta,
                            segments,
                            sqerrs_per_seg,
                            weighted=weighted,
                            verbose=verbose,
                        )

                    if selected_action is None:
                        if verbose:
                            print(
                                f"[iter {it}] no beneficial split/drop candidate "
                                "and no forced drop+split candidate -> stop",
                                flush=True,
                            )
                        termination_reason = "no_split_drop_or_forced_candidate"
                        break

                segments = selected_action["segments"]
                if selected_action["action"] == "split_boundary":
                    cuts_history.append((selected_action["segment_idx"], selected_action["boundary_idx"]))
                else:
                    if selected_action["action"].startswith("forced_"):
                        cuts_history.append((selected_action["segment_idx"], selected_action["boundary_idx"]))
                    dropped_record = selected_action["dropped_record"]
                    if dropped_record is not None:
                        dropped_record["iter"] = it
                        dropped_record["reason"] = selected_action["action"]
                        dropped_history.append(dropped_record)

                action_record.update({
                    "action": selected_action["action"],
                    "action_segment_idx": int(selected_action["segment_idx"]),
                    "action_boundary_idx": int(selected_action["boundary_idx"]),
                    "action_point_idx": (
                        np.nan if selected_action["point_idx"] is None else int(selected_action["point_idx"])
                    ),
                    "action_max_sqerr": float(selected_action["max_sqerr"]),
                    "action_score_before": float(selected_action["score_before"]),
                    "action_score_after": float(selected_action["score"]),
                    "action_sse_after": float(selected_action["sse"]),
                    "action_dropped_points": int(selected_action["dropped_points"]),
                    "action_drop_point_penalty_used": float(selected_action["drop_point_penalty_used"]),
                    "action_candidate_rank": int(selected_action.get("candidate_rank", 1)),
                    "action_forced": bool(selected_action.get("forced", False)),
                })
            elif split_selection_mode_normalized in {"legacy", "split_only"}:
                worst_seg_idx, worst_idx, worst_err = find_best_effective_split(
                    segments, sqerrs_per_seg, verbose=verbose
                )

                if worst_seg_idx is None:
                    if verbose:
                        print(f"[iter {it}] no effective split candidate -> stop", flush=True)
                    termination_reason = "no_effective_legacy_split_candidate"
                    break

                if verbose:
                    print(
                        f"[iter {it}] split(seg={worst_seg_idx}, idx={worst_idx}, max_sqerr={worst_err:.6e})",
                        flush=True
                    )

                cuts_history.append((worst_seg_idx, worst_idx))
                segments = apply_split_force(segments, worst_seg_idx, worst_idx)
                action_record.update({
                    "action": "legacy_split_boundary",
                    "action_segment_idx": int(worst_seg_idx),
                    "action_boundary_idx": int(worst_idx),
                    "action_max_sqerr": float(worst_err),
                    "action_forced": False,
                })
            else:
                raise ValueError(
                    "split_selection_mode must be 'evaluate_split_or_drop' or 'legacy'/'split_only'."
                )

            segments, dropped_history = cleanup_segments(
                segments,
                min_seg_len=min_seg_len,
                verbose=verbose,
                iter_no=it,
                dropped_history=dropped_history,
            )

        new_drop_records = dropped_history[n_drop_before:]
        dropped_this_iter = int(sum(rec["length"] for rec in new_drop_records))
        cumulative_dropped += dropped_this_iter
        remaining_points = int(sum(len(seg.raw) for seg in segments)) if len(segments) > 0 else 0

        if len(segments) == 0:
            if verbose:
                print(f"[iter {it}] all segments removed after cleanup -> stop", flush=True)
            termination_reason = "cleanup_removed_all_segments"

            summary_history.append({
                "iter": it,
                "stage": "cleanup_stop_all_removed",
                "objective_value": np.nan,
                "rmseQ": np.nan,
                "rmseV": np.nan,
                "V_error_max": np.nan,
                "n_segments": 0,
                "dropped_this_iter": dropped_this_iter,
                "cumulative_dropped": cumulative_dropped,
                "remaining_points": 0,
                "best_so_far": False,
                "rmseQ_improvement": np.nan,
                "objective_improvement": np.nan,
                "no_improve_count": np.nan,
                **action_record,
            })
            break

        new_sig = _segment_signature(segments)
        if new_sig == old_sig:
            if verbose:
                print(
                    f"[iter {it}] split+cleanup produced no effective change -> stop",
                    flush=True,
                )
            termination_reason = "cleanup_no_effective_change"

            summary_history.append({
                "iter": it,
                "stage": "cleanup_stop_no_change",
                "objective_value": np.nan,
                "rmseQ": np.nan,
                "rmseV": np.nan,
                "V_error_max": np.nan,
                "n_segments": len(segments),
                "dropped_this_iter": dropped_this_iter,
                "cumulative_dropped": cumulative_dropped,
                "remaining_points": remaining_points,
                "best_so_far": False,
                "rmseQ_improvement": np.nan,
                "objective_improvement": np.nan,
                "no_improve_count": np.nan,
                **action_record,
            })
            break

        if verbose:
            print(
                f"[iter {it}] segment lengths = {[len(seg.raw) for seg in segments]}",
                flush=True,
            )

        objective_fn, objective_meta = build_current_objective(segments)
        objective_label = objective_meta["label"]

        use_de_this_iter = (
            it == 1
            or (de_refit_period is not None and de_refit_period > 0 and it % de_refit_period == 0)
        )

        if verbose:
            print(
                f"[iter {it}] refit mode = {'DE+LBFGSB' if use_de_this_iter else 'LBFGSB only'}",
                flush=True,
            )

        if use_de_this_iter:
            theta0_iter = run_global_optimization(
                segments,
                bounds=optimization_bounds,
                verbose=verbose,
                de_workers=de_workers,
                maxiter=de_maxiter,
                popsize=de_popsize,
                tol=1e-4,
                seed=_stage_seed("segmentation_iter", it),
                print_every=10,
                weighted=weighted,
                objective_fn=objective_fn,
                objective_label=objective_label,
            )

            theta = run_local_optimization(
                theta0_iter,
                segments,
                bounds=optimization_bounds,
                verbose=verbose,
                maxiter=local_maxiter_iter,
                ftol=1e-8,
                print_every=10,
                weighted=weighted,
                objective_fn=objective_fn,
                objective_label=objective_label,
            )
        else:
            theta = run_local_optimization(
                theta,
                segments,
                bounds=optimization_bounds,
                verbose=verbose,
                maxiter=local_maxiter_iter,
                ftol=1e-8,
                print_every=10,
                weighted=weighted,
                objective_fn=objective_fn,
                objective_label=objective_label,
            )

        rmseQ, sqerrs_per_seg, rmseV, V_error_max, offsets = _evaluate_segment_metrics(
            theta,
            segments,
            search_range,
            weighted=weighted,
        )

        rmseV_history.append(rmseV)
        rmseQ_history.append(rmseQ)
        V_error_max_history.append(V_error_max)
        objective_value = float(objective_fn(theta))
        objective_history.append(objective_value)

        v_error_improvement = prev_V_error_max - float(V_error_max)

        summary_history.append({
            "iter": it,
            "stage": "point_drop_iter" if point_drop_mode else "iter",
            "objective_value": objective_value,
            "rmseQ": rmseQ,
            "rmseV": rmseV,
            "V_error_max": V_error_max,
            "n_segments": len(segments),
            "dropped_this_iter": dropped_this_iter,
            "cumulative_dropped": cumulative_dropped,
            "remaining_points": remaining_points,
            "best_so_far": False,
            "rmseQ_improvement": np.nan,
            "objective_improvement": np.nan,
            "no_improve_count": np.nan,
            "V_error_max_improvement": v_error_improvement,
            **action_record,
        })

        if verbose:
            print(
                f"[iter {it}] {objective_label}={objective_value:.6f}, "
                f"RMSE_Q={rmseQ:.6f}, RMSE_V={rmseV:.6f}, "
                f"max|V_err|={V_error_max:.6f}, "
                f"remaining_points={remaining_points}",
                flush=True
            )

        last_completed_iter = it
        last_used_de = use_de_this_iter

        if segmentation and rmsev_stop_active and rmseV <= rmsev_stop_threshold:
            if verbose:
                print(
                    f"[stop] RMSE_V={rmseV:.6f} <= {rmsev_stop_threshold:.6f}",
                    flush=True,
                )
            termination_reason = "rmsev_stop_threshold"
            break
        if segmentation and (not rmsev_stop_active) and V_error_max <= V_error_max_threshold:
            if verbose:
                print(
                    f"[stop] max|V_err|={V_error_max:.6f} <= {V_error_max_threshold:.6f}",
                    flush=True,
                )
            termination_reason = "v_error_max_threshold"
            break

        if segmentation and (not point_drop_mode) and rmseQ <= rmseq_segmentation_stop_threshold:
            point_drop_mode = True
            if verbose:
                print(
                    f"[iter {it}] switch to point-drop mode "
                    f"(RMSE_Q={rmseQ:.6f} <= {rmseq_segmentation_stop_threshold:.6f})",
                    flush=True,
                )

    if termination_reason is None and max_iters > 0 and last_completed_iter >= max_iters:
        termination_reason = "max_iters"

    if segmentation and len(segments) > 0 and last_completed_iter > 0 and not last_used_de:
        if verbose:
            print("[final refit] run DE+LBFGSB on final segments", flush=True)

        theta0_iter = run_global_optimization(
            segments,
            bounds=optimization_bounds,
            verbose=verbose,
            de_workers=de_workers,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-4,
            seed=_stage_seed("final_refit"),
            print_every=10,
            weighted=weighted,
            objective_fn=objective_fn,
            objective_label=objective_label,
        )

        theta = run_local_optimization(
            theta0_iter,
            segments,
            bounds=optimization_bounds,
            verbose=verbose,
            maxiter=local_maxiter_iter,
            ftol=1e-8,
            print_every=10,
            weighted=weighted,
            objective_fn=objective_fn,
            objective_label=objective_label,
        )

        rmseQ, sqerrs_per_seg, rmseV, V_error_max, offsets = _evaluate_segment_metrics(
            theta,
            segments,
            search_range,
            weighted=weighted,
        )

        rmseV_history.append(rmseV)
        rmseQ_history.append(rmseQ)
        V_error_max_history.append(V_error_max)
        objective_value = float(objective_fn(theta))
        objective_history.append(objective_value)

        summary_history.append({
            "iter": last_completed_iter,
            "stage": "final_refit",
            "objective_value": objective_value,
            "rmseQ": rmseQ,
            "rmseV": rmseV,
            "V_error_max": V_error_max,
            "n_segments": len(segments),
            "dropped_this_iter": 0,
            "cumulative_dropped": cumulative_dropped,
            "remaining_points": int(sum(len(seg.raw) for seg in segments)),
            "best_so_far": False,
            "rmseQ_improvement": np.nan,
            "objective_improvement": np.nan,
            "no_improve_count": np.nan,
        })

    if segmentation and len(segments) > 0:
        merged_segments, _, merged_count = merge_adjacent_segments_by_offset(
            segments,
            offsets,
            offset_gap_threshold=post_merge_offset_gap_threshold,
        )
        if merged_count > 0:
            if verbose:
                print(
                    f"[post-merge] merged {merged_count} adjacent segment boundary(ies) "
                    f"with |offset diff| <= {post_merge_offset_gap_threshold:.3f}",
                    flush=True,
                )
            segments = merged_segments
        else:
            if verbose:
                print("[post-merge] no adjacent segments met the offset-gap criterion", flush=True)

        objective_fn, objective_meta = build_current_objective(segments)
        objective_label = objective_meta["label"]
        rmseQ, sqerrs_per_seg, rmseV, V_error_max, offsets = _evaluate_segment_metrics(
            theta,
            segments,
            search_range,
            weighted=weighted,
        )
        rmseV_history.append(rmseV)
        rmseQ_history.append(rmseQ)
        V_error_max_history.append(V_error_max)
        objective_value = float(objective_fn(theta))
        objective_history.append(objective_value)
        summary_history.append({
            "iter": last_completed_iter + 1 if last_completed_iter > 0 else 1,
            "stage": "post_merge_eval",
            "objective_value": objective_value,
            "rmseQ": rmseQ,
            "rmseV": rmseV,
            "V_error_max": V_error_max,
            "n_segments": len(segments),
            "dropped_this_iter": 0,
            "cumulative_dropped": cumulative_dropped,
            "remaining_points": int(sum(len(seg.raw) for seg in segments)),
            "best_so_far": False,
            "rmseQ_improvement": np.nan,
            "objective_improvement": np.nan,
            "no_improve_count": np.nan,
            "merged_count": merged_count,
        })
        if verbose:
            print(
                f"[post-merge eval] {objective_label}={objective_value:.6f}, "
                f"RMSE_Q={rmseQ:.6f}, RMSE_V={rmseV:.6f}, "
                f"max|V_err|={V_error_max:.6f}, n_segments={len(segments)}",
                flush=True,
            )

    kept_indices = (
        np.sort(np.concatenate([seg.idx for seg in segments])).astype(int)
        if len(segments) > 0
        else np.array([], dtype=int)
    )
    dropped_indices = np.setdiff1d(np.arange(total_input_len, dtype=int), kept_indices)

    return {
        "theta": theta,
        "segments": [seg.raw for seg in segments],
        "segment_indices": [seg.idx.copy() for seg in segments],
        "rmseV_history": rmseV_history,
        "rmseQ_history": rmseQ_history,
        "objective_history": objective_history,
        "V_error_max_history": V_error_max_history,
        "cuts_history": cuts_history,
        "offsets_final": offsets,
        "dropped_history": dropped_history,
        "dropped_indices": dropped_indices,
        "kept_indices": kept_indices,
        "summary_history": summary_history,
        "rmseQ_weighted": bool(weighted),
        "objective_mode": objective_meta["mode"],
        "objective_config": objective_meta,
        "random_seed": base_random_seed,
        "optimization_bounds": optimization_bounds,
        "warm_start_theta": None if warm_start_theta_arr is None else warm_start_theta_arr.copy(),
        "V_error_max_threshold": float(V_error_max_threshold),
        "rmsev_stop_threshold": None if not rmsev_stop_active else float(rmsev_stop_threshold),
        "segmentation_stop_metric": "RMSE_V" if rmsev_stop_active else "V_error_max",
        "split_selection_mode": split_selection_mode_normalized,
        "min_split_segment_len": int(effective_min_split_segment_len),
        "drop_point_penalty": None if drop_point_penalty is None else float(drop_point_penalty),
        "force_drop_split_on_no_candidate": bool(force_drop_split_on_no_candidate),
        "termination_reason": termination_reason,
    }







def infer_health_mode(theta):
    x0_aged = theta[0]
    y0_aged = theta[1]
    Q_NE_aged = theta[2]
    Q_PE_aged = theta[3]

    f_inv_aged = get_OCV_inv_extrapolated(x0_aged, y0_aged, Q_NE_aged, Q_PE_aged)

    aged_dict = {
        "Q_cell": f_inv_aged(4.2) - f_inv_aged(2.5),
        "Q_Li": x0_aged * Q_NE_aged + y0_aged * Q_PE_aged,
        "Q_PE": np.round(Q_PE_aged, 2),
        "Q_NE": np.round(Q_NE_aged, 2),
    }
    return pd.DataFrame([aged_dict])


