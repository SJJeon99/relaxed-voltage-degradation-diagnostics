from __future__ import annotations

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

try:
    from .analysis_core import get_OCV_extrapolated, get_OCV_inv_extrapolated
    from .fit_presets import materialize_rest_uncertainty_iterative_kwargs
    from .reproducibility import DEFAULT_PROJECT_RANDOM_SEED, derive_seed
    from .rest_OCV_segmentation import compute_dQ_rmse, infer_health_mode, iterative_segmentation
except ImportError:
    from src.analysis_core import get_OCV_extrapolated, get_OCV_inv_extrapolated
    from src.fit_presets import materialize_rest_uncertainty_iterative_kwargs
    from src.reproducibility import DEFAULT_PROJECT_RANDOM_SEED, derive_seed
    from src.rest_OCV_segmentation import compute_dQ_rmse, infer_health_mode, iterative_segmentation




def unwrap_result_item(item):
    if isinstance(item, dict) and "result_iter" in item:
        return item["result_iter"]
    return item


def get_DMs_from_result(result_dict):
    dm_df = infer_health_mode(result_dict["theta"])
    return {
        "Q_cell": float(dm_df["Q_cell"].iloc[0]),
        "Q_Li": float(dm_df["Q_Li"].iloc[0]),
        "Q_PE": float(dm_df["Q_PE"].iloc[0]),
        "Q_NE": float(dm_df["Q_NE"].iloc[0]),
    }




def collect_voltage_errors(result_dict):
    segments = result_dict["segments"]
    offsets = result_dict["offsets_final"]
    theta = result_dict["theta"]
    f_curve = get_OCV_extrapolated(*theta)

    error_list = []
    for seg, off in zip(segments, offsets):
        if seg is None or off is None or len(seg) == 0:
            continue
        error_list.append(np.asarray(seg[:, 1] - f_curve(seg[:, 0] + off), dtype=float))

    if error_list:
        return np.concatenate(error_list)
    return np.array([], dtype=float)




def estimate_noise_std_from_base_result(
    base_result,
    *,
    use_robust_noise_std: bool = True,
    noise_clip_percentiles=(20, 80),
):
    base_errors = collect_voltage_errors(base_result)
    if base_errors.size == 0:
        return np.nan, base_errors

    if use_robust_noise_std:
        lo, hi = np.percentile(base_errors, noise_clip_percentiles)
        central = base_errors[(base_errors >= lo) & (base_errors <= hi)]
        if central.size >= 2:
            return float(np.std(central, ddof=1)), base_errors

    return float(np.std(base_errors, ddof=1)), base_errors




def build_voltage_resampled_segments_from_base_result(
    base_result,
    noise_std,
    *,
    rng,
    segment_generation_mode="direct_voltage_noise",
):
    mode = str(segment_generation_mode).strip().lower()

    if mode == "direct_voltage_noise":
        qv_iter = []
        for arr in base_result["segments"]:
            if arr is None or len(arr) == 0:
                continue
            arr = np.asarray(arr, dtype=float)
            arr_new = np.column_stack((arr[:, 0], arr[:, 1])).copy()
            arr_new[:, 1] += rng.normal(loc=0.0, scale=noise_std, size=arr.shape[0])
            qv_iter.append(arr_new)
        return qv_iter

    if mode == "baseline_ocv_resample":
        theta = np.asarray(base_result["theta"], dtype=float)
        f_inv = get_OCV_inv_extrapolated(*theta)
        offsets = list(base_result.get("offsets_final", []))
        qv_iter = []

        for seg_idx, arr in enumerate(base_result["segments"]):
            if arr is None or len(arr) == 0:
                continue

            arr = np.asarray(arr, dtype=float)
            voltage_original = np.asarray(arr[:, 1], dtype=float)
            q_baseline = np.asarray(f_inv(voltage_original), dtype=float)

            offset = offsets[seg_idx] if seg_idx < len(offsets) else 0.0
            if offset is None or not np.isfinite(offset):
                offset = 0.0

            q_resampled = q_baseline - float(offset)
            voltage_noisy = voltage_original + rng.normal(loc=0.0, scale=noise_std, size=arr.shape[0])
            qv_iter.append(np.column_stack((q_resampled, voltage_noisy)))

        return qv_iter

    raise ValueError(
        "Unknown segment_generation_mode="
        f"{segment_generation_mode!r}. Expected 'direct_voltage_noise' or 'baseline_ocv_resample'."
    )


def compute_rmsev_from_result(result_dict):
    segments = result_dict["segments"]
    offsets = result_dict["offsets_final"]
    theta = result_dict["theta"]
    f_curve = get_OCV_extrapolated(*theta)

    se_sum = 0.0
    n_sum = 0

    for seg, off in zip(segments, offsets):
        if seg is None or off is None or len(seg) == 0:
            continue
        err_v = seg[:, 1] - f_curve(seg[:, 0] + off)
        se_sum += float(np.sum(err_v**2))
        n_sum += len(err_v)

    return np.sqrt(se_sum / n_sum) if n_sum > 0 else np.nan


def compute_rmseq_from_result(result_dict):
    segments = result_dict["segments"]
    theta = result_dict["theta"]
    weighted = bool(result_dict.get("rmseQ_weighted", False))

    try:
        return float(compute_dQ_rmse(theta, segments, weighted=weighted))
    except Exception:
        pass

    if weighted:
        history = result_dict.get("rmseQ_history", [])
        return float(history[-1]) if history else np.nan

    f_inv = get_OCV_inv_extrapolated(*theta)
    se_sum = 0.0
    n_sum = 0

    for seg in segments:
        if seg is None or len(seg) < 2:
            continue

        q_true = np.asarray(seg[:, 0], dtype=float)
        v_meas = np.asarray(seg[:, 1], dtype=float)
        q_from_v = np.asarray(f_inv(v_meas), dtype=float)

        dq_true = np.diff(q_true)
        dq_model = np.diff(q_from_v)
        err_q = dq_true - dq_model

        se_sum += float(np.sum(err_q**2))
        n_sum += len(err_q)

    return np.sqrt(se_sum / n_sum) if n_sum > 0 else np.nan


def build_uncertainty_summary_df(base_result, uncertainty_results):
    rows = []

    base_dms = get_DMs_from_result(base_result)
    rows.append({
        "sample_id": 0,
        "type": "base",
        "x0": base_result["theta"][0],
        "y0": base_result["theta"][1],
        "Q_NE_theta": base_result["theta"][2],
        "Q_PE_theta": base_result["theta"][3],
        "Q_cell": base_dms["Q_cell"],
        "Q_Li": base_dms["Q_Li"],
        "Q_PE": base_dms["Q_PE"],
        "Q_NE": base_dms["Q_NE"],
        "RMSEQ": compute_rmseq_from_result(base_result),
        "RMSEV": compute_rmsev_from_result(base_result),
        "RMSEQ_hist_last": base_result["rmseQ_history"][-1] if base_result["rmseQ_history"] else np.nan,
        "RMSEV_hist_last": base_result["rmseV_history"][-1] if base_result["rmseV_history"] else np.nan,
        "V_error_max_hist_last": (
            base_result["V_error_max_history"][-1] if base_result["V_error_max_history"] else np.nan
        ),
        "n_segments": len(base_result["segments"]),
        "n_offsets": len(base_result["offsets_final"]),
        "n_points_total": int(sum(len(seg) for seg in base_result["segments"] if seg is not None)),
    })

    for item in uncertainty_results:
        result_iter = unwrap_result_item(item)
        if result_iter is None:
            continue

        dms = get_DMs_from_result(result_iter)
        sample_id = item["iter"] if isinstance(item, dict) and "iter" in item else len(rows)

        rows.append({
            "sample_id": sample_id,
            "type": "uncertainty",
            "x0": result_iter["theta"][0],
            "y0": result_iter["theta"][1],
            "Q_NE_theta": result_iter["theta"][2],
            "Q_PE_theta": result_iter["theta"][3],
            "Q_cell": dms["Q_cell"],
            "Q_Li": dms["Q_Li"],
            "Q_PE": dms["Q_PE"],
            "Q_NE": dms["Q_NE"],
            "RMSEQ": compute_rmseq_from_result(result_iter),
            "RMSEV": compute_rmsev_from_result(result_iter),
            "RMSEQ_hist_last": result_iter["rmseQ_history"][-1] if result_iter["rmseQ_history"] else np.nan,
            "RMSEV_hist_last": result_iter["rmseV_history"][-1] if result_iter["rmseV_history"] else np.nan,
            "V_error_max_hist_last": (
                result_iter["V_error_max_history"][-1] if result_iter["V_error_max_history"] else np.nan
            ),
            "n_segments": len(result_iter["segments"]),
            "n_offsets": len(result_iter["offsets_final"]),
            "n_points_total": int(sum(len(seg) for seg in result_iter["segments"] if seg is not None)),
        })

    return pd.DataFrame(rows).sort_values(["type", "sample_id"]).reset_index(drop=True)


def summarize_uncertainty_dataframe(summary_df):
    uncertainty_df = summary_df[summary_df["type"] == "uncertainty"].copy()
    if uncertainty_df.empty:
        return pd.DataFrame()

    rows = []
    for column in ["Q_cell", "Q_Li", "Q_PE", "Q_NE", "RMSEQ", "RMSEV", "V_error_max_hist_last"]:
        series = uncertainty_df[column].dropna()
        if series.empty:
            continue
        rows.append({
            "metric": column,
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if len(series) >= 2 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
        })

    return pd.DataFrame(rows)




def run_single_iter(
    i,
    base_result,
    data_list,
    noise_std,
    V_error_max_threshold,
    *,
    segmentation=False,
    iterative_kwargs=None,
    verbose: bool = False,
    iterative_verbose: bool = False,
    n_iter_total: int | None = None,
    random_seed: int | None = DEFAULT_PROJECT_RANDOM_SEED,
):
    iter_seed = derive_seed(random_seed, "rest_uncertainty_iter", i)
    rng = np.random.default_rng(iter_seed)
    iterative_kwargs = materialize_rest_uncertainty_iterative_kwargs(base_result, iterative_kwargs)
    if iter_seed is not None and "random_seed" not in iterative_kwargs:
        iterative_kwargs["random_seed"] = derive_seed(iter_seed, "fit")
    warm_start_theta = iterative_kwargs.get("warm_start_theta")
    objective_stop_threshold = iterative_kwargs.get("objective_stop_threshold")
    segment_generation_mode = str(
        iterative_kwargs.pop("segment_generation_mode", "direct_voltage_noise")
    ).strip().lower()

    input_n_segments = len(data_list)
    input_n_points = int(sum(len(arr) for arr in data_list if arr is not None))

    qv_iter = build_voltage_resampled_segments_from_base_result(
        base_result,
        noise_std,
        rng=rng,
        segment_generation_mode=segment_generation_mode,
    )

    try:
        if verbose:
            progress_label = f"{i + 1}/{n_iter_total}" if n_iter_total else f"{i + 1}"
            print(f"[rest_uncertainty] iter {progress_label} start", flush=True)
            print(
                "[rest_uncertainty] "
                f"iter {progress_label} segment_generation_mode={segment_generation_mode}",
                flush=True,
            )
            if warm_start_theta is not None:
                warm_start_arr = np.asarray(warm_start_theta, dtype=float)
                print(
                    "[rest_uncertainty] "
                    f"iter {progress_label} init theta="
                    f"({warm_start_arr[0]:.6f}, {warm_start_arr[1]:.6f}, "
                    f"{warm_start_arr[2]:.4f}, {warm_start_arr[3]:.4f})",
                    flush=True,
                )
            if objective_stop_threshold is not None:
                print(
                    "[rest_uncertainty] "
                    f"iter {progress_label} objective_stop_threshold={float(objective_stop_threshold):.6f}",
                    flush=True,
                )
        result_iter = iterative_segmentation(
            qv_iter,
            V_error_max_threshold=V_error_max_threshold,
            segmentation=segmentation,
            verbose=iterative_verbose,
            **iterative_kwargs,
        )

        segments_fin = result_iter["segments"]
        theta = result_iter["theta"]
        errors = collect_voltage_errors(result_iter)
        result_iter["V_error_distribution"] = errors

        debug_info = {
            "iter": i + 1,
            "input_n_segments": input_n_segments,
            "input_n_points": input_n_points,
            "segment_generation_mode": segment_generation_mode,
            "warm_start_theta": (
                None if warm_start_theta is None else [float(x) for x in np.asarray(warm_start_theta, dtype=float)]
            ),
            "objective_stop_threshold": (
                None if objective_stop_threshold is None else float(objective_stop_threshold)
            ),
            "random_seed": iter_seed,
            "output_n_segments": len(segments_fin),
            "output_n_points": int(sum(len(seg) for seg in segments_fin if seg is not None)),
            "x0": theta[0],
            "y0": theta[1],
            "Q_NE": theta[2],
            "Q_PE": theta[3],
            "RMSEQ_hist_last": result_iter["rmseQ_history"][-1] if result_iter["rmseQ_history"] else np.nan,
            "RMSEV_hist_last": result_iter["rmseV_history"][-1] if result_iter["rmseV_history"] else np.nan,
            "V_error_max_hist_last": (
                result_iter["V_error_max_history"][-1] if result_iter["V_error_max_history"] else np.nan
            ),
        }

        if verbose:
            progress_label = f"{i + 1}/{n_iter_total}" if n_iter_total else f"{i + 1}"
            print(
                "[rest_uncertainty] "
                f"iter {progress_label} ok | "
                f"rmseQ={debug_info['RMSEQ_hist_last']:.6f} | "
                f"rmseV={debug_info['RMSEV_hist_last']:.6f}",
                flush=True,
            )

        return {
            "iter": i + 1,
            "QV_iter": qv_iter,
            "result_iter": result_iter,
            "debug_info": debug_info,
            "status": "ok",
        }

    except Exception as exc:
        return {
            "iter": i + 1,
            "QV_iter": qv_iter,
            "result_iter": None,
            "debug_info": {
                "iter": i + 1,
                "input_n_segments": input_n_segments,
                "input_n_points": input_n_points,
                "random_seed": iter_seed,
                "error_message": repr(exc),
            },
            "status": "failed",
        }
    finally:
        if verbose and "exc" in locals():
            progress_label = f"{i + 1}/{n_iter_total}" if n_iter_total else f"{i + 1}"
            print(f"[rest_uncertainty] iter {progress_label} failed -> {exc!r}", flush=True)




def run_rest_ocv_uncertainty(
    base_result,
    *,
    n_iter: int = 20,
    noise_std: float | None = None,
    V_error_max_threshold: float | None = None,
    use_robust_noise_std: bool = False,
    noise_clip_percentiles=(20, 80),
    segmentation: bool = False,
    iterative_kwargs=None,
    segment_generation_mode="direct_voltage_noise",
    n_jobs: int = -1,
    verbose: bool = False,
    iterative_verbose: bool = False,
    joblib_verbose: int = 0,
    random_seed: int | None = DEFAULT_PROJECT_RANDOM_SEED,
):
    base_errors = collect_voltage_errors(base_result)

    if noise_std is None:
        noise_std, _ = estimate_noise_std_from_base_result(
            base_result,
            use_robust_noise_std=use_robust_noise_std,
            noise_clip_percentiles=noise_clip_percentiles,
        )

    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be a positive finite value.")

    if V_error_max_threshold is None:
        history = base_result.get("V_error_max_history", [])
        V_error_max_threshold = float(history[-1]) if history else 0.02

    data_list = [np.asarray(seg, dtype=float) for seg in base_result["segments"] if seg is not None and len(seg) > 0]
    iterative_kwargs = {} if iterative_kwargs is None else dict(iterative_kwargs)
    iterative_kwargs.setdefault("segment_generation_mode", segment_generation_mode)
    normalized_n_jobs = int(n_jobs)
    objective_stop_threshold = iterative_kwargs.get("objective_stop_threshold")
    effective_segment_generation_mode = str(
        iterative_kwargs.get("segment_generation_mode", segment_generation_mode)
    ).strip().lower()

    if verbose:
        threshold_suffix = (
            f" | objective_stop_threshold={float(objective_stop_threshold):.6f}"
            if objective_stop_threshold is not None else ""
        )
        print(
            "[rest_uncertainty] "
            f"start | n_iter={n_iter} | n_jobs={normalized_n_jobs} | "
            f"noise_std={float(noise_std):.6f} | "
            f"V_error_max_threshold={float(V_error_max_threshold):.6f}"
            f" | segment_generation_mode={effective_segment_generation_mode}"
            f"{threshold_suffix} | random_seed={random_seed}",
            flush=True,
        )

    if normalized_n_jobs == 1:
        raw_results = [
            run_single_iter(
                i,
                base_result,
                data_list,
                noise_std,
                V_error_max_threshold,
                segmentation=segmentation,
                iterative_kwargs=iterative_kwargs,
                verbose=verbose,
                iterative_verbose=iterative_verbose,
                n_iter_total=n_iter,
                random_seed=random_seed,
            )
            for i in range(n_iter)
        ]
    else:
        raw_results = Parallel(
            n_jobs=normalized_n_jobs,
            backend="loky",
            max_nbytes=None,
            verbose=int(joblib_verbose),
        )(
            delayed(run_single_iter)(
                i,
                base_result,
                data_list,
                noise_std,
                V_error_max_threshold,
                segmentation=segmentation,
                iterative_kwargs=iterative_kwargs,
                verbose=verbose,
                iterative_verbose=iterative_verbose,
                n_iter_total=n_iter,
                random_seed=random_seed,
            )
            for i in range(n_iter)
        )

    ok_results = [item for item in raw_results if item["status"] == "ok" and item["result_iter"] is not None]
    failed_results = [item for item in raw_results if item["status"] != "ok"]
    summary_df = build_uncertainty_summary_df(base_result, ok_results)
    stats_df = summarize_uncertainty_dataframe(summary_df)
    debug_df = pd.DataFrame([item["debug_info"] for item in raw_results])

    if verbose:
        print(
            "[rest_uncertainty] "
            f"done | successful_samples={len(ok_results)} | failed_samples={len(failed_results)}",
            flush=True,
        )

    return {
        "base_result": base_result,
        "base_errors": base_errors,
        "noise_std": float(noise_std),
        "V_error_max_threshold": float(V_error_max_threshold),
        "objective_stop_threshold": (
            None if objective_stop_threshold is None else float(objective_stop_threshold)
        ),
        "random_seed": None if random_seed is None else int(random_seed),
        "segment_generation_mode": effective_segment_generation_mode,
        "results": ok_results,
        "failed_results": failed_results,
        "summary_df": summary_df,
        "stats_df": stats_df,
        "debug_df": debug_df,
    }






























