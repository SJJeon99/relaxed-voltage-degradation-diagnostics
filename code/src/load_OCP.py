from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter

try:
    from .paths import get_ocp_data_root
except ImportError:
    from src.paths import get_ocp_data_root


OCP_DATA_DIR = get_ocp_data_root()
DEFAULT_AHC_PATH = OCP_DATA_DIR / "OCPn_halfcell.mat"
DEFAULT_CHC_PATH = OCP_DATA_DIR / "OCPp_halfcell.mat"


@dataclass(frozen=True)
class OCPBundle:
    df_AHC_OCVchg: pd.DataFrame
    df_AHC_OCVdis: pd.DataFrame
    df_CHC_OCVchg: pd.DataFrame
    df_CHC_OCVdis: pd.DataFrame
    function_OCPp: interp1d
    function_OCPn: interp1d
    function_OCPp_smooth: interp1d
    function_OCPn_smooth: interp1d
    function_OCPp_smooth_extrapolated: interp1d
    function_OCPn_smooth_extrapolated: interp1d


def get_data(mat_data):
    idx_charge = 3
    idx_discharge = 5

    V_ch = mat_data["data"]["V"][idx_charge][0]
    t_ch = mat_data["data"]["t"][idx_charge][0]
    I_ch = mat_data["data"]["I"][idx_charge][0]
    t_ch -= t_ch[0]
    t_ch = np.array(t_ch).flatten().astype(float)
    I_ch = np.array(I_ch).flatten().astype(float)
    Q_ch = -cumulative_trapezoid(I_ch, t_ch, initial=0) / 3600

    V_dch = mat_data["data"]["V"][idx_discharge][0]
    t_dch = mat_data["data"]["t"][idx_discharge][0]
    I_dch = mat_data["data"]["I"][idx_discharge][0]
    t_dch -= t_dch[0]
    t_dch = np.array(t_dch).flatten().astype(float)
    I_dch = np.array(I_dch).flatten().astype(float)
    Q_dch = -cumulative_trapezoid(-I_dch, t_dch, initial=0) / 3600

    Q_ch = Q_ch.flatten().astype(float)
    V_ch = V_ch.flatten().astype(float)
    Q_dch = Q_dch.flatten().astype(float)
    V_dch = V_dch.flatten().astype(float)

    df_ch = pd.DataFrame({"Capacity": Q_ch, "Voltage": V_ch})
    df_dch = pd.DataFrame({"Capacity": Q_dch, "Voltage": V_dch})
    return df_ch, df_dch


def savgol_filter_nan_aware(data, window_length=300, polyorder=2):
    filtered_data = data.copy()
    isnan = np.isnan(data)
    not_nan_idx = np.where(~isnan)[0]

    if len(not_nan_idx) == 0:
        return filtered_data

    diff_idx = np.diff(not_nan_idx)
    split_points = np.where(diff_idx > 1)[0] + 1
    segments = np.split(not_nan_idx, split_points)

    for seg in segments:
        if len(seg) < polyorder + 2:
            continue

        seg_data = data[seg]
        wl = min(window_length, len(seg))

        if wl % 2 == 0:
            wl = max(wl - 1, polyorder + 2)
        if wl < polyorder + 2:
            wl = polyorder + 2
        if wl % 2 == 0:
            wl += 1
        wl = min(wl, len(seg))

        try:
            filtered_segment = savgol_filter(seg_data, window_length=wl, polyorder=polyorder)
            filtered_data[seg] = filtered_segment
        except Exception as exc:
            print(f"Warning: segment {seg[0]}-{seg[-1]} savgol filter skipped due to {exc}")

    return filtered_data


def _load_mat_file(path: str | Path):
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"OCP data file not found: {resolved_path}")
    return scipy.io.loadmat(resolved_path)


def build_ocp_bundle(
    ahc_path: str | Path = DEFAULT_AHC_PATH,
    chc_path: str | Path = DEFAULT_CHC_PATH,
    *,
    y1: float = 0.215685,
    soc_points: int = 15000,
    smooth_window_length: int = 200,
    smooth_polyorder: int = 2,
) -> OCPBundle:
    mat_data_AHC = _load_mat_file(ahc_path)
    mat_data_CHC = _load_mat_file(chc_path)

    df_AHC_OCVchg, df_AHC_OCVdis = get_data(mat_data_AHC)
    df_CHC_OCVchg, df_CHC_OCVdis = get_data(mat_data_CHC)

    df_AHC_OCVdis = df_AHC_OCVdis.copy()
    df_CHC_OCVdis = df_CHC_OCVdis.copy()

    df_AHC_OCVdis["SOC"] = (
        (df_AHC_OCVdis["Capacity"] - df_AHC_OCVdis["Capacity"].iloc[-1])
        / (df_AHC_OCVdis["Capacity"].iloc[0] - df_AHC_OCVdis["Capacity"].iloc[-1])
    )

    df_CHC_OCVdis["SOC"] = y1 + 1 - (
        (
            (df_CHC_OCVdis["Capacity"] - df_CHC_OCVdis["Capacity"].iloc[-1])
            / (df_CHC_OCVdis["Capacity"].iloc[0] - df_CHC_OCVdis["Capacity"].iloc[-1])
        )
        * (1 - y1)
        + y1
    )

    soc_pe = df_CHC_OCVdis["SOC"].values
    pe = df_CHC_OCVdis["Voltage"].values
    spline_pe = UnivariateSpline(soc_pe, pe, k=1, s=0.0003)
    soc_pe_extrapolated = np.linspace(0, soc_pe.min() - 0.001, 1000)
    pe_extrapolated = spline_pe(soc_pe_extrapolated)
    soc_pe_full = np.concatenate([soc_pe_extrapolated, soc_pe])
    pe_full = np.concatenate([pe_extrapolated, pe])

    function_OCPp = interp1d(
        soc_pe_full,
        pe_full,
        kind="linear",
        fill_value="extrapolate",
    )
    function_OCPn = interp1d(
        df_AHC_OCVdis["SOC"],
        df_AHC_OCVdis["Voltage"],
        fill_value="extrapolate",
    )

    soc = np.linspace(0, 1, soc_points)
    ocpp_smoothed = savgol_filter_nan_aware(
        function_OCPp(soc),
        window_length=smooth_window_length,
        polyorder=smooth_polyorder,
    )
    ocpn_smoothed = savgol_filter_nan_aware(
        function_OCPn(soc),
        window_length=smooth_window_length,
        polyorder=smooth_polyorder,
    )

    function_OCPp_smooth = interp1d(
        soc,
        ocpp_smoothed,
        kind="linear",
        fill_value=np.nan,
        bounds_error=False,
    )
    function_OCPn_smooth = interp1d(
        soc,
        ocpn_smoothed,
        kind="linear",
        fill_value=np.nan,
        bounds_error=False,
    )
    function_OCPp_smooth_extrapolated = interp1d(
        soc,
        ocpp_smoothed,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    function_OCPn_smooth_extrapolated = interp1d(
        soc,
        ocpn_smoothed,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    return OCPBundle(
        df_AHC_OCVchg=df_AHC_OCVchg,
        df_AHC_OCVdis=df_AHC_OCVdis,
        df_CHC_OCVchg=df_CHC_OCVchg,
        df_CHC_OCVdis=df_CHC_OCVdis,
        function_OCPp=function_OCPp,
        function_OCPn=function_OCPn,
        function_OCPp_smooth=function_OCPp_smooth,
        function_OCPn_smooth=function_OCPn_smooth,
        function_OCPp_smooth_extrapolated=function_OCPp_smooth_extrapolated,
        function_OCPn_smooth_extrapolated=function_OCPn_smooth_extrapolated,
    )


@lru_cache(maxsize=1)
def load_default_ocp_bundle() -> OCPBundle:
    return build_ocp_bundle()


_DEFAULT_OCP_BUNDLE = load_default_ocp_bundle()

df_AHC_OCVchg = _DEFAULT_OCP_BUNDLE.df_AHC_OCVchg
df_AHC_OCVdis = _DEFAULT_OCP_BUNDLE.df_AHC_OCVdis
df_CHC_OCVchg = _DEFAULT_OCP_BUNDLE.df_CHC_OCVchg
df_CHC_OCVdis = _DEFAULT_OCP_BUNDLE.df_CHC_OCVdis
function_OCPp = _DEFAULT_OCP_BUNDLE.function_OCPp
function_OCPn = _DEFAULT_OCP_BUNDLE.function_OCPn
function_OCPp_smooth = _DEFAULT_OCP_BUNDLE.function_OCPp_smooth
function_OCPn_smooth = _DEFAULT_OCP_BUNDLE.function_OCPn_smooth
function_OCPp_smooth_extrapolated = _DEFAULT_OCP_BUNDLE.function_OCPp_smooth_extrapolated
function_OCPn_smooth_extrapolated = _DEFAULT_OCP_BUNDLE.function_OCPn_smooth_extrapolated


