"""Reference values of each cell from a fit of the OCV model to its C/40 OCV curve.

Model: OCV(Q) = U_PE(y0 - Q/Q_PE) - U_NE(x0 + Q/Q_NE), with model charge
Q = offset - (discharged capacity). Writes output/ocv_reference/reference_values.csv
(Ah per cell).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from src.load_OCP import (function_OCPn_smooth_extrapolated as OCP_NE,    # noqa: E402
                          function_OCPp_smooth_extrapolated as OCP_PE)
from src.rest_OCV_segmentation import infer_health_mode                   # noqa: E402

IN_DIR = ROOT / "data" / "ocv"
OUT_DIR = ROOT / "output" / "ocv_reference"

PARAMS = ["x0", "y0", "Q_NE", "Q_PE"]
MODES = ["Q_cell", "Q_Li", "Q_PE", "Q_NE"]
SEED = 42
OFFSET_RANGE = (-300.0, 300.0)
OFFSET_XATOL = 1e-3
DE_STRATEGY, DE_POPSIZE, DE_TOL, DE_MAXITER = "best1bin", 16, 1e-4, 30
LOCAL_MAXITER, LOCAL_FTOL = 200, 1e-8
REFIT_RMSE_MV = 5.0
BOUND_TOL_REL = 1e-5

# Bounds on (x0, y0, Q_NE, Q_PE). A cell is refitted with the next, wider set
# when a parameter ends at a bound.
BOUNDS_SEQUENCE = [
    ((0.00, 0.02), (0.80, 0.99), (40.0, 70.0), (70.0, 90.0)),
    ((0.00, 0.15), (0.40, 1.00), (25.0, 100.0), (35.0, 120.0)),
    ((0.00, 0.35), (0.20, 1.00), (15.0, 130.0), (20.0, 160.0)),
    ((0.00, 0.60), (0.05, 1.00), (5.0, 160.0), (10.0, 220.0)),
]


def ocv_voltage(theta, q):
    x0, y0, q_ne, q_pe = np.asarray(theta, dtype=float)
    q = np.asarray(q, dtype=float)
    return OCP_PE(y0 - q / q_pe) - OCP_NE(x0 + q / q_ne)


def _v_mse_at_offset(q_neg, v, theta, offset) -> float:
    err = v - ocv_voltage(theta, q_neg + offset)
    return float(np.mean(err * err))


def best_offset(q_neg, v, theta) -> tuple[float, float]:
    res = minimize_scalar(lambda off: _v_mse_at_offset(q_neg, v, theta, off),
                          bounds=OFFSET_RANGE, method="bounded",
                          options={"xatol": OFFSET_XATOL})
    return float(res.x), float(np.sqrt(res.fun))


def objective(theta, q_neg, v) -> float:
    if not np.all(np.isfinite(theta)):
        return 1e6
    _, rmse = best_offset(q_neg, v, np.asarray(theta, dtype=float))
    return rmse if np.isfinite(rmse) else 1e6


def at_bound(theta, bounds):
    for j in range(len(PARAMS)):
        lo, hi = bounds[j]
        tol = BOUND_TOL_REL * (hi - lo)
        if theta[j] <= lo + tol or theta[j] >= hi - tol:
            return True
    return False


def fit(q_data, v, bounds):
    q_neg = -np.asarray(q_data, dtype=float)
    de = differential_evolution(objective, bounds=bounds, args=(q_neg, v),
                                strategy=DE_STRATEGY, popsize=DE_POPSIZE, tol=DE_TOL,
                                maxiter=DE_MAXITER, seed=SEED, polish=False,
                                updating="deferred", workers=1)
    loc = minimize(objective, x0=de.x, args=(q_neg, v), method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": LOCAL_MAXITER, "ftol": LOCAL_FTOL})
    theta = np.asarray(loc.x if (loc.success and loc.fun <= de.fun) else de.x, dtype=float)
    offset, rmse = best_offset(q_neg, v, theta)
    return theta, float(offset), float(rmse)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(IN_DIR.glob("Cell*_ocv_discharge.csv"),
                   key=lambda p: int(p.stem.split("_")[0][4:]))
    rows = []
    for path in paths:
        cell = path.stem.split("_")[0]
        d = pd.read_csv(path, comment="#")
        q_data = d["capacity_Ah"].to_numpy(float)
        v = d["voltage_V"].to_numpy(float)
        for k, bounds in enumerate(BOUNDS_SEQUENCE):
            theta, offset, rmse = fit(q_data, v, bounds)
            if not (at_bound(theta, bounds) or (k == 0 and rmse * 1000.0 > REFIT_RMSE_MV)):
                break
        h = infer_health_mode(list(theta))
        row = {"cell_id": cell, "x0": float(theta[0]), "y0": float(theta[1])}
        row.update({m: float(h[m].iloc[0]) for m in ["Q_NE", "Q_PE", "Q_cell", "Q_Li"]})
        row["rmse_mV"] = rmse * 1000.0
        rows.append(row)
        print(f"{cell}: RMSE {row['rmse_mV']:.3f} mV, Q_cell {row['Q_cell']:.3f} Ah", flush=True)
    with (OUT_DIR / "reference_values.csv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("# Reference values from the C/40 OCV curves, Ah per cell.\n")
        pd.DataFrame(rows).to_csv(fh, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
