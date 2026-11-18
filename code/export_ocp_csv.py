"""Write the electrode OCP tables used by the OCV model to CSV.

OCPp.csv: y, U_PE [V]    OCPn.csv: x, U_NE [V]
The tables are the nodes of the two interpolation functions that
``src/load_OCP.py`` builds from the half cell .mat files in ``data/ocp``.
Output folder: output/ocp (``--out-dir`` changes it).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ocp-dir", type=Path, default=ROOT / "data" / "ocp",
                        help="folder holding the two .mat files")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "ocp",
                        help="where to write OCPp.csv and OCPn.csv")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.dont_write_bytecode = True
    # Set before the import: load_OCP.py resolves and loads its .mat files at import.
    os.environ["OCP_DATA_ROOT"] = str(args.ocp_dir.resolve())
    sys.path.insert(0, str(HERE))
    import numpy as np
    import scipy
    from src import load_OCP

    f_pe = load_OCP.function_OCPp_smooth_extrapolated
    f_ne = load_OCP.function_OCPn_smooth_extrapolated
    tables = {
        "OCPp.csv": ("y", "U_PE [V]", f_pe),
        "OCPn.csv": ("x", "U_NE [V]", f_ne),
    }
    for name, (c0, c1, f) in tables.items():
        s = np.asarray(f.x, float)
        u = np.asarray(f.y, float)
        assert s.ndim == u.ndim == 1 and len(s) == len(u)
        assert np.all(np.diff(s) > 0) and np.all(np.isfinite(u))
        # The table rows are the function's own nodes: evaluating there returns them.
        assert np.max(np.abs(f(s) - u)) < 1e-12
        path = out_dir / name
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(f"{c0},{c1}\n")
            for a, b in zip(s, u):
                fh.write(f"{a:.17g},{b:.17g}\n")
        print(f"wrote {path}  rows={len(s)}  {c0}=[{s[0]:.6f}, {s[-1]:.6f}]  "
              f"{c1}=[{u.min():.6f}, {u.max():.6f}]")

    pe = load_OCP.df_CHC_OCVdis
    ne = load_OCP.df_AHC_OCVdis
    print(f"measured U_PE branch: {len(pe)} samples, y = {pe['SOC'].min():.6f} .. "
          f"{pe['SOC'].max():.6f}, V = {pe['Voltage'].min():.4f} .. {pe['Voltage'].max():.4f}")
    print(f"measured U_NE branch: {len(ne)} samples, x = {ne['SOC'].min():.6f} .. "
          f"{ne['SOC'].max():.6f}, V = {ne['Voltage'].min():.4f} .. {ne['Voltage'].max():.4f}")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
