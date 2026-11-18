"""Charge basis of the solver.

The solver works on the charge of ``P_PARALLEL`` cells in parallel. The run
scripts multiply the measured per cell charge by ``P_PARALLEL`` before fitting
and divide the fitted capacities by it afterwards. Voltage is per cell.
"""

from __future__ import annotations

P_PARALLEL: int = 2
