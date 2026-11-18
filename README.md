# Relaxed-voltage experiments: data and fitting code

Code and results of the paper **Battery Degradation Mode Diagnostics from Relaxed-Voltage Observations in the Field**
Sangjun Jeon, Gyeomson Park, Sungju Ahn, Jonghoon Kim, Seongmin Heo, Juhyun Song

## Cells

Nine NMC/graphite pouch cells, tested at 25 °C between 2.5 and 4.2 V. `Cell1`–`Cell9` are Cell 1–9 of the paper. Each cell ran three designed load protocols (A, B, C). Capacities and charges are in Ah per cell.

## Contents

```
data/
  ocv/                 C/40 discharge OCV curve of each cell (voltage, capacity)
  load_raw/            cycler files of the load protocols A, B, C for each cell
  load_protocols/      load profile of each protocol
  rest_observations/   rest observations (accumulated charge, relaxed voltage) used as fitting input
  ocp/                 half cell OCP: measured (.mat) and the tables used by the model (.csv)
code/                  fitting scripts and the modules they import (code/src/)
results/
  ocv_reference/       reference values from the C/40 OCV curves
  uncertainty_3mV_5mV/ uncertainty fits at 3 and 5 mV voltage noise
  uncertainty_10mV/    uncertainty fits at 10 mV voltage noise
```

- `data/load_raw/` files are in the cycler's export format (CP949 encoding).
- `data/rest_observations/rest_observations.json` is read by the code; `rest_observations.csv` holds the same observations as a table.
- `data/ocp/OCPp.csv` (`y`, `U_PE`) and `OCPn.csv` (`x`, `U_NE`) are the smoothed and extrapolated tables that the model evaluates, built from the measured half cell curves by `code/src/load_OCP.py`.
- Model: OCV(Q) = U_PE(y0 − Q/Q_PE) − U_NE(x0 + Q/Q_NE).

## Running the code

Python 3.12.

```
python -m pip install -r requirements.txt

python code/fit_ocv_reference.py           # reference values       -> output/ocv_reference/
python code/run_uncertainty_3mV_5mV.py     # uncertainty fits       -> output/uncertainty_3mV_5mV/
python code/run_uncertainty_10mV.py        #                        -> output/uncertainty_10mV/
python code/summarize_3mV_5mV.py           # error tables from the fits in output/
python code/summarize_10mV.py
python code/export_ocp_csv.py              # OCP tables             -> output/ocp/
```

Quick check with one run and one replicate:

```
python code/run_uncertainty_3mV_5mV.py --limit-runs 1 --n-iter 1 --sigmas 5 --jobs 1 --out output/check
```
