"""Fit the rest observations of each cell and protocol and estimate the uncertainty.

For each of the 27 runs (3 protocols x 9 cells) the OCV model is fitted once to
the rest observations, and then ``n_iter`` more times with Gaussian noise of
standard deviation sigma added to the voltages. Called by
``run_uncertainty_3mV_5mV.py`` and ``run_uncertainty_10mV.py``, which set the
output folder and sigma.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'uncertainty'
sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'

import argparse
import json
import pickle
import time
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / 'code'))
from src.fit_presets import get_runtime_fit_configs
from src.rest_OCV_segmentation import iterative_segmentation
from src.rest_OCV_uncertainty import run_rest_ocv_uncertainty
from src.reproducibility import derive_seed
from src.cell_basis import P_PARALLEL
from src.cell_ids import SEED_ID

P = float(P_PARALLEL)
SEED = 20260916
OFFSET_SEED = 42
CELLS = list(SEED_ID)  # Cell1 ... Cell9
SIGMAS = [5]
SOURCE = ROOT / 'data/rest_observations/rest_observations.json'
# Parameter bounds (x0, y0, Q_NE, Q_PE). The solver works on the charge of two
# cells in parallel, so its capacity bounds are twice the per cell bounds.
BOUNDS_SOLVER = ((0.0, 0.02), (0.8, 0.99), (100.0, 140.0), (120.0, 180.0))
BOUNDS_CELL = ((0.0, 0.02), (0.8, 0.99), (50.0, 70.0), (60.0, 90.0))


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def jwrite(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=serial), encoding='utf-8')


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f: pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load(path):
    with path.open('rb') as f: return pickle.load(f)


def configs(jobs, seed, n_iter=100):
    cfg = get_runtime_fit_configs('default')
    base = cfg['rest_segmentation']
    base.update(bounds=BOUNDS_SOLVER, segmentation=False, search_range=(-600, 600),
                de_workers=1, multistart_workers=1, random_seed=seed, verbose=False)
    unc = cfg['rest_uncertainty']
    unc.update(n_iter=n_iter, n_jobs=jobs, segmentation=False, segment_generation_mode='direct_voltage_noise',
               random_seed=seed, verbose=False)
    unc['iterative_kwargs'].update(bounds=BOUNDS_SOLVER, search_range=(-600, 600))
    return base, unc


def check_shape(result, segments):
    assert len(result['segments']) == len(segments)
    assert [len(x) for x in result['segments']] == [len(x) for x in segments]
    assert len(result['offsets_final']) == len(segments)
    assert all(x is not None and np.isfinite(x) for x in result['offsets_final'])
    assert not len(result.get('dropped_indices', []))
    for a, b in zip(result['segments'], segments):
        assert np.array_equal(np.asarray(a)[:, 0], np.asarray(b)[:, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=12, help='parallel workers for the replicates')
    parser.add_argument('--limit-runs', type=int, help='only the first N of the 27 runs')
    parser.add_argument('--sigmas', nargs='+', type=int, default=SIGMAS, help='voltage noise in mV')
    parser.add_argument('--base-only', action='store_true', help='only the first fit of each run')
    parser.add_argument('--progress-file', default='progress.json')
    parser.add_argument('--n-iter', type=int, default=100, help='replicates per run (published results: 100)')
    parser.add_argument('--out', help='output folder')
    args = parser.parse_args()
    global OUT
    if args.out: OUT = Path(args.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    assert set(args.sigmas) <= set(SIGMAS)
    payload = json.loads(SOURCE.read_text())
    runs = sorted(payload['runs'], key=lambda r: (r['protocol'], SEED_ID[r['cell_id']]))
    assert len(runs) == 27 and sorted({r['cell_id'] for r in runs}) == CELLS
    base_example, unc_example = configs(args.jobs, SEED, args.n_iter)
    settings = dict(cells=CELLS, protocols=['A', 'B', 'C'], sigmas_mV=SIGMAS, n_iter=args.n_iter, master_seed=SEED,
        source=str(SOURCE.relative_to(ROOT)),
        observations_per_run={'A': 119, 'B': 87, 'C': 72}, segments_per_run={'A': 26, 'B': 16, 'C': 8},
        segment_offset_range_Ah=[-8 / P, 8 / P],
        base_config=base_example, uncertainty_config=unc_example,
        bounds_per_cell=BOUNDS_CELL, bounds_solver=BOUNDS_SOLVER)
    settings_path = OUT / 'settings.json'
    if settings_path.exists():
        old = json.loads(settings_path.read_text())
        assert old['source'] == settings['source'] and old['bounds_solver'] == [list(x) for x in BOUNDS_SOLVER]
    else: jwrite(settings_path, settings)
    offsets = []; input_runs = []
    for run in runs:
        p, c = run['protocol'], run['cell_id']; seed = derive_seed(SEED, 'segmented_experiment', p, SEED_ID[c])
        rng = np.random.default_rng(derive_seed(OFFSET_SEED, 'input_offsets', p, SEED_ID[c]))
        segments = []
        for s in run['segments']:
            # A random charge offset is added to every segment, so that the fit
            # cannot use the charge reference between segments.
            qv = np.array(s['qv'], dtype=float)
            shifted = qv.copy(); delta = float(rng.uniform(-8, 8)); shifted[:, 0] = qv[:, 0] * P + delta
            assert np.allclose(np.diff(shifted[:, 0]) / P, np.diff(qv[:, 0]), atol=1e-12, rtol=0)
            segments.append(shifted)
            offsets.append(dict(protocol=p, cell_id=c, segment_idx=s['segment_idx'], n_points=len(qv),
                offset_Ah=delta / P))
        input_runs.append(dict(protocol=p, cell_id=c, seed=seed, segments=segments,
            segment_ids=[s['segment_idx'] for s in run['segments']]))
    offsets_path = OUT / 'segment_offsets.csv'
    if not offsets_path.exists():
        pd.DataFrame(offsets).to_csv(offsets_path, index=False)
    log_path = OUT / args.progress_file
    failures = []; completed = []; start = time.time()
    selected = input_runs[:args.limit_runs] if args.limit_runs else input_runs
    for ri, run in enumerate(selected, 1):
        tag = f"{run['protocol']}_{run['cell_id']}"; segments = run['segments']
        basecfg, unccfg = configs(args.jobs, run['seed'], args.n_iter)
        basedir = OUT / 'base_fits'; basedir.mkdir(exist_ok=True)
        bp = basedir / f'{tag}.pkl'
        try:
            if bp.exists(): base = load(bp)
            else:
                print(f'BASE {ri}/{len(selected)} {tag}: {len(segments)} segments, {sum(map(len, segments))} observations', flush=True)
                t0 = time.time(); base = iterative_segmentation(segments, **basecfg)
                check_shape(base, segments); base['wall_seconds'] = time.time() - t0
                dump(bp, base)
                print(f"BASE DONE {tag}: {base['wall_seconds']:.1f}s theta={np.asarray(base['theta'])}", flush=True)
            check_shape(base, segments)
            if args.base_only:
                completed.append(f'{tag}_base')
                jwrite(log_path, dict(status='running', current_run=tag, current_sigma_mV=None, completed=completed, failures=failures))
                continue
            for sigma in args.sigmas:
                dest = OUT / f'sigma_{sigma:02d}mV' / 'per_run'; dest.mkdir(parents=True, exist_ok=True)
                up = dest / f'{tag}_uncertainty.pkl'
                if up.exists():
                    print(f'RESUME {tag} {sigma}mV: saved result', flush=True); completed.append(f'{tag}_{sigma}mV'); continue
                jwrite(log_path, dict(status='running', current_run=tag, current_sigma_mV=sigma, completed=completed, failures=failures))
                print(f'UNC START {tag} {sigma}mV {args.n_iter} replicates', flush=True)
                t0 = time.time(); unc = run_rest_ocv_uncertainty(base, noise_std=sigma / 1000, **unccfg)
                for item in unc['results']: check_shape(item['result_iter'], segments)
                assert len(unc['results']) == args.n_iter and not unc['failed_results']
                unc['wall_seconds'] = time.time() - t0
                unc['run_metadata'] = dict(protocol=run['protocol'], cell_id=run['cell_id'], sigma_mV=sigma,
                    segment_ids=run['segment_ids'], reporting_divisor=P)
                dump(up, unc)
                completed.append(f'{tag}_{sigma}mV')
                jwrite(log_path, dict(status='running', current_run=tag, current_sigma_mV=sigma, completed=completed, failures=failures))
                print(f"UNC DONE {tag} {sigma}mV {unc['wall_seconds']:.1f}s {len(unc['results'])}/{args.n_iter}", flush=True)
        except Exception:
            err = traceback.format_exc(); print(err, flush=True); failures.append(dict(run=tag, error=err))
            jwrite(log_path, dict(status='running_with_failure', completed=completed, failures=failures))
    jwrite(log_path, dict(status='complete' if not failures else 'needs_attention', completed=completed, failures=failures, elapsed_seconds=time.time() - start))
    if failures: raise RuntimeError(f'{len(failures)} run failures; see {args.progress_file}')


if __name__ == '__main__': main()
