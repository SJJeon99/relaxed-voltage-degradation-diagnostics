"""Error tables of the 3 and 5 mV uncertainty fits, in Ah per cell."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output'/'uncertainty_3mV_5mV'
CELLS = [f'Cell{k}' for k in range(1,10)]
MODES = ['Q_cell','Q_Li','Q_PE','Q_NE']

def main():
    ref = pd.read_csv(ROOT/'results/ocv_reference/reference_values.csv', comment='#').set_index('cell_id')
    rows = []
    for sigma in [3,5]:
        for protocol in ['A','B','C']:
            for cell in CELLS:
                path = OUT/f'sigma_{sigma:02d}mV/per_run/{protocol}_{cell}_uncertainty.pkl'
                with path.open('rb') as f:
                    u = pickle.load(f)
                assert len(u['results']) == 100 and not u['failed_results']
                d = u['summary_df']
                d = d[d['type'].eq('uncertainty')]
                assert len(d) == 100
                for mode in MODES:
                    for i, value in enumerate(d[mode].to_numpy()/2):
                        reference = float(ref.loc[cell,mode])
                        rows.append(dict(sigma_mV=sigma,protocol=protocol,cell_id=cell,mode=mode,replicate=i+1,estimate_Ah=value,reference_Ah=reference,error_Ah=value-reference))
    d = pd.DataFrame(rows)
    assert np.isfinite(d[['estimate_Ah','reference_Ah','error_Ah']]).all().all()
    d.to_csv(OUT/'replicate_errors.csv',index=False)
    def metrics(g):
        e=g.error_Ah.to_numpy()
        return pd.Series(dict(n=len(e),MAE_Ah=np.abs(e).mean(),RMSE_Ah=np.sqrt(np.mean(e**2)),STD_error_Ah=np.std(e,ddof=1),bias_Ah=e.mean(),estimate_median_Ah=g.estimate_Ah.median()))
    for keys,name in [(['sigma_mV','protocol','mode'],'protocol_metrics.csv'),(['sigma_mV','protocol','cell_id','mode'],'cell_metrics.csv')]:
        records=[]
        for key,g in d.groupby(keys):
            records.append(dict(zip(keys,key))|metrics(g).to_dict())
        pd.DataFrame(records).to_csv(OUT/name,index=False)

if __name__=='__main__':
    main()
