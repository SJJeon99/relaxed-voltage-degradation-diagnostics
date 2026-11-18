"""Error tables of the 10 mV uncertainty fits, in Ah per cell."""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'uncertainty_10mV'
ref=pd.read_csv(ROOT/'results/ocv_reference/reference_values.csv',comment='#').set_index('cell_id')
cells=[f'Cell{k}' for k in range(1,10)]; modes=['Q_cell','Q_Li','Q_PE','Q_NE']; rows=[]
for protocol in 'ABC':
 for cell in cells:
  u=pickle.load(open(OUT/f'sigma_10mV/per_run/{protocol}_{cell}_uncertainty.pkl','rb'))
  d=u['summary_df']; d=d[d.type.eq('uncertainty')]
  assert len(d)==100 and not u['failed_results']
  for mode in modes:
   e=d[mode].to_numpy()/2-float(ref.loc[cell,mode])
   rows.append(dict(protocol=protocol,cell_id=cell,mode=mode,n=100,MAE_Ah=float(np.abs(e).mean()),RMSE_Ah=float(np.sqrt(np.mean(e**2))),STD_error_Ah=float(np.std(e,ddof=1)),abs_error_STD_Ah=float(np.std(np.abs(e),ddof=1)),p95_abs_error_Ah=float(np.percentile(np.abs(e),95)),max_abs_error_Ah=float(np.max(np.abs(e))),estimate_median_Ah=float(np.median(d[mode].to_numpy()/2))))
out=pd.DataFrame(rows); out.to_csv(OUT/'metrics_10mV.csv',index=False)
g=out.groupby(['protocol','mode'])[['MAE_Ah','RMSE_Ah','STD_error_Ah','abs_error_STD_Ah','p95_abs_error_Ah','max_abs_error_Ah']].mean().reset_index(); g.to_csv(OUT/'protocol_metrics_10mV.csv',index=False)
print(g.to_string(index=False))
