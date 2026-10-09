"""Read-only diagnostic of the full-column-rank assumption on selected models.

Uses distinct canonical columns and normalises their scales before numerical
rank detection. Does not change fitting, selection or any saved search outcome.
"""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from corrected_energy import rebuild
from corrected_experiments import factors
from energy_freerun import examples,select_tau
out=ROOT/'Resultados/corrected_v1'
runs=json.loads((out/'equal_calls/ops/runs.json').read_text());exs=examples();rows=[]
for r in runs:
    if not r['delivered']:continue
    s=select_tau(r['delivered']);ex=exs[r['example']];_,f=rebuild(s['genes'],ex)
    y=np.ravel(ex['data']['y_id']);u=np.ravel(ex['data']['u_id']);n=f['lag'];N=len(y)
    matrix=np.ones((N-n,len(f['terms'])+1))
    for i,m in enumerate(f['terms']):
        for v,lag,p in factors(m):matrix[:,i+1]*=(y if v=='y' else u)[n-lag:N-lag]**p
    norms=np.linalg.norm(matrix,axis=0);scaled=matrix/np.where(norms>0,norms,1)
    singular=np.linalg.svd(scaled,compute_uv=False);tol=max(scaled.shape)*np.finfo(float).eps*singular[0]
    rank=int(np.sum(singular>tol));rows.append(dict(example=r['example'],config=r['config'],seed=r['seed'],n_columns=matrix.shape[1],rank=rank,full_rank=rank==matrix.shape[1],smallest_scaled_singular_value=singular[-1],rank_tolerance=tol))
df=pd.DataFrame(rows);df.to_csv(out/'selected_rank_checks.csv',index=False)
summary=df.groupby(['example','config']).full_rank.agg(['sum','count']);summary.to_csv(out/'selected_rank_summary.csv');print(summary)
