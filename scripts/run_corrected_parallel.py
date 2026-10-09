"""Resume the frozen corrected protocol using independent worker processes.

Only orchestration differs from corrected_experiments.main. This helper runs only the final ops/equal_calls folder; do not run it if the serial runner has already reached that folder. Seeds are reset in
run_one, and each worker is single-threaded. Parent alone writes result files.
Runtime measurements under concurrent load are diagnostic, not benchmarks.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/green-mpl')
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import corrected_experiments as study
EXS=None

def task(args):
    global EXS
    if EXS is None:EXS=study.examples()
    scenario,arm,key,cfg,seed=args
    result=study.run_one(EXS[key],key,cfg,seed,study.COST_SCENARIOS[scenario],arm)
    result['runtime_concurrent']=True
    return scenario,arm,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--arm',choices=['native','equal_calls'],default='equal_calls');args=p.parse_args()
    out=ROOT/'Resultados/corrected_v1';protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['code_sha256']==hashlib.sha256((ROOT/'corrected_experiments.py').read_bytes()).hexdigest()
    collections={};tasks=[]
    for scenario in ['karatsuba','ops']:
        for arm in ['native','equal_calls']:
            if (scenario,arm)!=('ops',args.arm):continue
            folder=out/arm/scenario;folder.mkdir(parents=True,exist_ok=True)
            f=folder/'runs.json';runs=json.loads(f.read_text()) if f.exists() else []
            collections[scenario,arm]=runs
            seen={(r['example'],r['config'],r['seed']) for r in runs}
            cfgs=['SO','MO-lex','Green']+(['Pareto-terms','Pareto-nodes'] if arm=='equal_calls' else [])
            tasks.extend((scenario,arm,key,cfg,seed) for key in ['E1','E2','E3'] for cfg in cfgs for seed in protocol['seeds'] if (key,cfg,seed) not in seen)
    (out/('execution_'+args.arm+'.json')).write_text(json.dumps(dict(workers=args.workers,note='Independent single-threaded workers; runtime under concurrent load is not a benchmark.',resumed_missing=len(tasks)),indent=2))
    print('Remaining runs:',len(tasks),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for scenario,arm,r in pool.map(task,tasks,chunksize=1):
            if arm=='equal_calls':assert r['evaluations']==4510
            runs=collections[scenario,arm];runs.append(r)
            # Checkpoint every completed run; sorting makes output independent of execution order.
            runs.sort(key=lambda r:(r['example'],r['config'],r['seed']))
            f=out/arm/scenario/'runs.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(runs));tmp.replace(f)
            print(scenario,arm,r['example'],r['config'],r['seed'],'saved',len(runs),flush=True)
    print('COMPLETED',flush=True)
if __name__=='__main__':main()
