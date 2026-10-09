"""Isolated corrected NARX experiments; never edits the vendored package or old runs.

The same canonical lag semantics, least-squares estimator and simulation pipeline
apply to every configuration. Original operator and mutation schedules are kept.
Equal-call arm: 90 offspring and mandatory evaluation of every offspring, so all
configurations perform exactly 100 + 49*90 = 4510 fitness calls, failures included.
Term/node ablations use Pareto selection and the same equal-call protocol.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from copy import deepcopy
from functools import lru_cache

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from energy_freerun import COMMON, examples, nrmse
from green_mggp import GreenMGGP, arithmetic_cost, canonical_terms, gene_monomial, make_cost, COST_SCENARIOS
from mggp import MGGP
from deap.tools import sortNondominated
import numpy as np

VERSION='corrected-v1'

def factors(mono):
    return [(v[0],s+(v.startswith('y')),power) for (v,s),power in mono]

def terms_cost(ind):
    return float(len(canonical_terms(ind)))

def nodes_cost(ind):
    # Representation-level comparator: duplicated genes count, as standard node count does.
    return float(sum(len(g) for g in ind))

@lru_cache(maxsize=100000)
def layout(gene_terms):
    n=max((lag for m in gene_terms for _,lag,_ in factors(m)),default=0)
    terms=tuple(dict.fromkeys(gene_terms))
    # Straight-line scalar simulator: no circular buffers or divergence masking.
    expressions=[]
    for i,m in enumerate(terms):
        fs=[f'{v}[k-{lag}]' for v,lag,p in factors(m) for _ in range(p)]
        expressions.append('*'.join([f'coef[{i+1}]']+fs))
    expression=' + '.join(['coef[0]']+expressions)
    src=f'def simulate(y0,u,coef,n_samples):\n    y=list(y0)\n    for k in range({n},n_samples):\n        y.append({expression})\n    return y\n'
    namespace={};exec(src,namespace)
    return n,terms,namespace['simulate'],src

def fit(ind,y,u):
    gene_terms=tuple(gene_monomial(g) for g in ind)
    n,terms,sim,source=layout(gene_terms)
    y=np.ravel(y);u=np.ravel(u);N=len(y)
    if n>=N: raise ValueError('not enough samples')
    matrix=np.ones((N-n,len(ind)+1))
    for i,m in enumerate(gene_terms):
        for v,lag,p in factors(m):
            matrix[:,i+1]*=(y if v=='y' else u)[n-lag:N-lag]**p
    theta=np.linalg.lstsq(matrix,y[n:],rcond=None)[0]
    coef=[float(theta[0])]+[float(sum(theta[j+1] for j,g in enumerate(gene_terms) if g==m)) for m in terms]
    return dict(lag=n,terms=terms,theta=theta,coef=coef,simulate=sim,source=source)

def predict(fitted,y,u,horizon=None):
    y=np.ravel(y);u=np.ravel(u);n=fitted['lag'];sim=fitted['simulate'];coef=fitted['coef']
    if horizon is None:
        yp=np.asarray(sim(y[:n],u.tolist(),coef,len(y)))
        return yp[n:],y[n:]
    size=n+horizon;windows=len(y)//size
    if windows==0: raise ValueError('no complete windows')
    out=[]
    for w in range(windows):
        start=w*size
        out.extend(sim(y[start:start+n],u[start:start+size].tolist(),coef,size))
    return np.asarray(out),y[:windows*size]

def term_str(m):
    return '*'.join((f'{v}(k-{lag})' if lag else f'{v}(k)') for v,lag,p in factors(m) for _ in range(p))

class Corrected:
    def __init__(self,*args,equal_calls=False,**kwargs):
        self.equal_calls=equal_calls;self.eval_count=0;self.trace=[];self.failure_count=0
        super().__init__(*args,**kwargs)
    def evaluation(self,ind):
        self.eval_count+=1
        try:
            f=fit(ind,self.outputs,self.inputs)
            yp,yd=predict(f,self.outputs,self.inputs,self.k)
            error=float(np.sqrt(np.mean((yd-yp)**2)))
            if not math.isfinite(error): raise ValueError('non-finite simulation')
            ind.theta=list(f['theta'])
            return (error,float(self.new_evaluation(ind))) if self.new_evaluation else (error,)
        except (ValueError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError):
            self.failure_count+=1
            return (np.inf,np.inf) if self.new_evaluation else (np.inf,)
    def _record(self):
        seen=set();points=[];by_j={}
        for m in self._pop:
            j=float(m.fitness.values[0]);c=arithmetic_cost(m,self.report_rho)
            by_j.setdefault(j,set()).add(c)
            key=canonical_terms(m)
            if math.isfinite(j) and key not in seen:
                seen.add(key);points.append(dict(J=j/float(np.std(self.outputs)),C=c))
        self.trace.append(dict(front=[[p['C'],p['J']] for p in nd(points)],distinct=len(seen),
            sig=hashlib.sha256(repr([[str(g) for g in m] for m in self._pop]).encode()).hexdigest(),
            tie=any(len(cs)>1 for cs in by_j.values()),n_inf=sum(not math.isfinite(m.fitness.values[0]) for m in self._pop),evaluations=self.eval_count))
    def step(self,gen_number):
        is_green=isinstance(self,GreenMGGP)
        # Native Green keeps its original lambda=100 and fitness reuse.
        if is_green and not self.equal_calls:
            super().step(gen_number);self._record();return
        if is_green:
            pop=[ind for ind in self._pop if np.all(np.isfinite(ind.fitness.values))] or self._pop
            self._assign_rank_crowding(pop)
            offspring=[deepcopy(m) for m in self._crowded_tournament(pop,self.populationSize-self._hofSize)]
        else:
            offspring=[deepcopy(m) for m in self._toolbox.select(self._pop,self.populationSize-self._hofSize)]
        for i in range(0,len(offspring)-1,2):
            if np.random.random()<self.crossoverRate:
                cross=random.choice(self._crossList)
                offspring[i],offspring[i+1]=cross.cross(offspring[i],offspring[i+1])
                self._delAttr(offspring[i]);self._delAttr(offspring[i+1])
        for i in range(len(offspring)):
            if np.random.random()<self.mutationRate:
                mut=random.choice(self._mutList);offspring[i],=mut.mutate(offspring[i]);self._delAttr(offspring[i])
        candidates=offspring if self.equal_calls else [m for m in offspring if not m.fitness.valid]
        for m in candidates: m.fitness.values=self.evaluation(m)
        self._pop=self._survive(self._pop+offspring,self.populationSize) if is_green else self._hof.items+offspring
        self._hof.update(self._pop)
        self._logbook.record(gen=gen_number+1,evals=len(candidates),fitness=self._stats.compile(self._pop))
        self._record()
    def run_corrected(self,rho):
        self.report_rho=rho;self._toolbox.register('map',map)
        self.initPop();self._record();err_before=self._hof[0].fitness.values[0]
        for g in range(1,self.generations):
            self.step(g)
            err_current=self._hof[0].fitness.values[0]
            if g%9==0:
                # Same checkpoints and condition as the original package; handle 0/0 as no improvement.
                ratio=err_before/err_current if err_current else (np.inf if err_before else 1.0)
                if (ratio-1)*100<5: self.mutationRate=min(self.mutationRate+0.1,0.9)
                err_before=err_current
        return sortNondominated(self._pop,len(self._pop),first_front_only=True)[0] if self.new_evaluation else [self._hof[0]]

class CorrectedSO(Corrected,MGGP): pass
class CorrectedGreen(Corrected,GreenMGGP):
    def _survive(self,pool,n):
        # Node count is representation-dependent. Prefer the smallest tree representation
        # of each canonical structure before applying the common duplicate removal.
        if self.new_evaluation is nodes_cost:
            pool=sorted(pool,key=lambda m: nodes_cost(m))
        return super()._survive(pool,n)

def nd(points):
    out=[];best=np.inf
    for p in sorted(points,key=lambda p:(p['C'],p['J'])):
        if p['J']<best: out.append(p);best=p['J']
    return out

def describe(ind,ex,rho):
    d=ex['data'];f=fit(ind,d['y_id'],d['u_id'])
    try:
        yp,yd=predict(f,d['y_val'],d['u_val']);val=nrmse(yd,yp)
    except (ValueError,OverflowError):val=np.inf
    return dict(J=float(ind.fitness.values[0])/float(np.std(d['y_id'])),C=arithmetic_cost(ind,rho),
        objective_cost=float(ind.fitness.values[1]) if len(ind.fitness.values)>1 else None,
        p1=len(f['terms']),sumd=sum(sum(p for _,p in m) for m in f['terms']),val=val,
        lag=f['lag'],terms=sorted(term_str(m) for m in f['terms']),
        theta=[float(t) for t in f['theta']],genes=[str(g) for g in ind])

def run_one(ex,key,cfg,seed,rho,arm,generations=50):
    random.seed(seed);np.random.seed(seed)
    objective={'SO':None,'MO-lex':make_cost(rho),'Green':make_cost(rho),'Pareto-terms':terms_cost,'Pareto-nodes':nodes_cost}[cfg]
    cls=CorrectedSO if cfg in ('SO','MO-lex') else CorrectedGreen
    extra=dict(new_evaluation=objective,equal_calls=arm=='equal_calls')
    if cls is CorrectedGreen: extra['rho']=rho
    d=ex['data'];t0=time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        cwd=os.getcwd();os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mg=cls(inputs=d['u_id'],outputs=d['y_id'],**{**COMMON,'generations':generations},**ex['search'],**extra)
                delivered=mg.run_corrected(rho)
        finally:os.chdir(cwd)
    seen=set();dl=[]
    for m in delivered:
        k=canonical_terms(m)
        if k not in seen and np.all(np.isfinite(m.fitness.values)):
            dl.append(describe(m,ex,rho));seen.add(k)
    return dict(example=key,config=cfg,seed=seed,rho=rho,arm=arm,version=VERSION,runtime=time.perf_counter()-t0,
        evaluations=mg.eval_count,failures=mg.failure_count,delivered=dl,trace=mg.trace,
        pop_front=nd([dict(J=float(m.fitness.values[0])/float(np.std(d['y_id'])),C=arithmetic_cost(m,rho)) for m in mg._pop if math.isfinite(m.fitness.values[0])]),
        front1_sizes=getattr(mg,'front1_sizes',[]))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=30);parser.add_argument('--generations',type=int,default=50)
    parser.add_argument('--out',type=Path,default=ROOT/'Resultados/corrected_v1');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol=dict(version=VERSION,code_sha256=fingerprint,seeds=list(range(args.seeds)),generations=args.generations,
        common=COMMON,arms=['native','equal_calls'],equal_calls=100+(args.generations-1)*90,
        equal_offspring=90,ablation_costs={'terms':'distinct canonical nonconstant terms','nodes':'all tree nodes, including duplicated genes'},
        predictor='actual delayed factors, full initial history; no nan_to_num',estimator='numpy.linalg.lstsq, duplicate gene columns retained',
        energy='deferred; previous measurements cannot be attached to changed selected models')
    info=args.out/'protocol.json'
    if info.exists() and json.loads(info.read_text())!=protocol:raise RuntimeError('Protocol/code changed: use a new output directory')
    info.write_text(json.dumps(protocol,indent=2))
    exs=examples();total=0
    for scenario in ('karatsuba','ops'):
        rho=COST_SCENARIOS[scenario]
        for arm in ('native','equal_calls'):
            folder=args.out/arm/scenario;folder.mkdir(parents=True,exist_ok=True)
            file=folder/'runs.json';runs=json.loads(file.read_text()) if file.exists() else []
            done={(r['example'],r['config'],r['seed']) for r in runs}
            configs=['SO','MO-lex','Green']+(['Pareto-terms','Pareto-nodes'] if arm=='equal_calls' else [])
            for key,ex in exs.items():
                for cfg in configs:
                    for seed in range(args.seeds):
                        if (key,cfg,seed) in done:continue
                        r=run_one(ex,key,cfg,seed,rho,arm,args.generations)
                        if arm=='equal_calls':assert r['evaluations']==protocol['equal_calls'],r['evaluations']
                        runs.append(r);tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(runs));tmp.replace(file);total+=1
                        print(f'{scenario} {arm} {key} {cfg} seed={seed} evals={r["evaluations"]} seconds={r["runtime"]:.2f} saved={len(runs)}',flush=True)
    print('COMPLETED',total,flush=True)
if __name__=='__main__':main()
