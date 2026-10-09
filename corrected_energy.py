"""Prepare/measure corrected-model execution, isolated from historical energy data.

Default: rebuild and check only. --measure requires AC power and explicit
GMGGP_ENERGY_RUN=1. Run with no other agent activity or competing workloads.
The secondary implementation evaluates all genes with NumPy at the intended
lags; it is labelled numpy_allgenes, not the faulty unmodified package predictor.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
from corrected_experiments import ROOT, COMMON, CorrectedSO, fit, predict, gene_monomial, factors, layout
from energy_freerun import examples, select_tau, nrmse, timing_table, measure_block, net_per_sim, paired_ratios, machine_info
from power_meter import PowerMeter

LABELS=[('SO','SO','ops'),('MO-lex','MO-lex','ops'),('Green','Green','ops'),('MO-lex-K','MO-lex','karatsuba'),('Green-K','Green','karatsuba')]
SIMS=['canonical','numpy_allgenes']

def rebuild(genes,ex):
    cwd=os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mg=CorrectedSO(inputs=ex['data']['u_id'],outputs=ex['data']['y_id'],**{**COMMON,'generations':1,'populationSize':2},**ex['search'])
                model=mg.element.buildModelFromList(genes)
                fitted=fit(model,ex['data']['y_id'],ex['data']['u_id'])
        finally:os.chdir(cwd)
    return model,fitted

def make_functions(model,fitted,ex):
    y=np.ravel(ex['data']['y_val']);u=np.ravel(ex['data']['u_val']);n=fitted['lag']
    y0=[float(v) for v in y[:n]];u_list=[float(v) for v in u]
    sim=fitted['simulate'];coef=fitted['coef'];size=len(y)
    def canonical():return sim(y0,u_list,coef,size)[n:]
    gene_terms=tuple(gene_monomial(g) for g in model);theta=fitted['theta']
    def numpy_allgenes():
        out=np.empty(size);out[:n]=y[:n]
        for k in range(n,size):
            values=np.ones(len(gene_terms)+1)
            for i,m in enumerate(gene_terms):
                for v,lag,p in factors(m):
                    for _ in range(p):values[i+1]*=(out if v=='y' else u)[k-lag]
            out[k]=np.dot(values,theta)
        return out[n:]
    return canonical,numpy_allgenes

def prepare(root,arm):
    exs=examples();rows=[]
    for label,cfg,scenario in LABELS:
        file=root/arm/scenario/'runs.json';runs=json.loads(file.read_text())
        selected=[r for r in runs if r['config']==cfg]
        assert len(selected)==90,('Incomplete study',label,len(selected))
        for r in selected:
            s=select_tau(r['delivered']) if r['delivered'] else None
            row=dict(example=r['example'],config=label,seed=r['seed'],cost=scenario,
                C=s['C'] if s else np.nan,C_ops=s['p1']+s['sumd'] if s else np.nan,
                C_K=s['p1']+(729/64)*s['sumd'] if s else np.nan,
                p1=s['p1'] if s else np.nan,sumd=s['sumd'] if s else np.nan,
                val_saved=s['val'] if s else np.nan,genes=tuple(s['genes']) if s else (),terms=tuple(s['terms']) if s else ())
            if not s or not np.isfinite(s['val']):row.update(ok=False,reason='failed validation');rows.append(row);continue
            model,fitted=rebuild(s['genes'],exs[r['example']]);can,other=make_functions(model,fitted,exs[r['example']])
            a=can();b=other();truth=np.ravel(exs[r['example']]['data']['y_val'])[fitted['lag']:]
            val=nrmse(truth,a);diff=float(np.max(np.abs(a-b)))
            assert abs(val-s['val'])<1e-10,(label,r['seed'],val,s['val'])
            assert np.allclose(a,b,rtol=1e-9,atol=1e-9),(label,r['seed'],diff)
            assert len(fitted['terms'])==s['p1'] and sum(sum(p for _,p in m) for m in fitted['terms'])==s['sumd']
            row.update(ok=True,can=can,pkg=other,val_rebuilt=val,max_abs_diff=diff,n_genes=len(model),n_steps=len(a))
            rows.append(row)
    return rows

def measure(rows,out):
    groups=OrderedDict()
    for ex in ['E1','E2','E3']:
        for label,_,_ in LABELS:
            members=[r for r in rows if r['ok'] and r['example']==ex and r['config']==label]
            if not members:raise RuntimeError(f'Empty energy group: {ex} {label}')
            for sim in SIMS:groups[ex,label,sim]=[r['can' if sim=='canonical' else 'pkg'] for r in members]
    rng=random.Random(2027);blocks=[]
    def record(rnd,pos,ex,cfg,sim,fns):
        res=measure_block(fns,10.,100)
        if res['pm_samples']<2 or not np.isfinite(res['pm_J']):raise RuntimeError('powermetrics did not provide valid samples')
        blocks.append(dict(round=rnd,position=pos,example=ex,config=cfg,simulator=sim,n_models=len(fns) if fns else 0,**res))
        pd.DataFrame(blocks).to_csv(out/'energy_blocks.csv',index=False)
        print(rnd,pos,ex,cfg,sim,flush=True)
    for rnd in range(10):
        record(rnd,0,'','','idle',None);order=list(groups);rng.shuffle(order)
        for pos,key in enumerate(order,1):record(rnd,pos,*key,groups[key])
    record(10,0,'','','idle',None)
    return pd.DataFrame(blocks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--measure',action='store_true');p.add_argument('--analyse',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT/'Resultados/corrected_v1');p.add_argument('--arm',default='equal_calls',choices=['native','equal_calls'])
    args=p.parse_args();out=args.root/'energia'/args.arm;out.mkdir(parents=True,exist_ok=True)
    if args.measure:
        if os.environ.get('GMGGP_ENERGY_RUN')!='1':raise SystemExit('Set GMGGP_ENERGY_RUN=1 only for the idle AC-powered session.')
        power=subprocess.check_output(['/usr/bin/pmset','-g','batt'],text=True)
        if 'AC Power' not in power:raise SystemExit('Connect the computer to AC power before measuring.')
        if (out/'energy_blocks.csv').exists():raise SystemExit('Existing measurement preserved; choose a new --root for another session.')
    rows=prepare(args.root,args.arm)
    df=pd.DataFrame([{k:v for k,v in r.items() if k not in ['can','pkg']} for r in rows]);df.to_csv(out/'model_checks.csv',index=False)
    report=dict(n_records=len(rows),n_ok=sum(r['ok'] for r in rows),n_missing=sum(not r['ok'] for r in rows),
        max_sim_diff=max(r['max_abs_diff'] for r in rows if r['ok']),energy_measured=(out/'energy_blocks.csv').exists(),
        secondary_simulator='NumPy evaluation of every gene using corrected lags; not the unmodified package predictor')
    (out/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    if args.measure:
        info=machine_info(os.environ.get('GMGGP_COUNTRY','IRL'));info['simulators']=SIMS
        info['study_protocol_sha256']=hashlib.sha256((args.root/'protocol.json').read_bytes()).hexdigest()
        (out/'measurement_info.json').write_text(json.dumps(info,indent=2))
        timing=timing_table(rows);timing=timing.rename(columns={'t_package':'t_numpy_allgenes'});timing.to_csv(out/'timing_per_model.csv',index=False)
        blocks=measure(rows,out)
    elif args.analyse:
        blocks=pd.read_csv(out/'energy_blocks.csv')
    else:return
    per=net_per_sim(blocks,os.environ.get('GMGGP_COUNTRY','IRL'));per.to_csv(out/'energy_per_sim.csv',index=False)
    paired_ratios(per).to_csv(out/'energy_paired_ratios.csv',index=False)
    per.groupby(['example','config','simulator'])[['E_uJ','CO2_ug','t_us']].median().to_csv(out/'energy_summary.csv')
    from scipy.stats import mannwhitneyu
    from scripts.analyse_corrected import a12,holm
    tests=[]
    for ex in ['E1','E2','E3']:
        for sim in SIMS:
            group=per[(per.example==ex)&(per.simulator==sim)]
            green=group[group.config=='Green'].E_uJ.to_numpy()
            for baseline in ['SO','MO-lex','Green-K']:
                b=group[group.config==baseline].E_uJ.to_numpy()
                tests.append(dict(example=ex,simulator=sim,baseline=baseline,p=float(mannwhitneyu(green,b,alternative='two-sided').pvalue),A12=a12(green,b)))
    stats=pd.DataFrame(tests);stats['p_holm']=holm(stats.p.to_numpy());stats.to_csv(out/'energy_tests.csv',index=False)
    export_energy_artifacts(out,rows,per)
def export_energy_artifacts(out,rows,per):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from energy_freerun import ops_regression
    mpl.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42})
    timing=pd.read_csv(out/'timing_per_model.csv')
    checks=pd.DataFrame([{k:v for k,v in r.items() if k not in ['can','pkg']} for r in rows])
    timing=timing.merge(checks[['example','config','seed','terms']],on=['example','config','seed'])
    distinct=timing.groupby(['example','terms']).median(numeric_only=True).reset_index()
    distinct['canonical_ns']=1e9*distinct.t_canonical/distinct.n_steps
    calibration=ops_regression(distinct.dropna(subset=['canonical_ns']),'canonical_ns')
    (out/'timing_calibration.json').write_text(json.dumps(calibration,indent=2))
    ratios=paired_ratios(per)
    colors={'SO':'#555555','MO-lex':'#E69F00','Green':'#009E73','MO-lex-K':'#E69F00','Green-K':'#009E73'}
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.5))
    for ax,sim in zip(axes,SIMS):
        for i,(label,_,_) in enumerate(LABELS):
            values=[ratios[(ratios.example==ex)&(ratios.simulator==sim)&(ratios.config==label)].ratio.to_numpy() for ex in ['E1','E2','E3']]
            med=np.array([np.median(v) for v in values]);lo=np.array([np.percentile(v,25) for v in values]);hi=np.array([np.percentile(v,75) for v in values])
            ax.errorbar(np.arange(3)+(i-2)*.12,med,yerr=[med-lo,hi-med],fmt='o',color=colors[label],mfc='white' if label.endswith('-K') else colors[label],label=label)
        ax.axhline(1,color='0.7',lw=.6);ax.set_xticks(range(3),['E1','E2','E3']);ax.set_title(sim);ax.set_ylabel('Energy per simulation / SO')
    axes[1].legend(frameon=False,fontsize=6);fig.tight_layout();fig.savefig(out/'fig_energy.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.2))
    for ax,cost in zip(axes[:2],['C_K','C_ops']):
        ax.scatter(distinct[cost],distinct.canonical_ns,s=8,c='#009E73');ax.set_xlabel(r'$C_{\rho_K}$' if cost=='C_K' else r'$C_1$');ax.set_ylabel('Canonical ns/sample')
    axes[2].scatter(distinct.C_ops,1e9*distinct.t_numpy_allgenes/distinct.n_steps,s=8,c='#555555');axes[2].set_xlabel(r'$C_1$');axes[2].set_ylabel('NumPy ns/sample')
    fig.tight_layout();fig.savefig(out/'fig_exec_time.pdf');plt.close(fig)
    table=per.groupby(['example','config','simulator'])[['t_us','E_uJ','CO2_ug']].median().reset_index()
    from scripts.format_energy_table import format_energy_table
    tex=format_energy_table(per)
    (out/'table_energy.tex').write_text(tex)
    macro=[r'% Corrected energy only; do not combine with historical macros.']
    mapping={'E1':'EB','E2':'EC','E3':'ED'}
    for ex in mapping:
        v=ratios[(ratios.example==ex)&(ratios.simulator=='canonical')&(ratios.config=='Green')].ratio.median()
        macro.append(r'\newcommand{\Corrected'+mapping[ex]+r'EnergyRatio}{'+f'{v:.3f}'+'}')
    info=json.loads((out/'measurement_info.json').read_text())
    blocks=pd.read_csv(out/'energy_blocks.csv');idle=blocks[blocks.simulator=='idle']
    tests=pd.read_csv(out/'energy_tests.csv')
    values={'CorrectedMulAdd':f"{calibration['mul_over_add']:.3f}",
        'CorrectedRtwoOps':f"{calibration['R2_ops']:.4f}",
        'CorrectedRtwoOne':f"{calibration['R2_one']:.4f}",
        'CorrectedRtwoK':f"{calibration['R2_K']:.4f}",
        'CorrectedTadd':f"{calibration['t_add']:.2f}",
        'CorrectedTmul':f"{calibration['t_mul']:.2f}",
        'CorrectedNStructures':str(calibration['n']),
        'CorrectedCI':f"{info['carbon_intensity_g_per_kWh']:.3f}",
        'CorrectedIdleMin':f"{idle.pm_W.min():.3f}",
        'CorrectedIdleMax':f"{idle.pm_W.max():.3f}"}
    for ex,tag in mapping.items():
        for sim,st in [('canonical','Can'),('numpy_allgenes','Numpy')]:
            v=ratios[(ratios.example==ex)&(ratios.simulator==sim)&(ratios.config=='Green')].ratio.median()
            values['Corrected'+tag+st+'Reduction']=f'{100*(1-v):.1f}'
            pv=tests[(tests.example==ex)&(tests.simulator==sim)&(tests.baseline=='SO')].p_holm.iloc[0]
            values['Corrected'+tag+st+'P']=f'{pv:.3f}'
    v=ratios[(ratios.example=='E1')&(ratios.simulator=='numpy_allgenes')&(ratios.config=='Green')].ratio.median()
    values['CorrectedEBNumpyIncrease']=f'{100*(v-1):.1f}'
    for name,value in values.items():
        macro.append('\\newcommand{\\'+name+'}{'+value+'}')
    (out/'energy_macros.tex').write_text('\n'.join(macro)+'\n')

if __name__=='__main__':main()
