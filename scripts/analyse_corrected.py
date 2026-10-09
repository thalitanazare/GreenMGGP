"""Aggregate corrected experiments without starting searches or measuring power."""
from pathlib import Path
import argparse
import json
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energy_freerun import select_tau

def hv(points,ref=1.):
    best=ref;area=0.
    for c,j in sorted(points):
        if math.isfinite(j) and c<1 and j<best:area+=(1-c)*(best-j);best=j
    return area

def holm(values):
    out=np.empty(len(values));running=0
    for i,k in enumerate(np.argsort(values)):
        running=max(running,(len(values)-i)*values[k]);out[k]=min(1.,running)
    return out

def a12(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.mean((a[:,None]>b[None,:])+0.5*(a[:,None]==b[None,:])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'Resultados/corrected_v1');args=p.parse_args()
    protocol=json.loads((args.root/'protocol.json').read_text())
    expected=len(protocol['seeds'])*3
    allrows=[];alltests=[];sensitivity=[];theory=[];green_checks=[];taurows=[];energy=[]
    truth={'y(k-1)','u(k)','u(k)*y(k-1)'}
    for arm in ['native','equal_calls']:
        for scenario in ['karatsuba','ops']:
            folder=args.root/arm/scenario;runs=json.loads((folder/'runs.json').read_text())
            configs=['SO','MO-lex','Green']+(['Pareto-terms','Pareto-nodes'] if arm=='equal_calls' else [])
            assert len(runs)==expected*len(configs),(arm,scenario,len(runs))
            rows=[]
            for r in runs:
                dl=r['delivered'];ref=(5 if r['example']!='E3' else 8)*(1+r['rho']*(4 if r['example']!='E3' else 8))
                base={k:r[k] for k in ['example','config','seed','evaluations','failures','runtime']}
                selected=select_tau(dl) if dl else None
                # Validation failures remain missing; hypervolume excludes them naturally.
                row=dict(**base,arm=arm,scenario=scenario,n_delivered=len(dl),HV_id=hv([(m['C']/ref,m['J']) for m in dl]),
                    HV_val=hv([(m['C']/ref,m['val']) for m in dl]),HV_pop=hv([(m['C']/ref,m['J']) for m in r['pop_front']]),
                    sel_C=selected['C'] if selected else np.nan,sel_J=selected['J'] if selected else np.nan,
                    sel_val=selected['val'] if selected and math.isfinite(selected['val']) else np.nan,
                    best_val=min((m['val'] for m in dl if math.isfinite(m['val'])),default=np.nan),
                    recovered=bool(selected and truth<=set(selected['terms'])),front_has_truth=any(truth<=set(m['terms']) for m in dl),
                    selected_terms=' + '.join(selected['terms']) if selected else '',validation_failed=not selected or not math.isfinite(selected['val']))
                rows.append(row)
                for error_ref in [1.,.2,.1]:
                    sensitivity.append(dict(arm=arm,scenario=scenario,example=r['example'],config=r['config'],seed=r['seed'],error_reference=error_ref,
                        HV_val=hv([(m['C']/ref,m['val']) for m in dl],error_ref)))
                for tau in [0.,.01,.02,.05,.1,.2,.5,1.]:
                    sel=select_tau(dl,tau) if dl else None
                    taurows.append(dict(arm=arm,scenario=scenario,example=r['example'],config=r['config'],seed=r['seed'],tau=tau,
                        C=sel['C'] if sel else np.nan,val=sel['val'] if sel else np.nan,recovered=bool(sel and truth<=set(sel['terms']))))
                if arm=='equal_calls' and selected and (r['config'] in ['SO','MO-lex','Green']) and (scenario=='ops' or r['config']!='SO'):
                    energy.append(dict(example=r['example'],config=r['config']+('-K' if scenario=='karatsuba' else ''),seed=r['seed'],scenario=scenario,
                        C=selected['C'],val=selected['val'],genes=selected['genes'],lag=selected['lag']))
            df=pd.DataFrame(rows);df.to_csv(folder/'per_run_metrics.csv',index=False)
            group=df.groupby(['example','config'])
            summary=group[['HV_id','HV_val','HV_pop','sel_C','sel_val','best_val','evaluations','failures','runtime','n_delivered']].median()
            summary['recovered_fraction']=group.recovered.mean();summary['front_has_truth_fraction']=group.front_has_truth.mean()
            summary['validation_failures']=group.validation_failed.sum();summary.to_csv(folder/'summary_medians.csv');allrows.extend(rows)
            tests=[]
            for ex in ['E1','E2','E3']:
                for baseline in [c for c in configs if c!='Green']:
                    for metric in ['HV_id','HV_val','HV_pop','sel_C','sel_val']:
                        g=df[(df.example==ex)&(df.config=='Green')][metric].dropna().to_numpy()
                        b=df[(df.example==ex)&(df.config==baseline)][metric].dropna().to_numpy()
                        tests.append(dict(arm=arm,scenario=scenario,example=ex,baseline=baseline,metric=metric,n_green=len(g),n_baseline=len(b),
                            p=float(mannwhitneyu(g,b,alternative='two-sided').pvalue) if len(g)*len(b) else np.nan,A12=a12(g,b) if len(g)*len(b) else np.nan))
            testdf=pd.DataFrame(tests);testdf['p_holm']=holm(testdf.p.fillna(1.).to_numpy());testdf.to_csv(folder/'stats_tests.csv',index=False);alltests.extend(testdf.to_dict('records'))
            for ex in ['E1','E2','E3']:
                so={r['seed']:r for r in runs if r['example']==ex and r['config']=='SO'}
                lex={r['seed']:r for r in runs if r['example']==ex and r['config']=='MO-lex'}
                for seed in protocol['seeds']:
                    a,b=so[seed],lex[seed]
                    div=next((i for i,(x,y) in enumerate(zip(a['trace'],b['trace'])) if x['sig']!=y['sig']),None)
                    theory.append(dict(arm=arm,scenario=scenario,example=ex,seed=seed,lex_identical=div is None,first_div=div,
                        prior_tie=div is not None and div>0 and (a['trace'][div-1]['tie'] or b['trace'][div-1]['tie'])))
                for r in [r for r in runs if r['example']==ex and r['config']=='Green']:
                    ref=(5 if ex!='E3' else 8)*(1+r['rho']*(4 if ex!='E3' else 8))
                    values=[hv([(c/ref,j) for c,j in t['front']]) for t in r['trace']]
                    drops=np.flatnonzero(np.diff(values)<-1e-10)
                    for index in drops:
                        assert r['front1_sizes'][int(index)]>100,('Conditional HV regression',arm,scenario,ex,r['seed'])
                    green_checks.append(dict(arm=arm,scenario=scenario,example=ex,seed=r['seed'],max_U1=max(r['front1_sizes']),capacity_always_holds=max(r['front1_sizes'])<=100,hv_drop_count=len(drops)))
                    assert len(r['trace'])==protocol['generations']
            if arm=='equal_calls':assert (df.evaluations==protocol['equal_calls']).all()
    # SO and cost-independent ablations must have identical populations across weights.
    for arm in ['native','equal_calls']:
        a=json.loads((args.root/arm/'ops/runs.json').read_text());b=json.loads((args.root/arm/'karatsuba/runs.json').read_text())
        b={(r['example'],r['config'],r['seed']):r for r in b}
        for r in a:
            if r['config'] in ['SO','Pareto-terms','Pareto-nodes']:
                q=b[(r['example'],r['config'],r['seed'])]
                assert [t['sig'] for t in r['trace']]==[t['sig'] for t in q['trace']]
    sensitivity_df=pd.DataFrame(sensitivity)
    sensitivity_df.to_csv(args.root/'hv_reference_sensitivity.csv',index=False)
    sens_summary=[]
    for (arm,scenario,ex,ref),group in sensitivity_df.groupby(['arm','scenario','example','error_reference']):
        green=group[group.config=='Green'].HV_val.to_numpy()
        for cfg in group.config.unique():
            if cfg=='Green':continue
            baseline=group[group.config==cfg].HV_val.to_numpy()
            sens_summary.append(dict(arm=arm,scenario=scenario,example=ex,error_reference=ref,baseline=cfg,
                green_median=float(np.median(green)),baseline_median=float(np.median(baseline)),A12=a12(green,baseline)))
    sens_summary=pd.DataFrame(sens_summary);sens_summary.to_csv(args.root/'hv_reference_summary.csv',index=False)
    pd.DataFrame(theory).to_csv(args.root/'theory_checks.csv',index=False)
    pd.DataFrame(green_checks).to_csv(args.root/'green_theory_checks.csv',index=False)
    pd.DataFrame(taurows).to_csv(args.root/'tau_sensitivity.csv',index=False)
    pd.DataFrame(alltests).to_csv(args.root/'stats_tests_all.csv',index=False)
    (args.root/'energy_selected_models.json').write_text(json.dumps(energy,indent=2))
    report=['# Corrected NARX study','',f"Completed {len(allrows)} runs with {len(protocol['seeds'])} seeds per configuration and example.",
        '', 'The equal-call arm uses 4510 fitness evaluations per run (100 initial + 49 × 90 offspring), including unchanged offspring and failed evaluations. Native-arm fitness reuse and offspring counts are retained as a diagnostic. Both use the corrected, common polynomial evaluator.',
        '', 'Corrections use the actual maximum factor delay for estimation and initial history, avoid circular indexing, support delayed mixed products and input-only structures, and retain divergence as failure. Thus these are new experiments, not a correction applied retrospectively to the old search.',
        '', 'Pareto-terms minimises distinct canonical terms; Pareto-nodes minimises all tree nodes. Node-count duplicate removal retains the smallest representation of a canonical structure before survival. Hypervolume and the final τ-rule always use the arithmetic cost of each scenario, so the ablations are assessed in the same error–arithmetic-cost plane.',
        '', '## Median results (primary weight, equal calls)', '']
    df=pd.DataFrame(allrows);part=df[(df.arm=='equal_calls')&(df.scenario=='ops')]
    report.append('```text\n'+part.groupby(['example','config'])[['HV_id','HV_val','sel_C','sel_val','best_val','evaluations']].median().to_string()+'\n```')
    report+=['','Validation failures (excluded from validation-error medians, retained as failures in the raw records):','', '```text\n'+part.groupby(['example','config']).validation_failed.sum().to_string()+'\n```']
    report+=['','## Reference-point sensitivity (equal calls, primary weight; Green versus MO-lex)','', '```text\n'+sens_summary[(sens_summary.arm=='equal_calls')&(sens_summary.scenario=='ops')&(sens_summary.baseline=='MO-lex')][['example','error_reference','green_median','baseline_median','A12']].to_string(index=False)+'\n```']
    report+=['','## Interpretation','', 'On E1, selected-model validation errors are at machine precision; differences below $10^{-14}$, including changes of rank-based effects at tight hypervolume references, should not be interpreted as practical accuracy differences.', 'The corrected selected-model validation errors, particularly on E3, must be assessed independently of the historical results. Low identification error and low arithmetic cost do not establish good free-run generalisation. The best validation error available in a returned set is included as a diagnostic; it is not used to alter the predeclared identification-only selection rule. Treat changes to the headline claims as results to inspect, including any loss of advantage in the corrected experiments or ablations. Holm correction covers all metrics/examples/baselines within an arm and weight, with validation failures excluded from validation-error tests and counted separately. The old manuscript and energy measurements remain historical and must not be combined with these selected models.',
        '', '## Energy deferred','',f'{len(energy)} selected-model records prepared for the five original energy groups. New energy, timing and CO₂e claims require a new session; no new power measurements have been made.']
    validation_ablation=pd.DataFrame(alltests)
    validation_ablation=validation_ablation[(validation_ablation.arm=='equal_calls')&(validation_ablation.scenario=='ops')&(validation_ablation.baseline=='Pareto-terms')&(validation_ablation.metric=='HV_val')]
    if (validation_ablation.p_holm>=0.05).all():
        report+=['','No significant difference in validation hypervolume between Green and Pareto-terms was detected on any of the three examples after the declared Holm correction. The arithmetic objective therefore has not demonstrated a consistent advantage over term count on this indicator in these settings.']
    (args.root/'REPORT.md').write_text('\n'.join(report)+'\n')
    export_figures(part,args.root)
    print('Analysis complete:',args.root,flush=True)

def export_figures(df,folder):
    import matplotlib as mpl
    mpl.use('Agg');import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42})
    configs=['SO','MO-lex','Green','Pareto-terms','Pareto-nodes'];colors=['#555555','#E69F00','#009E73','#0072B2','#CC79A7']
    fig,axes=plt.subplots(2,3,figsize=(7.16,4.5))
    for j,ex in enumerate(['E1','E2','E3']):
        for i,metric in enumerate(['HV_val','sel_C']):
            ax=axes[i,j]
            boxes=ax.boxplot([df[(df.example==ex)&(df.config==c)][metric].dropna() for c in configs],patch_artist=True)
            for patch,color in zip(boxes['boxes'],colors):patch.set_facecolor(color);patch.set_alpha(.65)
            ax.set_xticks(range(1,6),configs,rotation=35,ha='right');ax.set_title(ex)
            if j==0:ax.set_ylabel('Validation hypervolume' if i==0 else 'Selected operations')
    fig.tight_layout();fig.savefig(folder/'fig_corrected_ablation.pdf');plt.close(fig)
    table=df.groupby(['example','config'])[['HV_id','HV_val','sel_C','sel_val','evaluations']].median()
    table=table.rename(columns={'HV_id':'HV (id)','HV_val':'HV (val)','sel_C':'Cost','sel_val':'NRMSE','evaluations':'Evaluations'})
    tex=table.to_latex(float_format=lambda v:f'{v:.3f}',escape=True)
    (folder/'table_corrected_ablation.tex').write_text(r'\begin{table}[t]\centering'+'\n'+r'\caption{Corrected equal-evaluation ablation: medians over 30 runs with $\rho=1$.}'+'\n'+tex+r'\end{table}'+'\n')
if __name__=='__main__':main()
