"""Presentation-only energy table from saved per-round measurements."""
import pandas as pd

def format_energy_table(per):
    summary=per.groupby(['example','config','simulator'])[['t_us','E_uJ','CO2_ug']].median()
    lines=[r'\begin{table}[!htbp]',r'\centering',r'\caption{Execution of the corrected selected models: median time, net energy and estimated carbon per simulation over ten measurement rounds. Canonical execution evaluates distinct monomials; all-gene NumPy execution uses corrected delays. Suffix K denotes searches under $\rho_K$.}',r'\label{tab:corrected_energy}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llrrrrrr}',r'\toprule',r' & & \multicolumn{3}{c}{Canonical polynomial} & \multicolumn{3}{c}{All-gene NumPy} \\',r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',r'Ex. & Method & Time ($\mu$s) & Energy ($\mu$J) & CO$_2$e ($\mu$g) & Time ($\mu$s) & Energy ($\mu$J) & CO$_2$e ($\mu$g) \\',r'\midrule']
    for i,ex in enumerate(['E1','E2','E3']):
        if i:lines.append(r'\midrule')
        for cfg in ['SO','MO-lex','Green','MO-lex-K','Green-K']:
            vals=[]
            for sim in ['canonical','numpy_allgenes']:
                vals.extend(f'{summary.loc[(ex,cfg,sim),key]:.3g}' for key in ['t_us','E_uJ','CO2_ug'])
            lines.append(' & '.join([ex,cfg]+vals)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    return '\n'.join(lines)+'\n'
