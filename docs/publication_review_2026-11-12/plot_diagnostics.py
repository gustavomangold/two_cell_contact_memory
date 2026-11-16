"""Standalone figure from derived CSVs; no simulations or dataset changes."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'finite_size_diagnostics.csv').open()))
boots=list(csv.DictReader((root/'peak_grid_bootstrap.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
colors=plt.get_cmap('viridis')(np.linspace(.1,.85,4))
for n,color in zip([8,32,128,512],colors):
    group=sorted([r for r in rows if int(r['N'])==n],key=lambda r:float(r['tau']))
    tau=np.array([float(r['tau']) for r in group])
    mean=np.array([float(r['M']) for r in group]);sem=np.array([float(r['M_sem']) for r in group])
    axes[0,0].plot(tau,mean,'o-',label=f'N={n}',color=color,ms=4)
    axes[0,0].fill_between(tau,mean-sem,mean+sem,color=color,alpha=.12)
axes[0,0].set_xscale('symlog',linthresh=1e-5)
axes[0,0].set(xlabel='Contact-memory time, τ',ylabel='Local demixing index, M',ylim=(0,1))
axes[0,0].legend(frameon=False)
group=sorted([r for r in rows if int(r['N'])>=4 and float(r['tau'])==.001],key=lambda r:int(r['N']))
n=np.array([int(r['N']) for r in group])
axes[0,1].errorbar(n,[float(r['psi_lowq4']) for r in group],
                   yerr=[float(r['psi_lowq4_sem']) for r in group],fmt='o-',capsize=2,label='Measured')
axes[0,1].plot(n,[float(r['psi_lowq4_shuffled']) for r in group],'--',color='#64748b',label='Balanced random-label expectation')
axes[0,1].set(xscale='log',yscale='log',xlabel='Particle number, N',ylabel='Normalized composition power, Ψₗₒw q')
axes[0,1].legend(frameon=False,fontsize=8)
for typ,color,label in [('green','#198754','Green'),('red','#d9534f','Red')]:
    axes[1,0].errorbar(n,[float(r[f'largest_{typ}_fraction']) for r in group],
                       yerr=[float(r[f'largest_{typ}_fraction_sem']) for r in group],
                       fmt='o-',color=color,capsize=2,label=label)
axes[1,0].set(xscale='log',xlabel='Particle number, N',ylabel='Largest same-type cluster / type population',ylim=(0,1))
axes[1,0].legend(frameon=False)
selection=[next(float(b['bootstrap_grid_maximum_frequency']) for b in boots if int(b['N'])==x and float(b['tau'])==.001) for x in n]
axes[1,1].plot(n,selection,'o-',color='#7c3aed')
axes[1,1].set(xscale='log',xlabel='Particle number, N',ylabel='Bootstrap selection frequency for τ=0.001\n(on the sampled grid)',ylim=(0,1.05))
for ax in axes.flat:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
for ax in [axes[0,1],axes[1,0],axes[1,1]]:
    ax.set_xticks(n,labels=[str(x) for x in n])
fig.savefig(root/'finite_size_diagnostics.png',dpi=200)
fig.savefig(root/'finite_size_diagnostics.pdf')
plt.close(fig)
print('Saved finite_size_diagnostics.png and .pdf')
