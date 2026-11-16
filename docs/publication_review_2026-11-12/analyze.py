"""Read existing paper sweep only; derive finite-size diagnostics, no simulations."""
from pathlib import Path
import csv,json
import numpy as np

ROOT=Path('/home/gustavo/Documents/doutorado/repositorios/two_cell_contact_memory')
OUT=Path(__file__).resolve().parent
data_root=ROOT/'data/exactly-like-paper'
runs=[]
for folder in sorted(data_root.glob('N*_phi*_D*_tau*')):
    meta=json.loads((folder/'settings.json').read_text())
    for path in sorted(folder.glob('sample_*.npz')):
        with np.load(path) as data:
            n=int(data['N']); L=float(data['L']); tau=float(data['tau'])
            positions=data['final_positions']; signs=np.r_[np.ones(n//2),-np.ones(n//2)]
            z=np.exp(2j*np.pi*positions/L)
            psi=float((np.abs((signs[:,None]*z).sum(axis=0))**2).mean()/n**2)
            shuffle=float(((n*n-np.abs(z.sum(axis=0))**2)/(n-1)).mean()/n**2)
            # Include diagonal modes so an oblique domain is not invisible.
            q=np.array([[1,0],[0,1],[1,1],[1,-1]])
            low=np.exp(2j*np.pi*(positions@q.T)/L)
            psi_lowq4=float((np.abs((signs[:,None]*low).sum(axis=0))**2).mean()/n**2)
            shuffle_lowq4=float(((n*n-np.abs(low.sum(axis=0))**2)/(n-1)).mean()/n**2)
            delta=positions[:,None]-positions[None,:]
            delta-=L*np.floor(delta/L+.5)
            same=(signs[:,None]==signs[None,:])
            i,j=np.nonzero(np.triu((np.sum(delta**2,axis=-1)<1.1**2)&same,1))
            parents=np.arange(n)
            def find(x):
                while parents[x]!=x:
                    parents[x]=parents[parents[x]];x=parents[x]
                return x
            for a,b in zip(i,j):
                a,b=find(a),find(b)
                if a!=b:parents[a]=b
            sizes=np.bincount([find(x) for x in range(n)],minlength=n)
            largest_green=max(sizes[find(x)] for x in range(n//2))/(n//2)
            largest_red=max(sizes[find(x)] for x in range(n//2,n))/(n//2)
            M=float(data['final_metrics'][0]); baseline=(n/2-1)/(n-1)
            runs.append(dict(N=n,tau=tau,seed=int(data['seed']),D=float(data['D']),L=L,
                             M=M,M_corrected=(M-baseline)/(1-baseline),
                             coverage=float(data['final_metrics'][3]),psi=psi,psi_shuffled=shuffle,
                             psi_lowq4=psi_lowq4,psi_lowq4_shuffled=shuffle_lowq4,
                             psi_excess=psi-shuffle,largest_green_fraction=largest_green,
                             largest_red_fraction=largest_red,source_hash=str(data['source_hash']),
                             neighbor_method=str(data['neighbor_method']) if 'neighbor_method' in data else meta.get('neighbor_method','unknown')))
def mean_sem(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    return (float(values.mean()) if len(values) else np.nan,
            float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else np.nan,len(values))
with (OUT/'per_seed_diagnostics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(runs[0]));writer.writeheader();writer.writerows(runs)
summary=[]
metrics=['M','M_corrected','coverage','psi','psi_shuffled','psi_excess','psi_lowq4','psi_lowq4_shuffled','largest_green_fraction','largest_red_fraction']
for n,tau in sorted(set((r['N'],r['tau']) for r in runs)):
    group=[r for r in runs if r['N']==n and r['tau']==tau]
    row=dict(N=n,tau=tau,total_runs=len(group),neighbor_method=group[0]['neighbor_method'])
    for metric in metrics:
        mean,sem,count=mean_sem([r[metric] for r in group])
        row[metric]=mean;row[metric+'_sem']=sem;row[metric+'_runs']=count
    summary.append(row)
with (OUT/'finite_size_diagnostics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
rng=np.random.default_rng(20261112);boots=[]
for n in sorted({r['N'] for r in runs if r['N']>=4}):
    group=[r for r in runs if r['N']==n]
    taus=sorted({r['tau'] for r in group}); seeds=sorted({r['seed'] for r in group})
    lookup={(r['seed'],r['tau']):r['M'] for r in group}
    matrix=np.array([[lookup.get((seed,tau),np.nan) for tau in taus] for seed in seeds])
    choices=rng.integers(0,len(seeds),size=(5000,len(seeds)))
    bootstrap_means=np.nanmean(matrix[choices],axis=1)
    best=np.argmax(bootstrap_means,axis=1)
    for index,tau in enumerate(taus):
        boots.append(dict(N=n,tau=tau,bootstrap_grid_maximum_frequency=float((best==index).mean()),bootstrap_replicates=5000))
with (OUT/'peak_grid_bootstrap.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(boots[0]));writer.writeheader();writer.writerows(boots)
print('Analyzed',len(runs),'saved runs;',len(summary),'parameter points; no simulations run.')
print('Source hashes:',sorted({r['source_hash'] for r in runs}))
for row in summary:
    if row['tau']==.001:
        b=next((b['bootstrap_grid_maximum_frequency'] for b in boots if b['N']==row['N'] and b['tau']==.001),None)
        print('N',row['N'],'M',round(row['M'],4),'psi',round(row['psi'],5),'shuffled',round(row['psi_shuffled'],5),
              'green_max/Ngreen',round(row['largest_green_fraction'],3),'bootstrap peak .001',b)
v=1000.;Dr=360000.;Dt=1.
continuous=Dt+v*v/(2*Dr)
for h in [1e-5,5e-6,2.5e-6,1.25e-6]:
    discrete=Dt+v*v*h/4/np.tanh(Dr*h/2)
    print('Free-ABP dt',h,'EM Deff',discrete,'continuous',continuous,'ratio',discrete/continuous)
