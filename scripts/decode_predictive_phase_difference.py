"""Explainable compass decoder for the original 512-cell predictive population.

Learn only linear spatial-phase readouts. At inference, cancel current phase
from predictive phase and solve the known grid wave-vector geometry. Position,
direction, and future trajectory are NOT provided to the inference function.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from codes import GridPopulation


class PhaseDifferenceDecoder:
    def __init__(self,population,modules=(1,2,3)):
        self.pop=population
        self.modules=tuple(modules)
        self.readouts={}
        self.indices={}
        self.geometry=[]
        for m in self.modules:
            k=population.K[np.flatnonzero(population.module==m)[0]].cpu().numpy()
            if population.pgc_shift*np.linalg.norm(k,axis=-1).max()>=np.pi:
                raise ValueError('Selected module aliases the full20cm direction shift')
            self.geometry.append(population.pgc_shift*k)
            for label,pool in [('current',population.gc_idx),('predictive',population.pgc_idx)]:
                self.indices[m,label]=pool[population.module[pool]==m]
        self.geometry=np.concatenate(self.geometry)
        self.inverse=np.linalg.pinv(self.geometry)

    def fit(self,rates,positions,directions):
        for m in self.modules:
            k=self.pop.K[np.flatnonzero(self.pop.module==m)[0]].cpu().numpy()
            for label in ('current','predictive'):
                x=positions+(self.pop.pgc_shift*directions if label=='predictive' else 0)
                phase=x@k.T
                target=np.concatenate((np.cos(phase),np.sin(phase)),axis=-1)
                features=np.column_stack((rates[:,self.indices[m,label]],np.ones(len(rates))))
                # Ordinary least squares, independent calibration set.
                self.readouts[m,label]=np.linalg.lstsq(features,target,rcond=1e-6)[0]
        return self

    def predict(self,rates):
        differences=[]
        for m in self.modules:
            z={}
            for label in ('current','predictive'):
                features=np.column_stack((rates[:,self.indices[m,label]],np.ones(len(rates))))
                y=features@self.readouts[m,label]
                z[label]=y[:,:3]+1j*y[:,3:]
            differences.append(np.angle(z['predictive']*z['current'].conj()))
        phase_difference=np.concatenate(differences,axis=-1)
        return phase_difference@self.inverse.T


def load_decoder(seed,folder=ROOT/'results/ring_heading/phase_decoder'):
    pop=GridPopulation(n_cells=512,pgc_frac=.33,seed=seed,device='cpu')
    decoder=PhaseDifferenceDecoder(pop)
    with np.load(Path(folder)/f'seed{seed}.npz') as archive:
        for m in decoder.modules:
            for label in ('current','predictive'):
                decoder.readouts[m,label]=archive[f'm{m}_{label}_readout']
    return decoder


def experiment(seed,out,n_train=4096,n_test=4096):
    rng=np.random.default_rng(seed+59713)
    pop=GridPopulation(n_cells=512,pgc_frac=.33,seed=seed,device='cpu')
    def samples(n,offset=0):
        pos=rng.uniform(-1.25,1.25,(n,2)).astype(np.float32)+offset
        angle=rng.uniform(-np.pi,np.pi,n)
        direction=np.column_stack((np.cos(angle),np.sin(angle))).astype(np.float32)
        return pos,direction
    def rates(pos,direction,scale=1):
        return pop.activity(torch.from_numpy(pos),torch.from_numpy(direction),shift_scale=scale).numpy()
    pos,direction=samples(n_train)
    decoder=PhaseDifferenceDecoder(pop).fit(rates(pos,direction),pos,direction)
    results=[]
    archives={}
    for offset in (0.,10.):
        pos,direction=samples(n_test,offset)
        actual=np.arctan2(direction[:,1],direction[:,0])
        for scale in (1.,0.,-1.,.5):
            code=rates(pos,direction,scale)
            for noise in (0.,.01,.03):
                noisy=np.clip(code+noise*rng.standard_normal(code.shape),0,1)
                decoded=decoder.predict(noisy)
                magnitude=np.linalg.norm(decoded,axis=1)
                estimate=np.arctan2(decoded[:,1],decoded[:,0])
                error=np.arctan2(np.sin(estimate-actual),np.cos(estimate-actual))
                expected=scale*direction
                row=dict(spatial_offset_m=offset,shift_scale=scale,channel_noise_sd=noise,
                    direction_mae_deg=float(np.mean(np.abs(error))*180/np.pi),
                    decoded_vector_magnitude=float(np.mean(magnitude)),
                    vector_rmse_against_scaled_direction=float(np.sqrt(np.mean((decoded-expected)**2))),
                    fraction_vector_magnitude_above_0_1=float(np.mean(magnitude>.1)))
                results.append(row)
                if noise==0 and offset==0:
                    archives[f'scale{scale}_decoded']=decoded
                    archives['true_direction']=direction
    result=dict(seed=seed,n_cells=512,predictive_cells=pop.n_pgc,n_train=n_train,n_test=n_test,
        modules=list(decoder.modules),inference_inputs='512 population rates only',
        learned='Linear current/predictive spatial-phase readouts; fitted using calibration position labels.',
        fixed='Complex phase cancellation and inversion of known grid wave-vector geometry.',rows=results)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/f'seed{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    for (m,label),weights in decoder.readouts.items():
        archives[f'm{m}_{label}_readout']=weights
    np.savez_compressed(out/f'seed{seed}.npz',**archives)
    print(json.dumps(dict(seed=seed,rows=[r for r in results if r['channel_noise_sd']==0]),indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[1])
    parser.add_argument('--out',type=Path,default=ROOT/'results/ring_heading/phase_decoder')
    args=parser.parse_args()
    torch.set_num_threads(2)
    for seed in args.seeds:
        experiment(seed,args.out)
