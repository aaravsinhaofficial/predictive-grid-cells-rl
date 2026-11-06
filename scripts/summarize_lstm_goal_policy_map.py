"""Audit the frozen original-policy steering-map cohort and its heldout fit."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_goal_state_pulse import write_json


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(x,signed_contrast=True):
    x=np.asarray(x,float);n=len(x);mean=float(x.mean())
    ci=float(t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)) if n>1 else 0.
    signs=np.array(list(itertools.product((-1.,1.),repeat=n)))
    p=float((np.abs((signs*x).mean(1))>=abs(mean)-1e-12).mean()) if signed_contrast else None
    return dict(n_seeds=n,mean=mean,ci95=[mean-ci,mean+ci],exact_two_sided_sign_flip_p=p,
                positive_seed_count=int((x>0).sum()),seed_values=x.tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/rl_map_mechanism/policy_map_confirmation')
    args=parser.parse_args(); manifest_path=ROOT/'docs/research/original_rl_goal_policy_map_sources.json'
    manifest=json.loads(manifest_path.read_text())
    for path,digest in manifest['sources'].items():assert sha(ROOT/path)==digest,path
    models,rows,missing,failures=[],[],[],[]
    for seed in range(1,9):
        for context,geometry in enumerate(manifest['contexts']):
            path=args.out/f'context{context}/pgc33_s{seed}.json'
            if not path.exists():missing.append([seed,context]);continue
            model=json.loads(path.read_text());models.append(model)
            assert model['seed']==seed and model['frames_trained']==1_000_000_000
            assert model['original_RL_checkpoint_sha256']==manifest['checkpoint_sha256'][str(seed)]
            assert sha(ROOT/f'models/pgc33/pgc33_lstm_s{seed}_1000M.pt')==manifest['checkpoint_sha256'][str(seed)]
            assert model['source_sha256']==manifest['sources']['scripts/lstm_goal_policy_map.py']
            for key,digest in model['dependencies_sha256'].items():assert digest==manifest['sources'][key]
            assert model['grid_side']==7 and model['heading_count']==64 and model['scene_seed']==989131+context
            np.testing.assert_allclose(model['goal_center'],geometry[:2],atol=1e-7)
            np.testing.assert_allclose(model['arrival_direction'],[np.cos(geometry[2]),np.sin(geometry[2])],atol=1e-7)
            row=dict(seed=seed,context=context,descriptive=seed<3)
            for step in (15,60):
                for field in ('all_translation','all_translation_unweighted','dominant_translation'):
                    fit=model['field_models'][str(step)][field]; prefix=f'{field}_t{step}_'
                    for key in ('native_minus_ordinary_along_arrival_m','native_minus_ordinary_across_arrival_m',
                                'target_model_heldout_improvement_deg'):row[prefix+key]=fit[key]
                    for name in ('vary_target_shared_offset','shared_target_vary_offset'):
                        row[prefix+name+'_heldout_mae_deg']=fit[name]['mean_heldout_angle_error_deg']
                        if not fit[name]['optimizer_success'] or fit[name]['point_bound_reached']:
                            failures.append(dict(seed=seed,context=context,step=step,field=field,model=name,
                                success=fit[name]['optimizer_success'],bound_reached=fit[name]['point_bound_reached']))
            rows.append(row)
    result=dict(status='final' if not missing else 'partial',completed_contexts=len(rows),missing=missing,
                source_manifest_sha256=sha(manifest_path),frozen_sources_and_checkpoints_verified=True,
                optimizer_or_bound_flags=failures,context_rows=rows,seed_rows=[],confirmation={},descriptive={})
    for seed in range(1,9):
        rr=[r for r in rows if r['seed']==seed]
        if len(rr)!=4:continue
        result['seed_rows'].append(dict(seed=seed,descriptive=seed<3,
            **{key:float(np.mean([r[key] for r in rr])) for key in rr[0] if key not in ('seed','context','descriptive')}))
    for name,seeds in (('confirmation',list(range(3,9))),('descriptive',[1,2])):
        selected=[r for r in result['seed_rows'] if r['seed'] in seeds]
        if len(selected)!=len(seeds):continue
        result[name]={key:summary([r[key] for r in selected],signed_contrast=not key.endswith('heldout_mae_deg'))
                      for key in selected[0] if key not in ('seed','descriptive')}
    if not missing:
        primary=result['confirmation']['all_translation_t60_native_minus_ordinary_along_arrival_m']
        fit_quality=result['confirmation']['all_translation_t60_target_model_heldout_improvement_deg']
        primary_flags=[f for f in failures if f['seed']>=3]
        result['positive_primary_displacement_supported']=primary['mean']>0 and primary['exact_two_sided_sign_flip_p']<=.05
        result['effective_target_model_gate_passed']=fit_quality['positive_seed_count']>=5 and fit_quality['mean']>0 and not primary_flags
        result['interpretation']=('A displaced effective target is a supported description of the clamped steering field; not proof of an encoded neuronal goal coordinate.'
                                 if result['effective_target_model_gate_passed'] else
                                 'Report steering-field modulation; the prespecified effective-target interpretation gate failed.')
    write_json(args.out/'summary.json',result)
    if missing:
        print(json.dumps(dict(status='partial',completed=len(rows),missing=missing)));return
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.6),layout='constrained')
    a=axes[0]
    a.add_patch(plt.Circle((0,0),.30,color='#eeeeee',zorder=0))
    a.plot(0,0,'x',color='black',ms=9,mew=2);a.plot(-.29,0,'o',color='#2777b6',ms=8)
    a.plot(-.09,0,'o',color='#e46b2e',ms=8)
    a.annotate('',xy=(-.09,.07),xytext=(-.29,.07),arrowprops=dict(arrowstyle='->',color='#e46b2e',lw=2))
    a.text(-.19,.11,'20 cm',ha='center');a.text(-.29,-.07,'Pickup',ha='center',fontsize=10)
    a.text(-.09,-.12,'Predictive point',ha='center',fontsize=10);a.text(.04,.015,'Goal',ha='left',fontsize=10)
    a.set(xlim=(-.39,.35),ylim=(-.33,.33),aspect='equal',title='Shift the predictive goal inputs')
    a.set_axis_off()
    a=axes[1];key='all_translation_t60_native_minus_ordinary_along_arrival_m'
    for row in result['seed_rows']:
        ispilot=row['seed']<3
        a.scatter(row['seed'],row[key]*100,facecolors='white' if ispilot else '#e46b2e',edgecolors='#888888' if ispilot else '#e46b2e',s=50,zorder=3)
    a.axhline(0,color='.5',lw=.8);a.axhline(20,color='.7',ls=':',lw=1)
    a.axvspan(.5,2.5,color='.95',zorder=0)
    a.set(xticks=list(range(1,9)),xlabel='Trained agent seed',ylabel='Shift along the supplied cue (cm)',title='Effective steering target shifts' if result['effective_target_model_gate_passed'] else 'Target-fit displacement is uncertain')
    a.text(1.5,a.get_ylim()[1],'Pilots',ha='center',va='top',fontsize=9,color='.4')
    a=axes[2]
    xkey='all_translation_t60_vary_target_shared_offset_heldout_mae_deg'; ykey='all_translation_t60_shared_target_vary_offset_heldout_mae_deg'
    for row in result['seed_rows']:
        if row['seed']>=3:a.plot([0,1],[row[xkey],row[ykey]],'-o',color='#777777',alpha=.5,lw=1,ms=4)
    means=[result['confirmation'][k]['mean'] for k in (xkey,ykey)]
    a.plot([0,1],means,'-o',color='#e46b2e',lw=3,ms=8,label='Mean of six new seeds')
    a.set(xticks=[0,1],xticklabels=['Target\nchanges','Heading offset\nchanges'],ylabel='Heldout direction error (degrees)',title='Which explanation predicts better?')
    a.legend(frameon=False,fontsize=8);a.grid(axis='y',alpha=.15)
    for ext in ('png','pdf','svg'):fig.savefig(args.out/f'goal_code_steering.{ext}')
    plt.close(fig)
    primary=result['confirmation'][key];q=result['confirmation']['all_translation_t60_target_model_heldout_improvement_deg']
    text=['# Predictive goal code changes the original agent’s steering','',
          '![Goal-code steering test](goal_code_steering.png)','',
          'We kept the original reward-trained LSTM, actual grid maps, rendered room and six-action policy. At fixed physical positions, we changed only the predictive part of its saved goal snapshot and measured how its movement choices changed across facing directions. These are clamped responses, not navigation trajectories.','',
          f'The six previously untested agents shifted their fitted effective steering target by **{primary["mean"]*100:.1f} cm along the supplied predictive direction** on average across four fixed contexts (95% interval {primary["ci95"][0]*100:.1f} to {primary["ci95"][1]*100:.1f} cm; exact two-sided p={primary["exact_two_sided_sign_flip_p"]:.4f}). Seeds1–2 are earlier descriptive pilots and are excluded from this test.','',
          f'The target-change explanation improved heldout direction prediction by {q["mean"]:.2f}° versus a change of heading offset alone, and was better in {q["positive_seed_count"]}/6 new agents. **Effective-target interpretation gate: {"passed" if result["effective_target_model_gate_passed"] else "failed"}.**','',
          result['interpretation'],'',
          'The fitted point summarizes a direction field. It is not a directly measured neuronally encoded goal location, and these tests do not demonstrate an autonomous ring attractor. All four translation actions enter the readout; all goal variants use the same position weights; a free common angular offset accounts for policy heading-response bias. Uniform weighting and the former dominant-action readout are retained as robustness checks.','',
          'All four contexts and all eight trained agents are retained. Source/checkpoint hashes and optimizer/bound checks are recorded in [summary.json](summary.json). The [frozen protocol](../../../docs/research/original_rl_goal_policy_map_protocol.md) states the endpoint and interpretation gate.']
    (args.out/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='final',primary=primary,model_gate=result['effective_target_model_gate_passed'],heldout_improvement=q)))


if __name__=='__main__':main()
