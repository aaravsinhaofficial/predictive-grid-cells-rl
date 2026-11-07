"""Independent lightweight saved-JSON review; no model or array re-evaluation."""
from pathlib import Path
from collections import Counter
import hashlib,json,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/native_input_observability';OUT=BASE/'report'
INITIAL='initial_probability_RMS_at_least_1e_minus4';EVOLVED='evolved_probability_RMS_at_least_1e_minus4'
WINDOWS={'left':(2,7,14,29,59),'fb_B':(2,14,30,60),'fb_F':(2,14,30,60)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def describe(values):
 a=np.asarray(values);return dict(n=len(a),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),minimum=float(a.min()),maximum=float(a.max()))if len(a)else dict(n=0)
def totals(rows,statuses):
 valid=[r for r in rows if r.get('physical_valid')]
 return dict(expected=len(rows),screened=sum(r['status']!='screen_pending'for r in rows),physical_valid=len(valid),reference_visible=sum(r.get('gates',{}).get(INITIAL,False)for r in rows),evolved_visible=sum(r.get('gates',{}).get(EVOLVED,False)for r in rows),both_visible=sum(r.get('gates',{}).get(INITIAL,False)and r.get('gates',{}).get(EVOLVED,False)for r in rows),geometry_qualified=sum(r.get('geometry_qualified',False)for r in rows),conditional_completed=sum('conditional_metrics'in r for r in rows),candidates=sum(r['status']=='observed_candidate'for r in rows),diagnostic_probability_centered_pass=sum(r.get('diagnostic_probability_centered_pass',False)for r in rows),diagnostic_logit_centered_pass=sum(r.get('diagnostic_logit_centered_pass',False)for r in rows),statuses={s:sum(r['status']==s for r in rows)for s in statuses},overlapping_failed_gates=dict(Counter(g for r in rows for g in r.get('failed_gates',[]))),initial_probability_RMS=describe([r['initial_probability_RMS']for r in valid]),evolved_probability_RMS=describe([r['evolved_probability_RMS']for r in valid]),evolved_antipodal_TV=describe([r['evolved_antipodal_TV']for r in valid]))
def main():
 s=read(OUT/'summary.json');v=read(OUT/'validation.json');p=ROOT/'docs/research/native_input_observability_protocol.json'
 assert s['complete']and s['all_cohorts_complete']and v['complete']and s['expected_primary_rows']==len(s['rows'])==6656
 assert sha(p)==s['protocol_sha256']==v['protocol_sha256']and sha(OUT/'summary.json')==v['summary_sha256']
 for rel,h in v['arithmetic_source_sha256'].items():assert sha(ROOT/rel)==h
 for name,r in read(OUT/'artifacts.json').items():assert sha(OUT/name)==r['sha256']and(OUT/name).stat().st_size==r['bytes']
 records=[];reset={};numeric=0
 for condition in ('gc100','pgc33'):
  folder=BASE/condition;terminal=read(folder/'summary.json');execution=read(folder/'execution.json');screen=read(folder/'screen_summary.json');conditional=read(folder/'conditional_summary.json');cohort=read(folder/'cohort.json')
  audited=next(r for r in v['cohorts']if r['condition']==condition)
  assert terminal['complete']and execution['complete']and execution['exit_code']==0 and execution['summary_sha256']==sha(folder/'summary.json')==audited['summary_sha256']
  assert terminal['screen_summary_sha256']==sha(folder/'screen_summary.json')and terminal['conditional_summary_sha256']==sha(folder/'conditional_summary.json')
  assert screen['complete']and conditional['complete']and conditional['rows']==[] and terminal['candidates']==terminal['geometry_qualified_rows']==0
  rows=[r for r in s['rows']if r['condition']==condition];raw={(r['family'],r['regime'],r['retention_updates']):r for r in screen['rows']};seen=set()
  for r in rows:
   key=(r['family'],r['regime'],r['retention_updates']);assert key not in seen;seen.add(key);a=raw[key];g=a['primary_geometry'];assert r['physical_valid']==a['physical_valid']is True
   for field in ('context','prefix','kind','branch','exact_sham_of','geometry_qualified','latest_observation_updates_after_cue'):assert r[field]==a[field]
   assert r['gates']==g['gates']and r['failed_gates']==[k for k,vv in g['gates'].items()if not vv]
   for name in ('initial_probability_RMS','evolved_probability_RMS','radius_retention'):assert r[name]==g[name]
   assert r['departure_p90']==g['residual_to_reference_curve_over_radius']['p90']
   status='initial_probability_below_floor'if not g['gates'][INITIAL]else'evolved_probability_below_floor'if not g['gates'][EVOLVED]else'geometry_failed'if not all(g['gates'].values())else'conditional_pending'
   assert r['status']==status and not r['geometry_qualified']
   assert r['physical_phase']==a['physical_phase']
   for name in ('probability','logit'):
    assert r['diagnostic_'+name+'_centered_gates']==a['diagnostic_'+name+'_centered']['gates']
    assert r['diagnostic_'+name+'_centered_pass']==a['diagnostic_'+name+'_centered']['necessary_geometry_pass']
  assert seen==set(raw)=={tuple(k)for k in cohort['expected_primary_keys']}and len(rows)==3328
  statuses=list(s['populations'][condition]['named']['statuses']);numeric_rows=[r for r in rows if not r['exact_sham_of']];numeric+=len(numeric_rows)
  assert totals(rows,statuses)==s['populations'][condition]['named']and totals(numeric_rows,statuses)==s['populations'][condition]['numerical']
  for regime,windows in WINDOWS.items():
   for t in windows:
    entry=next(w for w in s['windows']if(w['condition'],w['regime'],w['retention_updates'])==(condition,regime,t));assert {k:v for k,v in entry.items()if k not in ('condition','regime','retention_updates')}==totals([r for r in rows if r['regime']==regime and r['retention_updates']==t],statuses)
  for branch in ('pre_cue_reset_native','pre_cue_reset_ordinary'):
   subset=[r for r in rows if r['branch']==branch];visible=[r for r in subset if r['gates'][INITIAL]and r['gates'][EVOLVED]]
   reset[condition+'/'+branch]=dict(denominator=len(subset),both_visible=len(visible),contexts=sorted({r['context']for r in visible}),kinds=sorted({r['kind']for r in visible}),cases=[{k:r[k]for k in ('context','prefix','kind','branch','regime','retention_updates','initial_probability_RMS','evolved_probability_RMS')}for r in visible])
  records.append(dict(condition=condition,terminal_sha256=sha(folder/'summary.json'),execution_sha256=sha(folder/'execution.json'),screen_sha256=sha(folder/'screen_summary.json'),conditional_sha256=sha(folder/'conditional_summary.json'),named_rows=len(rows),numerical_rows=len(numeric_rows),statuses=s['populations'][condition]['named']['statuses']))
 assert numeric==4160 and sum(x['artifact_hashes_checked']for x in v['cohorts'])==5706
 assert reset['pgc33/pre_cue_reset_native']['both_visible']==20 and reset['pgc33/pre_cue_reset_native']['contexts']==[3,5]and reset['pgc33/pre_cue_reset_native']['kinds']==['probe_glimpse']
 assert all(x['both_visible']==0 for k,x in reset.items()if k!='pgc33/pre_cue_reset_native')
 result=dict(complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),review_source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),full_array_validation_sha256=sha(OUT/'validation.json'),protocol_sha256=sha(p),rows_independently_matched=6656,numerical_rows=4160,models=records,reset_cue_sensitivity=reset,all_population_window_percentiles_counts_and_overlapping_gates_exact=True,figures_visually_reviewed=['probability_sensitivity.png','geometry_criteria.png'],interpretation='20/832 PGC reset-native cases meet the sensitivity floor versus zero reset-ordinary; one original model, two contexts, probe condition only. This is readable cue-specific information, not a ring, transverse recovery, or navigation benefit. All primary geometry cases fail; no conditional kicks or held-out queries ran.',scope='Light saved-JSON independent review, bound to the separately completed full-profile/physics/provenance arithmetic audit. No model/environment/new query and no duplicate heavy array audit.',artifacts={x.name:sha(x)for x in OUT.iterdir()if x.is_file()and x.name not in('artifacts.json','independent_review.json')})
 (OUT/'independent_review.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'artifacts.json').write_text(json.dumps({x.name:dict(sha256=sha(x),bytes=x.stat().st_size)for x in sorted(OUT.iterdir())if x.is_file()and x.name!='artifacts.json'},indent=2)+'\n');print('PASS',sha(OUT/'independent_review.json'))
if __name__=='__main__':main()
