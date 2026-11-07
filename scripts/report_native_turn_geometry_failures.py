"""Explain a completed turn-geometry screen using its already audited saved rows."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/native_turn_geometry'
SPACING='max_edge_below_10percent_radius'
DEPARTURE='p90_invariance_residual_below_5percent'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(rows):
    failures=collections.Counter(tuple(sorted(k for k,v in r['gates'].items()if not v))for r in rows)
    return dict(cases=len(rows),pass_all=sum(all(r['gates'].values())for r in rows),
        pass_except_spacing=sum(all(v for k,v in r['gates'].items()if k!=SPACING)for r in rows),
        pass_except_spacing_and_departure=sum(all(v for k,v in r['gates'].items()if k not in(SPACING,DEPARTURE))for r in rows),
        individual_gate_failures=dict(collections.Counter(k for r in rows for k,v in r['gates'].items()if not v)),
        exact_failure_sets=[dict(gates=list(k),count=v)for k,v in sorted(failures.items())])


def primary(row):
    return row['window_updates']==(2 if row['regime']!='left'and row['observable']=='full_state'else 3)


def summarize(data):
    groups=collections.defaultdict(list)
    for r in data['rows']:
        groups[(r['regime'],r['observable'],r['centered'],r['window_updates'],r['metric'])].append(r)
    result=[]
    for (regime,observable,centered,window,metric),rows in sorted(groups.items()):
        native=[r for r in rows if r['branch']=='native']
        result.append(dict(condition=data['condition'],endpoint=data['endpoint'],regime=regime,
            observable=observable,centered=centered,window_updates=window,metric=metric,
            primary=primary(rows[0]),all=counts(rows),native=counts(native),
            native_median_radius_retention=float(np.median([r['radius_retention']for r in native])),
            native_median_p90_departure=float(np.median([r['residual_to_reference_curve_over_radius']['p90']for r in native])),
            native_median_max_edge_over_radius=float(np.median([r['geometry']['max_edge_over_radius']for r in native]))))
    return result


def render(data,out,endpoint):
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    regimes=('left','fb_B','fb_F');colors=('#2673b8','#e56b2f')
    for model,color,offset in zip(data,colors,(-.14,.14)):
        for j,regime in enumerate(regimes):
            rows=[r for r in model['rows']if r['regime']==regime and r['observable']=='full_state'and
                  r['metric']=='raw'and r['branch']=='native'and primary(r)]
            assert len(rows)==64
            jitter=np.linspace(-.08,.08,len(rows));x=j+offset+jitter
            values=([r['radius_retention']for r in rows],
                    [r['residual_to_reference_curve_over_radius']['p90']for r in rows])
            for ax,y in zip(axes,values):
                ax.scatter(x,y,s=10,alpha=.4,color=color,linewidths=0,
                           label=('Grid'if model['condition']=='gc100'else'PGC + grid')if j==0 else None)
                ax.plot([j+offset-.09,j+offset+.09],[np.median(y)]*2,color=color,lw=2.5)
    for ax in axes:
        ax.set_xticks(range(3),('Turning','F/B · B','F/B · F'))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_title('Angle-dependent spread remains')
    axes[0].set_ylabel('Remaining spread / initial spread')
    axes[0].axhline(1,color='#888888',ls=':',lw=1)
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].set_title('The full state leaves its initial curve')
    axes[1].set_ylabel('Curve departure / initial spread')
    axes[1].set_yscale('log')
    axes[1].axhline(.05,color='#555555',ls='--',lw=1)
    axes[1].text(2.45,.05,'5% criterion',ha='right',va='bottom',fontsize=8,color='#555555')
    fig.suptitle(f'Native input · {endpoint} checkpoints',fontsize=14)
    fig.text(.5,.015,'64 physical families per group · all contexts and both turn histories · short task window',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96))
    for suffix in ('png','pdf'):
        fig.savefig(out/f'full_state_persistence_{endpoint}.{suffix}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-directory',default='report_original',choices=('report_original','report_all'))
    args=parser.parse_args();out=BASE/args.report_directory
    validation=json.loads((out/'validation.json').read_text())
    assert validation['complete']and validation['summary_sha256']==sha(out/'summary.json')
    results=[];groups=[]
    for audit in validation['models']:
        path=BASE/audit['condition']/audit['endpoint']/'summary.json'
        assert audit['complete']and sha(path)==audit['summary_sha256']
        data=json.loads(path.read_text());assert data['complete']and len(data['rows'])==33_280
        results.append(data);groups.extend(summarize(data))
    endpoints=list(dict.fromkeys(r['endpoint']for r in results))
    for endpoint in endpoints:render([r for r in results if r['endpoint']==endpoint],out,endpoint)
    record=dict(complete=True,source_sha256=sha(__file__),validation_sha256=sha(out/'validation.json'),
        description='Gate-failure accounting. Omitted-gate counts describe refinement opportunities; they are not new nominations.',
        raw_rows=sum(len(r['rows'])for r in results),
        necessary_geometry_pass_rows=sum(r['necessary_geometry_pass']for m in results for r in m['rows']),
        normal_test_rows=sum(r['normal_test_run']for m in results for r in m['rows']),groups=groups)
    (out/'failure_patterns.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=['## What prevented nomination','',
        f"Across all {record['raw_rows']:,} saved screening rows, {record['necessary_geometry_pass_rows']:,} passed every necessary geometry check. The staged protocol therefore ran {record['normal_test_rows']:,} broad disturbance tests. A geometry rejection is not a measured failure of disturbance recovery.",'',
        'The strongest full-state pattern is retained angular spread combined with drift away from the starting curve. The figures below include the native branch only, keeping all 64 physical families per population and regime; reset controls remain in the complete counts. Turning spans three updates (0.2 s), and the full-state F/B comparison spans two updates (0.133 s) from its separate periodic reference.','']
    for endpoint in endpoints:lines.extend([f'![Full-state persistence: {endpoint}](full_state_persistence_{endpoint}.png)',''])
    lines.extend(['| Network | Regime | Native families | Median spread retained | Median curve departure |',
                  '| --- | --- | --- | --- | --- |'])
    for r in groups:
        if r['primary']and r['metric']=='raw'and r['observable']=='full_state':
            name='Grid'if r['condition']=='gc100'else'PGC + grid'
            lines.append(f"| {name} · {r['endpoint']} | {r['regime']} | {r['native']['cases']} | {100*r['native_median_radius_retention']:.1f}% | {100*r['native_median_p90_departure']:.1f}% |")
    lines.extend(['','The 128-angle spacing guard also rejects many curves: neighbouring samples can be too far apart relative to the family radius. This is a finite-resolution limitation, not a finding that the true curve is discontinuous. Some centered hidden-output and actor-output families pass every other geometry check. They are legitimate targets for a separately specified denser sampling test; their current nomination remains negative.', '',
        'Distances use interpolation along every segment of the closed polygon, rather than just the nearest sampled vertex. Denser sampling could still alter the polygon approximation. The full-state drift and the centered-output results should therefore be reported separately.','',
        '| Network | Regime | Observable | All cases | Pass all except spacing | Pass all except spacing + departure |',
        '| --- | --- | --- | --- | --- | --- |'])
    for r in groups:
        if r['primary']and r['metric']=='raw'and r['centered']==(r['observable']!='full_state'):
            name='Grid'if r['condition']=='gc100'else'PGC + grid'
            lines.append(f"| {name} · {r['endpoint']} | {r['regime']} | {r['observable']} | {r['all']['cases']} | {r['all']['pass_except_spacing']} | {r['all']['pass_except_spacing_and_departure']} |")
    lines.extend(['','The omitted-gate columns are diagnostics, not relaxed criteria or positive results. Exact failure sets, native-only counts, both metrics, centered/noncentered outputs and every longer window are retained in [failure_patterns.json](failure_patterns.json).'])
    readme=(out/'README.md').read_text().split('\n## What prevented nomination\n')[0]
    readme=readme.replace('The256 cases per entry cover16','The 256 cases per entry cover 16').replace('not256','not 256')
    (out/'README.md').write_text(readme.rstrip()+'\n\n'+'\n'.join(lines)+'\n')
    (out/'artifacts.json').write_text(json.dumps({p.name:dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in out.iterdir()if p.is_file()and p.name!='artifacts.json'},indent=2)+'\n')
    print(json.dumps(dict(complete=True,raw_rows=record['raw_rows'],groups=len(groups),report=str(out))))


if __name__=='__main__':main()
