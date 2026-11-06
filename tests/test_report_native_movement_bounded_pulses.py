import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import report_native_movement_bounded_pulses as report


def test_robustness_comparison_uses_intersection_across_both_experiments(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    source=tmp_path/'scripts/report_native_movement_pulses.py';source.parent.mkdir();source.write_text('presentation source')
    def row(case,valid,gain):return dict(case_id=case,model='gc100_s1',mode='native',pulse='normal_plus',family='normal',stream='closed',horizon=30,joint_all_directions_available=valid,state_gain=gain,action_tv=.1)
    old=dict(complete=True,source_sha256=report.sha(source),response_rows=[row('a',True,10),row('b',False,200),row('c',True,100)],mode_effect_rows=[])
    path=tmp_path/'results/native_movement_pulses/report/summary.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(old))
    new=dict(models={'gc100_s1':{}},response_rows=[row('a',True,9),row('b',True,300),row('c',False,80)],mode_effect_rows=[])
    result=report.compare_original(new)
    assert len(result['rows'])==1 and result['rows'][0]['case_id']=='a'
    group=result['models']['gc100_s1']['closed|native|normal|30']
    assert group['original_state_gain']['mean']==10 and group['bounded_state_gain']['mean']==9
