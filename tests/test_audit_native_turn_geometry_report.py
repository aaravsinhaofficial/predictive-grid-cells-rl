"""Completion guards for the independent saved-array geometry audit."""
from pathlib import Path
import sys
import copy
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_native_turn_geometry_report as audit


def cohort():
    return (dict(scope=['original','clean','noise'],models=6,raw_rows=199680,primary=[None]*54),
        dict(models=[dict(condition=c,endpoint=e)for e in ('original','clean','noise')for c in ('gc100','pgc33')],
             inputs=[dict(condition='gc100'),dict(condition='pgc33')]),
        dict(raw_rows=199680,groups=[None]*780))


def test_all_six_unique_endpoints_required_even_when_totals_match():
    s,v,f=cohort();audit.check_cohort(s,v,f,'report_all')
    v['models'][-1]=copy.deepcopy(v['models'][0])
    with pytest.raises(AssertionError):audit.check_cohort(s,v,f,'report_all')
    s,v,f=cohort();v['models'].pop()
    with pytest.raises(AssertionError):audit.check_cohort(s,v,f,'report_all')


def test_expected_window_keys_keep_two_section_clock_and_all_cases():
    keys=audit.expected_row_keys()
    assert len(keys)==33280
    assert (0,0,'turn_only','native','fb_B',3,2,'all_hidden','raw',True)in keys
    assert (0,0,'turn_only','native','fb_B',2,2,'full_state','raw',False)in keys
    assert (15,1,'probe_glimpse','pre_cue_reset_ordinary','left',60,59,'all_actor','h_c_RMS',True)in keys
    assert (0,0,'turn_only','native','fb_F',3,3,'all_hidden','raw',True)not in keys
