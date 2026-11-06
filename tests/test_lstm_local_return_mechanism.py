from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import pytest
import lstm_local_return_mechanism as mechanism


def report(condition, numeric=False, certified=False, complete=True):
    return dict(condition=condition,passed=True,complete=complete,checks=dict(
        final_exact_stage4=True,final_summary=True,stage4_raw_evaluation=dict(gates=dict(
            hard_numeric_pass=numeric,hard_certified_memory_pass=numeric and certified))))


def test_integrity_pass_does_not_automatically_enable_neural_assays():
    pair={c:report(c) for c in mechanism.CONDITIONS}
    assert not mechanism.behavioral_gate(pair)['passed']
    pair['pgc33']=report('pgc33',numeric=True,certified=False)
    assert not mechanism.behavioral_gate(pair)['passed']
    pair['gc100']=report('gc100',numeric=True,certified=True)
    result=mechanism.behavioral_gate(pair)
    assert result['passed'] and set(result['by_condition'])==set(mechanism.CONDITIONS)


def test_incomplete_or_missing_model_cannot_select_one_favorable_final():
    pair={c:report(c,numeric=True,certified=True) for c in mechanism.CONDITIONS}
    pair['pgc33']['complete']=False
    with pytest.raises(AssertionError):mechanism.behavioral_gate(pair)
    pair.pop('pgc33')
    with pytest.raises(AssertionError):mechanism.behavioral_gate(pair)


def test_numerical_guard_scales_only_by_actual_state_and_preserves_absolute_flag():
    import numpy as np
    native=np.zeros((2,512));native[:,256:]=90
    candidate=native.copy();candidate[:,256:]+=.0004
    result=mechanism.numerical_consistency(native,candidate,100)
    assert result['pointwise_pass'] and not result['absolute_only_3e_minus4_pass']
    assert result['blocks']['cell']['native_max_abs_activity']==90
    candidate[:,0]=.0004
    assert not mechanism.numerical_consistency(native,candidate,100)['pointwise_pass']
    native[:,0]=2
    with pytest.raises(AssertionError):mechanism.numerical_consistency(native,candidate,100)
