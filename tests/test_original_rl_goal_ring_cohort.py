"""Full-opportunity first-return endpoint and v1/v2 dynamics identity."""
import ast
import importlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import original_rl_goal_ring_cohort as m


def test_primary_retains_failures_and_excludes_only_shared_late_discovery():
    rows=[dict(score=10,n_goals=1,first_goal_step=100,first_return_steps=-1),
          dict(score=20,n_goals=2,first_goal_step=1049,first_return_steps=150),
          dict(score=20,n_goals=2,first_goal_step=1050,first_return_steps=20),
          dict(score=0,n_goals=0,first_goal_step=-1,first_return_steps=-1)]
    result=m.summarize(rows,1350)
    assert result['episodes']==4
    assert result['eligible_first_returns']==2
    assert result['excluded_late_discoveries']==1
    assert result['eligible_first_return_capped20s_steps']==225
    assert result['eligible_first_return_within20s_fraction']==.5
    assert result['mean_score']==12.5


def test_v2_preserves_map_inverse_and_ring_dynamics_exactly():
    def definitions(path):
        return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=definitions(ROOT/'scripts/original_rl_goal_ring.py')
    new=definitions(ROOT/'scripts/original_rl_goal_ring_cohort.py')
    for name in ('ring_step','phase_rotate','MapInverse','validate_inverse','validate_ring'):
        assert old[name]==new[name]
