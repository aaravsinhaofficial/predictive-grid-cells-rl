import numpy as np
import torch

from scripts.lstm_continued_heading_memory import ANCHORS,RECOVERY,eligible_window
from scripts.lstm_heading_current_release import ordinary_current_input


def window():
    rows=[]
    for t in range(15):
        x=np.zeros((1,1287));x[:,768:1280]=.3
        rows.append(dict(known=np.array([True]),blind=np.array([False]),
                         remaining=np.array([15-t]),hit=np.array([False]),done=np.array([False]),
                         x=x,pos=np.array([[.1,.1]]),goal_center=np.array([[.9,.8]])))
    return rows


def test_only_complete_native_visible_intervals_qualify():
    rows=window();assert eligible_window(rows,0,.3)==(True,'accepted')
    rows[4]['blind'][0]=True;assert eligible_window(rows,0,.3)[1]=='visibility'
    rows=window();rows[8]['hit'][0]=True;assert eligible_window(rows,0,.3)[1]=='hit_or_reset'
    rows=window();rows[0]['remaining'][0]=14;assert eligible_window(rows,0,.3)[1]=='schedule'


def test_changed_goal_rejects_a_context():
    rows=window();rows[10]['x'][0,900]=.1
    assert eligible_window(rows,0,.3)[1]=='goal_changed'


def test_current_release_preserves_goal_vision_action_and_reward():
    class Pop:
        pgc_idx=np.arange(169)
        is_pgc=torch.arange(512)<169
        def activity(self,pos,direction,velocity,shift_scale):
            assert shift_scale==0.
            return torch.arange(512,dtype=torch.float32)[None,:]/512
    x=torch.randn(1,1287,dtype=torch.float64)
    changed,audit=ordinary_current_input(Pop(),x,[.1,.2],[0.,1.])
    assert audit['allowed_changed_slots']==audit['numerically_changed_slots']==169
    assert audit['saved_goal_max_change']==audit['other_input_max_change']==0.
    torch.testing.assert_close(changed[:,:256],x[:,:256],atol=0,rtol=0)
    torch.testing.assert_close(changed[:,425:],x[:,425:],atol=0,rtol=0)


def test_primary_recovery_stays_inside_trained_four_seconds():
    assert max(ANCHORS)+max(RECOVERY)==60
    assert max(ANCHORS)==30 and max(RECOVERY)==30
