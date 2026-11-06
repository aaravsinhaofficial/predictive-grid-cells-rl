import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from write_feedback_restricted_actions import json_values


def test_numpy_scalars_change_type_without_changing_value():
    data={'possible':np.bool_(True),'count':np.int64(9),'distance':np.float64(.601),'nested':[np.bool_(False)]}
    result=json_values(data)
    assert json.loads(json.dumps(result))=={'possible':True,'count':9,'distance':.601,'nested':[False]}
    assert result['distance']==data['distance']
