"""Run all fixed original-RL steering-map contexts without selecting outcomes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CONTEXTS=((0.,0.,0.),(.25,-.20,1.2),(-.20,.25,-1.),(.20,.20,2.6))


def main():
    manifest=json.loads((ROOT/'docs/research/original_rl_goal_policy_map_sources.json').read_text())
    for path,digest in manifest['sources'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    output=ROOT/'results/rl_map_mechanism/policy_map_confirmation'
    for seed in (3,4,5,6,7,8,1,2):
        for context,(x,y,angle) in enumerate(CONTEXTS):
            folder=output/f'context{context}';folder.mkdir(parents=True,exist_ok=True)
            result=folder/f'pgc33_s{seed}.json'
            if result.exists():
                old=json.loads(result.read_text())
                assert old['source_sha256']==manifest['sources']['scripts/lstm_goal_policy_map.py']
                continue
            command=[sys.executable,str(ROOT/'scripts/lstm_goal_policy_map.py'),'--seed',str(seed),
                '--side','7','--headings','64','--steps','15','60','--goal-x',str(x),'--goal-y',str(y),
                '--arrival-angle',str(angle),'--scene-seed',str(989131+context),'--device','cpu','--threads','2',
                '--out',str(folder)]
            with (folder/f'pgc33_s{seed}.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({'completed_seed':seed,'context':context}),flush=True)


if __name__=='__main__':main()
