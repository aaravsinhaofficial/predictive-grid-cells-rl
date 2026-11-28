"""Fill the numbers of the code-variation write-up from results/code_variations/summary.json.

README.template.md (the short page) and DETAILS.template.md (methods, all
statistics, caveats) hold the text; every number is a placeholder {{...}}
evaluated against the summary, so the pages cannot drift from the data.

Helpers (room = 'open' | 'door' | 'lstm'; mark = frames in millions):
  sc(room, mark, code)          mean training score            '31.2'
  gl(room, mark, code)          the same in goals per episode  '3.1'
  c(room, mark, 'a - b')        the paired comparison a - b (a dict)
  d(x)    '+1.2 ± 0.9'          ci(x)  '−0.9 to +3.3'          dci(x) '+1.2 (95 % CI −0.9 to +3.3)'
  p(x)    'p = 0.21'            ph(x)  Holm-corrected p        k(x)   '6 of 8'
  within(x)                     '± 2.1' : the half-width of the 95 % confidence interval
  pct(a, b)                     how many per cent a is above b
  la(group, m)                  learned look-ahead of module m (0-3) in cm, group e.g. 'lzero_open'
  laf(group, m)                 the same as a fraction of the module spacing
  S, D (door_eval.json), L (lstm_rooms.json) are available as dicts.

  python -m scripts.codes.render_readme
"""
import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fmt_p(pv):
  if pv is None or pv != pv:
    return 'p –'
  return 'p < 0.001' if pv < 0.001 else f'p = {pv:.3f}' if pv < 0.1 else f'p = {pv:.2f}'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dir', default=os.path.join(ROOT, 'results', 'code_variations'))
  args = ap.parse_args()
  S = json.load(open(os.path.join(args.dir, 'summary.json')))

  def opt(name):
    p = os.path.join(args.dir, 'data', name)
    return json.load(open(p)) if os.path.exists(p) else {}
  D, L = opt('door_eval_summary.json'), opt('lstm_rooms_summary.json')

  def n(x, nd=1):
    return '??' if x is None or x != x else f'{x:.{nd}f}'

  def sg(x, nd=1):
    return '??' if x is None or x != x else f'{x:+.{nd}f}'.replace('-', '−')

  def blk(room, mark):
    return S[room][str(mark)]

  def score(room, mark, code):
    return blk(room, mark)['codes'][code]['mean']

  def comp(room, mark, name):
    b = blk(room, mark)
    for t in b.get('named', []) + b.get('descriptive', []):
      if t['name'] == name:
        return t
    a, ref = [x.strip() for x in name.split(' - ')]
    if ref in b.get('vs', {}) and a in b['vs'][ref]:
      return b['vs'][ref][a]
    raise KeyError(f'no comparison {name!r} in {room} at {mark} M')

  helpers = dict(
      S=S, D=D, L=L, n=n, sg=sg, fmt_p=fmt_p, blk=blk, score=score,
      sc=lambda room, mark, code, nd=1: n(score(room, mark, code), nd),
      gl=lambda room, mark, code, nd=1: n(score(room, mark, code) / 10.0, nd),
      c=comp,
      d=lambda x, nd=1: f'{sg(x["diff"], nd)} ± {n(x["sem"], nd)}',
      ci=lambda x, nd=1: f'{sg(x["diff"] - x["ci95"], nd)} to {sg(x["diff"] + x["ci95"], nd)}',
      dci=lambda x, nd=1: f'{sg(x["diff"], nd)} (95 % CI {sg(x["diff"] - x["ci95"], nd)} to {sg(x["diff"] + x["ci95"], nd)})',
      within=lambda x, nd=1: f'± {n(x["ci95"], nd)}',
      p=lambda x: fmt_p(x.get('p')),
      ph=lambda x: fmt_p(x.get('p_holm')),
      pf=lambda x: fmt_p(x.get('p_flip')),
      k=lambda x: f'{x["pos"]} of {x["n"]}',
      pct=lambda a, b, nd=0: n(100.0 * (a / b - 1.0), nd),
      la=lambda group, m, nd=1: n(S['learned']['groups'][group]['mean_cm'][m], nd),
      laf=lambda group, m, nd=2: n(S['learned']['groups'][group]['frac'][m], nd),
      sum=sum, max=max, min=min, len=len, abs=abs, str=str, int=int, sorted=sorted, round=round, zip=zip,
      range=range, enumerate=enumerate, float=float)

  for tpl in ('README', 'DETAILS'):
    src = os.path.join(args.dir, f'{tpl}.template.md')
    if not os.path.exists(src):
      continue
    text = open(src).read()
    bad = []

    def sub(m):
      try:
        return str(eval(m.group(1), {'__builtins__': {}}, helpers))
      except Exception as e:          # noqa: BLE001
        bad.append(f'{m.group(1)!r}: {type(e).__name__}: {e}')
        return '??'
    out = re.sub(r'\{\{(.+?)\}\}', sub, text)
    open(os.path.join(args.dir, f'{tpl}.md'), 'w').write(out)
    print(f'{tpl}.md: {len(re.findall(r"{{", text))} numbers filled, {len(bad)} failed')
    for b in bad:
      print('   ', b)


if __name__ == '__main__':
  main()
