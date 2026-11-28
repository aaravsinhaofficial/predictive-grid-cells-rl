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

  # cost: what the controller's watchers counted (list prices), per block
  costs = {}
  for tag in ('oc', 'os', 'dr', 'tr', 'lstm', 'trl'):
    for name in ('cost.remote.json', 'cost.json'):
      p = os.path.join(ROOT, 'runs_codes', tag, name)
      if os.path.exists(p):
        try:
          costs[tag] = json.load(open(p))['usd']
          break
        except Exception:      # noqa: BLE001
          pass
  helpers['costs'] = costs

  # ---------------------------------------------------------------- tables for the details page
  LAB = S.get('labels', {})
  F = opt('free_lookaheads.json')

  def lab(code):
    return f'{LAB.get(code, code)} (`{code}`)'

  def pv(x):
    return '–' if x is None or x != x else '< 0.001' if x < 0.001 else f'{x:.3f}' if x < 0.1 else f'{x:.2f}'

  def pm(g, nd=1):
    return f'{n(g["mean"], nd)} ± {n(g["sem"], nd) if g.get("sem") is not None else "–"}'

  def dcell(t, nd=1):
    if t is None or t.get('diff') is None:
      return ['–'] * 4
    cis = '–' if t.get('ci95') is None else f'{sg(t["diff"] - t["ci95"], nd)} to {sg(t["diff"] + t["ci95"], nd)}'
    return [f'{sg(t["diff"], nd)} ± {n(t["sem"], nd) if t.get("sem") is not None else "–"}', cis, pv(t.get('p')),
            f'{t["pos"]} of {t["n"]}']

  def t_codes(room, mark, codes, ref, b=None, score='training score'):
    """One row per code: score, and the paired difference to the reference code."""
    b = b or blk(room, mark)
    holm = {t['a_code']: t.get('p_holm') for t in b.get('named', []) if t.get('b_code') == ref}
    rows = [f'| code | agents | {score} ± SEM | difference to {LAB.get(ref, ref)} ± SEM | 95 % CI | p | p (Holm) | agents higher |',
            '|---|---|---|---|---|---|---|---|']
    for c in codes:
      if c not in b['codes']:
        continue
      g = b['codes'][c]
      d = ['–'] * 4 if c == ref else dcell(b.get('vs', {}).get(ref, {}).get(c))
      rows.append(f'| {lab(c)} | {g["n"]} | {pm(g)} | {d[0]} | {d[1]} | {d[2]} | {pv(holm.get(c)) if c in holm else "–"} | {d[3]} |')
    return '\n'.join(rows)

  def t_named(room, mark, names):
    """One row per comparison 'a - b'."""
    rows = ['| comparison | difference ± SEM | 95 % CI | p | p (Holm) | p (sign flip) | agents higher |', '|---|---|---|---|---|---|---|']
    for nm in names:
      try:
        t = comp(room, mark, nm)
      except KeyError:
        continue
      a, b_ = [x.strip() for x in nm.split(' - ')]
      d = dcell(t)
      rows.append(f'| {LAB.get(a, a)} − {LAB.get(b_, b_)} | {d[0]} | {d[1]} | {d[2]} | {pv(t.get("p_holm")) if "p_holm" in t else "–"} | '
                  f'{pv(t.get("p_flip"))} | {d[3]} |')
    return '\n'.join(rows)

  def t_learned(groups):
    names = dict(lzero='start at 0 cm', lu20='start at 20 cm', lzeroX='control, start at 0 cm', lu20X='control, start at 20 cm')
    rooms = dict(open='open room', door='door room', mix='room mixture, LSTM agents')
    rows = ['| room | start | agents | 40 cm module | 57 cm module | 81 cm module | 115 cm module |', '|---|---|---|---|---|---|---|']
    for g in groups:
      if g not in S['learned']['groups']:
        continue
      v = S['learned']['groups'][g]
      code, room = g.rsplit('_', 1)
      cells = [(f'{n(m_)} ± {n(s_)} cm ({n(100 * f_, 0)} %)' if v['n'] > 1 else f'{n(m_)} cm').replace('-', '−')
               for m_, s_, f_ in zip(v['mean_cm'], v['sem_cm'], v['frac'])]
      rows.append(f'| {rooms[room]} | {names[code]} | {v["n"]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)

  def t_learned_scores(pairs):
    """Score of the agents with trainable look-aheads against the fixed code they started from (same seeds)."""
    rooms = dict(open='open room', door='door room', mix='room mixture, LSTM agents')
    rows = ['| room | frames | trainable code | score ± SEM | fixed code | difference ± SEM | 95 % CI | agents higher |', '|---|---|---|---|---|---|---|---|']
    for env, mark, code, ref in pairs:
      v = S['learned'].get('scores', {}).get(env, {}).get(str(mark))
      if not v or code not in v['codes'] or ref not in v.get('vs', {}):
        continue
      d = dcell(v['vs'][ref].get(code))
      rows.append(f'| {rooms[env]} | {mark} M | {dict(lzero="start at 0 cm", lu20="start at 20 cm")[code]} | {pm(v["codes"][code])} | '
                  f'{LAB.get(ref, ref)} | {d[0]} | {d[1]} | {d[3]} |')
    return '\n'.join(rows)

  def t_lesions(room, budget, codes):
    ft = S[room].get('final_test', {}).get(str(budget))
    if not ft:
      return '??'
    rows = ['| code | agents | test score ± SEM | predictive cells off, beyond 169 random cells off | look-aheads set to zero | whole code off | goal code off |',
            '|---|---|---|---|---|---|---|']
    for c in codes:
      le = ft['lesions'].get(c)
      if not le or c not in ft['codes']:
        continue
      # stored as points lost; shown as the change of the score (negative = the agent got worse)
      cell = lambda k: '–' if k not in le or le[k].get('diff') is None else f'{sg(-le[k]["diff"])} ± {n(le[k].get("sem"))}'   # noqa: E731
      rows.append(f'| {lab(c)} | {ft["codes"][c]["n"]} | {pm(ft["codes"][c])} | {cell("pgc_specific")} | {cell("pgc_as_gc")} | '
                  f'{cell("no_code")} | {cell("no_goal")} |')
    return '\n'.join(rows)

  def t_groups(X, rows_, codes, ref, nd=1, unit=''):
    """Rows = measures or room groups of a summary {row: {codes, vs}}, columns = codes, last columns = differences to ref."""
    others = [c for c in codes if c != ref]
    head = '| | ' + ' | '.join(LAB.get(c, c) for c in codes) + ' | ' + ' | '.join(f'{LAB.get(c, c)} − {LAB.get(ref, ref)}' for c in others) + ' |'
    out = [head, '|' + '---|' * (1 + len(codes) + len(others))]
    for key, title in rows_:
      if key not in X:
        continue
      g = X[key]
      sc_ = [f'{n(g["codes"][c]["mean"], nd)}{unit}' if c in g['codes'] else '–' for c in codes]
      df_ = []
      for c in others:
        t = g.get('vs', {}).get(ref, {}).get(c)
        df_.append('–' if not t or t.get('diff') is None else f'{sg(t["diff"], nd)} ± {n(t["sem"], nd)} (p {pv(t.get("p"))})')
      out.append(f'| {title} | ' + ' | '.join(sc_) + ' | ' + ' | '.join(df_) + ' |')
    return '\n'.join(out)

  def t_free():
    rows = ['| run | room | frames | forward, mean ± SD over cells | sideways, mean ± SD | forward mean per module (40 / 57 / 81 / 115 cm) |', '|---|---|---|---|---|---|']
    for name, v in sorted(F.items()):
      kind = 'control' if name.startswith('cfreeX') else 'trained'
      rows.append(f'| {kind}, agent {name.rsplit("_s", 1)[1]} | {name.split("_")[1]} | {v["frames"] / 1e6:.0f} M | {sg(v["fwd_mean"])} ± {n(v["fwd_sd"])} cm | '
                  f'{sg(v["lat_mean"])} ± {n(v["lat_sd"])} cm | ' + ' / '.join(sg(x) for x in v['fwd_by_module']) + ' cm |')
    return '\n'.join(rows)

  helpers.update(F=F, lab=lab, pv=pv, pm=pm, t_codes=t_codes, t_named=t_named, t_learned=t_learned,
                 t_learned_scores=t_learned_scores, t_lesions=t_lesions, t_groups=t_groups, t_free=t_free)
  helpers['COST_LINE'] = (f'About USD {sum(costs.values()) + 12:.0f} of cloud compute at list prices '
                          f'(spot L4 GPUs, counted by the fleet watcher, plus about USD 12 for tests, the controller '
                          f'and data transfer), of USD 600 allowed.') if costs else '??'
  helpers.setdefault('DETAILS_RESULTS', '')

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
