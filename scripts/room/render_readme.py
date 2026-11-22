"""Fill the numbers of the room-change write-up from results/room_change/summary.json.

results/room_change/README.template.md holds the text; every number is a
placeholder {{...}} evaluated against the summary, so the README cannot drift
from the data. Helpers (st = a paired statistic of the summary):

  S                         the summary (dict);  X['excl3'] the summary of a sensitivity analysis;
                            SC = scratch.json (from-scratch reference runs)
  xst(key, state, group, metric)  the same statistic in X[key]
  st(state, group, metric)  S['stats'][state][group][metric]
  pc(state, room, metric)   S['per_condition'][state][room][metric]
  gp(x)   'grid 21.5, PGC 24.4'      d(x)   '+3.0 ± 2.8'      ci(x)  '+3.0 (95 % CI −3.6 to +9.5)'
  p(x)    'p = 0.32'   pv(x)  '0.32'   k(x)   '6/8'             full(x) 'd, p, k' in one go
  chg(key, group, metric)  S['change'][key][group][metric] (gc / pgc / diff, each a one-sample statistic)
  one(x)  '+12.3 ± 2.1' of a one-sample statistic;  npos(state, rooms)  rooms in which PGC scores higher
  row(label, x)   a table row  | label | grid | PGC | difference | p | k/n |
  n(value, digits)  plain number;  pct(value) percentage

  python -m scripts.room.render_readme            # writes results/room_change/README.md
"""
import argparse
import glob
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
  ap.add_argument('--dir', default=os.path.join(ROOT, 'results', 'room_change'))
  ap.add_argument('--template', default=None)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()
  S = json.load(open(os.path.join(args.dir, 'summary.json')))
  # sensitivity analyses (python -m scripts.room.analyse --exclude 3): X['excl3'] etc.
  X = {os.path.basename(f)[8:-5]: json.load(open(f)) for f in glob.glob(os.path.join(args.dir, 'summary_*.json'))}
  scr = os.path.join(args.dir, 'scratch.json')          # from-scratch reference runs (scripts/room/scratch_analyse.py)
  SC = json.load(open(scr)) if os.path.exists(scr) else {}
  nd_default = [1]

  def n(x, nd=None):
    nd = nd_default[0] if nd is None else nd
    return '??' if x is None or x != x else f'{x:.{nd}f}'

  def sg(x, nd=None):
    nd = nd_default[0] if nd is None else nd
    return '??' if x is None or x != x else f'{x:+.{nd}f}'.replace('-', '−')

  def est(x):          # paired statistics carry 'diff', one-sample ones 'mean'
    return x.get('diff', x.get('mean'))

  helpers = dict(
      S=S, X=X, SC=SC, n=n, sg=sg,
      xst=lambda key, state, group, metric: X[key]['stats'][state][group][metric],
      st=lambda state, group, metric: S['stats'][state][group][metric],
      pc=lambda state, room, metric: S['per_condition'][state][room][metric],
      gp=lambda x, nd=None: f'grid {n(x["gc"], nd)}, PGC {n(x["pgc"], nd)}',
      vs=lambda x, nd=None: f'{n(x["gc"], nd)} vs {n(x["pgc"], nd)}',
      d=lambda x, nd=None: f'{sg(est(x), nd)} ± {n(x["sem"], nd)}',
      ci=lambda x, nd=None: f'{sg(est(x), nd)} (95 % CI {sg(est(x) - x["ci95"], nd)} to {sg(est(x) + x["ci95"], nd)})',
      p=lambda x: fmt_p(x.get('p')),
      pp=lambda x: fmt_p(x.get('p_perm')),
      k=lambda x: f'{x["pos"]}/{x["n"]}',
      full=lambda x, nd=None: f'{sg(est(x), nd)} ± {n(x["sem"], nd)}, {fmt_p(x.get("p"))}, PGC higher in {x["pos"]}/{x["n"]} seeds',
      pv=lambda x: fmt_p(x.get('p')).replace('p = ', '').replace('p ', ''),
      row=lambda label, x, nd=None: (f'| {label} | {n(x["gc"], nd)} | {n(x["pgc"], nd)} | {sg(x["diff"], nd)} ± {n(x["sem"], nd)} | '
                                     + fmt_p(x.get("p")).replace('p = ', '').replace('p ', '') + f' | {x["pos"]}/{x["n"]} |'),
      pct=lambda x, nd=0: '??' if x is None else f'{100 * x:.{nd}f} %',
      chg=lambda key, group, metric: S['change'][key][group][metric],
      one=lambda x, nd=None: f'{sg(x["mean"], nd)} ± {n(x["sem"], nd)}',
      npos=lambda state, rooms: sum(S['per_condition'][state][r]['score']['diff'] > 0 for r in rooms),
      changed_rooms=[r for r in S['per_condition'].get('before', {}) if r != 'open'],
      trained_rooms=S['groups']['changed'], new_rooms=S['groups']['new_rooms'] + S['groups']['new_rooms_goal_moved'],
      extra=(S.get('long_m') or 0) - 1000, long_b=(S.get('long_m') or 0) / 1000.0,
      sum=sum, max=max, min=min, len=len, abs=abs, str=str, int=int, sorted=sorted, round=round, zip=zip, enumerate=enumerate,
      range=range)
  helpers['fmt_p'] = fmt_p

  def conf_table():          # the planned contrasts of S['confirmatory'] as table rows
    rows = []
    for f in S.get('confirmatory', []):
      nd = 4 if 'share' in f['label'] else 1
      rows.append(f'| {f["label"]} | {sg(f["estimate"], nd)} | {sg(f["estimate"] - f["ci95"], nd)} to {sg(f["estimate"] + f["ci95"], nd)} | '
                  + fmt_p(f['p_perm']).replace('p = ', '').replace('p ', '') + ' | '
                  + fmt_p(f['p_holm']).replace('p = ', '').replace('p ', '') + f' | {f["pos"]}/{f["n"]} |')
    return '\n'.join(rows)
  helpers['conf_table'] = conf_table

  def ref_table():           # the four reference agents (seeds 1-2): score per state
    lm = S.get('long_m') or 0
    lab = [('before', '1 B'), ('same', '1.25 B, all in the familiar room'), ('after', '1.25 B, last 250 M in the changed rooms'),
           ('same_long', f'{lm / 1000:g} B, all in the familiar room'), ('long', f'{lm / 1000:g} B, last {lm - 1000} M in the changed rooms')]
    rows = []
    for a, v in S.get('reference', {}).items():
      name = ('grid' if a.startswith('gc100') else 'PGC') + ', seed ' + a[-1]
      rows += [f'| {name} | {t} | {n(v[k]["familiar"])} | {n(v[k]["changed"])} | {n(v[k]["new_rooms"])} |' for k, t in lab if k in v]
    return '\n'.join(rows)
  helpers['ref_table'] = ref_table

  def pcell(x):
    return fmt_p(x.get('p')).replace('p = ', '').replace('p ', '')

  def scr_table():           # from-scratch agents in their own room against the pretrained agents in the same rooms
    rows = []
    T = SC.get('tests', {})
    for mark in sorted(T, key=int):
      sr = T[mark].get('single_rooms')
      if sr and len(sr['rooms']) >= 9:
        x = sr['own_room']
        part = '' if len(sr['rooms']) == 11 else f' ({len(sr["rooms"])} of 11 rooms)'
        rows.append(f'| from scratch, {mark} M frames in that room alone{part} | {n(x["gc"])} | {n(x["pgc"])} | {sg(x["diff"])} ± {n(x["sem"])} | {pcell(x)} | {x["pos"]}/{x["n"]} rooms |')
    lab = {'1000M': 'trained 1 B frames in the familiar room, frozen', '1250M': 'the same agents + 250 M frames on the mixture of changed rooms',
           '1500M': 'the same agents + 500 M frames on the mixture'}
    for tag in ('1000M', '1250M', '1500M'):
      r = SC.get('finetuned_reference', {}).get(tag)
      if r and 'stat_rooms' in r:
        x = r['stat_rooms']
        rows.append(f'| {lab[tag]} | {n(x["gc"])} | {n(x["pgc"])} | {sg(x["diff"])} ± {n(x["sem"])} | {pcell(x)} | {x["pos"]}/{x["n"]} seeds |')
    return '\n'.join(rows)
  helpers['scr_table'] = scr_table

  def scr_mark():            # the latest frame mark at which every single-room pair has been tested
    T = SC.get('tests', {})
    ok = [m for m in sorted(T, key=int) if len(T[m].get('single_rooms', {}).get('rooms', [])) >= int(os.environ.get('ROOM_SCR_MIN', 11))]
    return ok[-1] if ok else None
  helpers['scr_mark'] = scr_mark

  def scr_ref():             # the fine-tuned state closest in changed-room frames to the from-scratch mark
    m_ = int(scr_mark() or 0)
    return '1500M' if (m_ >= 400 and '1500M' in SC.get('finetuned_reference', {})) else '1250M'
  helpers['scr_ref'] = scr_ref
  helpers['scr_ref_m'] = lambda: int(scr_ref()[:-1]) - 1000
  helpers['SR'] = lambda: SC['tests'][scr_mark()]['single_rooms']          # single-room block at the reported mark
  helpers['ST'] = lambda: SC['tests'][scr_mark()]

  LEARNED = 35.0             # own-room score below which a from-scratch agent counts as not (fully) having learned its room

  def scr_unlearned():       # 'D1 (grid 14.4, PGC 14.1); D3 (PGC 12.4) ...': agents below LEARNED in their own room
    out = []
    for e, v in SR()['per_room'].items():
      low = [f'{lab} {n(v[k])}' for k, lab in (('gc', 'grid'), ('pgc', 'PGC')) if v[k] < LEARNED]
      if low:
        out.append(('the open room with a moving goal' if e == 'R' else e) + ' (' + ', '.join(low) + ')')
    return '; '.join(out) if out else 'none'
  helpers['scr_unlearned'] = scr_unlearned

  def scr_learned():         # PGC - grid over the rooms that both agents of the pair have learned (one-sample statistic)
    d = [v['pgc'] - v['gc'] for v in SR()['per_room'].values() if v['gc'] >= LEARNED and v['pgc'] >= LEARNED]
    m_ = sum(d) / len(d)
    sem = (sum((x - m_) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
    return f'{sg(m_)} ± {n(sem)} over the {len(d)} rooms'
  helpers['scr_learned'] = scr_learned
  helpers['SR'] = lambda: SC['tests'][scr_mark()]['single_rooms']
  SR = helpers['SR']

  def scr_room_table(mark=None):   # one row per room: from scratch (own room) and the pretrained agents in that room
    mark = mark or scr_mark()
    if mark is None:
      return '| (tests pending) | | | | | | |'
    pr = SC['tests'][mark]['single_rooms']['per_room']
    fr = SC.get('finetuned_reference', {})
    tags = [t for t in ('1000M', '1250M', '1500M') if t in fr]
    rows = []
    for e, v in pr.items():
      name = 'open room, goal moves' if e == 'R' else e
      rows.append(f'| {name} | {v["seed"]} | {n(v["gc"])} | {n(v["pgc"])} | '
                  + ' | '.join(f'{n(fr[t]["gc100"]["per_room"][e])}, {n(fr[t]["pgc33"]["per_room"][e])}' for t in tags) + ' |')
    g = [v['gc'] for v in pr.values()]
    q = [v['pgc'] for v in pr.values()]
    rows.append(f'| **mean of the 11 rooms** | | **{n(sum(g) / len(g))}** | **{n(sum(q) / len(q))}** | '
                + ' | '.join(f'**{n(fr[t]["gc100"]["rooms"])}, {n(fr[t]["pgc33"]["rooms"])}**' for t in tags) + ' |')
    return '\n'.join(rows)
  helpers['scr_room_table'] = scr_room_table

  def scr_room_header():
    fr = SC.get('finetuned_reference', {})
    lab = {'1000M': 'pretrained, frozen', '1250M': 'pretrained + 250 M', '1500M': 'pretrained + 500 M'}
    tags = [t for t in ('1000M', '1250M', '1500M') if t in fr]
    return ('| room | seed | from scratch: grid | from scratch: PGC | ' + ' | '.join(lab[t] + ': grid, PGC' for t in tags) + ' |\n'
            + '|---|---|---|---|' + '---|' * len(tags))
  helpers['scr_room_header'] = scr_room_header

  def mix_table():           # from scratch on the mixture (seeds 1-2) against the pretrained agents of the same seeds
    rows = []
    T = SC.get('tests', {})
    for mark in sorted(T, key=int):
      m = T[mark].get('mixture', {})
      if len(m) == 4:
        rows.append(f'| from scratch on the mixture, {mark} M frames | {n(m["gc100_s1"]["changed"])}, {n(m["gc100_s2"]["changed"])} | '
                    f'{n(m["pgc33_s1"]["changed"])}, {n(m["pgc33_s2"]["changed"])} | {n(m["gc100_s1"]["familiar"])}, {n(m["gc100_s2"]["familiar"])} | '
                    f'{n(m["pgc33_s1"]["familiar"])}, {n(m["pgc33_s2"]["familiar"])} |')
    lab = {'1000M': 'trained 1 B frames in the familiar room, frozen', '1250M': 'the same agents + 250 M frames on the mixture', '1500M': 'the same agents + 500 M frames on the mixture'}
    for tag in ('1000M', '1250M', '1500M'):
      r = SC.get('finetuned_reference', {}).get(tag)
      if r and all(len(r[c]['per_seed']) >= 2 for c in ('gc100', 'pgc33')):
        g, q = r['gc100']['per_seed'], r['pgc33']['per_seed']
        rows.append(f'| {lab[tag]} | {n(g[0]["changed"])}, {n(g[1]["changed"])} | {n(q[0]["changed"])}, {n(q[1]["changed"])} | '
                    f'{n(g[0]["familiar"])}, {n(g[1]["familiar"])} | {n(q[0]["familiar"])}, {n(q[1]["familiar"])} |')
    return '\n'.join(rows)
  helpers['mix_table'] = mix_table

  def seeds(x, key, nd=1):   # per-seed values of one code, in seed order
    return ', '.join(n(v, nd) for v in x[key + '_seeds'])
  helpers['seeds'] = seeds

  def sub(m):
    try:
      return str(eval(m.group(1), dict(helpers, __builtins__={})))      # helpers as globals: visible inside comprehensions
    except Exception as e:           # a number that does not exist (yet) stays visible
      return f'??[{m.group(1)}: {type(e).__name__}]'

  text = open(args.template or os.path.join(args.dir, 'README.template.md')).read()
  # optional blocks: <!--IF long--> ... <!--END--> are kept only when the summary has that state
  def block(m):
    key = m.group(1)
    if key in ('scratch', 'noscratch'):      # from-scratch section: only when every single-room pair has a frozen test
      return m.group(2) if (helpers['scr_mark']() is not None) == (key == 'scratch') else ''
    return m.group(2) if key in S.get('stats', {}) or S.get(key) or key in X else ''
  # block form (markers on their own lines) first, then the inline form inside a sentence; blocks do not nest
  text = re.sub(r'^<!--IF (\w+)-->\n(.*?)^<!--END-->\n', block, text, flags=re.S | re.M)
  text = re.sub(r'<!--IF (\w+)-->(.*?)<!--END-->', block, text, flags=re.S)
  out = re.sub(r'\{\{(.+?)\}\}', sub, text)
  path = args.out or os.path.join(args.dir, 'README.md')
  open(path, 'w').write(out)
  missing = len(re.findall(r'\?\?', out))
  print(f'wrote {path} ({missing} unresolved placeholders)')


if __name__ == '__main__':
  main()
