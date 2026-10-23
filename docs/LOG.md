# Session recovery notes

Project: perfect grid-cell vs predictive-grid-cell population codes fed to a
Banino-style RL agent (first experiment of the project "Behavioral
optimization reveals principles of neural coding"). See README.md for the
layout.

## 2026-10-10 — build day
- Code written and unit/smoke-tested (`tests/`, 10 passing). Validation of the
  codes with the paper pipeline's GridScorer: `runs/validation_pgc33_s1/`
  (grid cells peak at ~0 shift, PGCs at +16.5 cm for a configured 20 cm).
- Throughput: ~29k fps single process, ~17.5k fps each with 4 concurrent.
- Finding: with the SI Table 2 optimizer settings (lr 1e-4, RMSProp eps 0.1)
  the policy is still uniform after 4M frames (0.1 goals/episode = random
  walk); the earlier banino FakeEnv runs needed >100M frames to leave chance.
  25M-frame optimizer probe (gc100/lstm, seed 11), return after ~20M frames:
  SI default lr 1e-4 eps 0.1: 1.4 (chance) | lr 2e-4 eps 0.1: 1.0 (chance) |
  forward-biased init, eps 0.1: 3.8 (higher chance line, no learning) |
  lr 1e-4 **eps 1e-5**: 7.4 and rising from ~17M | lr 3e-4 **eps 1e-5**: 21
  (2.1 goals/episode) by 18M. => RMSProp eps 0.1 (Mnih-2016 value) makes the
  effective step ∝ the tiny sparse-reward gradients; a normalised step
  (eps 1e-5, the standard A2C value) is required. Confirmation probes E (ff
  policy, lr 3e-4) and F (lr 7e-4) running; the matrix will use
  `--train_args "--lr 3e-4 --rms_eps 1e-5"` (or 7e-4 if F is faster and stable).
- **Saturation trap found (16:40).** With eps 1e-5 and lr >= 3e-4 the LSTM
  agent's "fast learning" was degenerate: the vision embedding exploded
  (|e| mean 32, max 484 at 22M frames; 138/1403 at lr 7e-4), 87 % of LSTM
  gate pre-activations exceeded +/-15 (exactly 0/1 in float32) and the policy
  output became EXACTLY independent of the population code (probes intact =
  no_code = no_goal bit-identical; code input-weight columns still at init).
  lr 1e-4 eps 1e-5 (probe B) stayed healthy to 25M (|e| 1.1, no saturation,
  return 16.8) but the embedding scale was growing. The ff policy at lr 3e-4
  did NOT blow up (|e| 0.08) and its logits depend strongly on the code.
  Added `--encoder_norm layernorm` (LayerNorm on the 256-d embedding) and
  per-log diagnostics (e_abs_mean/max, pre_abs_mean, frac_saturated,
  drive_vision/drive_code) to train.py. Round-2 probes (scratch `probe2/`):
  H LN lr3e-4 eps1e-5 | I lr3e-4 eps1e-3 | J LN lr3e-4 eps1e-3 |
  K LN lr1e-4 eps1e-5 | L ff LN lr3e-4 eps1e-5.
- Reliance at 25M (probe B, lstm, healthy): intact 16.6, no_code 14.8,
  no_goal 16.3, untrained 1.7 -> the paper-style LSTM policy uses the code
  only weakly (vision + dead-reckoning), as in the Aug-2026 attempt; the ff
  policy is the code-dependent regime.
- Round-2 probes (~13M frames; return / saturation): H LN+lr3e-4+eps1e-5: 26,
  none | I lr3e-4 eps1e-3: 8 | J LN lr3e-4 eps1e-3: 17.5 | K LN lr1e-4 eps1e-5:
  13.7 | L ff LN lr3e-4 eps1e-5: 44 (4.4 goals/ep). **Decision: LayerNorm +
  lr 3e-4 + eps 1e-5 for every run** (scheduler default `--train_args`).
- ff probe E (20M): intact 23.2 / no_code 4.0 / no_goal 24.1 / untrained 1.2:
  the ff policy is code-dependent (-83 % without the code) but at 20M frames
  it does not yet use the goal snapshot (code-driven search, path eff 0.28).
- Review findings fixed (17:30): motion direction now only updates when the
  agent moves faster than v_min=0.05 m/s (was noise-dominated while turning,
  scrambling space-mode PGCs); landmark/stripe layout re-drawn every episode
  (paper: intra-maze cues differ per episode; a fixed layout let one frame
  reveal position to ~10 cm); episode clocks staggered at reset (envs were
  phase-locked); previous action/reward reset at episode start (-1 sentinel =
  no action); --resume refuses a changed experiment definition; probes
  validated; eval files carry provenance (canonical only for ckpt_final +
  200 episodes); validate_codes handles conditions without PGCs and reports
  running-speed cm conversion + plateau centres (PGC: 19.5 +/- 1.2 cm);
  --goal_code_shift zero control added; update pass encodes frames once.
- Confirmation runs on the revised env (gc100/pgc33 x lstm/ff, chosen
  optimizer): all four learn by 3.8M frames with zero saturation (stopped
  early to free the GPU for wave 1).
- **Review verdicts + design critique (16:40)** adopted: (a) raycast vision
  localises to ~7 cm from one frame even with per-episode textures (room
  geometry + fixed wall colours), so ff+raycast is a "vision+code" arm; the
  code-only arm is ff + heading-only vision (`ffb`); (b) the PGC part of g*
  is a better-centred goal marker (0.17 vs 0.29 m from the goal centre) ->
  goal-code factor: pgc33 goal intact vs zero, plus gc100 with a look-ahead
  snapshot; (c) homing had not emerged at 20M (no_goal probe costs nothing)
  -> pilot wave with checkpoint probes decides budget/arms; (d) new probes:
  no_current, sham0-4 (count-matched lesion), shuffled_current, fake_goal;
  chance line = mean over 3 random inits; evaluation episodes are paired
  across runs (layouts keyed by slot/episode/draw).
- **WAVE 1 LAUNCHED 16:58 (box clock)** (tmux `popcode`): gc100 / pgc33 / none x
  {ffb, ff, lstm}, seed 1, 150M frames, checkpoint probes every 25M
  (`runs/<run>/eval/<probe>@<frames>f_64ep.json`). Decision rules before
  wave 2: an arm is interpretable only if gc100 - none and intact - no_code
  are both >= ~30 % of the learned gain; homing emerged if no_goal costs
  >= 30 % of intact (or path efficiency rises well above no_goal). Wave 2 =
  gc100 vs pgc33 (+ pgc33 goal-zero) x 8 seeds in the arms that pass, runs
  extended to the chosen budget; wave 3 = controls.

## How to resume the experiment
```
tmux new -d -s popcode "cd /home/ec2-user/optimization && /home/ec2-user/banino/.venv/bin/python scripts/launch_matrix.py --wave 1 --max_concurrent 9"
/home/ec2-user/banino/.venv/bin/python scripts/status.py          # progress
/home/ec2-user/banino/.venv/bin/python -m popcode.analyze --runs_dir runs --out results
```
The scheduler is idempotent (skips finished runs, resumes from resume.pt,
extends runs whose --frames grew, probes checkpoints, evaluates finished runs).
`--wave 2` / `--wave 3` add the later waves (wave-1 runs are extended in place).

## 2026-10-10 20:15 — wave-1 checkpoint probes (one seed, 64-episode probes at 25/50/75/100M)
Reliance guard mostly FAILS in the 2.5 m arena: at 100M, no-code agents score
about as well as coded ones (lstm: none 41 vs gc100 40; ffb: none 23 vs gc100
20; ff: none 32 vs gc100 45 but equal at 75M). Zeroing the code at test costs
gc100 only 2-11 points, zeroing the goal memory costs <= 6 -> agents mostly
solve the task by efficient SEARCH (post-discovery goal rate of the no-code
ff agent is the highest, 4.5-5.4 per 1000 steps), and goal-homing has barely
emerged by 100M. A search-dominated task caps any code advantage. pgc33 vs
gc100 at this stage: no consistent difference (ff 31 vs 45, ffb 25 vs 19,
lstm 35 vs 38; single seed, noisy).
Action: launched a harder-arena pilot (tmux `popcode_a4`, `runs_a4/`):
4 m arena, goal radius 0.25 m, 120 s episodes (1800 steps), ff + ffb arms x
gc100/pgc33/none, 100M frames, checkpoint probes. Search by sweeping is then
~6x slower, so goal memory / position code should become load-bearing.
Wave 1 (2.5 m) continues to 150M for its final 200-episode evaluation.

## 2026-10-10 22:15 — pilot verdict and WAVE 2 LAUNCH
2.5 m arena, final 200-episode paired evals at 150M (seed 1):
ff: gc100 46.5 (no_code 47.9) | pgc33 51.7 (no_code 43.2, no_goal 40.7) | none 40.6
ffb: gc100 18.6 (no_code 21.7) | pgc33 26.3 (no_code 21.5) | none 22.0
lstm: gc100 49.2 (no_code 37.9) | pgc33 40.6 (no_code 37.6) | none 38.7
-> code reliance and goal-homing DO emerge by 150M in some agents (pgc33_ff,
gc100_lstm); direction of the gc-vs-pgc difference flips between arms at
n=1, as expected from the +/-8 checkpoint-to-checkpoint scatter. Decision:
run the full wave 2 in the 2.5 m arena at 300M frames, all three arms.
- CLOUD (gcloud project connectome-ai-research, bucket
  connectome-ai-research-popcode, fleet tag `main`): 96 runs on 32 x
  g2-standard-8 (1 L4 each, 3 runs per VM, ~$27/h, est. ~$220 total):
  gc100 / pgc33 / pgc33goalzero x 3 arms x seeds 2-8 (63), none x 3 arms x
  seeds 2-3 (6), pgc100 / gc100vel / gc100lookahead x 3 arms x seeds 1-3 (27).
  Each VM syncs to gs://.../runs/<vm>/ every 10 min and shuts down when done.
  `python scripts/gcp_fleet.py status|collect|cleanup --tag main`.
- LOCAL (tmux `popcode_w2`): seed-1 runs extended 150M -> 300M (old evals
  demoted to eval/<probe>@150003200f_200ep.json) + pgc33goalzero seed 1;
  4 m arena pilot (`runs_a4`, tmux `popcode_a4`) continues to 100M as a hedge.
- Cloud test VM (popcode-test): L4 trains at ~19k fps single-process; full
  15-probe evaluation 12.5 min; results + DONE marker synced; VM deleted.

## 2026-10-10 22:40 UTC (5:40 pm CST) — local box saturated; seed-1 extensions moved to cloud
Local GPU at 100 % util / CPU load 31 of 32 (4 m pilot x6 + another user's
fly-brain process). The 12 local seed-1 wave-2 runs were NOT started locally;
instead their run dirs are uploaded to gs://.../runs_ext/popcode-ext-0N/ and
4 L4 VMs (fleet tag `ext`) resume them 150M -> 300M (+ 3 fresh
pgc33goalzero seed-1 runs). Collect with
`python scripts/gcp_fleet.py collect --tag ext --dest runs`. The local GPU
now only runs the 4 m pilot (`runs_a4`). User asked for all times in CST
(box clock is UTC).
- 22:50 UTC: user: "don't reduce anything or scale; cloud VMs are OK". Added the
  PGC-shift dose series (pgc33 shift 10 / 40 cm x 3 arms x seeds 1-3 = 18
  runs) on 6 VMs, fleet tag `dose` (collect with
  `python scripts/gcp_fleet.py collect --tag dose --dest runs`). Fleet now:
  32 (main) + 6 (dose) + 4 (ext, pending) L4 VMs, ~$36/h, projected total
  ~$265 for wave 2 + controls + dose series.
- 17:40-17:55 CST incident: the first fleet monitor treated provisioning VMs
  as finished and deleted ext-00/ext-03 while they booted; the relaunch then
  created duplicates (VM names are unique per ZONE, the zone-retry recreated
  existing names in zone a). Fixed: monitor collects/deletes only when all
  DONE markers exist; launcher skips existing names and never zone-retries
  on 'already exists'. Duplicates deleted; ext fleet = ext-00 (a), ext-01 (b),
  ext-02 (c), ext-03 (b). Bucket state (runs_ext/) was not corrupted (the
  short-lived VMs never reached their first 10-min sync).

## 2026-10-10 23:50 UTC (6:50 pm CST) — 4 m arena pilot verdict: shelved
Final 200-episode evals at 100M (seed 1): ff: gc100 13.1 | pgc33 9.0 | none
13.0; ffb: gc100 9.4 | pgc33 8.5 | none 9.0. No homing in any agent (no_goal
= intact), no-code agents equal to grid agents. The harder arena only slows
search; agents would need far more than 100M frames (the paper's 1e9) to
exploit a goal memory there. Kept as results_a4/ (negative pilot); no further
runs. Wave 2 (2.5 m arena, 300M frames, 8 seeds) is the primary experiment.

## 2026-10-11 00:15 CST — WAVE 2 RESULT (126 runs, 300M frames, 8 seeds primary)
Primary paired contrast pgc33 - gc100 (final 200-episode paired evals):
  ffb (code-only FF):  27.8 -> 31.2, +3.38 +/- 0.92, CI [1.6, 5.0], p=0.008 (Wilcoxon 0.039)
  ff  (FF + vision):   68.2 -> 71.8, +3.6 +/- 5.5, p=0.53; path eff +0.043 +/- 0.014 p=0.018;
                       early AUC +2.3 +/- 1.0 p=0.05; frames-to-80% -35M p=0.09
  lstm (paper):        66.4 -> 66.6, +0.2 +/- 6.4, p=0.98; path eff +0.057 +/- 0.030 p=0.10
Controls: none = 21.4 / 41.0 / 41.9 (code load-bearing everywhere); pgc100 in ffb
36.0 (+10.8, p=0.008, n=3); gc100+vel -0.1; gc100 lookahead-goal +2.0 (ffb),
+18 (ff, n=3, p=0.2); shift10/40 +2.0/+2.5 (ffb, ns); goal-zero PGC = PGC.
PGC-specific lesion (sham - no_pgc): +6.3 / +6.5 / +4.0 in ffb / ff / lstm.
Homing (intact - no_goal): ~0 in ffb (search strategy), 30 / 27 in ff / lstm.
Outputs: results/RESULTS.md, results/summary.json, results/*.png; runs/ (126 dirs).
Cloud: ~$270 spent; fleet_monitor deletes the last VMs after collection.

## 2026-10-11 00:40 CST — verification and repository
Three independent re-analyses (raw eval JSONs, no project code) reproduce
every number in results/RESULTS.md to machine precision; integrity audit of
the 72 primary runs passed (configs, 300M checkpoints, 15 canonical probes,
bit-identical populations within seed, effective episode pairing). Flagged:
LSTM runs carry <= 0.4 % saturated gates (benign; reliance positive in all);
the ff/lstm paired differences are dominated by a few seeds whose gc100
partner under-used the code (those arms are nulls anyway); the lstm
shift-10/40 rows were initially built from partial evaluations (the last
dose VM finished later) and have been re-collected and re-analysed.
Figures regenerated by scripts/make_figures.py; shareable data packaged by
scripts/package_data.py (data/). Cloud spend ~$270; all VMs deleted.

## 2026-10-11 ~1:40 pm CST — follow-up experiment (user request)
Only the code-only arm (ff + heading-only vision), where the code was
load-bearing and PGCs helped. (1) Arrangement of the 169 PGCs among the 512
cells: candidate layouts implemented in codes.py (`--pgc_layout` uniform /
fine / coarse / middle / weights; `--pgc_shift_rule spacing` = constant phase
shift of 0.5 x module spacing; `--pgc_shift_jitter`), chosen after a
literature + cohort-data survey (workflow). (2) Concentration series 10 / 25 /
30 / 35 / 40 % on the best arrangement. (3) Band-cell pilot (`--cell_type
band`: one cosine per cell along one of the module's lattice directions;
predictive bands shifted like PGCs): band / pband33 / pband100 x 4 seeds,
running LOCALLY (tmux popcode_band) while the cloud token is refreshed.
Budget: $300 (arrangements 40 runs + concentrations 40 runs ~ $140 on L4s).
- 2:25 pm CST: arrangement wave launched on GCP (fleet tag `v2a`, 14 L4 VMs,
  40 runs = pgc33fine / pgc33coarse / pgc33middle / pgc33phase / pgc33jitter
  x seeds 1-8, code-only arm, 300M frames). Some us-central1 zones were out of
  L4 capacity; 4 VMs were placed in us-west1-a / us-east4-a / us-central1-c.
  Monitor: tmux `fleet_monitor_v2a` (runs/fleet_monitor_v2a.log). Band pilot
  continues locally (tmux `popcode_band`).
- ~2:50 pm CST: survey verdict (3 agents; Ouchi & Fujisawa 2024 read directly;
  cohort + MEC data re-analysed): PGC shift = 0.27 +/- 0.11 of each cell's
  spacing (constant phase, invariant under grid rescaling), mixed within every
  module (layer-segregated, not module-segregated), CV of shifts ~0.4-0.6;
  MEC: fewest PGCs in the finest module; half-spacing shift predicted worst.
  Addendum fleet `v2a2` (11 VMs, 32 runs, 8 seeds each): pgc33phase27
  (0.27 x spacing), pgc33bio (0.27 x spacing + lognormal jitter 0.4),
  pgc33mec (module weights 1:2:2:1), pgc33retro (20 cm BEHIND: control for
  heading information vs prediction). auto_wave_b now waits for v2a + v2a2
  and picks the best of 9 arrangements by mean final score.

## 2026-10-11 ~10:30 pm CST — arrangement wave finished; figures
All 72 arrangement runs (v2a + v2a2) collected by 8:35 pm. With 200-episode
finals, no arrangement beats uniform significantly (best: quarter-spacing
shift 32.8 vs 31.2, +1.6 +/- 1.8; MEC-weighted 32.6, +1.4 +/- 2.4). Backward,
half-spacing and fine-only codes fall to grid-only level and their agents stop
relying on the PGCs (lesion cost 1-3 vs 6-12). MEC-weighted learns fastest
(-20M frames to a training score of 25, p = 0.016) but has the widest seed
spread. auto_wave_b picked pgc33phase27 (highest mean) and launched the
concentration fleet `v2b` (14 VMs, 40 runs) at 8:53 pm; expected done ~3 am.
Band pilot at ~235M frames locally. Spend so far ~$127 of $300.
New figures: fig0 (how the modules are built), fig7 (all arrangements), and
figures/arrangements/<cond>.png (one per arrangement); write-up in
docs/arrangements.md. Follow-up code (pgc_layout / shift rules / band cells,
make_matrix_v2.py, auto_wave_b.sh) committed; results/ and data/ will be
refreshed when the concentration and band runs are in.

## 2026-10-11 ~11:40 pm CST — package renamed popcode/ -> code/
User request. `git mv popcode code`; every module path (`-m code.train`,
`from code.common import ...`) updated in code/, tests/, scripts/ and README.
Cloud resource names are unchanged (bucket connectome-ai-research-popcode,
bucket prefix popcode/, VM names popcode-<tag>-NN) and so are tmux session
names. A local, untracked symlink `popcode -> code` keeps the jobs that were
already running (band-pilot scheduler, auto_wave_b, fleet_monitor_v2b) working
until they finish; auto_wave_c was restarted with the new name. Note that
`code` shadows the standard-library module of the same name when Python is
started from the repository root (only matters for tools that import it, e.g.
pdb); everything here is run via `python -m code.<module>` from the root.

## 2026-10-12 ~2:50 am CST — concentration series + band pilot finished
Fleet v2b (14 VMs, 40 runs) collected 2:46 am, VMs deleted 2:47 am (~$70).
Quarter-spacing code, 8 seeds, paired vs 33 %: 10 % 28.0 (-4.8, p=0.009),
25 % 32.7 (-0.0), 30 % 29.9 (-2.8, ns), 33 % 32.8, 35 % 31.3 (-1.5, ns),
40 % 30.6 (-2.1, ns). Best = 33 % -> auto_wave_c: no rerun (guard: rerun only
if another percentage beats 33 % with p < 0.05). Band pilot (local, 4 seeds)
finished 1:02 am: band 25.0 vs grid 26.6 (ns); band + 33 % predictive 30.4
(+5.4 vs band, p=0.04, 4/4); 100 % predictive band 32.4 (+7.4, p=0.009, 4/4);
predictive bands ~= PGCs at 33 %, -4.6 at 100 % (p=0.05). auto_wave_b
regenerated results/, figures/ (fig8, fig9) and data/ (250 runs). Follow-up
spend ~$200 of $300. Nothing is running; all VMs deleted.

## 2026-10-20 ~11:30 pm CST — wave E: controls, 600 M extensions, long runs
Code: conditions `gc67` (the pgc33 population with its 169 PGC slots held at
zero: tests whether the PGC advantage could come from having fewer grid-cell
inputs) and `pgc50` (256 grid + 256 PGCs: class-size-matched lesions), probes
`pgc_half0-2` / `gc_half0-2`, weights-only extension of finished runs
(`train.py --resume` with `--warmup_updates 200` at lr 0; provenance in
config.json `extensions`), `scripts/make_matrix_v3.py controls|extend`,
`scripts/extension_report.py`, `scripts/auto_wave_d.sh` (post-processing),
`scripts/auto_wave_e.sh` (fleet collection), `scripts/relative_benefit.py`
(results/relative_benefit.md, Fig. 10: PGC/grid ratio per arm, per-cell lesion
costs, secondary endpoints) and docs/shift_vs_spacing.md (source check of the
shift-vs-spacing statements against Ouchi & Fujisawa 2024 and the RNN cohort).
Runs (matrices in runs/waveE/): GCP fleet `ctl` (24 L4 VMs, 72 runs, 300 M):
gc67 and pgc50 in all three arms x 8 seeds, pgc50phase27 / pgc50mec /
pgc50retro in the code-only arm x 8 seeds -> runs/. Fleet `ext6` (8 VMs):
gc100 / pgc33 x ffb, ff x seeds 3-8 extended 300 M -> 600 M. Fleet `long`
(5 VMs): gc100 / pgc33 ff seeds 1-2 -> 1.5 B; gc100 / pgc33 / pgc33phase27 /
pgc33mec / pgc33retro ffb seeds 1-2 -> 1 B. Local (tmux `popcode_ext_lstm`):
gc100 / pgc33 lstm seeds 1-8 -> 600 M. Extended runs live in runs_ext/
(copies of config, final checkpoint, logs and evaluations; the 300 M finals
are kept as eval/<probe>@300006400f_200ep.json); runs/ keeps the canonical
300 M results. us-central1 L4 quota was full after `ctl` + `ext6`, so the
`long` VMs were placed in us-west1 / us-east1 by a retry loop (tmux
`popcode_long_retry`). Driver: tmux `auto_wave_e` (runs/auto_wave_e.log)
collects each fleet when its DONE markers are complete, deletes the VMs, waits
for the local scheduler, then runs `scripts/auto_wave_d.sh --now` (200-episode
evaluations of the 600 M and 1 B checkpoints of the long runs, analyze ->
results/, extension_report -> results_ext/EXTENSION.md + Fig. 11, figures,
data/). Projected cost ~$400 of a $500 budget; expected to finish Oct 22
(controls, 600 M) and Oct 22-23 (long runs).

## 2026-10-21 ~11 pm CST — wave E finished: controls, 600 M extensions, long runs
All three fleets finished and were collected; the five `long` VMs were
deleted at 10:45 pm CST. Cloud cost of the wave ~$235 (driver estimate; the
controls and 600 M fleets ran faster than projected). runs/ now holds 322
runs at 300 M (the 72 control runs added). runs_ext/ holds 54 extended runs:
40 to 600 M (gc100 / pgc33: seeds 3-8 in the code-only and code+vision arms,
seeds 1-8 in the LSTM arm), ten to 1 B (code only, seeds 1-2: gc100, pgc33,
pgc33phase27, pgc33mec, pgc33retro) and four to 1.5 B (code+vision, seeds
1-2: gc100, pgc33), each with the full probe set at its final checkpoint and,
for the long runs, at 600 M and 1 B (200 episodes). Post-processing
(scripts/auto_wave_d.sh --now): results/RESULTS.md and data/ regenerated over
the 322 runs, results_ext/EXTENSION.md + extension.json, Fig. 11 (now two
rows: paired gap per seed per budget on top, training curves per arm with the
saturating fits below). Fixes: the checkpoint-probe scheduler in
scripts/launch_matrix.py (and the recovery path in code/train.py) matched only
`ckpt_0*.pt`, so the periodic 64-episode probes of the long runs stopped at
975 M; the glob is now `ckpt_[0-9]*.pt`, and the missing probes of the four
1.5 B runs (1.0-1.5 B) were evaluated locally. Milestone labels round to 2
decimals ('1.5B', not '1.50001B').
Results (README 3.7): the paired gap pgc33 - gc100 is significant at 600 M in
every arm (code only +3.4 -> +11.5, wider in 8/8 seeds, change p 0.002;
code+vision +3.6 -> +16.7, change p 0.065; LSTM +0.2 -> +10.0, change p
0.016), with path efficiency and post-discovery goal rate following and the
time to the first goal unchanged. At 600 M both vision arms are still rising
(grid ~5, PGC ~8-9 points per 100 M); the two code+vision seeds trained to
1.5 B level off (95 % of the fitted asymptote at ~450-600 M for grid,
~650-800 M for PGC), their fitted plateaus 13.9 and 15.5 apart. Code only: at
600 M (seeds 3-8) the grid code climbs about half as fast as pgc33 (2.9 vs
5.9 points per 100 M, p 0.012); at 1 B (2 seeds) the predictive codes still
climb 3-8 points per 100 M, one grid seed has been flat since 600 M and the
other rose late (24.1 -> 40.0); gap +24.9 at 1 B. The retrospective code
scores 63 / 46 at 1 B (grid 40 / 27), but its two extended seeds were
already level with the forward shift at 300 M and its PGC-specific lesion
cost is inconsistent (-8 / +11), so the 300 M conclusion of README 3.4
stands.

## 2026-10-22 ~12:15 am CST — wave F launched: code-only runs to 2 B, LSTM runs to 1 B
Code-only arm, seeds 1-2 (gc100, pgc33, pgc33phase27, pgc33mec, pgc33retro):
1 B -> 2 B on fleet `long2` (5 L4 VMs, 2 runs each). LSTM arm, gc100 / pgc33
seeds 1-8: 600 M -> 1 B, seeds 1-4 locally (tmux `popcode_ext_lstm2`, 8
concurrent, ~9.8 k fps per run) and seeds 5-8 on fleet `lstm1b` (4 L4 VMs,
2 runs each). All 16 LSTM runs were first started locally, but 16 concurrent
runs saturate the local GPU (~5.3 k fps per run, ~21 h), so after 5 minutes
they were stopped, their metrics and episode logs cut back to 600 M, and half
moved to the cloud, where an LSTM run trained at ~13 k fps in the last wave.
Both extensions continue in 'resume' mode from the resume.pt written at the
end of the previous segment (optimizer and RNG state, checked identical to
the final checkpoint's weights; for the code-only runs taken from the `long`
fleet's bucket copy, since `collect` skips resume.pt), so every extended run
has a single warm restart, at 300 M. Matrices in runs/waveF/; L4 stockouts
handled by retry loops (runs/waveF/retry_*.sh). Driver: tmux `auto_wave_f`
(scripts/auto_wave_e.sh long2:5:runs_ext lstm1b:4:runs_ext) collects both
fleets into runs_ext, deletes the VMs, waits for the local scheduler and runs
scripts/auto_wave_d.sh --now, which now also evaluates a 1.5 B milestone (the
first checkpoint at or after each milestone, within 25 M); tmux
`popcode_milestone_1500` evaluates the 1.5 B checkpoints of the code-only runs
from the bucket while the fleet is still training. Expected: LSTM ~9 am CST
(cloud) and ~12:30 pm CST (local), code-only fleet ~3:30 pm CST, results
~4 pm CST Oct 22; cost ~$65 (long2) + ~$30 (lstm1b).

## 2026-10-23 ~1 pm CST — wave F finished
Cloud LSTM runs (seeds 5-8) reached 1 B at ~8 am CST Oct 22 and were
collected and their VMs deleted at 8:09 am; local LSTM runs (seeds 1-4)
finished at ~11:35 am - 12 pm and were evaluated by 12:22 pm CST. The code-only fleet finished between
3:40 and 4:01 pm CST and the VMs stopped themselves, but the gcloud login had
expired at 10:40 am, so collection, VM deletion and post-processing waited
until the login was renewed (collected 1:02 pm CST Oct 23, post-processed by
1:06 pm). Cloud cost of wave F about $90 (the driver's counter stopped at $69
while the login was expired); waves E + F about $325.
Results (README 3.7): LSTM gap +10.3 +/- 3.7 at 1 B (8 seeds, p = 0.026),
unchanged from 600 M; late-checkpoint mean (64-episode probes every 25 M over
800 M - 1 B) +13.3 +/- 3.1, all 8 seeds, p = 0.004; path efficiency and
post-discovery goal rate higher in all 8 seeds. Code only at 2 B (2 seeds):
predictive codes have slowed to under 2 points per 100 M and end at 63-85
(late-checkpoint means over the last 300 M); grid ends at 63 (began climbing
after 600 M) and 32 (stayed low); gap +12.6 / +42.3 in late-checkpoint
means, +36.4 / +39.8 at the final checkpoint. At 2 B silencing the shifted
cells no longer costs more than silencing 169 random cells in either the
forward or the retrospective code.
