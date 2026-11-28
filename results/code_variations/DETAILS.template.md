# Code variations: details

Methods, every statistic, sources and caveats behind the [short page](README.md). All tables are in [RESULTS.md](RESULTS.md).

The comparisons, the outcome and the training lengths were fixed in [PLAN.md](PLAN.md) before the cloud runs started (commit `799d9932`). The trainable look-ahead in the LSTM agent was added to the plan about an hour after the launch, before any result was looked at (commit `9fd6682e`). The free look-ahead per cell is not in the plan and is exploratory. Six local pilot runs in the door room (one seed, stopped at 20 M frames) and 18 short test runs came before the plan and are not counted.

**Runs added after a first look (not in the plan).** Early looks at 200 and 250 M frames showed the codes with spread far ahead and still rising. Three blocks were added for that reason: the ten codes of the 300 M block continued to 600 M, and two codes (spread × 3, and Ouchi & Fujisawa with spread, shear and moved fields) in the door room and in the LSTM agents. They were started only after the planned 300 M readout had confirmed the effect. The rule, written down before that readout, was that spread × 3 must score above the grid-only code with p < 0.05. Everything these runs show is descriptive. The planned 300 M readout is not affected: it uses the scores and tests recorded before the runs were continued.

## What was run

| block | room | agent | codes | agents per code | frames |
|---|---|---|---|---|---|
| open room, core | open | code only | 8 | 8 | 600 M |
| open room, screen | open | code only | 10 | 8 | 300 M |
| door room | wall with a door + obstacle | code only | 6, plus no code with 3 agents for 300 M | 8 | 600 M |
| trainable look-ahead | open and door | code only | 2 starts, plus 4 control runs | 4 | 600 M |
| free look-ahead per cell | open and door | code only | 1, plus 2 control runs | 2 | 300 M |
| LSTM agents | mix of 21 changed rooms | LSTM + first-person vision | 4 | 8 | 500 M |
| trainable look-ahead, LSTM agents | mix of 21 changed rooms | LSTM + first-person vision | 2 starts, plus 2 control runs | 4 | 500 M |
| added: open room, screen continued | open | code only | the same 10 | 8 | 300 M → 600 M |
| added: door room | wall with a door + obstacle | code only | 2 | 8 | 600 M |
| added: LSTM agents | mix of 21 changed rooms | LSTM + first-person vision | 2 | 8 | 500 M |

263 planned training runs and 32 added ones. The continued runs are the same 80 runs trained further from the state saved at 300 M frames (network, optimizer and random-number state). Their episodes start afresh at that point, as they do whenever a spot machine is stopped and a run resumes. All but the six "free look-ahead" runs ran on spot L4 GPUs in the cloud.

**The agent.** The code-only agent is the one of the main study in which predictive cells first showed a benefit: a feed-forward policy (two layers of 256 units) whose image shows only its heading. It has no memory and cannot see walls, so the code is its only source of position. Its inputs are the image, the code at the current place, the code remembered from the last time the goal was reached, the last action and the last reward. Training is unchanged from the main study (A2C, 32 parallel rooms, learning rate 3e-4).

**The rooms.** The open room is the 2.5 m room of the main study. The door room is layout DO1 of the room-change study: a wall across the room at x = −0.4 m with one 0.6 m doorway, and a 0.5 m obstacle. It is the same for every agent and every episode. Because the agent cannot see, it has to learn where the wall, the door and the obstacle are from the code.

**The task** is unchanged: 90 s episodes, one hidden goal per episode, 10 points per arrival, and a new random start place after every arrival. The LSTM agents train in a mix of 21 rooms, one per episode: ten layouts with a door wall, an obstacle or both, each with and without a goal that moves once in mid-episode, and the open room with a moving goal.

## The codes

Every four-module code is the 512-cell population of its seed with one thing changed. Only `mecreal` changes several things at once. These codes share the phase draws and the module orientations. The codes with predictive cells share the same 169 predictive cells (43 / 42 / 42 / 42 per module), except `p100`, where all 512 cells are predictive. The one-module codes `m1` and `m1var` are a separate population with its own phases. Module spacings are 40.0 / 56.8 / 80.7 / 114.5 cm.

| name | what changes | exact setting | source |
|---|---|---|---|
| `gc100` | nothing: grid cells only | | |
| `u20` | 169 predictive cells look 20 cm ahead | | code of the main study |
| `prop10` | look-ahead = 10 % of module spacing | 4.0 / 5.7 / 8.1 / 11.5 cm | follow-up 1 |
| `prop27` | look-ahead = 27 % of module spacing | 10.8 / 15.3 / 21.8 / 30.9 cm | mean of Ouchi & Fujisawa |
| `mec` | per cell, a fraction of module spacing | normal, mean 0.27, SD 0.11, kept between 0.01 and 0.5 | Ouchi & Fujisawa 2024, Fig. 3G |
| `gard` | per cell, one of the 108 recorded look-aheads in cm | mean 8.3 cm, median 6 cm, the same list in every module | lab analysis of the Gardner et al. 2022 recordings |
| `rand` | per cell, uniform between 0 and 40 cm | mean 20 cm | follow-up 6 |
| `two` | half of each module's predictive cells 10 cm, half 30 cm | mean 20 cm | follow-up 6 |
| `side` | 20 cm to the left of the direction of travel | | follow-up 6 |
| `lr30` | 20 cm, 30° left of ahead for half the predictive cells, 30° right for the others | | loosely after theta sweeps (Vollan et al. 2025). They report about 20° for grid modules and up to about 30° with the most cells. There the whole module alternates from one theta cycle to the next. Here half the cells always look left and half always right. |
| `p100` | all 512 cells look 20 cm ahead | | reference |
| `gcvar` | grid only, per-cell spacing and orientation spread | spacing SD 6 % of module spacing, orientation SD 1° | Redman et al. 2025, their model of module R12 |
| `gcvar3` | the same, three times larger | 18 %, 3° | beyond the recorded range |
| `gcshear` | grid only, each module's lattice sheared along a wall | shear 0.13, ellipticity 1.14 | Stensola et al. 2015 |
| `gcjit` | grid only, every firing field moved a little | about 6 % of the spacing, 2.4 to 6.9 cm | our own choice, see below |
| `mecreal` | `mec` with spread, shear and jitter together | | |
| `m1`, `m1var` | grid only, one module of 512 cells at 81 cm, without and with the `gcvar` spread | | Redman et al. 2025 |
| `none` | no code | | floor |

How each change is built (`code/codes_var.py`):

- **Look-ahead.** A predictive cell fires for the place a fixed distance ahead of the agent along its direction of travel. The remembered goal code uses the same look-aheads.
- **Spread within a module.** Each cell draws its own spacing and orientation from a normal distribution around its module's values.
- **Shear.** The lattice of a module is sheared along one of the two wall directions (chosen at random per module). A shear of 0.13 turns the lattice axes by up to 7.4°, depending on how the lattice lies to the wall. It makes the pattern elliptical (long axis / short axis 1.14).
- **Jitter.** Each cell's map is warped by a smooth random displacement field, different for every cell, with a wavelength of 1.5 to 2.5 grid spacings. The root-mean-square move of a field is 6 % of the spacing. The mean move is 5.4 %.
- **Trainable look-ahead.** One forward distance per module, shared by the predictive cells of that module. It is trained by the same loss as the policy, with its own learning rate (1e-4). In the free version each of the 512 cells of the grid-only code has its own forward and sideways distance, starting at zero. In a control run every look-ahead gradient gets a random sign at each update, so the look-aheads move by noise alone.
- Each change draws from its own random stream. Turning one on never alters another, and with all of them off the population is bit-identical to the one of the main study (checked on the cells and on short deterministic training runs).
- `code/codes.py`, the population of the main study, is not changed. Earlier studies in this repository record its exact text by checksum. The variations are in `code/codes_var.py`. The cloud runs were made with the same code when it still sat in `code/codes.py` (commits `799d9932` to `9fd6682e`).

## Outcome and tests

As fixed in the plan. The outcome is the training score: the mean reward per episode over the last 25 M frames before the mark (about 4,600 episodes per agent). For the LSTM agents it is the last 50 M frames (about 9,300 episodes). The 200-episode test of the final network is reported beside it in RESULTS.md. In the runs of October the test was the noisier of the two (SD of a paired difference 4.7 against 3.1 points), because a network's score moves from checkpoint to checkpoint.

Comparisons are paired within seed. Named comparisons use a paired t-test with Holm correction inside their list. The exact sign-flip p is given uncorrected. Everything else is descriptive and shown with a 95 % confidence interval.

## Results

In every table a positive difference means that the first code scores higher. "Agents higher" counts the agents for which it does. Scores are points per 90 s episode, ± the standard error over agents.

### Open room after 600 M frames (follow-ups 1, 3 and 6)

The four named comparisons, Holm-corrected together:

{{t_named('open', 600, ['prop10 - u20', 'mec - u20', 'gard - u20', 'side - u20'])}}

All eight codes against the 20 cm code:

{{t_codes('open', 600, ['gc100', 'u20', 'prop10', 'prop27', 'mec', 'gard', 'rand', 'side'], 'u20')}}

The same against the grid-only code:

{{t_codes('open', 600, ['gc100', 'u20', 'prop10', 'prop27', 'mec', 'gard', 'rand', 'side'], 'gc100')}}

Further comparisons, descriptive and not corrected:

{{t_named('open', 600, ['prop27 - u20', 'rand - u20', 'prop27 - rand', 'mec - prop27', 'gard - prop10', 'u20 - gc100'])}}

**Many distances against one.** Three pairs of codes have about the same mean look-ahead and differ in whether it is one distance per module or many: random against 20 cm, Ouchi & Fujisawa against 27 %, lab recordings against 10 %. Averaged over the three pairs, many minus one is {{d(blk('open', 600)['many_vs_one'])}} points (95 % CI {{ci(blk('open', 600)['many_vs_one'])}}, {{p(blk('open', 600)['many_vs_one'])}}, {{k(blk('open', 600)['many_vs_one'])}} agents higher).

### Open room after 300 M frames (follow-ups 4, 5 and 6)

The four named comparisons, Holm-corrected together:

{{t_named('open', 300, ['gcvar - gc100', 'gcshear - gc100', 'gcjit - gc100', 'mecreal - mec'])}}

All 18 codes against the grid-only code. The eight codes of the 600 M block are read here at 300 M:

{{t_codes('open', 300, ['gc100', 'gcvar', 'gcvar3', 'gcshear', 'gcjit', 'm1', 'm1var', 'u20', 'prop10', 'prop27', 'mec', 'mecreal', 'gard', 'rand', 'two', 'side', 'lr30', 'p100'], 'gc100')}}

The codes with predictive cells against the 20 cm code:

{{t_codes('open', 300, ['u20', 'prop10', 'prop27', 'mec', 'mecreal', 'gard', 'rand', 'two', 'side', 'lr30', 'p100'], 'u20')}}

Further comparisons, descriptive and not corrected:

{{t_named('open', 300, ['gcvar3 - gc100', 'm1var - m1', 'm1 - gc100', 'two - u20', 'two - rand', 'lr30 - u20', 'p100 - u20', 'prop27 - rand', 'u20 - gc100'])}}

**Is a code with spread simply more active?** No. In all 18 codes a cell has the same mean activity ({{n(CS['gc100']['mean'], 3)}}) and the same variation of activity (SD {{n(CS['gc100']['sd'], 3)}}). The table gives two properties of the codes themselves, computed from 4,500 random places without any agent (`scripts/codes/code_stats.py`, mean of the 8 seeds). Similarity is the correlation of two population vectors. "Distant" means more than 40 cm apart, and the number is the similarity of the most alike 0.1 % of such pairs. The readout is a linear fit of x and y, tested on places it was not fitted on. An error near 100 cm means that position cannot be read out linearly at all.

{{t_stats(['gc100', 'gcvar', 'gcvar3', 'gcshear', 'gcjit', 'm1', 'm1var', 'u20', 'mec', 'mecreal', 'p100'])}}

Neither orders the scores. Moved fields make position as easy to read out linearly as spread does, and they do not help. One module with spread still mixes up distant places, and it scores highest. Which property of the codes with spread the agent uses is open.

**Check against the runs of October.** The grid-only and the 20 cm code were trained in October with the same seeds and settings, on other machines. New minus old at 300 M frames: grid only {{d(blk('open', 300)['rerun_check']['gc100'])}} points ({{k(blk('open', 300)['rerun_check']['gc100'])}} agents higher), 20 cm {{d(blk('open', 300)['rerun_check']['u20'])}} ({{k(blk('open', 300)['rerun_check']['u20'])}}). A training run is not bit-reproducible across machines, so this difference is the size of pure rerun noise.

<!--IF has('open', 600, 'gcvar', 'gcvar3', 'gcshear', 'gcjit', 'm1', 'm1var', 'mecreal', 'two', 'lr30', 'p100')-->
### Open room after 600 M frames, all 18 codes (added, not in the plan)

The ten codes of the 300 M block were trained on to 600 M frames. All 18 codes against the grid-only code:

{{t_codes('open', 600, ['gc100', 'gcvar', 'gcvar3', 'gcshear', 'gcjit', 'm1', 'm1var', 'u20', 'prop10', 'prop27', 'mec', 'mecreal', 'gard', 'rand', 'two', 'side', 'lr30', 'p100'], 'gc100')}}

The codes with predictive cells against the 20 cm code:

{{t_codes('open', 600, ['u20', 'prop10', 'prop27', 'mec', 'mecreal', 'gard', 'rand', 'two', 'side', 'lr30', 'p100'], 'u20')}}

The comparisons of the 300 M block again at 600 M, descriptive and not corrected:

{{t_named('open', 600, ['gcvar - gc100', 'gcvar3 - gc100', 'gcvar3 - gcvar', 'gcshear - gc100', 'gcjit - gc100', 'mecreal - mec', 'm1 - gc100', 'm1var - m1', 'm1var - gc100', 'two - u20', 'two - rand', 'lr30 - u20', 'p100 - u20'])}}

![all lattice codes at 600 M](figures/open_lattice_600.png)

*Open room after 600 million frames, the codes of Figure 7 of the short page. Read it like that figure.*

<!--END-->
### Door room after 600 M frames

The three named comparisons, Holm-corrected together:

{{t_named('door', 600, ['u20 - gc100', 'prop10 - u20', 'mec - u20'])}}

All six codes against the 20 cm code:

{{t_codes('door', 600, ['gc100', 'u20', 'prop10', 'mec', 'gard', 'rand'], 'u20')}}

The same against the grid-only code:

{{t_codes('door', 600, ['gc100', 'u20', 'prop10', 'mec', 'gard', 'rand'], 'gc100')}}

After 300 M frames, with the agents that get no code (3 agents, trained for 300 M frames only):

{{t_codes('door', 300, ['none', 'gc100', 'u20', 'prop10', 'mec', 'gard', 'rand'], 'gc100')}}

**Does the room change which code is better?** The difference 10 % minus 20 cm in the door room, minus the same difference in the open room, agent by agent: {{d(blk('door', 600)['room_interaction'])}} points (95 % CI {{ci(blk('door', 600)['room_interaction'])}}, {{p(blk('door', 600)['room_interaction'])}}).

<!--IF has('door', 600, 'gcvar3', 'mecreal')-->
**Spread in the door room (added, not in the plan).** Spread × 3 and the Ouchi & Fujisawa code with spread, shear and moved fields, 8 agents each, 600 M frames:

{{t_codes('door', 600, ['gc100', 'gcvar3', 'u20', 'mec', 'mecreal'], 'gc100')}}

{{t_named('door', 600, ['gcvar3 - gc100', 'mecreal - mec', 'mecreal - gc100'])}}

<!--END-->
**Trips through the door.** The trained agents were tested for 200 episodes in the door room. Every trip to a goal the agent had already found once was sorted by whether the start and the goal lie on the same side of the wall.

{{t_groups(D, [('score', 'test score'), ('reach_same', 'goal reached, same side (share of trips)'), ('reach_door', 'goal reached, through the door (share of trips)'), ('steps_same', 'steps to the goal, same side'), ('steps_door', 'steps to the goal, through the door'), ('wall_contact', 'share of steps pushing against a wall')], ['gc100', 'u20', 'prop10', 'mec', 'gard', 'rand'], 'u20', 2)}}

### Tests of the final networks and lesions

Each final network was tested for 200 episodes. The lesion columns give the change of the test score when part of the code is switched off at test time. A negative number means the agent got worse.

- **Predictive cells off.** The 169 predictive cells are silenced. The number is the loss beyond that of silencing 169 random cells (mean of three random sets).
- **Look-aheads set to zero.** The predictive cells keep firing but for the agent's own place.
- **Whole code off** and **goal code off** silence the current code or the remembered goal code.

Open room, 600 M frames:

{{t_lesions('open', 600, ['gc100', 'u20', 'prop10', 'prop27', 'mec', 'gard', 'rand', 'side'])}}

Open room, 300 M frames:

{{t_lesions('open', 300, ['gcvar', 'gcvar3', 'gcshear', 'gcjit', 'm1', 'm1var', 'mecreal', 'two', 'lr30', 'p100'])}}

<!--IF has('open', 600, 'gcvar3', 'm1var', 'mecreal') and 'gcvar3' in S['open']['final_test'].get('600', {}).get('codes', {})-->
The same codes after 600 M frames (added):

{{t_lesions('open', 600, ['gcvar', 'gcvar3', 'gcshear', 'gcjit', 'm1', 'm1var', 'mecreal', 'two', 'lr30', 'p100'])}}

<!--END-->

Door room, 600 M frames:

{{t_lesions('door', 600, ['gc100', 'u20', 'prop10', 'mec', 'gard', 'rand'])}}

### Trainable look-aheads (follow-up 7)

Look-ahead of each module at the end of training, mean ± SEM over agents, and as a share of the module's spacing. The control runs (one per row) learn from a gradient with scrambled signs.

{{t_learned(['lzero_open', 'lu20_open', 'lzeroX_open', 'lu20X_open', 'lzero_door', 'lu20_door', 'lzeroX_door', 'lu20X_door'])}}

Score of the agents with a trainable code against the agents with a fixed code and the same seeds:

{{t_learned_scores([('open', 600, 'lzero', 'gc100'), ('open', 600, 'lzero', 'u20'), ('open', 600, 'lu20', 'u20'), ('door', 600, 'lzero', 'gc100'), ('door', 600, 'lzero', 'u20'), ('door', 600, 'lu20', 'u20')])}}

**One look-ahead vector per cell (exploratory, not in the plan).** All 512 cells of the grid-only code get their own forward and sideways look-ahead, starting at zero.

{{t_free()}}

### LSTM agents in the mix of changed rooms

Training score after 500 M frames, against the grid-only code:

{{t_codes('lstm', 500, ['gc100', 'u20', 'prop10', 'mec'], 'gc100')}}

The same against the 20 cm code:

{{t_codes('lstm', 500, ['gc100', 'u20', 'prop10', 'mec'], 'u20')}}

After 250 M frames:

{{t_codes('lstm', 250, ['gc100', 'u20', 'prop10', 'mec'], 'gc100')}}

<!--IF has('lstm', 500, 'gcvar3', 'mecreal')-->
**Spread in the LSTM agents (added, not in the plan).** Spread × 3 and the Ouchi & Fujisawa code with spread, shear and moved fields, 8 agents each, 500 M frames:

{{t_codes('lstm', 500, ['gc100', 'gcvar3', 'u20', 'mec', 'mecreal'], 'gc100')}}

{{t_named('lstm', 500, ['gcvar3 - gc100', 'mecreal - mec', 'mecreal - gc100'])}}

<!--END-->
**Tests room by room.** The final networks were tested for 200 episodes in each room: the open room, the 21 rooms of the training mix, and 6 room conditions that were never trained.

{{t_groups(L, [('open', 'open room'), ('trained', 'the 21 training rooms'), ('door', 'rooms with a door wall'), ('obstacle', 'rooms with an obstacle'), ('both', 'rooms with both'), ('goal_moved', 'rooms with a moving goal'), ('new', 'rooms never trained')], ['gc100', 'u20', 'prop10', 'mec'], 'gc100')}}

**Lesions in the LSTM agents** (test in the training mix):

{{t_lesions('lstm', 500, ['gc100', 'u20', 'prop10', 'mec'])}}

**Trainable look-aheads in the LSTM agents:**

{{t_learned(['lzero_mix', 'lu20_mix', 'lzeroX_mix', 'lu20X_mix'])}}

{{t_learned_scores([('mix', 500, 'lzero', 'gc100'), ('mix', 500, 'lzero', 'u20'), ('mix', 500, 'lu20', 'u20')])}}

## Sources

| topic | source | what it says |
|---|---|---|
| look-ahead of predictive grid cells | Ouchi & Fujisawa 2024, *Science* 385:776 | 305 predictive grid cells in six rats, next to 558 phase-precession and 241 phase-locked grid cells. Look-ahead 22.5 ± 10.2 cm (mean ± SD), 0.27 ± 0.11 of the cell's own spacing (quartiles 0.21 / 0.27 / 0.34, range 0.01 to 0.50, read off their Fig. 3G). The look-ahead follows the animal's path, scales with grid spacing and not with time or speed, and is the same in the four directions of travel tested. A sideways part was not measured. The paper gives no share of predictive cells. From its Table S1 it is 305 of 1,104 classified grid cells in the six rats (27.6 %), and 448 of 1,339 (33.5 %) with three further rats of a second experiment. Both shares are our own sums. Per rat the share runs from 16 % to 66 %. |
| the same, lab analysis | Gardner et al. 2022, *Nature* 602:123 (recordings) | 108 predictive grid cells in nine module recordings from three rats. This is an unpublished lab analysis. Gardner et al. themselves report no look-aheads. 87 of the 108 come from one rat whose three modules were recorded on two days, so some cells may be counted twice. Look-ahead mean 8.3 cm, median 6 cm, quartiles 4 and 10 cm, measured in 2 cm steps. Divided by the module spacings read off Gardner's Extended Data Fig. 2 (57 to 126 cm) this is a mean of 0.095 and a median of 0.067 of the spacing. The look-ahead is about the same in centimetres in small and large modules (module medians 5 to 9 cm). |
| the same, mice | Liu et al. 2026, *Advanced Science* | Two-photon imaging in mice. Prospective cells look 12.4 cm ahead at 56 cm spacing (0.22). They also report cells that look behind. |
| theta sweeps | Vollan et al. 2025, *Nature* 639:995 | Within each theta cycle the position decoded from a grid module sweeps forward by 19.7 ± 0.5 % of the module spacing (mean ± SEM over 31 modules, r = 0.95 between sweep length and spacing), alternating about 20° to the left and right of the head direction (19.3 ± 3.5°, mean ± SEM). |
| several look-ahead times at once | Jensen et al. 2026, *eLife* 15:RP109757; El-Gaby et al. 2024, *Nature* 636:671 | The model the PI mentioned. Jensen et al. is a model of prefrontal cortex in which different groups of neurons represent different steps into the future at the same time. El-Gaby et al. recorded such neurons in the frontal cortex of mice. Both count the future in steps of a task, not in centimetres. Jensen et al. contrast their model with grid cells, which they describe as coding one place at a time. |
| spread within a module | Redman et al. 2025, *eLife* 13:RP100652 | In the longest recorded module (R12, spacing 85 cm) cells differ in spacing by more than measurement noise explains. Two halves of one cell's recording differ by 1.9 cm and 1.2°, two different cells by 6.9 cm and 2.0°. Their model of that module uses a spacing SD of 5 cm (6 %) and an orientation SD of 1°. Over all eight modules the gap is smaller: 1.7 cm and 1.4° within a cell against 3.6 cm and 1.9° between cells. In the model the spread improves decoding of position from one module. With two neighbouring modules the error is already near zero and the spread does not change it much. With two modules that are not neighbours (spacing ratio 2) the spread helps again. |
| ellipticity and shear | Stensola et al. 2012, *Nature* 492:72; Stensola et al. 2015, *Nature* 518:207 | Mean ellipticity 1.16 (968 cells), shared by the cells of a module. Grid axes sit 7.5° off the walls. Stensola et al. 2015 report that the shear needed to turn a grid by 7.5° (tan 7.5° = 0.13) also gives an ellipticity of 1.16 in their simulated grids, close to the 1.17 they measured. For the same shear our code gives 1.14. We read the abstract, the figure legends and the extended data of the 2015 paper, not its main text. |
| moved fields | Krupic et al. 2018, *Science* 359:1143; Hägglund et al. 2019, *Curr Biol* 29:1047 | Krupic et al. use 5.7 cm as the field shift expected by chance between repeated trials in the same box. In a large box the spacing is about 5 % larger in the centre than along the walls (Hägglund). None of the papers we read gives a number for how far single fields sit off their lattice, so the 6 % used here is our own choice. |
| trainable spatial codes | Kumar et al. 2025, ICML; Banino et al. 2018, *Nature* 557:429 | Kumar et al. train place fields with the reward signal. Fields crowd at the reward and at the start, and stretch backward along the path. In Banino et al. the grid network learned alongside the agent, but with its own supervised loss. No reward gradient reached it. Among the papers we read, none trains the look-ahead of a grid code with a reward signal. We did not search systematically. |

## Caveats

- **The agent.** Most of these tests use the code-only agent, not the LSTM agents of the room-change study. It is the only agent in which eight seeds can resolve a difference of a few points. The SD of a paired difference is about 3 points at 300 M frames, against about 15 for the LSTM agents. At 600 M it is about 4 against about 10. What holds for the code-only agent need not hold for an agent that sees and remembers.
- **One door room.** One layout for all agents. A different layout could give a different answer.
- **Look-ahead and walls.** The look-ahead point is the position plus the shift along the direction of travel. It is not stopped by walls. With a 20 cm look-ahead the point is a place the agent itself can never be (outside the room, inside a wall, or closer than 8 cm to one) for about 11 % of positions and directions in the open room and 21 % in the door room. With a 4 cm look-ahead this is 2 % and 4 %.
- **Goal size.** The remembered goal code is taken where the agent enters the goal zone, a circle of 30 cm radius. A look-ahead of about that length marks a point near the goal's centre. The useful look-ahead is therefore tied to the size of the goal.
- **Jitter size.** The 6 % field jitter is our own choice, not a measured value.
- **Recorded look-aheads.** The two sources differ by a factor of about three. They use different animals, tasks and criteria. Both analyses leave out cells that still show a clear grid pattern at zero shift. In Ouchi & Fujisawa the rule is stricter, which probably thins out small look-aheads. That is our reading of their criteria, not a statement in the paper. In the lab analysis the smallest step is 2 cm and a fifth of the cells sit on it.
- **Trainable look-aheads.** Each look-ahead moves by at most 1 mm per update. Over 600 M frames chance alone moves it by about 2 cm. Differences of a few centimetres between agents or starts are therefore noise. The control runs show the size of that noise.
- **Runs added later.** The continuation to 600 M frames and the two codes added in the door room and the LSTM agents were decided after early looks at the scores. They are not planned tests.
- **Tests run again.** A spot machine can be switched off while it writes a test result, which leaves an empty file. {{n_refilled}} of the 200-episode lesion tests were lost this way and were run again from the saved final network, with the same test on a different GPU (`data/refilled_tests.json`). No training score is affected.
- **Cost.** {{COST_LINE}}

## Files

| what | where |
|---|---|
| plan fixed before the runs | `PLAN.md` |
| all tables | `RESULTS.md` |
| numbers of the write-up | `summary.json`, `data/door_eval_summary.json`, `data/lstm_rooms_summary.json` |
| score of every run at every mark | `data/scores.csv` |
| learning curves in 10 M-frame bins | `data/curves.csv` |
| final tests and lesions | `data/final_tests.csv` |
| trainable look-aheads over training | `data/learned_lookaheads.csv` |
| look-aheads learned per cell | `data/free_lookaheads.json` |
| door-room trips that need the door and trips that do not | `data/door_eval.json` |
| LSTM agents tested room by room | `data/lstm_rooms.csv` |
| codes | `code/codes_var.py`, `code/codes_data.py` (recorded look-aheads), `scripts/codes/matrix.py` (run list) |
| training with trainable look-aheads | `code/train.py --learn_shift` |
| analysis, figures, this page | `scripts/codes/analyse.py`, `door_eval.py`, `extras.py`, `figures.py`, `render_readme.py` |
| tests run again after a machine stopped | `scripts/codes/fill_tests.py`, `data/refilled_tests.json` |
| properties of the codes themselves | `scripts/codes/code_stats.py`, `data/code_stats.json` |
| cloud runs | `scripts/codes/fleet.py`, `scripts/codes/launch.py` |

To train one agent: `python -m code.train --cond pgc33 --pgc_shift_rule spacing --pgc_shift_frac 0.10 --seed 1 --policy ff --vision heading_only --encoder_norm layernorm --lr 3e-4 --rms_eps 1e-5 --rooms DO1 --frames 600000000 --out <dir>` (the 10 % code in the door room). `python scripts/launch_matrix.py --matrix_json <file> --dry_run` prints the command of every run of a block written by `python -m scripts.codes.matrix <block> --out <file>`. The run directories are not in the repository.
