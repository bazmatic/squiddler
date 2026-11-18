# Campaign Hunter search experiment

Run269, started 2026-11-14, continues `run268/iter0196` for 300 iterations. The previous run was paused at a saved
checkpoint and then marked stopped; its weights and optimizer state remain in its folder. The shipped policy is
unchanged. The user chose stronger hunting incentives and focused search practice while retaining efficient flight.

## Training changes

- `hunt_progress`: 20 → 40 points per maze width closer to the nearest enemy by route.
- `unseen_progress`: 0 → 20 additional points per maze width of ship progress when no enemy is directly visible.
  Each comparison freezes the preceding enemy position and compares the two ship positions against that same target
  and current wall geometry. Enemy movement alone earns nothing; retreat costs the same as approach earns. This
  uses the existing 0.2-maze-width per-decision cap. The interval that first acquires sight can earn its final search
  credit; subsequent visible combat does not receive the extra reward. No new observations or steering overrides.
- Half the level-cycle slots repeat search levels 10, 16, 19, 22 and 25, interleaved with the original campaign
  cycle (1–29, 36, 38, 40). Those five also keep their ordinary slots. The real levels run to their normal endings,
  retaining combat and exit practice. Levels 10, 16, 19 and 22 start with the nearest enemy behind a wall; level 25
  starts with its enemy out of range. These are existing campaign layouts, not simplified or scripted ship controls.
- Thrust stays **−0.5**, idling **−1.5**, turn changes **−1**, and wall crashes **−80**. Other settings are inherited.

The new reward defaults to zero, so existing runs retain their recorded reward settings. The trainer settings,
rewards panel and replays read it through the campaign reward table. The exact weighted level cycle and settings
are recorded in `run269/config.json` and `.astrovoid-local/neural/campaign-search-experiment.json`.

## Baseline and comparison

The starting checkpoint SHA256 is `962c1fba6fc60db59232247117f14f744fa6ee374f3a01303a5896b89c90e600`.
It was evaluated greedily on the five search levels with seeds **400–402**, outside training seeds 300–399,
at a 180-second horizon. All 15 episodes started without direct enemy sight:

| Measure | Starting checkpoint |
| --- | ---: |
| Episodes acquiring enemy sight | 80% |
| Mean time to contact, with misses charged the full 180 s | 44.65 s |
| Wall strikes per episode | 4.87 |
| Cleared exits | 20% |
| Scored out | 66.67% |

Level 16 never acquired sight in these three baseline episodes. Level 25 first acquired sight after 32.87 seconds
on average. The baseline JSON, including all episode rows, is `.astrovoid-local/neural/campaign-search-baseline.json`.

Search metrics now appear in training logs and the dashboard: hidden-start contact rate, time to first sighting,
and total time without enemy sight. Misses count the full configured horizon, including an early score-out, so dying
sooner does not look like faster searching. Compare contact rate and time together with wall strikes, thrust use,
cleared exits and score-outs; a larger shaped training return alone is not evidence of better maneuvering.

Evaluate a candidate using the same cases:

```sh
venv/bin/python -m tools.campaign_eval .astrovoid-local/neural/rl/run269/iter0299.npz \
  --levels 10,16,19,22,25 --seeds 3 --first-seed 400 --workers 4 --horizon 180 \
  --json .astrovoid-local/neural/campaign-search-candidate.json
```

Also check the ordinary campaign levels before replacing shipped weights. Training has been launched; improved
behavior has not yet been established and no candidate has been installed.

## Reward audit after run269 iteration 146

The reported flight matches the current run269 record for iteration 146, zero-based episode 12, level 29, seed 316.
It uses `iter0145.npz` (the policy that flew iteration 146, before that iteration's update). A sampled replay fired
52 shots, registered two enemy hits and one kill, and spent every decision turning. It scored out after about
18.6 seconds. A greedy replay also turned continuously, so sampling noise alone does not explain it.

Two accounting errors were confirmed and corrected:

- General hunt progress compared successive distances to live moving enemies. Pinning the ship in place still
  earned about 25.9 hunt points from enemy motion. Both hunt and exit progress now compare the two ship positions
  against the preceding goal's frozen position, using the same current wall geometry. Changes of goal, enemy motion
  and destroyed walls alone earn no progress. The unseen bonus uses this same signed movement count.
- `turn_change` charged every control change, including releasing the turn control. It now charges only reversing
  the last nonzero turn direction, including reversals across a neutral pause. Starting and stopping a turn are free.

Thrust and all numerical reward weights remain unchanged. Shooting in the sampled example cost 15.6 points; the
new unseen bonus paid zero because enemies remained visible. Sparse kills still offer little guidance on accurate
aiming. A separate hit/aiming reward would be an additional experiment, not part of this accounting correction.

Run269 was already paused at iteration 146 and was left paused. These source changes apply to future training;
they do not repair an already learned spinning policy. The pre-fix source and replay audits are preserved under
`run269/reward-audit-before`. Repeated flights can vary slightly with creature simulation, but the spinning and
stationary-progress defect reproduced consistently. Historical replays using the updated reward code recompute
their returns under the corrected rules, rather than the original reward accounting. The run also currently
records the ordinary campaign level cycle; its current episode settings should not be confused with the weighted
cycle it started with.

After correction, the pinned-ship replay's hunt progress is exactly zero. Regression tests exercise both bugs
through real campaign episodes. Validation: 3,536 tests passed, one skipped, and the seven previously known
`test_ppo_loop.py`/`test_ppo_resume.py` failures remain. The original saved actor still spins; it has not been retrained
under the corrected rules. Diagnostic artifacts and the scratch audit script are kept in the ignored
`run269/reward-audit-after` folder, outside production code.

## Branch from run261

The user selected run261 as the new starting point. Run270 starts from `run261/iter0499` for 300 iterations with
the current run269 settings and the corrected reward accounting. In particular, thrust remains −0.5, hunt progress
40 and unseen progress 20; it does not inherit run261's historical −0.4 thrust cost. Run269 was marked stopped with
all files preserved. The source checkpoint hash, settings source and reward corrections are recorded in
`run270/branch-provenance.json`; `working-tree.patch` preserves the source changes used at launch.

The small checkpoint probe showed run261 fought more successfully than run269 on the sampled cases, but it also
already spun continuously. This branch is a user-selected starting point for relearning with corrected rewards;
it is not a claim that run261 already meets the maneuvering goal.

## Shooting audit of run271/iter0299

Four greedy flights on 2026-11-15 used a 60-second horizon. Level 14 (seed 317) fired 208 shots and registered 30
enemy hits; level 29 (seed 316) fired 209 shots and registered six enemy hits. Levels 16 and 25 (seed 400) fired no
shots. Among newly observed bullets in the combat cases, none was fired while no enemy was directly visible.

The existing constant-velocity shot-quality predictor rated 145/205 observed bullets in level 14 and 187/208 in
level 29 below 0.05 quality. This is a geometry diagnostic, not an actual-hit counter: it accounts for target motion,
range and blocking walls but cannot predict future maneuvers. Bullets removed during their creation frame are not
observed by this after-update trace, so observed-bullet counts can be slightly lower than shots fired.

These cases point to poor aim/fire timing rather than a general failure to recognize enemy presence. The campaign
currently pays kills and charges every bullet equally, with no graded aiming or nonlethal-damage reward. A focused
next experiment would add signed aiming-progress credit and small bounded credit for real damaging hits, retaining
the existing bullet cost and thrust setting. No new reward was added or training started during this audit.

Detailed rows and the read-only probe are preserved in `run271/shooting-audit`.

## Reusing the other hunter's shooting incentives

The original hunter already has `hit_reward` and `shot_price`: credit when its bullets reach an enemy, and a cost
when no directly seen enemy is within 20 degrees of the nose. Both pilots now use the same bearing classifier in
`learning/shooting_rewards.py`; the original hunter's behavior and settings are unchanged.

Campaign Hunter exposes two additional settings (campaign penalties use negative points):

| Setting | Meaning | Default |
| --- | --- | ---: |
| `enemy_hit` / `--enemy-hit` | Points per player projectile contact that reduces enemy health | 0 |
| `unaimed_shot` / `--unaimed-shot` | Extra points per bullet without an enemy in the 20-degree sight cone | 0 |

Hit credit uses the encounter's actual damage callback, rather than its raw contact counter. Ally damage and
contacts that do not reduce health earn nothing. The unaimed check is the original rule: it uses the decision's
direct-sight snapshot for that interval's bullets, not a future-motion intercept, a remembered enemy or a guarantee
of hitting. It also charges wall-clearing shots without an aligned visible enemy. The normal `shot` cost still
applies to every bullet.

A conservative trial is `enemy_hit=1`, `unaimed_shot=-0.3`, with the existing `shot=-0.3` and thrust unchanged. This
makes an unaimed miss cost 0.6, an aimed miss cost 0.3, and a damaging aimed hit earn 0.7 before any kill credit.
These values are a proposed experiment, not changes to an existing run; both new settings default to zero.

The dashboard now charts damaging hits per episode and unaimed bullet share, with explanations of the definitions
and the zero-shot case. These can be checked against kills and bullets fired to distinguish accuracy from simply
holding fire. No new training run or policy installation was performed as part of this change.

The user subsequently authorized training: run272 continues `run271/iter0299` for 300 iterations with `enemy_hit=1`,
`unaimed_shot=-0.3`, the existing `shot=-0.3`, and thrust −0.5.

## Necessary block destruction

`breach_block` adds credit only when the player completely destroys a breakable block on the cheapest planned
breach route to an enemy, while no living enemy has an open route. An available route to any enemy suppresses the
bonus. This prevents demolition of unrelated rock from becoming a substitute for hunting. A route can require
several blocks: each completed relevant block can count, even before the whole route is open.

Partial damage pays nothing. Ally/enemy destruction, metal, the unbreakable outer ring, unrelated blocks and
destruction after all enemies are dead earn nothing. Ordinary bullets, weapon projectiles (including rail shots)
and blasts preserve their source when damaging a wall. A final-hit observer evaluates connectivity immediately
before removal, after destruction is confirmed; stale faces and blocks already removed in a batch cannot pay twice.
During a batched curved-wall rebuild, one blast uses its pre-break breach plan until the geometry generation changes.

The setting defaults to zero for old runs. The initial trial uses **`breach_block=3` points per completed block**,
below the 10-point kill reward. It supplements the existing shooting costs, without rewarding intermediate hits or
waiving any shooting penalty. The dashboard's “Necessary wall blocks destroyed” chart reports the qualifying count.
Run272 was paused after iteration 3 to add and verify this rule; its resume records the new setting and rule boundary.

## Wall impact severity

Campaign Hunter now uses the allied hunter's impact measure for its `wall_crash` training term:
`hypot(vx_after - vx_before, vy_after - vy_before) / ship.max_speed`, summed over unshielded wall contacts.
Velocity is sampled immediately around wall collision resolution, excluding thrust and ordinary flight drag.
Glancing and low-speed contacts therefore cost less than hard impacts. Repeated impacts accumulate; a clear
wall check resets the recorded impulse, and shielded contacts remain exempt.

The current coefficient stays at **−80 per impact unit**, with thrust and all other coefficients unchanged.
A unit means velocity change equal to top speed, so a full-speed rebound can exceed one unit (about 1.85 with
the current restitution). This changes the training measurement, not collision physics or in-game scoring.
The existing wall-strike chart retains the raw contact count; the new wall-impact chart reports summed severity
and explains its units. Historical logs retain their original values; new rollouts use the revised measurement.
No training session was started for this change.

Regression coverage uses real campaign collisions to compare head-on, glancing and slow impacts, checks the
shield exemption and verifies that old impulses cannot leak into later clear checks. The affected tests pass.
Validation: 94 affected tests passed. The full suite finished with 3,562 passed, one skipped and the same seven
pre-existing failures in `test_ppo_loop.py` and `test_ppo_resume.py`; no new failures. The dashboard was reloaded
and its served metadata includes the revised reward help and severity-chart explanation.

Run273 subsequently started from `run272/iter0299` for 300 iterations. Its first two iterations inadvertently
inherited the previous shooting settings. It was paused after iteration 1 and resumed with the proposed
`enemy_hit=2`, `unaimed_shot=-0.6`, and `breach_block=6`; the config records this exact change boundary.
Ordinary shots remain −0.3 (an unaimed shot therefore costs −0.9 total), idle −1.5, thrust −0.5, wall impact −80,
and turn reversals −1. Signed intercept-alignment shaping and focused aiming/breach lessons have not been added.

## Steering collapse and manual recovery

Run273 iteration 92 episode 16 (level 17, seed 324, sampled policy `iter0091`, sample seed 2224) replayed
identically twice. Its 773 decisions contained 724 left turns, 49 right turns and no straight decisions; one
ten-second stretch made 10.27 net revolutions without a kill. Its last kill was at 22.77 seconds and it scored
out at 77.28 seconds. Total return was −1,212.09, including −555 idle, −460.52 wall impact and −87 reversals.
The enemy-route compass remained available. This was a bad learned policy, not positive reward for spinning.
The turn head's neutral logits were so low that its float32 action probability underflowed to zero on this trace;
the same suppression was already present in run261. Resetting recurrent memory or neutralizing the last-turn
inputs did not restore the neutral action. The reversal penalty charges switching sides, while sustained turning
has no direct cost; this can reinforce the directional habit, but does not prove it originally caused collapse.

The user suggested manually editing the weights. Run273 was paused after iteration 135. Separate candidate
policies changed only `turn_w` and `turn_b`: scale left/right outputs, remove their mean logit imbalance on the
reported trace, and replace the neutral row by the midpoint of left/right plus a 0.35 bias. This restores a
meaningful straight action and retains differences between left/right preferences. Backbone, observation
normalization, thrust and fire weights were checked for exact equality with the source.

The selected scale was 0.2. Ten sampled held-out episodes per variant (levels 1, 6, 17, 25, 40; seeds 400/401;
60-second horizon) compared it against `run273/iter0135`. Windows of ten seconds with at least 90% of decisions
turning the same way fell from 16 to zero. The mean fraction of starting enemies killed rose from 46.1% to 89.1%;
exits stayed at 4/10 and score-outs at 4/10. Straight decisions rose from zero to 36.3%. These are small diagnostic
samples used to choose a recovery initialization, not proof of trained steering quality or a full campaign gate.

Run274 starts from the edited copy of `run273/iter0135` for the remaining 164 iterations of the authorized 300.
All reward settings are inherited unchanged. Run273's original checkpoints remain intact and its state is stopped.
Run274 stores the edited start, exact edit parameters, source/edited hashes, calibration observations, comparison
results and scripts, plus reward-source snapshots. This continuation uses the edited policy as its PPO reference
so the reference penalty does not force it back toward the collapsed turn head. Focused steering lessons and
intercept-alignment reward shaping have not been added by this manual-weight experiment.

The dashboard subsequently replaced the raw wall-strike and severity charts with **Wall collision penalty per
episode**: positive points deducted, averaged across episodes (zero is best). Its data endpoint derives this from
logged severity times the iteration's signed penalty coefficient, undoing later coefficient edits for earlier rows.
For older rows without severity it shows the actual historical flat-per-contact penalty. Raw logs are preserved;
unknown coefficients do not produce invented penalty values. The raw contact and impulse fields remain available
in the data but are hidden from charts. Dashboard and HTTP-server regression checks: 100 passed.

## Pursuit reward charts and short lessons

Campaign rollouts now return actual signed per-term `reward_totals`. Their sum matches the episode return.
The log and dashboard expose mean Hunt progress, Unseen pursuit and Exit progress **points per episode**.
Hunt progress is the base credit; unseen pursuit is additional credit and is not included in that base chart.
Retreat subtracts points, and episode duration affects the opportunity to earn. These charts appear only in logs
that record reward totals; historical reward totals cannot be recovered from old summary logs without replays.

`--pursuit-lesson` enables a short training-only search using the real maze, ship physics and senses. Ordinary
opponents and pickups are replaced by one stationary Urchin target behind a fixed wall, on a verified open route
two to eight cells away. Its placement and the ship's starting heading use the episode seed. The lesson ends at
first enemy sight, the time limit or score-out. `search_contact` pays once at first sight only for an initially
unseen target; it defaults to zero and never pays in normal campaign episodes. A visible start is rejected as a
completion. The trial uses +100 contact points, leaving thrust −0.5, idle −1.5, wall impact −80 and all other
existing coefficients unchanged. There is no reward for keeping an enemy visible or repeatedly rediscovering it.

An initial survey showed that waiting succeeds in four of five naturally unseen starts, so simple natural-level
truncation was rejected as a pursuit lesson. Run277 tried 30 iterations on levels 3/10/22. The sampled fixed-seed
search improved slightly while greedy contact fell from 3/18 to 1/18; a small normal-campaign exit check fell
from 4/10 to 1/10. It was rejected for continuation. An additional check found that level22's targets were hidden
only by moving gates, so waiting could reveal them. Placement now requires **fixed-wall** occlusion and still
checks live gates for route connectivity. A regression first failed on level22, then passed with this fix. Run277
retains its original source snapshots for reproducing the superseded experiment; current placement rejects it.

Run278 repeats the 30-iteration trial from `run274/iter0163` on validated fixed-wall levels 3/10, with 16 episodes
per iteration and a 30-second limit. Training uses seeds 300–399. Both greedy and sampled evaluations use the same
12 level/seed cases (levels 3/10 × seeds 400–405), with misses charged the full limit. The before/after protocol,
evaluations and exact source snapshots are kept in the run folder. No long run or policy installation is implied
by a successful sampled training metric; greedy pursuit and normal campaign behavior must also be checked.

Run278 completed all 30 iterations. On the paired 12-case fixed-wall evaluation, greedy contact improved from
3/12 to 12/12 and mean first sighting (misses charged 30 seconds) from 22.7 to 0.9 seconds. Sampled contact stayed
at 12/12 while mean first sighting improved from 1.64 to 1.21 seconds. The separate 10-case greedy normal-campaign
check (levels 1/6/17/25/40, seeds 400/401, 60 seconds) was mixed: exits fell 4/10 to 3/10, cleared exits 4/10 to
2/10, and wall contacts increased 2.2 to 13.8 per episode. Its two hidden-start cases improved from no contact
to both finding enemies (mean first sighting 60 to 0.9 seconds). These are small diagnostic samples, not a full
campaign gate. Run278 is retained as a pursuit-trained candidate; run274 remains the unmodified campaign baseline.
No long run or shipped-policy replacement was performed. Any future campaign continuation from the candidate
must explicitly disable `pursuit_lesson` and set `search_contact=0` to return to full levels and ordinary rewards.
Exact before/after data and the decision are in `run278/evaluation-comparison.json`.

Validation after the gate-occlusion fix: 62 focused tests passed; the full suite finished with 3,573 passed,
one skipped and the same seven pre-existing failures in `test_ppo_loop.py` / `test_ppo_resume.py`. The dashboard
serves the actual signed reward charts from the new logs. No continuous enemy-in-view reward was introduced.

The user then explicitly authorized a proper campaign session. Run279 starts from `run278/iter0029` for 300
iterations, restoring run274's 32-level cycle, 24 episodes per iteration, 180-second horizon and 10 workers.
`pursuit_lesson=False` and `search_contact=0` restore full levels with no contact or visibility bonus. Existing
shooting incentives, signed progress rewards and severity-scaled wall penalties are inherited unchanged.
Its continuation record notes run278's mixed normal-campaign evaluation, and preserves the exact source snapshots.

Run279 iteration135 episode23 (level40, seed363, policy `iter0134`, sample seed3263) exposed weak jelly defense.
Two replays matched exactly: score-out at 12.17 seconds, 20 body contacts, seven shots, two kills. Enemies were
visible for all 122 decisions; 22 decisions were both aligned within 20 degrees and weapon-ready, yet only three
selected fire. On the same observation trace, mean firing probability while aligned was 54.5% for run274's end,
20.0% after the pursuit trial and 12.9% at run279/iter0134. This supports a shooting regression associated with
pursuit-only training (which ended before combat while charging shots), further reinforced during campaign PPO.

A diagnostic override changed only fire selection when an enemy was aligned, retaining sampled neural steering
and thrust. At the original ending time it had 43 shots, eight kills and one enemy contact; it then survived all
180 seconds. Its discounted return improved from −311.52 to −182.81, despite a lower undiscounted total over the
much longer episode. This demonstrates useful firing opportunities and working weapons; raw totals across such
different durations must not be used to infer that training prefers the early death in this example.

The ship's recent/directional hit inputs were zero throughout: `ShipBody.take_hit()` is an unimplemented stub
returning None. It still senses falling score-health, enemies and motion, but not the allied hunter's explicit
knock direction. Body hits also have no direct campaign reward term; their cost is through game score, subsequent
events and eventual score-out. Diagnostic data are saved in run279's `diagnostic-iter0135-episode23` JSON/NPZ.
No production code, weights or active training settings were changed during this diagnosis.

## Hit feedback repair and mixed combat practice

The user authorized repairing hit feedback and combat practice alongside pursuit. Run279 paused and saved
iteration152 before the observation change; its checkpoints and resume state are preserved, and the run is stopped
while the practice branch is evaluated. The ship now records the strongest unshielded non-wall knock between
pilot observations. Enemy rams use actual collision-induced velocity change, bullets use the final strength-scaled
change, blasts use their actual push, and a no-impulse bite still signals a hit. `ShipBody.take_hit` reads and clears
the ship's record; the existing senses rotate and normalize the impulse into forward/side channels. Physics,
damage, score and weapon behavior are unchanged. Shielded contacts and walls do not populate these channels.

Campaign lifetime rule `ship_hit_feedback` preserves observation semantics: pre-repair runs without the rule
replay/resume with False, while new runs record True. Live gameplay uses the repaired feedback. Run280 records
True from its first iteration. The original iteration135/episode23 trace still matches exactly with historical
inputs; enabling repaired inputs supplies 18 hit observations where the historical trace had zero. Regression
tests cover real rams, weak projectile scaling, blasts, shield/wall exclusions, consumption and old/new replay rules.

`--combat-practice-levels` mixes full combat levels into a pursuit-lesson run. Run280 starts from
`run279/iter0152` for 30 iterations of 16 episodes, levels 3/10 as pursuit drills and level40 as full jelly combat,
with a 45-second horizon. Contact bonuses apply only to drills. Kill/exit/score-out metrics describe combat
episodes; lesson contact rate describes drills; other totals average the whole batch. All existing reward weights
are unchanged. The user proposed a missed-shot penalty, but has not authorized implementing it; none was added.

Fixed-seed evaluation baseline with repaired inputs enabled: greedy pursuit finds targets in 9/12 cases
(levels3/10, seeds400–405); level40 greedy combat averages 17.5 kills and no score-outs over six 45-second cases,
while sampled combat averages 11 kills and two score-outs. A separate 10-case 60-second campaign check is saved
for comparison. Exact protocol, source snapshots and baseline data are in run280. Future full campaign reentry
must disable pursuit lessons and the lesson-only contact bonus; the combat level list is inert with lessons off.

Final code validation: 177 initial focused checks passed, 93 replay/feedback checks passed after adding lifetime
compatibility, and the full suite has 3,579 passed, one skipped and the same seven pre-existing PPO test failures.

Run280 completed. Fixed-seed pursuit improved from 9/12 to 12/12, but the intended combat improvement failed:
level40 greedy mean kills fell 17.5 to 8.5 (score-outs 0/6 to 1/6); sampled mean kills fell 11 to 10 (score-outs
stayed 2/6). The separate normal-campaign check regressed from 3/10 exits to 2/10, with score-outs increasing from
3/10 to 6/10. Its weights were rejected rather than used for continuation. Exact comparisons are in run280's
`evaluation-comparison.json`. The hit feedback repair remains valid independently of these training results.

Run281 continues full campaign training from the retained `run279/iter0152` checkpoint for the remaining 147
iterations of the originally authorized 300 (153 saved iterations before the repair, plus 147 afterward).
It records `ship_hit_feedback=True`, uses the full selected level cycle, 24 episodes, 180-second horizon and
10 workers, and keeps all existing reward coefficients. Pursuit drills and the contact bonus are off. No
missed-shot penalty was added. This preserves the pre-practice policy while training on the repaired observations.

## Declined firing opportunity penalty

The user then authorized implementing the proposed penalty for **not firing when a useful shot is available**.
`missed_opportunity` defaults to zero for historical runs. The trial coefficient is **−0.1 per weapon firing
interval** spent declining a valid opportunity, prorated by actual elapsed simulation time, including a truncated
last decision. It is not a cost for a bullet that misses, and it does not charge during cooldown or when the
trigger is held. Changing decision frequency does not change the cost for the same amount of eligible time.

`learning/campaign_shot_opportunity.py` reconstructs the decision's directly seen enemy positions, checks actual
weapon readiness and ammunition, and predicts whether a projectile fired along the chosen heading (after the
frame's chosen turn) would intersect an enemy moving at its present velocity before expiry. It uses real muzzle
offsets, weapon speed, spread, lifetime, target collision geometry and projectile radius, and excludes a wall or
gate collision before the target. Shots do not inherit ship velocity. Inventory readiness uses actual weapon
state, including the basic gun; legacy readiness respects the game's strict cooldown boundary. Invulnerable
targets are excluded. Explosive/bouncing weapon predictions are excluded; seekers count only when their direct
ballistic path works. Future target turns, articulated motion and moving gates can invalidate any local linear
prediction, so this is not a promise of a future hit. The rule does not steer, aim or fire for the pilot.

Rollouts record firing-interval equivalents declined and actual signed penalty points. New Shooting charts show
both with explanations. The existing hit-feedback repair was rechecked: real rams, scaled bullets and blasts
reach the hit senses, shielded contacts and walls do not, and historical runs retain their original zero-input
behavior. `ship_hit_feedback=True` remains active in the continuation.

Validation: 95 focused checks passed initially and 78 passed after adding the turn-before-fire geometry check.
The full suite had 3,586 passed, one skipped and the same seven pre-existing PPO test failures before that final
focused addition. A real level40 replay with cost zero versus −0.1 produced identical controls and observations;
the enabled rule measured 11 declined firing-interval equivalents and deducted 1.1 points.

Run281 was already paused after iteration8. Its nine iterations and checkpoints are preserved. Run282 starts from
`run281/iter0008` for the remaining 138 iterations, enabling `missed_opportunity=-0.1` with all other reward
coefficients unchanged. The separate run preserves historical reward/replay semantics. Original campaign
accounting is 153 iterations in run279 + nine in run281 + 138 in run282 = 300; practice trials are separate.
Exact opportunity logic, source snapshots and experiment details are recorded in run282.

## Rough aim without shooting costs, graded positive accuracy

The user refined the intended incentives: penalize declining a roughly aligned opportunity, make roughly aimed
shots neutral rather than penalized, and give increasingly positive feedback for tighter aim. New campaign runs
record lifetime rule `graded_shooting=True`; older runs default False and retain their original shooting rewards.
Run282 had already finished. Its latest recorded settings were missed opportunity −0.2, ordinary shot −0.1,
extra unaimed cost −0.6 and damaging hit +2. These settings and the completed run were preserved.

The rough window is **±30 degrees** around a directly seen enemy's current bearing or predicted intercept. Within
that window both training shooting costs are waived. Declining a ready, roughly aimed opportunity with a clear
reachable intercept still incurs the existing per-firing-interval penalty. The strict physical direct-hit
requirement remains only for historical runs. Shots outside the window keep ordinary and unaimed costs.

`shot_accuracy` defaults to **+0.3 maximum per actual emitted volley** in new graded runs. Credit rises linearly
from zero at 30 degrees intercept error to full credit at zero error, requiring a reachable clear intercept.
Damaging hits and kills keep their separate rewards. A spread volley uses its best lane and scores once, so it
cannot multiply the accuracy bonus by pellet count. Grading observes the actual volley at creation, with its
real muzzle, velocity, lifetime and weapon behavior; it does not autoaim or fire. Current target motion and
walls/gates are checked. Invulnerable targets cannot earn positive accuracy credit. Unsupported explosive/bounce
predictions earn no accuracy bonus but rough shooting toward a seen enemy is still exempt from shooting costs.
The temporary fire observer is restored after each rollout, including exceptional exits.

The accuracy chart reports actual signed accuracy points. Declined-shot descriptions now distinguish the new
rough check from historical strict-hit opportunities. Historical reward panels show effective accuracy credit
as zero. Completed logs and coefficients are not rewritten. New setup is recorded for review in
`.astrovoid-local/neural/graded-shooting/next-session.json`; no additional session was started.

Final live-state inspection found run283 already paused after iteration28 with the pre-grading shooting rules.
It was not modified or resumed. The prepared next-session record now starts from `run283/iter0028`, keeping its
latest coefficients and proposing the remaining 109 iterations with graded shooting and +0.3 maximum accuracy
credit. A new continuation is needed to adopt the new lifetime rule while preserving historical replay rewards.

Run283 was subsequently resumed under its preserved legacy rule `graded_shooting=False`; the newly available
accuracy log column therefore contained zeros despite the reward being disabled. The dashboard now omits that
column for legacy runs, and rollouts tag their shooting mode so summaries do not emit disabled accuracy values.
A genuine zero in an enabled run remains charted. Tests cover both states and preserve the cached/raw log rows.

After the user reported the flat chart, run283 saved iteration44. Run284 continues from `run283/iter0044` for its
remaining 93 iterations, recording `graded_shooting=True` and `shot_accuracy=0.3`. Other coefficients and saved
history remain unchanged. The disabled legacy chart is hidden, and new accuracy values belong to run284.
Focused chart/rollout/replay checks: 74 passed; local-server checks: 39 passed. The live data endpoint reported
`shot_accuracy_reward=6.781` at run284 iteration1, confirming the enabled chart receives nonzero values.

## Damaging hits dominate near-miss feedback

The user noted that a few near misses must not rival a direct hit and explicitly chose **+4 per damaging hit**.
The aim-only ceiling is reduced to **+0.05 per volley** and the normalized alignment score is squared:
`shot_accuracy * max(0, 1 - intercept_error_degrees/30)**2`. Perfect predicted alignment earns at most +0.05;
15 degrees earns +0.0125; 30 degrees earns zero. A damaging hit earns +4 independently, plus any aim/kill credit.
Even five maximum aim-only credits total just +0.25. Roughly aimed shots remain exempt from shooting costs.

New lifetime rule `accuracy_power` records 2 for the squared curve. Historical runs without the rule use 1,
preserving their linear rewards and replay returns. The user-facing accuracy help describes the distinction.
Focused checks passed (80), including real damaging-hit payment, the aim-only cap, nonlinear taper, unchanged
controls between linear/squared feedback and legacy replay defaults.

Run284 paused after iteration65, preserving 66 iterations. Run285 continues from `run284/iter0065` for its
remaining 27 iterations with `enemy_hit=4`, `shot_accuracy=0.05`, `accuracy_power=2` and all other coefficients
unchanged. Its exact reward rule and source snapshots are saved in the run folder.
Full regression suite: 3,590 passed, one skipped and the same seven pre-existing PPO failures. Run285 iteration1
averaged approximately +93.8 damaging-hit points versus +1.0 aim-only points per episode.

Validation: 79 focused checks passed, covering monotonic accuracy, free rough shots, negative outside-window
shots, missed rough opportunities, historical shooting behavior, volley-observer cleanup and replay rules.
The full suite finished with 3,589 passed, one skipped and the same seven pre-existing PPO failures. On a real
level40 comparison with identical weights/seeds, controls and observations matched exactly. Of seven emitted
shots, six became free of shooting costs and accuracy earned 0.877 points; only reward feedback changed.

### Level 11 diagnosis — run286 iteration113

Pinned checkpoint/config and headless probes are in `/tmp/level11-diagnosis/`; reproduction script is
`/tmp/diagnose_level11.py`. No training or reward settings were changed. Four greedy episodes (seeds400–403)
all scored out after 35.8–41.8 seconds, killed one non-mission enemy and left all five required wallcrawlers alive.
They fired 101–111 shots with seven damaging hits each. By comparison, level12 exited in all four greedy probes;
level10 also struggled (no exits within the 90-second diagnostic horizon).

Seed400 used thrust on only three of 375 decisions, turned left or neutral exclusively, and scored out at
37.42 seconds. Four damaging hits were spread across crawlers without a crawler kill. Its game-score costs were
74.83 time, 30 enemy bullets, 11.1 shots, four wall contacts and 0.18 fuel, offset by one +20 kill. Training idle
penalties are separate from these game-score costs. Four sampled level11 episodes also failed to exit within
90 seconds: three scored out, while one remained alive with one full-health crawler left.

Confirmed design mismatch: level11 requires wallcrawler kills, but the enemy compass and hunt-progress target
the nearest living enemy of any kind. The observation layout contains neither mission-target markers nor enemy
kind. Crawlers have four HP and a hit knocks them loose into flight, so successful play needs follow-up tracking
and pursuit. The observed low-thrust, low-hit behavior fails to finish them. Mission blindness is a plausible
contributor, not an isolated causal proof; no wallcrawler collision or hit-sensing defect was established here.

### Persistent campaign targets

User requested stickier targets after a seed400 trace showed 24 compass target changes in 37.4 seconds, and
compass disagreement with the nearest visible contact in 140 of 340 visible decisions. CampaignSenses now keeps
one active target across distance-order changes and lost sight. The selected enemy takes visible slot0 when
actually visible; hidden enemies are never inserted as visible contacts. Compass navigation and hunt-progress
reward use the same selected enemy. Target death/removal releases it immediately, as does an unreachable target
when an alternative is reachable. A chase more expensive than three times the cheapest alternative plus four
cells must stay that way for two seconds before switching. Recovery cancels that timer; level reset clears focus.

`sticky_targets` is a lifetime rule: new training runs and live CampaignPilot enable it; runs without the recorded
rule retain False. The existing run286 was not modified or restarted. Watch/replay forwards the recorded flag.
Campaign evaluation defaults to sticky targeting; `--nearest-targets` provides the historical comparison.
No reward coefficients, thrust controls, or policy weights were changed.

Pinned run286/iter0113, seeds400–403, identical 90-second horizons: level11 greedy mean kills rose from 1 to12.75,
and every case killed at least one crawler (previously none). No level11 exits occurred in either 90-second group.
Level12 retained four exits out of four. Extended sticky level11 episodes to the normal 180-second horizon:
one of four greedy and one of four sampled episodes exited. This small fixed-checkpoint probe is encouraging,
not proof of general training improvement. Seed400 focus switched five times over90 seconds, with zero cases
where a visible focus was placed outside slot0. Artifacts: `/tmp/sticky-target-evaluation/`,
`/tmp/sticky-target-evaluation-180/`; scripts `/tmp/evaluate_sticky_targets.py` and
`/tmp/evaluate_sticky_targets_180.py`.

Validation: 115 focused campaign/senses/replay/evaluation checks passed. Full suite initially had3598 passes,
one skip, the seven known PPO failures and one expected-rules assertion needing the new flag. Updated that
assertion; final34 targeting/replay/reward-table checks passed. A real historical run286 seed400 replay matched
the pre-change senses, controls and rewards arrays exactly. `git diff --check` passed.

User requested stopping the current run and starting a fresh one with persistent targeting. Run286 paused after
saving iteration178 and was marked stopped. Reloaded the trainer and started run287 from `run286/iter0178.npz`
for300 iterations,24 episodes per iteration and10 workers. Verified active training and the saved config's
`sticky_targets=True`, alongside hit feedback, graded shooting and accuracy_power2. All source argument values
except the new run's start/output paths are unchanged, including enemy_hit4, shot_accuracy0.05,
missed_opportunity−0.2 and thrust−0.5.

User requested a temporary learning-rate increase to help level11. Run287 saved/paused at iteration69, then
resumed with lr0.0002 (was0.0001), preserving weights and Adam moments. Config and every saved optimizer
parameter group verified at0.0002. Planned boosted iterations70–94 (25 iterations), then restore0.0001 and
continue toward the original300. Rewards, sticky targeting and all other training arguments are unchanged.
`run287/lr_boost.json` records the plan/status; `lr_boost.log` and `lr_boost.pid` track the detached controller
(`/tmp/astrovoid-lr-boost.py`, initially PID75205). It uses the normal pause/resume API around the phase boundary,
so it may finish one extra iteration depending on polling. It cancels if the user separately pauses/stops,
selects another run or changes the learning rate; it does not resume a manually paused run. Mocked API checks
verified automatic restoration and manual-pause cancellation. Baseline for later comparison: iter0069.npz.
