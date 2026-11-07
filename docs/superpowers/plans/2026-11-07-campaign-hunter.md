# Campaign Hunter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In Demo, a neural pilot (the Campaign Hunter) flies the real player ship through each level — hunting every enemy, collecting pickups and the Ally beacon, then leaving through the exit — trained with the hunter's PPO loop.

**Architecture:** The game gains a *ship pilot* seam: when `Game.ship_pilot` is set, it supplies the ship's turn/thrust/fire each frame in place of the keyboard, and exit completion and score-out stay live even in autoplay. A `CampaignPilot` wraps the existing `NeuralController` with `CampaignSenses` (the hunter's senses plus exit and beacon compasses). Training reuses `tools/ppo_neural.py`'s `train_rl`, made task-aware, with a new `campaign_hunter` task whose episodes fly a real level in a headless `Game` until the ship exits, scores out, or times out.

**Tech Stack:** Python 3.10, pygame 2.6, numpy, the project's PPO (`learning/ppo.py`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-campaign-hunter-design.md`

## Global Constraints

- Demo only: normal play, the summonable allied hunter, enemy Vibers and `assets/neural/hunter.npz` behave exactly as before.
- The level's normal exit rules apply; nothing shuts the exit for the pilot.
- Score-out (potential score reaching 0) is live for the piloted ship in training and in Demo.
- Reward pays the ship's **own** kills only (`source == 'player'`); the clear-before-exit bonus counts kills by anyone.
- No cheating: the piloted ship has exactly the player's handling, ammo, weapons, shield and scoring.
- New sense inputs are appended after the hunter's `FEATURE_NAMES` and start at zero weight.
- Campaign policy file: `assets/neural/campaign_hunter.npz`. Training levels: the Demo range `SETTINGS.autoplay.firstLevel..lastLevel` (1–40).
- Training must be deterministic and faster than real time: no wall-clock reads on the piloted path.
- Ask the user before starting or restarting any training run (one run at a time, via the astrovoid-trainer tools).
- Tests cover features, not internals: few, behavioural tests.

## Review Focus

1. **No campaign policy installed** → Demo behaves exactly as today (parked ship, hunter summoned, old end rules). Test in Task 4.
2. **Demo must not write the person's profile**: a Demo level completing or scoring out must not record level results, stars or high scores to `profile_manager`. Test in Task 4.
3. **Keyboard during Demo** must not steer the piloted ship (keys still do whatever they did in Demo before, e.g. leave Demo). Test in Task 1.
4. **Ally beacon in Demo** → the ordinary hunter is summoned beside it, sees the piloted ship as its friend, and does not end the level when it dies. Test in Task 4.
5. **One end per level**: reaching the exit in the same frame as scoring out, or the director's time limit firing during the exit explosion, must advance Demo exactly once. Test in Task 4.

---

## File Structure

- Modify `game.py` — ship-pilot seam, clock seam, exit/score-out live when piloted, Demo wiring.
- Modify `game_handlers/autoplay.py` — end rules for a piloted Demo.
- Create `hunter/campaign_senses.py` — `CAMPAIGN_FEATURE_NAMES`, `CampaignSenses`, `ShipBody`, `CampaignSurroundings`.
- Create `hunter/campaign_pilot.py` — `ShipCommand`, `CampaignPilot`, `load_campaign_policy`.
- Modify `hunter/neural/policy.py`, `tools/migrate_policy.py` — policy layouts other than the hunter's.
- Create `experiments/campaign_rollout.py` — `CampaignShaping`, `CampaignResult`, `fly_campaign`.
- Create `learning/adapters/campaign_hunter.py`; modify `learning/tasks.py`, `tools/ppo_neural.py`, `tools/trainer/settings.py` (only if `applies_to` needs the new options listed).
- Create `tools/campaign_eval.py` — the "done when" measurement.
- Tests: `tests/test_campaign_pilot_seam.py`, `tests/test_campaign_senses.py`, `tests/test_campaign_demo.py`, `tests/test_campaign_training.py`.

---

### Task 1: Ship-pilot and clock seams in `Game`

**Files:**
- Modify: `game.py` (player input block ~970–1104, exit check ~1183, score-out ~1203–1215, `start_level` ~462)
- Test: `tests/test_campaign_pilot_seam.py` (headless game from the `game` fixture pattern in `tests/test_level_arc.py:38`)

**Interfaces:**
- Produces:
  - `ShipCommand` — `@dataclass(frozen=True) class ShipCommand: turn: int; thrust: bool; fire: bool` (turn −1 left, 0, +1 right), defined in `hunter/campaign_pilot.py` (create the module with just this class now).
  - `Game.ship_pilot: Optional[object]` (default `None`), any object with `decide(self, game) -> ShipCommand`.
  - `Game.clock: Callable[[], float]` (default `time.time`) and `Game.clock_ms: Callable[[], int]` (default `pygame.time.get_ticks`); every read of wall time on the ship's fire cooldown and potential-score path goes through them.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_ship_pilot_flies_the_ship_and_the_keyboard_does_not(game, monkeypatch):
    # demo game (autoplay set), pilot always thrusts and turns right; a held LEFT key is ignored
    ...
    assert ship moved forward and ship.angle increased by the ship's rotation speed per frame
def test_a_piloted_ship_fires_on_the_game_clock(game):
    # clock_ms frozen: the plain gun fires once, then not again until clock_ms passes the cooldown
def test_a_piloted_ship_reaching_the_open_exit_completes_the_level_in_demo(game):
    # ship placed on the activated exit with autoplay set → game.exit_explosion_active is True
def test_a_piloted_ship_scores_out_in_demo(game):
    # game.clock advanced until potential score <= 0 → game.game_frozen is True (game-over sequence started)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_campaign_pilot_seam.py -v` — expect 4 FAIL (`ShipCommand` missing / attributes missing).

- [ ] **Step 3: Implement.** In the input block, when `self.ship_pilot is not None`: take `cmd = self.ship_pilot.decide(self)`, build `commands` from it (`ROTATE_LEFT`/`ROTATE_RIGHT`/`APPLY_THRUST`) and `fire_pressed = cmd.fire`, skipping the keyboard/controller reads and the autoplay suppression; everything after (execute, record, fire, `ship.update`) is unchanged. Replace `pygame.time.get_ticks()` there with `self.clock_ms()` and the `time.time()` reads in `start_level` and the score-out check with `self.clock()`. Change the exit and score-out guards from `self.autoplay is None` to `(self.autoplay is None or self.ship_pilot is not None)`.

- [ ] **Step 4: Run** the new tests and `python -m pytest -q tests/test_autoplay.py tests/test_level_arc.py` — expect all PASS.

- [ ] **Step 5: Commit** `git commit -m "Game: a ship pilot can fly the player ship; game clock seam"`

---

### Task 2: Campaign senses

**Files:**
- Create: `hunter/campaign_senses.py`
- Test: `tests/test_campaign_senses.py`

**Interfaces:**
- Consumes: `hunter.senses.Senses`, `FEATURE_NAMES`; `hunter.maze_route.MazeRoutes.route(maze, origin, targets) -> (index, route) | None`; `learning/adapters/player_pilot.py` `PlayerSenses` as the model for presenting a `Ship` with hunter attributes.
- Produces:
  - `CAMPAIGN_FEATURE_NAMES = FEATURE_NAMES + ('exit_compass_present', 'exit_compass_sin', 'exit_compass_cos', 'exit_compass_dist', 'exit_open', 'beacon_compass_present', 'beacon_compass_sin', 'beacon_compass_cos', 'beacon_compass_dist')`
  - `class ShipBody` — wraps a `Ship` and its `Game`, exposing what `Senses.observe` reads (`x, y, vx, vy, angle, max_speed, fire_remaining, fire_interval, burst_remaining, gun_upgrade_level, weapon`, …); `health` = remaining potential score as a fraction of `scoring.calculate_max_possible_score()`, times `settings.health`.
  - `class CampaignSurroundings(game)` — `maze`, `enemies()` = `game.roster.all()`, `threats()` = `game.projectiles`, `pickups()` = active non-beacon crystals, `friend()` = `game.hunter`.
  - `class CampaignSenses(Senses)` — `perceive(self, body: ShipBody, surroundings: CampaignSurroundings, last_controls, now) -> np.ndarray` of length `len(CAMPAIGN_FEATURE_NAMES)`: the hunter's vector followed by the exit compass (route waypoint furthest in sight, as `_enemy_compass` does; distance in maze widths; `present` = 1 always), `exit_open` = `float(game.exit_opened)`, and the beacon compass (straight line through walls like the crystal compass; all zero when no active `JevBeacon`).

- [ ] **Step 1: Write the failing tests**

```python
def test_campaign_senses_are_the_hunter_senses_then_exit_and_beacon():
    assert CAMPAIGN_FEATURE_NAMES[:len(FEATURE_NAMES)] == FEATURE_NAMES and len(vector) == len(CAMPAIGN_FEATURE_NAMES)
def test_the_exit_compass_follows_the_maze_route_not_the_straight_line(game):
    # ship with a wall between it and the exit: compass bearing matches the route's first visible waypoint,
    # and differs from the straight-line bearing to the exit
def test_exit_open_reports_the_portal(game):
    # 0.0 before game.exit_opened, 1.0 after
def test_the_beacon_compass_points_at_an_active_beacon_and_is_blank_without_one(game)
def test_ship_health_is_the_remaining_potential_score(game)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_campaign_senses.py -v` — expect FAIL (module missing).
- [ ] **Step 3: Implement** as specified in Interfaces; `CampaignSenses.perceive` calls `Senses.observe` for the shared part and appends.
- [ ] **Step 4: Run** the tests — expect PASS.
- [ ] **Step 5: Commit** `git commit -m "Campaign senses: the hunter's, plus exit and Ally beacon compasses"`

---

### Task 3: Policy layouts and the campaign start policy

**Files:**
- Modify: `hunter/neural/policy.py` (`NeuralPolicy.load`), `tools/migrate_policy.py`
- Test: `tests/test_campaign_senses.py` (append)

**Interfaces:**
- Produces:
  - `NeuralPolicy.load(cls, path, feature_names=FEATURE_NAMES)` — validates against the given layout (the legacy-prefix padding still applies only to the hunter layout).
  - `migrate(old_path, new_path, feature_names=FEATURE_NAMES) -> int`; CLI `python -m tools.migrate_policy OLD NEW [--layout hunter|campaign]`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_migrated_hunter_pilot_flies_the_same_with_campaign_senses(tmp_path):
    migrate('assets/neural/hunter.npz', tmp_path/'c.npz', CAMPAIGN_FEATURE_NAMES)
    hunter, campaign = NeuralPolicy.load('assets/neural/hunter.npz'), NeuralPolicy.load(tmp_path/'c.npz', CAMPAIGN_FEATURE_NAMES)
    # same hunter senses, arbitrary values in the new inputs → identical logits from step()
```

- [ ] **Step 2: Run** — expect FAIL (unexpected keyword).
- [ ] **Step 3: Implement** the `feature_names` parameter in both; `--layout campaign` maps to `CAMPAIGN_FEATURE_NAMES`.
- [ ] **Step 4: Run** — expect PASS; also `python -m pytest -q tests -k "policy or migrate"`.
- [ ] **Step 5: Commit** `git commit -m "Policies can carry the campaign sense layout"`

---

### Task 4: `CampaignPilot` and Demo wiring

**Files:**
- Modify: `hunter/campaign_pilot.py`, `game.py` (`_begin_autoplay_level` ~248, `_autoplay_level_over` ~267, `_hunter_sees_player` ~261, `complete_level` ~1249, draw ~1430), `game_handlers/autoplay.py`
- Test: `tests/test_campaign_demo.py` (use the `make_game` fixture pattern from `tests/test_autoplay.py:154`)

**Interfaces:**
- Consumes: `ShipCommand`, `Game.ship_pilot`, `CampaignSenses`, `ShipBody`, `CampaignSurroundings`, `hunter.neural.controller.NeuralController(policy, decision_frames, greedy=True, rng=None)` and its `tick(senses_factory)`, `to_action`.
- Produces:
  - `class CampaignPilot` — `__init__(self, policy, decision_frames, greedy=True, rng=None)`; `decide(self, game) -> ShipCommand`; `on_observed(self, listener)` where `listener(senses: np.ndarray, last_controls: tuple)` is called once per decision (used by the recorder in Task 5); `reset()` at each level start.
  - `load_campaign_policy(path=None) -> Optional[NeuralPolicy]` — default `assets/neural/campaign_hunter.npz`, loaded with `CAMPAIGN_FEATURE_NAMES`; `None` (one warning) when missing or invalid.
  - `AutoplayDirector.start_level(enemy_count, hunter_present=True, piloted=False)`; when `piloted`: no `'hunter_dead'` and no `'cleared'` ending; `'no_kill'` only while `enemies_left > 0`; `'time'` unchanged; new reasons `'exited'` and `'scored_out'` reported by the game.

- [ ] **Step 1: Write the failing tests**

```python
def test_demo_without_a_campaign_policy_is_unchanged(make_game, monkeypatch)       # Review Focus 1
def test_demo_flies_the_ship_with_the_campaign_pilot(make_game, monkeypatch)
    # stub policy installed → game.ship_pilot is a CampaignPilot, no hunter at level start, ship drawn
def test_reaching_the_exit_in_demo_moves_on_to_the_next_level_once(make_game)      # Review Focus 5
def test_scoring_out_in_demo_moves_on_to_the_next_level(make_game)
def test_demo_levels_never_touch_the_profile(make_game)                             # Review Focus 2
def test_the_ally_beacon_in_demo_summons_a_hunter_that_follows_the_ship(make_game)  # Review Focus 4
def test_the_no_kill_skip_waits_while_the_ship_heads_for_the_exit()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_campaign_demo.py -v` — expect FAIL.
- [ ] **Step 3: Implement.** `_begin_autoplay_level`: if `load_campaign_policy()` returns a policy, set `self.ship_pilot = CampaignPilot(policy, SETTINGS.game.neuralDecisionFrames)` (greedy), don't summon the hunter, and start the director with `piloted=True`; otherwise today's path with `ship_pilot = None`. `_hunter_sees_player` is True when piloted. When piloted, the ship is drawn as normal and `complete_level` in Demo skips profile recording and, instead of showing level-complete, hands `'exited'`/`'scored_out'` to `_autoplay_level_over`, which advances exactly once.
- [ ] **Step 4: Run** the tests plus `python -m pytest -q tests/test_autoplay.py` — expect PASS.
- [ ] **Step 5: Commit** `git commit -m "Demo: the Campaign Hunter flies the player ship when its pilot is installed"`

---

### Task 5: Campaign episodes and rewards

**Files:**
- Create: `experiments/campaign_rollout.py`
- Test: `tests/test_campaign_training.py`

**Interfaces:**
- Consumes: `experiments.harness._make_game()` / `reset()` (cached headless Game); `CampaignPilot.on_observed`; `Game.clock`, `Game.clock_ms`; kill credit via `collision_handler.level_kills` (`'player'` vs `'hunter'`, `game_handlers/collision_handler.py:59`); `experiments.student.StudentSpec`.
- Produces:
  - `@dataclass CampaignShaping` — `own_kill=10.0, pickup=3.0, beacon=5.0, exit=100.0, cleared_exit=100.0, score_out=-100.0, wall_crash=-1.0, compass_following=0.02` (starting values; tuned in Task 7). `compass_following` pays per decision when the ship's velocity points within 45° of the enemy compass (enemies remain) or the exit compass (none remain).
  - `@dataclass(frozen=True) CampaignResult` — `level, seed, end ('exited'|'scored_out'|'horizon'), elapsed, enemies_at_start, own_kills, ally_kills, cleared (bool), pickups, beacon (bool)`.
  - `fly_campaign(level, seed, spec, horizon=180.0, shaping=CampaignShaping(), greedy=False) -> dict` with keys `senses (n×len(CAMPAIGN_FEATURE_NAMES) float32), controls (n×3 int8), rewards (n float32), temperatures (n float32), result (CampaignResult)`; one row per pilot decision; terminal rewards land on the last row.
  - Episode setup: Demo-like (`game.autoplay` set so profile is untouched, campaign pilot built from `spec` with `CAMPAIGN_FEATURE_NAMES`, sampling with `spec.sample_seed`), live exit, beacons and score-out, deterministic clock (`clock` from a frame counter at 1/60 s; `clock_ms` likewise), `random.seed(seed)`; the ship's sound manager muted. Ends: `exit_explosion_active` → `'exited'`; `game_frozen` by score-out → `'scored_out'`; `horizon` seconds → `'horizon'`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_campaign_episode_is_deterministic_and_well_formed(tmp_path)
    # migrated start policy, level 1, seed 300, horizon 3 → two runs give equal arrays; shapes agree; rewards finite
def test_exiting_pays_and_clearing_first_pays_more(tmp_path)
    # stub pilot that drives to the open exit; compare a level with enemies left vs cleared (kill them via encounter.kill)
def test_only_the_ships_own_kills_pay(tmp_path)
    # a kill credited to 'hunter' adds no own_kill reward and counts in ally_kills
def test_scoring_out_ends_the_episode_with_its_penalty(tmp_path)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_campaign_training.py -v` — expect FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** — expect PASS; then `python -m pytest -q tests -k "harness or rollout or autoplay"` to confirm the hunter harness is untouched.
- [ ] **Step 5: Commit** `git commit -m "Campaign episodes: fly the real ship through a level, with player-like rewards"`

---

### Task 6: The `campaign_hunter` training task

**Files:**
- Create: `learning/adapters/campaign_hunter.py`
- Modify: `learning/tasks.py` (`TASKS`), `tools/ppo_neural.py` (`_rollout_job` ~75, `make_jobs` ~56, `train_rl` ~138, `main` ~488, parser), `tools/trainer/settings.py` if needed
- Test: `tests/test_campaign_training.py` (append)

**Interfaces:**
- Consumes: `fly_campaign`, `CampaignShaping`, `migrate`, `NeuralPolicy.load(path, feature_names)`.
- Produces:
  - `TASKS` entry `campaign_hunter` → module `learning.adapters.campaign_hunter`, label "Campaign Hunter", `Capabilities(fresh_weights=False, replay_mode=None, settings_scope='listed')`, `options` = the campaign shaping flags.
  - Adapter: `FEATURE_NAMES = CAMPAIGN_FEATURE_NAMES`; `environment(**settings)` returning an object whose `run(level, seed, student, **settings)` returns `fly_campaign(...)`.
  - `train_rl(..., task='hunter')`: jobs carry the task; `_rollout_job` uses `task(job_task).adapter().environment().run`; policies load/save with the adapter's `FEATURE_NAMES`; default levels for `campaign_hunter` are `SETTINGS.autoplay.firstLevel..lastLevel`; if `--start` is the hunter pilot, it is first migrated into the run folder as `iter_start.npz` with the campaign layout. Log lines for this task report: exit %, cleared-exit %, scored-out %, mean own kills, mean return.
  - CLI: `python -m tools.ppo_neural --task campaign_hunter [--own-kill ... --compass-following ...]` (one flag per `CampaignShaping` field, defaults from the dataclass).

- [ ] **Step 1: Write the failing test**

```python
def test_one_campaign_training_iteration_saves_a_campaign_policy(tmp_path)
    # train_rl(start='assets/neural/hunter.npz', out_dir=tmp_path, iterations=1, episodes=2, workers=1,
    #          task='campaign_hunter', levels=[1], horizon=2) → an iter*.npz that loads with CAMPAIGN_FEATURE_NAMES
```

- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** per Interfaces; the hunter task's behaviour and defaults are unchanged.
- [ ] **Step 4: Run** the test, `python -m pytest -q tests -k "ppo or trainer or tasks"` — expect PASS.
- [ ] **Step 5: Commit** `git commit -m "Trainer: campaign_hunter task"`

---

### Task 7: Measure, train, ship

**Files:**
- Create: `tools/campaign_eval.py`
- Copy (after the user approves): best checkpoint → `assets/neural/campaign_hunter.npz`

**Interfaces:**
- Produces: `python -m tools.campaign_eval POLICY [--levels 1-40] [--seeds 3]` → per-level and overall exit %, cleared-exit %, scored-out %, mean own kills, flying greedily via `fly_campaign(..., greedy=True)`.

- [ ] **Step 1:** Run the eval on the migrated start policy to get a baseline; record it in the run notes.
- [ ] **Step 2:** **Ask the user** before starting training; then start `campaign_hunter` via the astrovoid-trainer tools (or `tools/ppo_neural.py` if the MCP tools don't yet take a task — add task support there in that case).
- [ ] **Step 3:** Watch the log lines; if it stalls, adjust shaping (first suspects: `compass_following`, `exit`, `score_out`) and ask before restarting.
- [ ] **Step 4:** Evaluate candidate checkpoints with `campaign_eval`; the bar is the spec's: on most Demo levels it clears every enemy and exits. Show the user the numbers, then a Demo on the real display.
- [ ] **Step 5:** With the user's go-ahead, copy the chosen checkpoint to `assets/neural/campaign_hunter.npz` and commit `git commit -m "Ship the Campaign Hunter pilot"`.
