# Inchworm Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A rule-based `RoutedPolicy` that lets a walker checkpoint drive an Inchworm by default and hands over to a leaper checkpoint to leap, measured on held-out trials, then used by the game.

**Architecture:** `RoutedPolicy` has the same interface as `RecurrentPolicy` (`initial_state`, `choose`, `spec`, `mean`, `scale`), so `collect`, the evaluation scripts and `InchwormDriver` use it unchanged. Both specialists see every observation and keep their own memory; the rule only chooses whose action is used.

**Tech Stack:** Python, numpy, pytest, the repo `venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-11-25-inchworm-router-design.md`

## Global Constraints

- Nothing is trained; the specialists are existing checkpoints (walker: run336 `iter0999`; leaper: run342 `iter0150`), passed as paths, never committed (`.astrovoid-local` is untracked).
- The rule reads only the observation vector `observe()` returns (feature names in `learning.adapters.inchworm.FEATURES`), so it works unchanged in the game.
- Do not edit `physics/inchworm.py`, `creatures/inchworm_control.py`, `creatures/traits/inchworm.py`, `learning/adapters/inchworm.py` or `learning/adapters/inchworm_scenario.py`: their hash is in every run's manifest.
- Thresholds are tuned on training seeds, then confirmed once on separate held-out seeds. Hand-over hold is 3 s (30 decisions; decisions are 10 per second).
- Held-out sets are 150 trials per lesson (Crawl, Gap, About-face) and 240 for Leap.

## Review Focus

- Both grips free (worm in the air, or dead): the walker drives; no hand-over starts from the air.
- No goal sensed (`goal_present` 0): the walker drives, boredom never fires.
- Two specialists with different feature layouts or action heads: construction raises, no silent mismatch.
- A new episode starts with fresh counters and fresh specialist memory (`initial_state()` again), so no state leaks between episodes.
- Same seed gives the same actions (the router draws from the `rng` it is given, nothing else).

---

### Task 1: The rule and `RoutedPolicy`

**Files:**
- Create: `learning/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `learning.adapters.inchworm.FEATURES`, `SIGHT`; `RecurrentPolicy.choose(senses, state, masks=None, greedy=True, rng=None, temperature=1.0) -> (action, state)`; `.initial_state()`; `.spec`; `.mean`; `.scale`.
- Produces: `RoutedPolicy(walker, leaper, **thresholds)` with `.spec`, `.mean`, `.scale` (the walker's), `.initial_state() -> RouterState`, `.choose(senses, state, masks=None, greedy=True, rng=None, temperature=1.0) -> (action, RouterState)`; `RouterState` carries both specialists' states and the rule's counters, with `.driver` (`'walker'|'leaper'`) and `.handovers: int`. Thresholds with defaults: `reach_min=80.`, `reach_max=200.`, `bored_decisions=50`, `bored_progress=20.`, `hold_decisions=30`, `cooldown_decisions=20`.

- [ ] **Step 1: Write the failing tests** in `tests/test_router.py`, using two stub specialists (a class returning a fixed action tagged walker or leaper, a counter of calls, and a trivial `initial_state`) and a helper `senses(rear, front, goal_present, forward, side)` that builds a zero feature vector with those entries set by name. Tests, each asserting which stub's action came back:
  - `test_the_walker_drives_by_default` (planted, goal 40 px away)
  - `test_it_hands_to_the_leaper_when_planted_and_the_goal_is_in_leaping_reach` (goal 120 px away, one grip down)
  - `test_it_stays_with_the_leaper_until_the_worm_has_flown_and_landed` (leaper chosen through air steps with both grips free, then back to the walker once a grip is down again)
  - `test_it_hands_back_after_the_hold_even_if_the_worm_never_lands` (30 decisions)
  - `test_it_does_not_hand_over_again_during_the_cooldown`
  - `test_a_worm_gripping_without_getting_closer_is_bored_and_leaps_from_any_visible_distance` (goal 250 px, 50 decisions, distance not improving by 20 px)
  - `test_a_worm_that_is_closing_in_is_not_bored`
  - `test_both_grips_free_never_starts_a_handover`, `test_no_goal_never_hands_over_or_gets_bored`
  - `test_both_specialists_see_every_observation` (each stub's call count equals the number of steps)
  - `test_mismatched_specialists_are_rejected` (stubs with different `spec.features`)
  - `test_a_new_initial_state_forgets_the_counters`
  - `test_the_router_is_deterministic_for_a_seed` (two runs with `np.random.default_rng(3)` give equal action sequences, using the real `RecurrentPolicy` built by an existing test helper such as `random_policy` in `tests/test_neural_inchworm.py`)

- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_router.py -q`. Expected: FAIL, `learning.router` missing.

- [ ] **Step 3: Implement `RoutedPolicy` and `RouterState` in `learning/router.py`.** Distance to the goal is `SIGHT * hypot(goal_forward, goal_side)`. "Planted" is `rear_grip` or `front_grip` above 0.5. Rule order each call: update counters; if driven by the leaper, keep it until `hold_decisions` or the worm has been airborne (both grips free) and is planted again; otherwise hand over when planted, `goal_present`, past the cooldown, and either distance in `[reach_min, reach_max]` or bored. Bored means planted for `bored_decisions` consecutive decisions while the best (smallest) distance seen in that window improved by less than `bored_progress`. Raise `ValueError` in `__init__` if the specialists' `spec.features` or `spec.heads` differ.

- [ ] **Step 4: Run** the same command. Expected: all PASS.

- [ ] **Step 5: Commit** `learning/router.py tests/test_router.py` with message "Inchworm router: a rule that hands leaps to a specialist".

### Task 2: Held-out evaluation tool

**Files:**
- Create: `tools/router_eval.py` (run as `venv/bin/python -m tools.router_eval WALKER LEAPER [--seeds train|heldout]`)
- Test: `tests/test_router_eval.py`

**Interfaces:**
- Consumes: Task 1's `RoutedPolicy`; `learning.training.collect(env, policy, seed, scenario, ...) -> (episodes, events, actions)`; `inchworm_scenario.environment(horizon=40.)`.
- Produces: `evaluate(policy, lesson, count, seed_base) -> float` success rate over `count` trials of lesson stage (Crawl 1, About-face 8, Gap 9, Leap 11); `handover_rate(router, count, seed_base) -> float` fraction of Leap-lesson trials in which the router handed over at least once; a table printed for walker, leaper and router, for either the training seeds (`seed_base` 6000) or held-out seeds (`seed_base` 90000).

- [ ] **Step 1: Write the failing test** `test_evaluate_reports_a_success_rate_between_zero_and_one` and `test_the_two_seed_sets_do_not_overlap` (training and held-out seed ranges for the same lesson share no seed), using a small `count` and a random policy.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_router_eval.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `evaluate`, `handover_rate` and the CLI in `tools/router_eval.py`; levels as the existing scripts use (`level=k%4`).
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** with message "Held-out evaluation for the router".

### Task 3: Tune on training seeds, confirm on held-out

**Files:**
- Modify: `learning/router.py` (default thresholds only); `docs/superpowers/specs/2026-11-25-inchworm-router-design.md` (record the numbers)

- [ ] **Step 1:** Run `venv/bin/python -m tools.router_eval <walker> <leaper> --seeds train` and adjust `reach_max`, `bored_decisions`, `bored_progress`, `hold_decisions`, `cooldown_decisions` defaults one at a time, keeping Crawl, Gap and About-face within noise of the walker and Leap at or above the leaper.
- [ ] **Step 2:** Run once with `--seeds heldout`. Expected, from the spec: Crawl, Gap and About-face within noise of the walker (150 trials each); Leap within noise of the leaper's about 0.55 (240 trials); `handover_rate` clearly above 0 on a stuck Leap case. If Leap does not beat the walker without costing Crawl, record that in the spec and stop; do not retune on held-out seeds.
- [ ] **Step 3:** Record both tables in the spec's "Success" section. Run `venv/bin/python -m pytest tests/test_router.py tests/test_router_eval.py -q`. Expected: PASS.
- [ ] **Step 4: Commit** with message "Router thresholds and measured results".

### Task 4: Use the router in the game (only if Task 3 passed)

**Files:**
- Modify: `creatures/inchworm_neural.py` (`driver_for`)
- Test: `tests/test_router_game.py`

**Interfaces:**
- Consumes: `learning.router.RoutedPolicy`; `learning.adapters.inchworm.load_policy(path)`.
- Produces: environment variable `INCHWORM_ROUTER=<walker.npz>,<leaper.npz>`; when set, `driver_for(enc)` builds an `InchwormDriver` around `RoutedPolicy(load_policy(walker), load_policy(leaper))`; unset leaves today's behaviour unchanged.

- [ ] **Step 1: Write the failing tests** `test_driver_for_uses_the_router_when_the_variable_is_set` (with `monkeypatch.setenv` and two saved random checkpoints; the driver's `policy` is a `RoutedPolicy`), `test_driver_for_is_unchanged_without_the_variable`, `test_a_bad_router_path_falls_back_to_the_single_policy_with_a_warning`.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_router_game.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the branch in `driver_for` before the single-policy path; catch `(OSError, ValueError, KeyError, TypeError)` as the existing code does.
- [ ] **Step 4: Run** the new tests and `tests/test_neural_inchworm.py`. Expected: PASS.
- [ ] **Step 5: Play it:** `INCHWORM_ROUTER="<walker>,<leaper>" START_LEVEL=33 ./run.sh`; watch for leaps from a wall and for steady walking.
- [ ] **Step 6: Commit** with message "Game: INCHWORM_ROUTER runs a walker and a leaper".
