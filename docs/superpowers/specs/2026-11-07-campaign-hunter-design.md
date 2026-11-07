# Campaign Hunter — design

## Intent

Demo mode should look like someone actually playing the campaign. Today it is a highlights reel: the player's ship
sits parked and hidden, the allied hunter fights, and the level is skipped the moment the last enemy dies (or after
20 s without a kill, or after the time limit). Nobody ever reaches the exit.

The **Campaign Hunter** is a neural pilot that flies the **actual player ship** in Demo. It plays a level the way a
good player would: it hunts every enemy wherever it is, collects powerups and the Ally beacon, and, once the enemies
are dead, flies into the exit portal, completing the level with the normal tally before Demo moves on.

## Goals

- In Demo, the player ship is flown by the Campaign Hunter pilot instead of being parked.
- It uses everything a player has: the ship's own handling, ammo, weapon inventory, shield, score and stars.
- Reaching the exit completes the level exactly as it does for a person (exit explosion, tally, level complete), then
  Demo continues to the next level.
- Collecting the Ally beacon does what it does for a person: the ordinary allied hunter is summoned and fights
  alongside the ship.
- It seeks enemies anywhere in the maze, through walls (like the Jellies), and seeks powerups, including the beacon.
- **Done when** watching Demo is impressive: on most levels it kills all the enemies and makes it to the exit.

## Non-goals

- The Campaign Hunter is **Demo only**. Normal play, the summonable allied hunter, enemy Vibers (`neural_hunter`) and
  their pilot (`assets/neural/hunter.npz`) do not change.
- No weapon cycling: it fires whatever the last pickup equipped (pickups equip themselves, as they do for a person).
- No two-hunter support: the Campaign Hunter is the player, so the existing single allied-hunter slot is enough.

## Anti-goals

- Don't change the exit rules for it. The level's normal exit conditions apply; clearing enemies before leaving is a
  *style it is trained for*, not something the game enforces.
- Don't make it cheat: no extra speed, damage, health, or knowledge the player couldn't plausibly have beyond the
  through-walls senses described here.
- Don't regress the existing hunter training or the shipped hunter pilot.

## Decided

- **Flies the real `Ship`** (`entities/ship.py`), driven through the same commands a keyboard produces
  (`ROTATE_LEFT`, `ROTATE_RIGHT`, `APPLY_THRUST`, fire). The pilot's outputs map one to one: turn {left, none, right},
  thrust {off, on}, fire {off, on}.
- **Its own policy file**, separate from `assets/neural/hunter.npz` (e.g. `assets/neural/campaign_hunter.npz`), loaded
  only by Demo.
- **Starts from today's neural hunter pilot** (warm start), not from scratch, so it can already aim and dodge; expect
  early training to look clumsy while it adapts to the ship's handling.
- **Senses**: everything the hunter senses today, including its existing route-following enemy compass
  (`hunter/senses.py` `_enemy_compass`, which already points along the cheapest maze route to the nearest live enemy
  anywhere) and crystal compass, plus new inputs for:
  - the **exit** (a compass along the maze route, and whether the exit is open), and
  - the **Ally beacon** (a compass, when one is on the level).
  New inputs are appended and start at zero weight (as `tools/migrate_policy.py` does), so the warm-started pilot
  first flies as before. *(Builder's call:)* the existing `health` input carries the remaining potential score as a
  fraction of the level's maximum, the nearest thing a player has to health.
- **Dying means what it means for a player**: the ship has no health; its run on a level ends when the level's
  draining potential score reaches zero (`game.py` ~1213, `_start_game_over_sequence`; today skipped when
  `self.autoplay` is set). That score-out is **live for the Campaign Hunter, in training and in Demo**. In Demo the
  power-down / game-over plays and Demo moves on to the next level. If a very large maze sometimes can't be cleared
  in time, that is the real game; the drain is not changed for it.
- **Reward** is roughly what a good player wants: **its own kills** pay (kills by the summoned allied hunter do not);
  pickups and fetching the beacon pay; scoring out costs; finishing through the exit pays a lot, and more when every
  enemy was cleared first (by anyone, the ally included); wall crashes cost what they cost the player. Kill credit
  therefore has to be tracked per killer (player ship vs. allied hunter) in the training environment.
- **Episodes** end on reaching the exit, scoring out, or a time limit — not on the last kill.
- **Demo flow**: a level ends when the ship exits (normal completion) or scores out. The existing time limit stays as
  a safety net. *(Builder's call, recorded here:)* the no-kill skip applies only while enemies remain, so it can't cut
  short the flight to the exit after the last kill.
- The allied hunter summoned by the beacon uses the ordinary hunter pilot and sees the Campaign Hunter's ship as its
  friend, as it does a person's.

## Constraints

- One training run at a time, managed through the astrovoid-trainer tools (`tools/ppo_neural.py`, `tools/trainer/`).
  Ask before starting or restarting a run.
- The current training harness (`experiments/harness.py`) is built around flying a *hunter creature*: it parks and
  hides the player, disables exit completion and beacons, and ends episodes when the enemies are cleared. The
  Campaign Hunter needs a training environment where the real ship is flown on real levels with exit, beacons and
  the allied hunter live.
- The ship's `fire` path today reads the wall clock (`pygame.time.get_ticks()` in `Game` for the plain gun's
  cooldown); training must be deterministic and run faster than real time.
- Policy files check their input layout (`feature_names`); the campaign pilot's layout differs from the hunter's.
- Training levels: the Demo range (`autoplay.firstLevel`..`lastLevel`, currently 1–40).

## Left to the builder

- How the training environment is structured (a new `learning/tasks.py` task such as `campaign_hunter`, reusing the
  hunter adapter where it fits), and how the trainer UI offers it.
- Exact reward weights and episode time limit; tune against the "done when" bar.
- Exact encoding of the new senses (route vs straight line for the beacon, how "exit open" is shown).
- How Demo wires the pilot into the ship (a pilot-driven command source replacing keyboard input while in Demo).
- How the HUD shows Demo (it should simply look like the player's game).
- Whether the exit compass appears only when the exit is open or always (with an "open" flag).
