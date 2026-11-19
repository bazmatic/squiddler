# Tests

About 200 test files cover the game, the creatures and the trainer. Run them headless:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy venv/bin/python -m pytest tests/ -q
```

Run one area by name, for example `tests/test_inchworm*.py`, `tests/test_hunter_*.py` or `tests/test_level_*.py`.

## What's where

| Files | Cover |
|---|---|
| `test_ship*`, `test_projectile*`, `test_ccd*`, `test_cave*`, `test_wall*` | Flight, shots, swept collision and the destructible curved rock |
| `test_encounter*`, `test_traits_*`, `test_*_creature*` | Creatures built from the catalogue, and each trait on its own |
| `test_jelly*`, `test_flocker*`, `test_wallcrawler*`, `test_inchworm*`, `test_sea_snake*`, `test_neural*` | Each creature's body and the neural brains that drive it |
| `test_hunter_*`, `test_rules_*`, `test_pilot*`, `test_duel*` | The hunter ship and its pilots |
| `test_level_*`, `test_editor*` | Level files, the campaign arc and the level editor |
| `test_scoring*`, `test_star*` | Score, stars and par |
| `test_ppo*`, `test_training*`, `test_trainer*`, `test_replay*`, `test_dashboard*` | The training pipeline, the trainer and exact replays |

## Writing tests

Test what a feature does, not how it's built. A few tests that would catch a real regression are worth more than many that break whenever the code is rearranged.

The game's random brains make runs differ. Tests that need repeatable creature behaviour set the brain's policy variable empty (for example `INCHWORM_POLICY=""`) to use the scripted controller, or seed the brain explicitly. No test calls a paid service or needs an API key.

Anything that draws on screen behaves differently under the dummy video driver (it has no alpha channel), so check visual changes in the real game too.
