# Developer guide

## Setup

Python 3.10 to 3.13 (3.13 recommended). `run.sh` expects the virtual environment at `venv/`.

```bash
python3.13 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import pygame.mixer; pygame.mixer.init(); print('mixer ok')"
./run.sh                          # or: python main.py
```

| Requirements file | For |
|---|---|
| `requirements.txt` | The game, tests and PyInstaller |
| `requirements-train.txt` | Training brains (PyTorch) and the training manager's MCP tools |

### Troubleshooting

- **`NotImplementedError: mixer module not available`**: pygame was built from source without SDL2_mixer, which happens on Python 3.14 (it has no prebuilt pygame wheels). Recreate the venv with Python 3.13.
- **`objc: Class SDL... is implemented in both ...` on macOS**: `opencv-python` bundles its own SDL2. The warnings are noisy but harmless.
- **`./run.sh: venv/bin/python: No such file or directory`**: create the venv first.

## Launch switches

Set these in the environment, for example `WINDOWED=1 START_LEVEL=13 ./run.sh`.

| Variable | Effect |
|---|---|
| `START_LEVEL=N` | Start at level N |
| `WINDOWED=1` | Run in a window (shrunk to fit the screen); `WINDOW_SIZE=1352x878` sets an exact size |
| `NO_POWER_DRAIN=1` | Power never drains, so a level never fails |
| `LEVEL_EDITOR=1` | Adds an **EDITOR** item to the main menu |
| `ASTROVOID_OBSERVER=1` | The player becomes an invisible, invulnerable observer, for watching creatures and pilots |
| `ASTROVOID_HUNTER_HOTKEY=1` | **H** summons the hunter ally beside you |
| `AUTOPLAY=1` | Skip the menus and let the Campaign Hunter fly the ship through levels unattended (Demo; see [Hunter ally](hunter.md)) |
| `JELLY_POLICY`, `FLOCKER_POLICY`, `WALLCRAWLER_POLICY`, `INCHWORM_POLICY` | Path to a different brain; set empty for the scripted behaviour |
| `SCRIPTED_JELLIES=1` | Spawn the old scripted Jellies instead of neural ones |

## Level editor

With `LEVEL_EDITOR=1`, choose **EDITOR**, pick a level and lay it out cell by cell: rock, metal walls, every enemy type, the start, the exit, the ally beacon and gates. **S** saves an exact `layout` into `levels/N.json` (git is the undo); **X** reverts to the generated level.

| Key | Action |
|---|---|
| Arrows | Move the cursor |
| Tab, Shift+Tab (or `[` `]`) | Next or previous palette item |
| Space | Place |
| Delete, Backspace | Remove |
| R | Turn a gate 45° |
| F | Flip a motor gate's direction |
| S / X / Esc | Save / revert / back |

The file format is in [levels/README.md](../levels/README.md).

## Project layout

```
main.py, game.py        Entry point and the game coordinator
config/                 settings.json (all tuning) and enemies.json (one entry per enemy kind)
levels/                 One JSON file per designed level
creatures/              Enemies built from a Body plus Traits, and the Encounter that runs them
  traits/               Behaviours: brains, propulsion, combat, flocking, spawning, ...
entities/               Bodies (how each creature looks), the ship, projectiles, exit
learning/               Neural brains: observations, actions, NumPy inference, training scenarios, PPO
  adapters/             One adapter per species (jelly, flocker, wallcrawler, inchworm) plus its scenario
assets/neural/          The bundled trained brains (.npz)
hunter/                 The hunter ally's pilots (neural and rules)
maze/                   Cave generation, rock blocks and their destruction
physics/                Gates and articulated bodies (pymunk)
weapons/                The weapon catalogue
rendering/, sounds/     Drawing, HUD, menus and procedural sound
scoring/                Score, stars and par
editor/                 The level editor
tools/                  Training manager, demo videos, replay viewer, build smoke test
experiments/            The rules lab and the neural pilot's first lessons
tests/                  pytest suite
```

See [ARCHITECTURE.md](../ARCHITECTURE.md) for how these fit together and [GLOSSARY.md](../GLOSSARY.md) for the vocabulary.

## Configuration

All tuning lives in `config/settings.json`, and each enemy kind has one entry in `config/enemies.json`. `config.py` loads them into typed dataclasses (`config.SETTINGS`) and also exposes the older uppercase names such as `config.SCREEN_WIDTH`. Restart the game after editing.

Player movement is tuned under `ship`, and hunter movement (allied and enemy) under `hunterShip`.

## Tests

```bash
venv/bin/python -m pytest tests/
```

See [tests/README.md](../tests/README.md).

## Building releases

PyInstaller can't cross-compile, so GitHub builds both platforms: run **Build Squiddler** from the Actions tab, or push a `v*` tag. The workflow (`.github/workflows/build.yml`) builds on macOS (Apple Silicon) and Windows, starts each build headless to check it runs (`tools/smoke_build.py`), and drafts a release with both zips for you to publish.

To build locally for your own platform:

```bash
./build.sh            # macOS/Linux; build.bat on Windows; or: pyinstaller pyinstaller.spec
```

The output goes to `dist/`. On macOS it is `Squiddler.app` (zip it with `ditto -c -k --keepParent dist/Squiddler.app Squiddler-macos.zip` to keep its symlinks). On Windows and Linux it is a `Squiddler/` folder, which must be kept together. `ASTROVOID_PYINSTALLER_MODE=onefile` builds a single executable instead. The spec leaves out torch and scipy, which the game never imports.
