# How to play

Fly from the start to the green exit portal before your power runs out. Thrust, hits and time all drain power. The ship keeps its momentum, so plan your braking.

Most levels lock the exit until you complete their mission, which is announced as the level starts. On the early levels the mission is to destroy every Egg. Later missions ask for more: every Flighthouse, every Wallcrawler or every Flocker, and from level 9 most levels (and every boss level) stay locked until every enemy is destroyed, including whatever hatches or is left behind. A level that must be cleared of every enemy gives unlimited ammo. The portal dims and powers down while it is locked, and brightens with a power-up sound when it opens.

The same basics are in the game itself: **HOW TO PLAY** on the main menu has four pages (goal and controls, power, crystals and weapons, stars and enemies). Their copy is `rendering/how_to_play_pages.py`, and every figure in it is read from the settings.

The game opens with a five-second animated splash; any key skips it, and `ui.splashEnabled` in `config/settings.json` turns it off.

## Stars

Reaching the exit gives one star for each goal you met. A level you don't finish gives none.

| Star | Earned when |
|---|---|
| No hits | No enemy bullet or body touched your ship (hits on an active shield don't count). |
| Half the enemies | At least half the level's enemies were killed. |
| Every enemy | No enemy is left alive when you reach the exit. |
| Every crystal | No weapon crystal is left uncollected (a level where none dropped earns it). |
| Par | You finished within the level's par time. |

- Enemies that appear during the level count too. Kills and crystals by your hunter ally count as yours.
- Par is worked out when the level starts: the shortest route flown at half top speed, plus time for each enemy and crystal. The settings are the `game.par*` keys in `config/settings.json`. The clock starts at your first move.
- During play, the stars you'd earn by exiting now are shown above the level number. A chime plays when a goal is met or lost.

A score is still kept (100 per level, less time, collisions, shots and fuel, plus kill bonuses), but stars decide your rating. **LEVELS** on the main menu lets you replay any unlocked level to improve your best stars, score and time without affecting your progress. Press **R** (controller **Y**) to restart a level at once.

## Weapons

Your basic gun has unlimited ammo. Destroyed enemies can drop a weapon crystal. Fly near it to pull it in; touching it adds the weapon and selects it. **Down / S** (controller **A**) cycles through your loaded weapons. Crystals vanish after six seconds, and every level starts with only the basic gun.

| Weapon | Behaviour | Ammo per pickup |
|---|---|---:|
| Twin Blaster | Rapid parallel blue bolts | 60 volleys |
| Scattergun | Five heavier orange pellets; two hits each | 20 volleys |
| Railgun | Pierces creatures and instantly destroys rock; stopped by metal walls | 12 shots |
| Seeker | Green missiles that home on visible enemies | 18 missiles |
| The Inflator | Balloons a creature to five times its size; any further hit pops it | 18 shots |
| Grenade Launcher | Bouncing shells with a proximity fuse; they chain Belemnite explosions and can hurt you | 10 grenades |
| Shield | Invulnerable while active; press Fire to toggle it | 10 seconds |

Tuning and drop weights are in `weapons/catalog.py`; the drop chance is `powerups.crystalSpawnChance` in `config/settings.json`.

## Walls

The caves are solid rock, drawn and felt as smooth curved walls. You bounce off the curve, not the grid behind it, and the bulges come from the level's seed, so a level always looks the same.

- Four hits break a block of rock (`maze.wallHitPoints`). Each hit chips and cracks it; the last bursts it into rubble. Tunnel through thick rock one block at a time.
- The outer ring cracks but never breaks. Riveted **metal** walls never break at all.
- Enemy shots stop at walls without damaging them.

The shape is tuned under `maze.cave` in `config/settings.json`.

## Gates

A gate is a plank on a pin that swings under real physics. **Free gates** settle wherever they're pushed; **motor gates** turn steadily and push back. Ships and creatures shove them aside, bullets nudge them, and a wall can jam them until you shoot it away. They're tuned under `obstacles` in `config/settings.json` and placed with the level editor.

## Enemies

Creatures marked 🧠 are driven by a trained neural network (see the [README](../README.md#meet-the-creatures)).

| Enemy | |
|---|---|
| Urchin | Sits still and fires when you're in range. Shots knock it about. |
| Crab | Patrols and shoots. |
| Jelly 🧠 | Fragile, fast chaser that hunts in teams. |
| Egg | Grows, then hatches Squidlets. On the early levels the exit stays locked while one is alive. Squidlets' guns grow stronger with the level, and past level 24 they take more hits, up to three. |
| Flocker 🧠 | Flies and fires as a flock and perches on walls. **Flighthouses** sweep a searchlight and launch more when they see you. |
| Anemone | Rooted to the rock; pulls your ship in. |
| Squiddler | A purple ship that replays your previous attempt at the level. |
| Belemnite | A spiky shelled chaser that explodes when it dies, hurting everything near it, walls included. |
| Wallcrawler 🧠 | A gun on twelve legs that walks on walls. Knock it loose and it can't fire while it tumbles. |
| Inchworm 🧠 | A physical six-link worm that crawls, rears and bites. It eats any creature except other inchworms. |
| Vibelet, Viber 🧠 | Ships flown by the neural pilot. A **Vibehouse** launches Vibelets. |
| Terminator | A ship flown by the hand-written rules pilot, hunting only you. |
| Squiddler Boss, Mother, Boss Viber | Bosses. The Squiddler Boss splits when destroyed, and the Mother lays Eggs. Their guns grow stronger with the level like every other enemy's, and past level 24 they take more hits each level, as do the Squidlets they leave behind. |

Every enemy kind has one entry in `config/enemies.json`.

## Levels

Levels 1–24 are a designed arc that introduces one new threat at a time, with boss levels along the way. Later levels are generated endlessly and get harder without getting more crowded. Every sixth one is a boss arena. The full progression and the level file format are in [levels/README.md](../levels/README.md).

From level 10 (except boss levels) an **ally beacon** can drop from enemies once you've killed 65% of them. Fly into it to summon an allied hunter ship (see [Hunter ally](hunter.md)).
