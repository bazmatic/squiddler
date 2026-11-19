# Level files

Levels 1 to 24 are a designed arc: each has a file here, `{level}.json`, that says exactly what is on it. Levels 25–32 also have custom files. Later levels come from the formulas in `level_rules.py` unless a file overrides them.

A file can exist for any level, and can state as much or as little as it likes. Outside an explicit `enemies` section, omitted settings come from the formulas. Within `enemies`, omitted types mean zero.

## Format

```json
{
  "seed": 107,
  "maze": { "complexity": "normal", "grid_size": 20 },
  "enemies": {
    "static": 2, "patrol": 1, "aggressive": 2, "replay": 1,
    "flocker": 0, "flighthouse": 0, "egg": 0, "anemone": 2,
    "split_boss": 0, "mother_boss": 0, "rules_hunter": 0
  },
  "hunter": false
}
```

- **seed**: random seed for the maze, enemy positions and the bulges of the curved cave rock (so a level's rock is the same every time, hand-made layouts included). Default: the level number. Arc levels use `100 + level`.
- **maze.complexity**: `empty` (perimeter walls only), `simple`, `normal`, `complex` or `extreme`.
- **maze.grid_size**: width and height in cells, 5 to 100. Bigger means more, smaller cells.
- **enemies**: counts for the types the file overrides; listing every enemy type is optional. Omitted types mean zero; an empty `enemies: {}` section gives an empty roster. If the whole section is absent, level formulas apply. Anemone counts in a file are taken as given; the formulas would hold them to what the maze has room for.
- **vibehouse**: a pink Flighthouse variant that launches Neural-piloted Vibelets when its searchlight sees the player or an ally. It has five hit points and uses the Flighthouse launch cadence. Add it through the editor or an `enemies.vibehouse` count.
- **rules_hunter**: Rules hunters on the level (the Rules pilot's hunter ship as an enemy, hostile only to the player). A level with one is a boss level. Level 25 has one.
- **neural_hunter**: Neural hunters on the level (the neural pilot's hunter ship as an enemy, in magenta, hostile only to the player). Like the Rules hunter it takes 12 hits and makes a boss level. Without an installed neural model (`assets/neural/hunter.npz`) the level starts without it.
- **belemnite**: Belemnites on the level: spiky squids in pointed shells that chase the player as Jellies do, never overlapping one another. They have no guns, take 3 hits, and explode when they die, hurting the player, the hunter, nearby walls and other enemies (so one can set off the next).
- **wallcrawler**: A gun walking on twelve little legs along curved rock or square steel. Bullets and explosions knock it loose; it cannot fire during its 0.6-second stunned tumble or its flight to the nearest wall. It has four hit points. Campaign encounters begin with a crowd of five on level 11 and rise to eight by level 21; level 27 pairs six with explosive Belemnites. Layouts place it through the editor; generated levels accept an `enemies.wallcrawler` count. Levels without a file carry five to eight on every fourth level from 11, the ones built around them.
- **inchworm**: A six-link worm that crawls on rock and steel on grips, rears up and bites. It eats everything alive except other inchworms: the player ship, hunters and every other creature, one hit point per bite. Layouts place it through the editor beside a wall; give it open floor (a 5x5 block with at most a few wall cells), because its arch jams in a narrow dead end. The campaign has two on levels 13, 16, 19, 22, 26 and 29, as a minority predator among the level's usual enemies; level 33 is the inchworm playtest and level 34 (Looper Garden) is built around eight of them, with four Wallcrawlers. Without `assets/neural/inchworm.npz` it walks with the scripted stride.
- **Snakes are solid to one another** (every size): bodies that meet are pushed apart, not left to pass through.
- **mini_sea_snake**: The sea snake at about half size: same body, same trained brain (it swims in its own lengths, so it works at any size) on a clock running 2.4 times faster, but 2 hits to kill and a smaller, shorter blast (1 hit to enemies and the ship). Level 35 only so far.
- **teeny_sea_snake**: The sea snake at about a third of its size: one hit kills it, and its blast is tiny (1 hit, a short reach). Same body and brain, but its own clock runs 3.5 times faster so it keeps up. Level 35 only so far.
- **sea_snake**: A free-swimming eight-link serpent that wriggles through open water after the player and stings on contact, which makes the ship flinch and throws it clear. It swims by bending its body alone. It takes 10 hits, and a shot knocks it about. When it dies it thrashes, bursts link by link, and a blast goes off 1.3 seconds after the kill (2 hits to enemies, one to the ship, shoving everything within 150 px), so back off. Layouts place it through the editor anywhere in open water; it is not planted on a wall. The campaign has two each on levels 28 and 31, the open ones, as a minority predator among the level's usual enemies (not level 23: its rotating gates jam a snake's physics); level 35 is the sea snake playtest. Without `assets/neural/sea_snake.npz` it swims with the scripted wave.
- **hunter**: `false` forbids an ally. `{"earned": true}` allows enemy kills to drop a beacon only after 65% of the starting enemies have been defeated (at least four kills). The usual 3% drop chance still applies after that threshold. No beacon starts on an earned level. An explicit `{"spawn_cell": [col, row]}` remains available for custom encounters.
- **layout**: an exact, hand-made level, written by the level editor (see the [developer guide](../docs/development.md#level-editor)). While it is present it replaces the generated maze, the random enemy positions and the `enemies` counts. Those are kept, untouched, so removing the layout restores the generated level. `beacon` takes the place of `hunter`. A layout that cannot be played (a cell out of range, an enemy on a wall, a missing start) is ignored with a warning and the level generates from its seed.

Generated mazes carve a connected network of broad corridors with continuous wall runs.
Complexity controls corridor density and the number of alternate routes; openings stay
at least roughly 70 pixels wide where the arena size allows. Corner rooms connect to
the same network as the rest of the maze. Seeds reproduce layouts, and training can
supply a different seed for each episode. Exact `layout` maps keep their authored geometry.

Level 31 introduces a Vibehouse in a central courtyard, with four offset entrances,
short walls for searchlight cover, and outer approach lanes. A flock of six Flockers, one Crab
and one Belemnite guard the approaches. Destroy the Vibehouse to open the exit;
ammo is unlimited, and there are no gates or allied beacons.

```json
"layout": {
  "walls":   [[3, 3], [4, 3]],
  "strong_walls": [[5, 3]],
  "start":   [1, 1],
  "exit":    [7, 7],
  "enemies": [{"type": "flocker", "cell": [2, 6]}],
  "beacon":  [3, 5],
  "gates":   [{"cell": [5, 5], "angle": 45, "motor": "cw"}, {"cell": [6, 3]}]
}
```

  **walls** are drawn as curved cave rock over their cells (see [Walls](../docs/gameplay.md#walls)); the cells still decide what is open. Seeded bulges may narrow passages; rock keeps gate swings clear and leaves the start, the exit, the beacon and each enemy as much room as square walls would.

  **strong_walls** (optional): wall blocks that no shot can destroy, drawn as riveted metal plate, and square (never curved). Use them where the player must not blast a way round, such as the walls of a pen whose only way in is its door. Everything that sees walls sees them; a shot at one strikes sparks and leaves no mark. A cell may be a wall or a strong wall, not both, and everywhere this page says "a wall" below, a strong wall counts too.

  **gates** (optional): each is a plank on a pin at the centre of `cell`. `angle` is in degrees clockwise on screen, a multiple of 45 from 0 up to but not including 180 (default 0, horizontal). `motor` is `"cw"` or `"ccw"` for a motor gate, and left out for a free gate. A gate may not share a cell with a wall, the start, the exit, the beacon, an enemy or another gate, and its plank (three cells long, so it covers its own cell and the one either side) may not cross a wall, leave the interior, cover the start, the exit, an anemone or a flighthouse, or share a cell with, cross or touch another gate's plank (planks that overlap would creep at rest); a layout that breaks this is ignored. Cells are not square on screen, so the game also measures each plank in pixels when the level starts: a gate whose plank would touch a wall or another plank there is left out (with a warning in the log). The level editor makes the same check on its own screen, so a gate placed there fits screens of that shape. A gate that doesn't fit the editor's screen (from a level saved on another screen, or written by hand) doesn't stop other edits; the cursor says "doesn't fit" over it, and R turns it to an angle that does.

  Cells are `[col, row]`; the border ring is always wall. Types: static, patrol, aggressive, replay, flocker, flighthouse, vibehouse, egg, anemone, split_boss, mother_boss, rules_hunter, neural_hunter, boss_viber, vibelet, belemnite. The editor does not check that the exit can be reached. A file the editor creates for a level that had none carries `"created_by_editor": true`, so reverting deletes it again.

## What each level is built around

Every ordinary level has one dominant active enemy type (Crabs, Jellies, Flockers, Wallcrawlers or, in the late levels, Belemnites), at least half of what moves on it. Jellies, Flockers and Wallcrawlers come in numbers: a level has none of them or at least four. A flock of up to eight shows the Flockers' flocking, and the boss levels bring a flock of six or eight. Urchins, Anemones, Eggs and Flighthouses are scenery around the theme and do not count towards it. Levels without a file follow the same rule, turning through the types the player has met.

## Missions

Each level's `exit_policy` is its mission, announced as the level starts. Levels 4 to 8 ask the player to destroy every Egg. After that the missions get harder: level 10 asks for every Flighthouse and Egg, level 11 for every Wallcrawler, level 15 for every Flocker, and levels 9, 13, 14, 16, 17, 19 and 20 for every enemy. A level that must be cleared of every enemy has unlimited ammo, so it can always be finished. Levels without a file keep the Egg mission up to `LAST_EGG_MISSION_LEVEL` in `level_rules.py`; after it, a level with Eggs must be cleared of every enemy.

## Boss levels

A level with at least one `split_boss`, `mother_boss` or `rules_hunter` is a boss level, whatever its number:

- the exit is locked until every enemy on the level is dead, including the ships a boss leaves behind and anything that hatches;
- ammo is infinite, so the level can always be finished;
- no ally beacon appears, so no hunter flies, unless the file places one.

The campaign introduces a single Split Boss at 12, two at 18, and a Mother at 24. Piloted hunter encounters also use clear-all exits and unlimited ammo. Level 30 introduces the Boss Viber with two Vibelets.

## The hunter

Collecting an ally beacon summons the hunter. Levels 1–9 and the boss and duel levels have no beacons. Elsewhere, levels use `{"earned": true}`: once you've killed 65% of the starting enemies (at least four), each kill has the usual 3% chance to drop a beacon. Levels without a file work the same way. A level file can also place a beacon outright with `{"spawn_cell": [col, row]}`.

## Campaign progression

Each level introduces at most one new creature:

| First level | New creature |
| --- | --- |
| 1 | Urchin |
| 2 | Crab |
| 3 | Jelly |
| 4 | Egg, and the Squidlets it hatches |
| 5 | Flocker |
| 7 | Anemone |
| 8 | Squiddler |
| 9 | Belemnite |
| 10 | Flighthouse |
| 11 | Wallcrawler |
| 12 | Squiddler Boss (boss level) |
| 13 | Inchworm |
| 19 | Vibelet |
| 21 | Viber |
| 23 | Terminator |
| 24 | Mother (boss level) |
| 30 | Boss Viber (boss level) |
| 31 | Vibehouse |

Level 6 is a practice arena, level 18 brings two Squiddler Bosses, and level 25 is a one-on-one fight with a Terminator. Ordinary levels through 23 grow by one enemy at a time, from four on level 1 to 24 on level 23, so there are no sudden swarms. `tests/test_level_arc.py` checks the first appearances, the gradual growth and the clear space around the start.

## Levels without a file

- **Ordinary levels from 25**: grid 32, `extreme`. The enemy count is 25 at level 25 and rises by one every two levels to 36 (`difficulty.maxEnemyCount`). Each added enemy goes to the next of aggressive, replay, flocker, static, patrol, anemone.
- **Every sixth level** (`difficulty.bossLevelInterval`) is a boss level in a 16-cell open arena: split bosses, then a mother boss, then both, gaining a boss each time round up to three.
- **Enemy speed and damage** grow 5% a level after level 4 and stop at 2.5 times.
- **An arc level whose file is missing** gets `3 + level` enemies of the types introduced by then.

## Level 28: Clockwork Reef

A gate-free, three-chamber hunt. Permanent reef walls put the main crossings at
opposite ends, creating an S-shaped route from the western start to the eastern
exit. Rock plugs in the other ends can be blasted into shortcuts. Belemnites
beside those plugs and the scattered cover make explosions useful, but dangerous
at close range.

Destroy both flighthouses (north in the middle chamber, south in the eastern
chamber) to open the exit. Other enemies may remain alive. Ammo is unlimited;
there is no hunter beacon and no neural model is required.

Start directly with `START_LEVEL=28 ./run.sh`.

## Level 29: Spiral Siege

A square spiral winds from the northwest start around the outside and into the
central exit chamber. Strong walls preserve the spiral; two three-block rock
sections let the player blast shortcuts through its coils. Belemnites beside
those sections make the shortcuts dangerous at close range.

Destroy both flighthouses, one on the outer turn and one in the core, to open
the exit. Ammo is unlimited. There are no gates, hunter beacons or neural enemies.

Start directly with `START_LEVEL=29 ./run.sh`.

## Level 32: Silkgrave Tunnel

A long crawler-only gauntlet: five folded cave passages lead from the quiet
northwestern entrance to the southeastern escape portal. Twenty Wallcrawlers
occupy alternating walls and widened lairs. Rock teeth interrupt the sightlines;
rounded turns and the chambers provide room to dodge and turn back. Permanent
reinforcement ribs preserve the winding route even when rock is blasted away.

Ammo is unlimited and the exit is open: reaching it is the objective, rather
than clearing every crawler. There are no other enemies, gates or hunter beacons.

Start directly with `START_LEVEL=32 ./run.sh`. To try a particular trained actor,
set `WALLCRAWLER_POLICY=/absolute/path/to/checkpoint.npz` on the same command.

## Level 33: Inchworm playtest

Four articulated inchworms crawl along two permanent horizontal platforms in an
open arena. Five bend muscles drive each body through alternating physical grips.
Approach a worm's head to provoke rearing, grabbing and feeding; shoot it to
interrupt the hold. Ammo is unlimited, the exit is open, and there are no other
enemies or allied beacons. Authored inchworms are placed on nearby wall surfaces
at setup so their first stride starts immediately.

Start directly with `START_LEVEL=33 ./run.sh`.

## Level 34: Looper Garden

Eight inchworms and four Wallcrawlers in five rock-walled rooms. Each room
has a permanent steel shelf with its inchworms: one on each corner shelf, four
along the long centre shelf of the middle hall. Inchworms are expensive to
simulate (about 0.6 ms a frame each), so the count stays low. The rooms connect
through narrow passages in the rock, and the way from the northwest start to
the southeast exit winds: down into the hall, along to its far west end, down
into the southwest room and east to the exit. The rock is all destructible, so
blasting through it shortens the route (48 cells to 38). The Wallcrawlers start
in the corners and hall ends, away from the shelves the worms keep to, though a
hungry worm will still eat one that wanders close.

The exit is open from the start, so the worms are a gauntlet to get past, not a
kill list; ammo is unlimited, and there are no gates or hunter beacons. The
shelves are square steel because a worm's arch jams in tight curved rock. On very small screens (960x640) the rock bulges close the
hall, and the way through has to be blasted.

Start directly with `START_LEVEL=34 ./run.sh`.

## Level 35: Sea snake playtest

Three free-swimming sea snakes, four mini and five teeny ones in an open arena with no walls. They are not
planted on rock like the crawlers. Ammo is unlimited and the exit is locked until every
snake is dead. There are no other enemies. See [the sea snake note](../docs/notes/sea-snake-body.md).

Start directly with `START_LEVEL=35 ./run.sh`.
