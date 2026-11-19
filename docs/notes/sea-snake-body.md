# Sea snake

Play the three-snake arena with `START_LEVEL=35 ./run.sh`. Ammo is unlimited and
the exit is open. The snakes swim straight for you through open water; they are
not planted on rock. Shoot one to knock it about, and keep clear of the whole
body, not just the head.

## Physical body

The snake has eight capsule links and seven bend joints, solved by Pymunk like the
Inchworm. Its size is 30 (about 170 px long and 20 px thick), and every constant in
`physics/sea_snake.py` scales with that radius. Each joint is a damped,
torque-limited angular spring toward a requested bend of up to 0.9 radians.

There is no thrust and no steering command. It swims by **anisotropic water drag**:
each link sheds 1% of its lengthwise velocity per frame but 150% of its sideways
velocity, so a wave of bending running from head to tail pushes the water back and
the body forward. Angular drag is 4% per frame. The scripted wave that swims
forward runs from head to tail, with one wavelength every 32 frames
(`wavelengthFrames`); the direction matters, because the opposite wave swims
backwards.

Because lengthwise drag is so light, a hit along the body makes it coast; a hit
across the body is soaked up almost at once.

## Controller interface

`physics/sea_snake.py` owns the mechanics: `SeaSnakeCommand(muscles)` carries seven
requested bends, and `step(command, dt, walls, gates, powered)` advances it. The
scripted controller, `creatures/sea_snake_control.py`, drives a travelling wave and
steers by biasing it. The neural driver replaces it when a trained brain exists.
A creature can also be given an explicit `snake_command`, which beats both.

## Behaviour

- **Sting.** Any part of the body touching the ship stings it, at most once every
  45 frames, unless the shield is up. A sting counts as a hit against the stars,
  makes the ship flinch and throws it 4 px/frame clear (`stingKnock`).
- **Hit points.** It takes 10 shots (`takesHits`), and a shot lands on any link.
- **Knockback.** A shot's momentum (its speed times the snake's mass) is scaled by
  `shotKnock` (1.6) and then split: 15% lands on the struck link, so the body whips
  and spins, and 10% is spread over the whole body (`WHIP` and `SHOVE` in
  `creatures/traits/sea_snake.py`). Together that is 0.4 times the shot's momentum.
  Keep the body share modest, because an end-on hit otherwise slides the snake
  backwards a long way.
- **Spawn radius.** The catalogue's `spawnRadius` is 100, not the 30 of its size,
  because the body reaches about 93 px from its centre. A snake placed closer to
  rock than that is born straddling it.
- **Cost.** About 0.2 ms a frame for physics and 0.35 ms a snake to draw.
- **Without a brain.** With no `assets/neural/sea_snake.npz` it swims with the
  scripted controller. `SEA_SNAKE_POLICY` names another file; empty selects the
  scripted controller.

## The brain

`learning/adapters/sea_snake.py` turns what the snake senses into numbers: each
joint's angle, spin and activation, its velocity along and across the body, which
links are touching something, eight wall rays from the head and eight from the
tail, the nearest surface, and the direction and distance to a target. It sees a
target only within ten body radii and with no wall in between, and a flag says
whether it sees one at all. Every length is in units of the snake's radius, so the
brain would carry over to a snake of another size. It makes a decision every six
frames and outputs seven bend requests.

It was trained with PPO (`tools.ppo_neural`, task `sea_snake`) through three
lessons, in order:

1. **Open arena:** swim to a still target.
2. **Moving target:** chase one that moves.
3. **Maze:** reach a target through a wall with gaps. Progress is shaped by the
   path through the gap, not the straight line, so the snake is rewarded for
   finding the way round.

A snake that wriggles badly is a failure however well it reaches the target, so
the reward and the deployment gate both score the wriggle
(`learning/sea_snake_wriggle.py`): curvature, a head-to-tail travelling wave,
rhythm, swimming progress and body spread. A tail that only flaps does not pass.

The shipped brain (run2, iteration 1999) passed the deployment gate:

| Lesson | Reached the target (gate) | Wriggle (gate 0.65) |
|---|---|---|
| Open arena | 100% (80%) | 0.85 |
| Moving target | 100% (80%) | 0.80 |
| Maze | 75% (50%) | 0.97 |

The scripted snake reaches the maze target only 8% of the time. The thresholds are
first guesses, and the training does not try to beat the scripted controller; it
only has to reach the target and wriggle like a snake.

Train it from the training manager (`./run-trainer.sh`), choosing the `sea_snake`
task. The trainer needs Python 3.11 or later. Copy a run's `best.npz` to
`assets/neural/sea_snake.npz` to ship it.

## Speed and solidity between snakes

A snake swims in its own body lengths, so a small one is slow in pixels. The `sea_snakes` trait's `speed` runs the
snake's own clock faster (minis 2.4, teenies 3.5; its physics, wave and brain all hurry together, in steps of no more
than one of its own frames so the brain keeps its beat). Each snake has its own solver, so snakes cannot collide there:
instead the crowd step (`keep_solid` in `creatures/traits/sea_snake.py`) pushes apart any links of different snakes
that come within a padded skin of each other, once a frame. It is a shove, not a jam. A brush at speed can sink in for a
frame; two snakes pressed on a third can still be squeezed deeper than that.

## Death

A dead snake takes 150 frames (2.5 s) to go, in four beats, all in `entities/sea_snake.py`:

1. **Thrash (24 frames).** The corpse keeps its muscles powered with a fading sine of spasms, and the body flares white.
2. **Chain burst (8 links, 6 frames apart, head first).** Each link flashes, throws a ring and sprays sparks in the
   snake's colours, and is no longer drawn; what is left keeps its body width and is cut off square. Each link makes
   its own bang, rising in pitch (`play_enemy_destroy(pitch)`).
3. **Finale (frame 78).** Embers fly wide and three rings and a flash spread out. This is also when the blast goes
   off: the `explodes` trait in `config/enemies.json` (`fuse` 78) must equal `FINALE`. The blast reaches 5 radii,
   deals 2 hits to every enemy it reaches (so snakes can set each other off), shoves things, and hurts the ship and the
   hunter ally like a Belemnite's. It breaks one hit of each wall block.
4. **Embers** drift and cool until the 150 frames end.
