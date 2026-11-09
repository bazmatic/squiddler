"""Neural Jellies hunting plain drifting practice targets in the real cave arenas, under the game's own Encounter,
Jelly body and wall-bounce code. Stage 1: navigation to a moving target. A catch is body contact; the target then
respawns elsewhere and the episode runs on to the horizon, so the Jellies must catch repeatedly. Stage 2: teams of
two to four Jellies on the same scene, with a small shaped reward for closing in from differing bearings (flanking)
and a penalty for Jellies bunched up on each other. Stage 3: no plain target; a training-only ship,
the `training_ship` creature kind (config/enemies.json: the allied hunter ship with a slower top speed and thrust, half
the fire rate and half-strength shots, a mortal three-point hull, flown by the frozen neural hunter pilot), hunts the
Jellies with the game's own gun, shots and hit rules while they hunt it. It is a friend-allegiance creature on the
roster, so Encounter.target_entity_for gives it to every Jelly. A Jelly touching it costs it a hit point through the
game's allied-hunter contact rule (CollisionHandler.handle_hunter_contacts, with the hull's immunity after each hit);
that is rewarded, a Jelly dying is penalised, and the episode ends when the ship dies, every Jelly dies, or at the
horizon. The stage-2 flank/bunch shaping carries on, around the ship."""
from collections import Counter, deque
from dataclasses import replace
import hashlib
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from learning.environment import Snapshot, Transition
from learning import jelly_rewards, jelly_curriculum
from learning.adapters.jelly import SPEC, CRITIC_SIZE, observe, critic, intention, teacher, enabled, facing, route_guide, load_transfer_policy
from learning.adapters.crawler_scenario import TRAIN_LEVELS as CRAWLER_TRAIN_LEVELS, HOLDOUT_LEVELS, SilentSound
from creatures.jelly_team import DECISION_FRAMES
from creatures.traits.rooted import wall_between

# Level 27 spawns targets and Jellies inside small pockets that are hard to reach, so it is no use for learning to hunt.
UNSUITABLE_TRAIN_LEVELS = (27,)
# More campaign arenas than the crawlers train on, so the Jelly learns caves in general and not these few layouts.
EXTRA_TRAIN_LEVELS = (7, 9, 10, 22, 25, 28, 29)
TRAIN_LEVELS = tuple(n for n in CRAWLER_TRAIN_LEVELS + EXTRA_TRAIN_LEVELS if n not in UNSUITABLE_TRAIN_LEVELS)

STAGES = (1, 2, 3)             # stage 2 (flanking) shares stage 1's scene and adds the flank/bunch shaping below;
                               # stage 3 swaps the plain target for the hunting training ship
TARGET_COUNT = 1               # one plain target: every Jelly in the team hunts the same one
TARGET_START_DISTANCE = 250.   # px: targets start at least this far from every Jelly (else the farthest open cell)
RESPAWN_DISTANCE = 300.        # px: a caught target reappears at least this far from every Jelly
JELLY_SPACING = 60.            # px between Jelly starts
CATCH_REWARD = jelly_rewards.DEFAULTS['catch_reward'] # to the Jelly that touches the target
TEAM_CATCH_SHARE = jelly_rewards.DEFAULTS['team_catch_share'] # every other living Jelly gets this share of CATCH_REWARD for a teammate's catch
PROGRESS_REWARD = jelly_rewards.DEFAULTS['progress_reward'] # per pixel a Jelly closes on its target over a decision (about .03 at full speed)
STUCK_WINDOW = 60              # frames (one second) over which displacement is measured
STUCK_DISTANCE = 10.           # px: moving less than this over the window, accelerating throughout it, is stuck
STUCK_ACCEL = .1               # an intent's accel above this counts as accelerating
STUCK_PENALTY = jelly_rewards.DEFAULTS['stuck_penalty'] # per decision spent stuck (.5 per second)
MAX_TEAM = 8                   # the critic's member slots
# Stage 2 shaping, kept small beside CATCH_REWARD: a perfect pincer held for a whole minute earns each Jelly 3, under
# a third of one catch, so catches dominate while two Jellies converging from opposite sides beats one rushing alone.
FLANK_REWARD = jelly_rewards.DEFAULTS['flank_reward'] # per decision to every living Jelly, times flank_spread of the Jellies in range
FLANK_RANGE = jelly_rewards.DEFAULTS['flank_range'] # px: Jellies this close to the target count towards its flank_spread
BUNCH_DISTANCE = jelly_rewards.DEFAULTS['bunch_distance'] # px between Jelly centres: closer than this is bunched (about two body widths apart)
BUNCH_PENALTY = jelly_rewards.DEFAULTS['bunch_penalty'] # per decision, to each Jelly of every pair closer than BUNCH_DISTANCE
STAGE2_TEAM_SIZES = (2, 3, 4)
RETENTION_SHARE = 4            # one stage-2 training case in this many is a stage-1 solo case, so navigation is kept
# Stage 3, the hunting ship. A Jelly's death costs more than one touch on the ship earns, so trading a life for a single
# hit (a suicide rush) loses (4 - 5 = -1), while the ship's whole life is worth far more than one death: its three hull
# points (training_ship energy.health) are 3 x 4 plus the kill (10), 22 to the touchers, before the team shares.
STAGE3_TEAM_SIZES = (2, 3, 4)  # teams against the ship; a team of one is allowed by reset but not in the case mix
SHIP_START_DISTANCE = 300.     # px: the ship starts at least this far from every Jelly (else the farthest open cell)
SHIP_KIND = 'training_ship'    # its stats live in config/enemies.json; its pilot is the frozen neural hunter below
SHIP_PILOT = Path(__file__).resolve().parents[2] / 'assets/neural/hunter.npz'
SHIP_HIT_REWARD = jelly_rewards.DEFAULTS['ship_hit_reward'] # to the Jelly whose touch costs the ship a hit point (none during hull immunity)
SHIP_KILL_REWARD = jelly_rewards.DEFAULTS['ship_kill_reward'] # on top, to the Jelly whose touch kills the ship (worth a stage 1-2 catch)
TEAM_SHIP_SHARE = jelly_rewards.DEFAULTS['team_ship_share'] # every other living Jelly gets this share of a teammate's hit and kill rewards
DEATH_PENALTY = jelly_rewards.DEFAULTS['death_penalty'] # to a Jelly the ship kills, on the decision it dies
TEAM_DEATH_SHARE = jelly_rewards.DEFAULTS['team_death_share'] # every surviving Jelly loses this share of DEATH_PENALTY for each teammate lost


def flank_spread(positions, target):
    """How evenly the Jellies' bearings from the target spread around it, in [0, 1]: one minus the length of the mean
    unit bearing vector. 0 when all share one bearing (or fewer than two have one), 1 for an even ring or a pincer."""
    bearings = [math.atan2(y-target[1], x-target[0]) for x, y in positions if (x, y) != tuple(target)]
    if len(bearings) < 2:
        return 0.
    mean = math.hypot(sum(map(math.cos, bearings)), sum(map(math.sin, bearings)))/len(bearings)
    return min(1., max(0., 1.-mean))


class JellyEnvironment:
    spec = SPEC
    adapter = sys.modules[__name__]   # the collector asks the adapter for enabled() and teacher_action()

    def __init__(self, horizon=60.0, arenas=None, scripted=False, rewards=None, lessons=None):
        self.rewards = dict(jelly_rewards.DEFAULTS, **(rewards or {}))
        self.horizon, self.arenas, self.scripted = horizon, arenas, scripted
        if self.rewards['boredom_window'] <= 0 or self.rewards['boredom_distance'] <= 0 or self.rewards['boredom_penalty'] < 0:
            raise ValueError('boredom needs a positive window/distance and a nonnegative penalty')
        self.lessons = lessons or jelly_curriculum.metadata()
        self.training_limits = False
        self.world = None
        self.render_frames = None

    @property
    def teacher_weight(self):
        """DAgger: while the Jelly flies its own episodes, the teacher is asked what it would do in every state it
        reaches, and the policy is pulled toward those answers with this weight (0: no teacher)."""
        return self.rewards['dagger_weight']

    def enable_training_limits(self):
        """Training rollouts may end an episode early (no_catch_timeout); evaluation and replay never do."""
        self.training_limits = True

    def stale(self):
        """No catch for no_catch_timeout seconds: the rest of the episode would only repeat a failure."""
        limit = self.rewards['no_catch_timeout']
        return self.training_limits and limit > 0 and self.frames - self.last_catch_frame >= limit*60

    def reset(self, seed, level=11, team_size=1, stage=1, lesson='cave'):
        import pygame
        import config
        import creatures
        import level_config
        from creatures.encounter import Encounter
        from creatures.jelly_team import get_jelly_team
        from entities.command_recorder import CommandRecorder
        from game_handlers.enemy_roster import EnemyRoster
        from game_handlers.collision_handler import CollisionHandler
        from scoring.system import ScoringSystem
        from maze.generator import Maze
        from physics.obstacle_world import ObstacleWorld
        from level_layout import parse_layout
        from maze.config import MazeComplexity
        if lesson not in jelly_curriculum.LESSONS:
            raise ValueError(f'unknown navigation lesson: {lesson}')
        self.lesson = lesson
        self.episode_horizon = min(self.horizon, self.lessons['episode_seconds'].get(lesson, self.horizon))
        if lesson != 'cave' and (stage != 1 or team_size != 1):
            raise ValueError('introductory navigation lessons are solo pursuit')
        if stage not in STAGES:
            raise ValueError(f'unknown jelly stage: {stage}')
        if type(team_size) is not int or not 1 <= team_size <= MAX_TEAM:
            raise ValueError(f'team size must be 1..{MAX_TEAM}')
        self.close()
        random.seed(seed)
        self.rng = rng = random.Random(seed)
        pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1352, 878))
        self._screen_settings = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT = 1352, 878
        arena = self.arenas.get(str(level)) if self.arenas else level_config.load_level_config(level)
        if lesson != 'cave':
            arena = self.lessons['arenas'][lesson]
        grid_size = arena['maze']['grid_size']
        complexity = MazeComplexity[arena['maze']['complexity'].upper()]
        layout = parse_layout(arena['layout'], grid_size) if 'layout' in arena else None
        if layout is not None:
            layout = replace(layout, enemies=(), beacon=None)
        maze = Maze(level, complexity, grid_size, layout=layout, seed=seed, curved=True)
        obstacles = ObstacleWorld(maze, layout.gates) if layout is not None and layout.gates else None
        maze.gate_segments = obstacles.segments() if obstacles else []
        from learning.adapters.practice_targets import TARGET_RADIUS
        cells = [maze.position_calculator.grid_center_to_screen(c, r)
                 for r in range(1, maze.grid_height-1) for c in range(1, maze.grid_width-1) if not maze.grid[r][c]]
        rng.shuffle(cells)
        self.cells = [p for p in cells if maze.is_open(p, TARGET_RADIUS+2)]
        from hunter.maze_route import MazeRoutes
        from creatures.wallcrawler_team import capsule_hits
        self._spawn_routes = MazeRoutes()
        self.cells = [p for p in self.cells if not any(
            w.active and capsule_hits(p, p, TARGET_RADIUS+2, w.start, w.end) for w in maze.gate_segments)]
        # Prefer the largest usable connected region, not a tiny sealed spawn pocket.
        remaining, largest = set(range(len(self.cells))), set()
        while remaining:
            first = min(remaining)
            component = set(self._spawn_routes.reachable_targets(maze, self.cells[first], self.cells))
            if len(component) > len(largest):
                largest = component
            remaining -= component
        self.cells = [p for i, p in enumerate(self.cells) if i in largest]
        starts = [p for p in self.cells if maze.is_open(p, 11)]
        scoring = ScoringSystem()
        scoring.power_drain_enabled = False
        handler = CollisionHandler(SilentSound(), scoring, CommandRecorder())
        handler.drop_pickup = lambda *a, **k: None   # training never generates pickups
        self.world = SimpleNamespace(ship=None, hunter=None, maze=maze, projectiles=[], roster=EnemyRoster(),
                                     scoring=scoring, sound_manager=SilentSound(),
                                     command_recorder=handler.command_recorder, collision_handler=handler,
                                     powerup_crystals=[], jev_beacon_allowed=lambda: False, combat_ruleset='legacy',
                                     additional_targets=[], jelly_policy=None, enemy_policy=ship_pilot,
                                     _hunter_sees_player=lambda: False)
        chosen = []
        for i in range(team_size):
            pos = next((p for p in starts if all(math.dist(p, q) >= JELLY_SPACING for q in chosen)), None)
            if pos is None:
                raise ValueError('arena has insufficient separated jelly starts')
            chosen.append(pos)
            c = creatures.build('neural_jelly', pos, level)
            c.agent_id = i
            self.world.roster.add(c)
        self.enc = Encounter(self.world.roster, self.world)
        self.enc.ready()
        self.team = get_jelly_team(self.enc)
        self.team.external = not self.scripted   # scripted: no policy, so the trait chases like the scripted Jelly
        self.team._next_id = team_size
        from learning.adapters.practice_targets import PracticeTarget
        self.targets = []    # the plain drifting targets of stages 1-2
        self.ship = None
        if stage == 3:
            # A friend-allegiance roster creature: target_entity_for gives it to every Jelly, its neural pilot hunts
            # the enemy creatures on the roster, and the Encounter steps it like any other creature.
            self.ship = creatures.build(SHIP_KIND, self.free_cell(SHIP_START_DISTANCE), level)
            self.world.roster.add(self.ship)
            self.enc.ready()
        else:
            for _ in range(TARGET_COUNT):
                target = PracticeTarget(self.free_cell(TARGET_START_DISTANCE), 0., 0., firing_enabled=False)
                self.launch(target)
                self.targets.append(target)
                self.world.additional_targets.append(target)
        self.obstacles = obstacles
        self.level, self.seed, self.team_size, self.stage = level, seed, team_size, stage
        self.agent_ids = tuple(range(team_size))
        self.frames = 0
        self.last_catch_frame = 0
        self.agent_catches = Counter()
        self.agent_returns = Counter()
        self.agent_stuck_frames = Counter()
        self.spread_frames = 0.      # flank_spread times frames, over decisions with two or more Jellies in range
        self.flank_frames = 0
        self.bunched_frames = 0      # Jelly-frames within the fixed 40px benchmark distance of another Jelly
        self.generation = Counter()   # how many times each target has respawned
        self.agent_ship_hits = Counter()
        self.living = set(self.agent_ids)   # Jellies alive after the last frame, so deaths are counted once
        self.dead = []                      # agent ids in the order they died
        self.ship_killed_frame = None
        self.contact_frame = None           # the frame of the first catch (stages 1-2) or ship hit (stage 3)
        self.finished = False
        self.trails = {c.agent_id: deque([c.get_pos()], maxlen=STUCK_WINDOW//DECISION_FRAMES+1)
                       for c in self.team.members()}
        self.pushes = {c.agent_id: deque(maxlen=STUCK_WINDOW//DECISION_FRAMES) for c in self.team.members()}
        self.idle_frames = Counter()
        self.idle_trails = {c.agent_id: deque([(0, c.get_pos())]) for c in self.team.members()}
        self.action_trace = []
        return self.snapshot()

    def reachable_cells(self, origin):
        """The spawn cells connected to `origin` by open cells. A body squeezed against a curved cave wall can sit in
        a grid cell whose centre is walled off from the rest of the arena, so if its own cell links nowhere, the
        neighbouring cells it can see are tried instead."""
        maze, routes = self.world.maze, self._spawn_routes
        found = routes.reachable_targets(maze, origin, self.cells)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            if found:
                break
            neighbour = (origin[0] + dx*maze.cell_size_x, origin[1] + dy*maze.cell_size_y)
            if not wall_between(origin, neighbour, maze):
                found = routes.reachable_targets(maze, neighbour, self.cells)
        return [self.cells[i] for i in found]

    def free_cell(self, distance, strict=True):
        """A random open cell at least `distance` from every living Jelly, else the one farthest from them. A teacher
        cannot demonstrate pursuit across genuinely disconnected pockets, so `strict` placement (an episode's start)
        rejects an arena with no reachable cell; a respawn mid-episode never ends the run for want of one."""
        jellies = [c.get_pos() for c in self.team.members()]
        def clearance(p):
            return min((math.dist(p, q) for q in jellies), default=float('inf'))
        cells = self.reachable_cells(jellies[0])
        if not cells:
            if strict:
                raise ValueError('arena has no reachable quarry spawn')
            cells = self.cells
        for p in self.rng.sample(cells, len(cells)):
            if clearance(p) >= distance:
                return p
        return max(cells, key=clearance)

    def launch(self, target):
        from learning.adapters.practice_targets import TARGET_SPEED_MIN, TARGET_SPEED_MAX
        heading, speed = self.rng.uniform(0, 2*math.pi), self.rng.uniform(TARGET_SPEED_MIN, TARGET_SPEED_MAX)
        target.vx, target.vy = math.cos(heading)*speed, math.sin(heading)*speed

    def respawn(self, target):
        target.x, target.y = self.free_cell(RESPAWN_DISTANCE, strict=False)
        self.launch(target)
        self.generation[id(target)] += 1

    def pursuit(self, c):
        """The Jelly's current target, its respawn generation and the distance to it: along the open route when there
        is one, so only getting nearer by the way round is progress (pushing at a wall that blocks the straight
        line is not)."""
        t = self.enc.target_entity_for(c)
        if t is None:
            return None
        distance = route_guide(c, self.enc, t.get_pos()).distance
        return id(t), self.generation[id(t)], math.dist(c.get_pos(), t.get_pos()) if distance is None else distance

    def snapshot(self):
        batch = {c.agent_id: observe(self.team, c) for c in self.team.members()}
        return Snapshot({i: v[0] for i, v in batch.items()}, {i: v[1] for i, v in batch.items()}, critic(self.team),
                        {i: critic(self.team, i) for i in batch})

    def terminal(self):
        """Every Jelly is dead, or (stage 3) the ship is."""
        return not self.team.members() or (self.ship is not None and not self.ship.active)

    def ended(self):
        return self.finished or self.frames >= self.episode_horizon*60 or self.terminal() or self.stale()

    def catch(self, events):
        """Body contact with a target is a catch: the toucher counts one, each other living Jelly a share of one."""
        caught = 0
        for target in self.targets:
            if not target.active:
                continue
            members = self.team.members()
            toucher = next((c for c in members
                            if math.dist(c.get_pos(), target.get_pos()) <= c.radius + target.radius), None)
            if toucher is None:
                continue
            caught += 1
            self.agent_catches[toucher.agent_id] += 1
            self.last_catch_frame = self.frames
            self.touched()
            for c in members:
                events[c.agent_id]['catches'] += 1. if c is toucher else self.rewards['team_catch_share']
            self.respawn(target)
        return caught

    def quarry(self):
        """What the Jellies hunt: the plain targets (stages 1-2) or the living ship (stage 3)."""
        return self.targets + ([self.ship] if self.ship is not None and self.ship.active else [])

    def contacts(self, events):
        """Stage 3: the game's allied-hunter contact rule, one Jelly at a time so the toucher is known. A touching
        Jelly bounces off and, unless the hull's immunity from the last hit still runs, costs the ship a hit point:
        the toucher counts a hit (and a kill on the last hull point), its teammates a share of each."""
        ship = self.ship
        for c in self.team.members():
            if not ship.active:
                break
            health = ship.health
            self.world.collision_handler.handle_hunter_contacts(ship, None, [c])
            if ship.health >= health:
                continue
            self.agent_ship_hits[c.agent_id] += 1
            self.touched()
            if not ship.active:
                self.ship_killed_frame = self.frames+1
            for o in self.team.members():
                credit = 1. if o is c else self.rewards['team_ship_share']
                events[o.agent_id]['ship_hits'] += credit
                if not ship.active:
                    events[o.agent_id]['ship_kills'] += credit

    def touched(self):
        """Note the first contact with the quarry (a catch or a ship hit), counted on the frame being stepped."""
        if self.contact_frame is None:
            self.contact_frame = self.frames+1

    def fly_shots(self):
        """Advance the shots in flight with the game's rules: walls and gates stop them, and Encounter.shoot hands a
        friendly shot to the first enemy creature it reaches, whose TakesHits does the damage."""
        live = []
        for shot in self.world.projectiles:
            shot.update(1)
            if not shot.active:
                continue
            if shot.check_wall_collision(self.world.maze.walls, self.world.maze.spatial_grid):
                shot.active = False
            elif self.obstacles and self.obstacles.bullet_hit(shot):
                shot.active = False
            elif self.enc.shoot(shot):
                shot.active = False
            if shot.active:
                live.append(shot)
        self.world.projectiles = live

    def deaths(self, events):
        """Jellies that died since the last frame: at stage 3 each counts a death, and every survivor a share of one
        for each."""
        living = {c.agent_id for c in self.team.members()}
        died = sorted(self.living - living)
        self.living = living
        if not died:
            return
        self.dead += died
        if self.stage == 3:
            for i in died:
                events[i]['deaths'] += 1
            for i in living:
                events[i]['deaths'] += self.rewards['team_death_share']*len(died)

    def step(self, actions):
        if self.ended():
            if self.frames or self.finished or not self.terminal():
                raise RuntimeError('step after episode ended')
            # Over before the first decision (the ship or every Jelly already dead): end the episode without crashing.
            self.finished = True
            rewards = {i: 0. for i in self.agent_ids}
            return Transition(self.snapshot(), 0., True, False, self.events(), agent_rewards=rewards)
        members = self.team.members()
        if self.scripted:
            accelerating = {c.agent_id: True for c in members}   # the scripted Drift always pushes
        else:
            faced = {c.agent_id: facing(c) for c in members}
            intents = {i: intention(a, faced[i]) for i, a in actions.items()}
            self.team.commit(intents)
            accelerating = {i: intent.accel > STUCK_ACCEL for i, intent in intents.items()}
        self.action_trace.append({str(i): [float(v) for v in a] for i, a in actions.items()})
        before = {c.agent_id: self.pursuit(c) for c in members}
        contacts_before = {c.agent_id: self.agent_catches[c.agent_id] + self.agent_ship_hits[c.agent_id] for c in members}
        # Per Jelly, the counts each reward term pays on (learning.jelly_rewards); paid together once the decision ends.
        events = {i: jelly_rewards.TABLE.row() for i in self.agent_ids}
        frames = 0
        for _ in range(DECISION_FRAMES):
            for target in self.targets:
                target.step(self.world.maze)
            self.enc.step(1)   # the ship, a roster creature, flies and fires here
            if self.ship is not None:
                self.contacts(events)
            if self.world.projectiles:
                self.fly_shots()
            self.world.roster.update_dying(1)
            if self.obstacles:
                self.obstacles.step(1/60, self.world.roster.alive())
                self.world.maze.gate_segments = self.obstacles.segments()
            self.catch(events)
            self.deaths(events)
            self.frames += 1
            frames += 1
            if self.render_frames is not None:
                self.render_frames(self)
            if self.ended():
                break
        for c in self.team.members():
            i = c.agent_id
            now, then = self.pursuit(c), before.get(i)
            if now is not None and then is not None and now[:2] == then[:2]:
                events[i]['closing_pixels'] += then[2]-now[2]
            trail, pushes = self.trails[i], self.pushes[i]
            trail.append(c.get_pos())
            pushes.append(accelerating.get(i, False))
            if (len(pushes) == pushes.maxlen and all(pushes)
                    and math.dist(trail[0], trail[-1]) < STUCK_DISTANCE):
                events[i]['stuck_decisions'] += 1
                self.agent_stuck_frames[i] += frames
        window = self.rewards['boredom_window'] * 60
        for c in self.team.members():
            i = c.agent_id
            trail = self.idle_trails[i]
            if self.agent_catches[i] + self.agent_ship_hits[i] > contacts_before.get(i, 0):
                trail.clear()   # Useful contact earns a fresh grace period.
            trail.append((self.frames, c.get_pos()))
            while len(trail) > 1 and trail[1][0] <= self.frames - window:
                trail.popleft()
            if (self.frames - trail[0][0] >= window
                    and max(math.dist(trail[0][1], pos) for _, pos in trail) < self.rewards['boredom_distance']):
                events[i]['bored_decisions'] += frames / DECISION_FRAMES
                self.idle_frames[i] += frames
        self.formation(events, frames)
        paid = jelly_rewards.TABLE.pay([events[i] for i in self.agent_ids], self.rewards)
        rewards = {i: float(r) for i, r in zip(self.agent_ids, paid)}
        self.agent_returns.update(rewards)
        terminated = self.terminal()
        truncated = (self.frames >= self.episode_horizon*60 or self.stale()) and not terminated
        return Transition(self.snapshot(), sum(rewards.values()), terminated, truncated, self.events(),
                          agent_rewards=rewards)

    def formation(self, events, frames):
        """Benchmark geometry stays fixed across runs; paid shaping uses the configured range and distance.

        The same fixed 200/40 px metrics feed charts and scripted promotion baselines, even in a reward experiment.
        """
        members = self.team.members()
        shaped = self.stage >= 2
        for target in self.quarry():
            if not target.active:
                continue
            positions = [(c.get_pos(), math.dist(c.get_pos(), target.get_pos())) for c in members]
            near = [pos for pos, distance in positions if distance <= FLANK_RANGE]
            if len(near) >= 2:
                self.spread_frames += flank_spread(near, target.get_pos())*frames
                self.flank_frames += frames
            if shaped:
                paid_near = [pos for pos, distance in positions if distance <= self.rewards['flank_range']]
                spread = flank_spread(paid_near, target.get_pos())
                for c in members:
                    events[c.agent_id]['flank_spread'] += spread
        bunched = set()
        for k, a in enumerate(members):
            for b in members[k+1:]:
                distance = math.dist(a.get_pos(), b.get_pos())
                if distance < BUNCH_DISTANCE:
                    bunched.update((a.agent_id, b.agent_id))
                if shaped and distance < self.rewards['bunch_distance']:
                    events[a.agent_id]['bunched_pairs'] += 1
                    events[b.agent_id]['bunched_pairs'] += 1
        self.bunched_frames += len(bunched)*frames

    def events(self):
        result = {'catches': sum(self.agent_catches.values()), 'agent_catches': dict(self.agent_catches),
                'seconds': self.frames/60, 'stuck_seconds': sum(self.agent_stuck_frames.values())/60,
                'agent_stuck_seconds': {i: f/60 for i, f in self.agent_stuck_frames.items()},
                'alive': len(self.team.members()), 'stage': self.stage,
                'angular_spread': self.spread_frames/self.flank_frames if self.flank_frames else 0.,
                'flank_seconds': self.flank_frames/60, 'bunched_seconds': self.bunched_frames/60,
                'agent_returns': {i: self.agent_returns[i] for i in self.agent_ids},
                'ship_hits': sum(self.agent_ship_hits.values()), 'agent_ship_hits': dict(self.agent_ship_hits),
                'ship_kills': int(self.ship_killed_frame is not None), 'jelly_deaths': len(self.dead),
                'survival_fraction': len(self.team.members())/self.team_size,
                'time_to_ship_kill': self.ship_killed_frame/60 if self.ship_killed_frame is not None else None,
                'time_to_contact': self.contact_frame/60 if self.contact_frame is not None else None}
        result['idle_seconds'] = sum(self.idle_frames.values()) / 60
        result['navigation_lesson'] = jelly_curriculum.LESSONS.index(self.lesson)
        return result

    def close(self):
        if hasattr(self, '_screen_settings'):
            import config
            config.SCREEN_WIDTH, config.SCREEN_HEIGHT = self._screen_settings
            del self._screen_settings
        if self.world is not None:
            self.enc.close()
            self.world = None


def teacher_action(env, agent_id, sampled):
    """The scripted Jelly's heading and accel labels (learning.adapters.jelly.teacher) for imitation bootstrap."""
    c = next(c for c in env.team.members() if c.agent_id == agent_id)
    return teacher(c, env.enc)


def environment(horizon=60.0, arenas=None, scripted=False, rewards=None, lessons=None, **ignored):
    """The trainer also passes `opponent` and `muted`; Jellies have no frozen opponent and no message channel."""
    return JellyEnvironment(horizon=horizon, arenas=arenas, scripted=scripted, rewards=rewards, lessons=lessons)


def environment_options(manifest):
    """Run-specific environment arguments for rollout workers and evaluation."""
    return dict(rewards=jelly_rewards.from_manifest(manifest), lessons=manifest['navigation'])


# Scenario policy belongs to this adapter, not to the optimiser or run manager.
INITIAL_STAGE = 1
IMITATION_HEADS = ('heading', 'accel')
REWARDS = {'catch': CATCH_REWARD, 'team_catch_share': TEAM_CATCH_SHARE, 'progress_per_pixel': PROGRESS_REWARD,
           'stuck_per_decision': -STUCK_PENALTY, 'stage2_flank_per_decision': FLANK_REWARD,
           'stage2_bunch_per_pair_decision': -BUNCH_PENALTY, 'stage3_ship_hit': SHIP_HIT_REWARD,
           'stage3_ship_kill': SHIP_KILL_REWARD, 'stage3_team_ship_share': TEAM_SHIP_SHARE,
           'stage3_death': -DEATH_PENALTY, 'stage3_team_death_share': TEAM_DEATH_SHARE}


def metadata(args):
    import dataclasses
    import config
    import enemy_kinds
    import level_config
    from creatures.jelly_team import MAX_SPEED, ACCEL_RATE, TURN_RATE
    from learning.adapters.practice_targets import TARGET_RADIUS, TARGET_SPEED_MIN, TARGET_SPEED_MAX
    global run_horizon
    run_horizon = args.horizon   # learning.run calls this in the process that later calls promote
    opponent = Path(getattr(args, 'opponent', '') or '')
    configured = jelly_rewards.from_args(args)
    rewards = {field: configured[name] * sign for name, (block, field, sign) in jelly_rewards.MANIFEST_FIELDS.items()
               if block == 'rewards'}
    # Version 6: the navigation curriculum (open pursuit, a barrier, then caves) is the only mode, and the boredom
    # rule is always recorded.
    result = dict(task=SPEC.task, scenario_version=11, policy_spec=SPEC.metadata(), observation_version=SPEC.version,
                rewards=rewards, reward_scope='individual', critic_context='focal_agent_first_v1',
                catch={'rule': 'body_contact', 'target_respawns': True, 'respawn_distance': RESPAWN_DISTANCE,
                       'team_share': configured['team_catch_share']},
                stuck={'window_frames': STUCK_WINDOW, 'distance_pixels': STUCK_DISTANCE,
                       'accelerating_above': STUCK_ACCEL, 'accelerating_throughout_window': True, 'scripted_counts_as_accelerating': True},
                practice_targets={'count': TARGET_COUNT, 'radius': TARGET_RADIUS, 'firing': False,
                                  'speed_range': [TARGET_SPEED_MIN, TARGET_SPEED_MAX],
                                  'start_distance': TARGET_START_DISTANCE},
                jelly={'max_speed': MAX_SPEED, 'accel_rate': ACCEL_RATE, 'turn_rate_radians': TURN_RATE,
                       'spacing': JELLY_SPACING, 'creature': dataclasses.asdict(enemy_kinds.kind('neural_jelly'))},
                flank={'reward_per_decision': configured['flank_reward'], 'range': configured['flank_range'], 'spread': '1_minus_mean_bearing_resultant',
                       'bunch_distance': configured['bunch_distance'], 'bunch_penalty_per_pair_decision': configured['bunch_penalty'],
                       'shaped_from_stage': 2},
                curriculum={'stage2_team_sizes': list(STAGE2_TEAM_SIZES), 'stage1_retention_every': RETENTION_SHARE,
                            'stage3_team_sizes': list(STAGE3_TEAM_SIZES),
                            'stage3_retention': 'one_in_retention_every_alternating_stage2_team_and_stage1_solo'},
                promotion={'1': 'training_catches_per_minute_and_stuck_fraction_beat_scripted_holdout_baseline',
                           '2': 'training_angular_spread_and_team_catches_per_minute_beat_scripted_holdout_team_baseline',
                           '3': 'final'},
                selection={'gate': 'positive catches in introductory lessons; scripted cave gate at the cave lesson',
                           'score': 'reward_weighted_catches_per_minute_times_unstuck_share',
                           'context': 'current_navigation_lesson; cave benchmark reported separately',
                           'validation_cases': validation_cases()},
                navigation=jelly_curriculum.metadata(),
                training_limits=dict(no_catch_timeout_seconds=configured['no_catch_timeout'],
                                     start_lesson=int(configured['start_lesson']),
                                     max_stage=int(configured['max_stage']),
                                     dagger_weight=configured['dagger_weight']),
                boredom=dict(window_seconds=configured['boredom_window'], distance_pixels=configured['boredom_distance']),
                ship={'kind': SHIP_KIND, 'creature': dataclasses.asdict(enemy_kinds.kind(SHIP_KIND)),
                      'pilot': 'frozen_neural_hunter_greedy',
                      'pilot_sha256': hashlib.sha256(SHIP_PILOT.read_bytes()).hexdigest() if SHIP_PILOT.is_file() else None,
                      'start_distance': SHIP_START_DISTANCE, 'contact_rule': 'collision_handler.handle_hunter_contacts',
                      'ends_episode_on_death': True},
                physics={name: dataclasses.asdict(getattr(config.SETTINGS, name)) for name in ('maze', 'obstacles')},
                arenas={str(n): level_config.load_level_config(n) for n in TRAIN_LEVELS+HOLDOUT_LEVELS},
                opponent_sha256=hashlib.sha256(opponent.read_bytes()).hexdigest() if opponent.is_file() else None,
                decision_frames=DECISION_FRAMES, horizon=args.horizon,
                train_levels=list(TRAIN_LEVELS), holdout_levels=list(HOLDOUT_LEVELS))
    return result


def training_cases(iteration, count, stage, bootstrap=False, curriculum=None):
    """Stage 1: solo Jellies. Stage 2: slots cycle through team sizes 2, 3, 4 plus one stage-1 solo case in every
    RETENTION_SHARE, so the navigation learnt in stage 1 keeps being rehearsed. Stage 3: the same slots, with teams of
    2, 3 and 4 against the ship in three of every four cases; the fourth (retention) slot alternates between a stage-2
    flanking case (team sizes cycling 2, 3, 4) and a stage-1 solo case. Over RETENTION_SHARE passes through
    TRAIN_LEVELS the mix is three stage-3 cases to one retention case, split between stage 2 and stage 1."""
    cases = []
    for i in range(count):
        index = iteration*count+i
        level, seed = TRAIN_LEVELS[index % len(TRAIN_LEVELS)], 100+i if bootstrap else 300+index % 100
        # The slot is the level's position plus the pass number, so each pass through TRAIN_LEVELS shifts every level
        # to the next slot; indexing by the case number alone would tie each level to a fixed team size whenever
        # len(TRAIN_LEVELS) shares a factor with RETENTION_SHARE. Over RETENTION_SHARE passes every level meets every
        # slot once, and the retention slot (0) is 1 case in 4 over that span.
        slot = (index % len(TRAIN_LEVELS) + index//len(TRAIN_LEVELS)) % RETENTION_SHARE
        if stage == 3 and slot:
            cases.append(dict(level=level, seed=seed, team_size=STAGE3_TEAM_SIZES[(slot-1) % len(STAGE3_TEAM_SIZES)],
                              stage=3))
        elif stage == 3 and (index//RETENTION_SHARE) % 2 == 0:
            # Retention cases alternate between a stage-2 team and a stage-1 solo case.
            size = STAGE2_TEAM_SIZES[(index//len(TRAIN_LEVELS)) % len(STAGE2_TEAM_SIZES)]
            cases.append(dict(level=level, seed=seed, team_size=size, stage=2))
        elif stage >= 2 and slot:
            cases.append(dict(level=level, seed=seed, team_size=STAGE2_TEAM_SIZES[(slot-1) % len(STAGE2_TEAM_SIZES)],
                              stage=stage))
        else:
            cases.append(dict(level=level, seed=seed, team_size=1, stage=1 if stage >= 2 else stage))
    if curriculum is not None:
        if curriculum['stage'] < 2:
            cases = [dict(c, stage=1, team_size=1) for c in cases]
        return jelly_curriculum.cases(cases, curriculum)
    return cases


def team_sizes(stage):
    return STAGE3_TEAM_SIZES if stage == 3 else STAGE2_TEAM_SIZES if stage == 2 else (1,)


VALIDATION_SEED = 900              # checkpoint selection (selection_score): never one of the final evaluation's seeds
EVALUATION_SEEDS = (1000, 1001)    # the final held-out report and the promotion baselines


def validation_cases():
    """What learning.run scores checkpoints on (selection_score): the real game's cave levels held out of training
    (HOLDOUT_LEVELS 15, 20 and 23, as the game builds them), at every stage, so selection sees flanking and ship
    hunting as well as navigation. Per arena: a solo Jelly on the plain target, a team on the plain target and a team
    against the training ship, the team sizes rotating over the arenas so each size appears once per stage; nine
    episodes. Its seed differs from evaluation_cases', so the final report is not on episodes selection has seen."""
    cases = []
    for stage in STAGES:
        sizes = team_sizes(stage)
        cases += [dict(level=n, seed=VALIDATION_SEED, team_size=sizes[j % len(sizes)], stage=stage)
                  for j, n in enumerate(HOLDOUT_LEVELS)]
    return cases


def evaluation_cases(stage=None):
    """The held-out cases of one stage (every HOLDOUT_LEVELS arena, seed and team size of the stage), or, with no
    stage, as learning.run's final report asks, of all three stages one after another."""
    if stage is None:
        return [c for s in STAGES for c in evaluation_cases(s)]
    return [dict(level=n, seed=s, team_size=k, stage=stage)
            for n in HOLDOUT_LEVELS for s in EVALUATION_SEEDS for k in team_sizes(stage)]


def summarise(rows):
    seconds = sum(r['seconds'] for r in rows)
    jelly_seconds = sum(r['seconds']*r['team_size'] for r in rows)
    catches = sum(r['catches'] for r in rows)
    stuck = sum(r['stuck_seconds'] for r in rows)
    returns = [r.get('return', sum(r.get('agent_returns', {}).values())) for r in rows]
    team = [r for r in rows if r['team_size'] >= 2]
    flank = sum(r.get('flank_seconds', 0.) for r in rows)
    hunted = [r for r in rows if r.get('stage') == 3]   # stage-3 rows: the retention cases have no ship
    kill_times = [r['time_to_ship_kill'] for r in hunted if r.get('time_to_ship_kill') is not None]
    timed = [r for r in rows if 'time_to_contact' in r]
    return dict(episodes=len(rows), catches=catches, seconds=seconds, stuck_seconds=stuck,
                catches_per_minute=catches*60/max(seconds, 1e-9),
                stuck_fraction=stuck/max(jelly_seconds, 1e-9),
                team_catches_per_minute=sum(r['catches'] for r in team)*60/max(sum(r['seconds'] for r in team), 1e-9),
                angular_spread=sum(r.get('angular_spread', 0.)*r.get('flank_seconds', 0.) for r in rows)/flank if flank else 0.,
                flank_seconds=flank, bunched_seconds=sum(r.get('bunched_seconds', 0.) for r in rows),
                bunched_fraction=sum(r.get('bunched_seconds', 0.) for r in rows)/max(jelly_seconds, 1e-9),
                catch_probability=sum(r['catches'] > 0 for r in rows)/max(1, len(rows)),
                # mean seconds to the first catch or ship hit; an episode without one counts its whole length
                time_to_contact=float(np.mean([r['seconds'] if r['time_to_contact'] is None else r['time_to_contact']
                                               for r in timed])) if timed else 0.,
                mean_episode_return=float(np.mean(returns)) if returns else 0.,
                ship_hits=sum(r.get('ship_hits', 0) for r in rows), ship_kills=sum(r.get('ship_kills', 0) for r in rows),
                jelly_deaths=sum(r.get('jelly_deaths', 0) for r in rows),
                # Jelly-weighted share alive at the end of the stage-3 episodes; 1 when there are none
                survival_fraction=(sum(r['survival_fraction']*r['team_size'] for r in hunted)
                                   / sum(r['team_size'] for r in hunted)) if hunted else 1.,
                # mean seconds to kill the ship over the episodes that killed it; 0 when none did (read with ship_kills)
                time_to_ship_kill=float(np.mean(kill_times)) if kill_times else 0.)


# What metrics repeats per stage, as stage<n>_<key>, for the stages present (a row without a stage is stage 1).
STAGE_KEYS = {1: ('episodes', 'catches', 'seconds', 'catches_per_minute', 'stuck_fraction', 'time_to_contact'),
              2: ('episodes', 'catches', 'seconds', 'catches_per_minute', 'stuck_fraction', 'time_to_contact',
                  'angular_spread', 'bunched_fraction'),
              3: ('episodes', 'seconds', 'stuck_fraction', 'time_to_contact', 'ship_hits', 'ship_kills',
                  'jelly_deaths', 'survival_fraction', 'time_to_ship_kill')}


def metrics(rows):
    """Catch rate (catches per minute), time to contact, stuck time, angular spread, Jelly survival and the rest over
    all rows, then the same per stage (stage1_catches_per_minute, stage3_survival_fraction, ...): flat numbers, as
    learning.run writes them into train.log lines."""
    summary = summarise(rows)
    summary['idle_seconds'] = sum(r.get('idle_seconds', 0) for r in rows)
    summary['idle_fraction'] = summary['idle_seconds'] / max(sum(r['seconds'] * r['team_size'] for r in rows), 1e-9)
    summary['navigation_lesson'] = max((r.get('navigation_lesson', 0) for r in rows), default=0)
    for stage, keys in STAGE_KEYS.items():
        part = [r for r in rows if r.get('stage', 1) == stage]
        if part:
            staged = summarise(part)
            summary.update({f'stage{stage}_{k}': staged[k] for k in keys})
    return summary


def chase_rate(summary):
    """Catches per minute over the plain-target (stage 1-2) episodes; stage-3 episodes have no target to catch."""
    seconds = summary.get('stage1_seconds', 0.)+summary.get('stage2_seconds', 0.)
    if not seconds:
        return summary['catches_per_minute']
    return (summary.get('stage1_catches', 0)+summary.get('stage2_catches', 0))*60/seconds


def passes_gate(summary, baseline):
    """The hard gate: on the same held-out cave-arena cases, a strictly higher plain-target catch rate than the
    scripted Jelly, and a strictly lower stuck share (or both never stuck)."""
    stuck, scripted_stuck = summary['stuck_fraction'], baseline['stuck_fraction']
    return (chase_rate(summary) > chase_rate(baseline)
            and (stuck < scripted_stuck or stuck == scripted_stuck == 0.))


def selection_score(summary, baseline=None):
    """learning.run keeps the checkpoint with the highest score on validation_cases(). A hard gate first: -inf unless
    the checkpoint beats the scripted Jelly on those same cases (passes_gate; the baseline is validation_baseline(),
    the scripted Jelly flown on validation_cases at the run's horizon). Passing checkpoints are ordered by what the
    rewards value, in catches per minute over every validation episode: catches, plus ship hits and kills and less
    Jelly deaths at their reward weights relative to a catch, discounted by the share of Jelly time spent stuck."""
    baseline = baseline or validation_baseline()
    if not passes_gate(summary, baseline):
        return -math.inf
    earned = (CATCH_REWARD*summary['catches']+SHIP_HIT_REWARD*summary['ship_hits']
              + SHIP_KILL_REWARD*summary['ship_kills']-DEATH_PENALTY*summary['jelly_deaths'])/CATCH_REWARD
    return earned*60/max(summary['seconds'], 1e-9)*(1-summary['stuck_fraction'])


_baselines = {}
run_horizon = 60.0   # the run's episode horizon (set by metadata), so promote's baseline matches the run's episodes


def scripted_metrics(cases, horizon, **options):
    """The scripted Jelly (Chase + Drift, no policy) flown on `cases` at `horizon`."""
    env = JellyEnvironment(horizon=horizon, scripted=True, **options)
    rows = []
    try:
        for case in cases:
            case = dict(case)
            env.reset(case.pop('seed'), **case)
            while True:
                result = env.step({})
                if result.terminated or result.truncated:
                    break
            rows.append(dict(result.events, **case))
    finally:
        env.close()
    return metrics(rows)


def scripted_baseline(stage=1, horizon=None):
    """The scripted Jelly on one stage's held-out evaluation cases at the run's horizon; cached per (stage, horizon)."""
    horizon = run_horizon if horizon is None else horizon
    key = (stage, horizon)
    if key not in _baselines:
        _baselines[key] = scripted_metrics(evaluation_cases(stage), horizon)
    return _baselines[key]


def validation_baseline(horizon=None):
    """The scripted Jelly on validation_cases() at the run's horizon, what selection_score gates against; cached."""
    horizon = run_horizon if horizon is None else horizon
    key = ('validation', horizon)
    if key not in _baselines:
        _baselines[key] = scripted_metrics(validation_cases(), horizon)
    return _baselines[key]


def promote(stage, summary, baseline=None):
    """Stage 1 -> 2 once catches per minute beat the scripted Jelly's held-out rate and the stuck share is lower.
    Stage 2 -> 3 once the training summary's angular_spread and team_catches_per_minute (team scenes only, the solo
    retention cases left out) both strictly beat the scripted team on the held-out team scenes, evaluation_cases(2).
    The scripted team is the stand-in for stage-1 behaviour in team scenes: every Jelly rushes the target on its own.
    Stage 3 is the last: it stays there."""
    if stage not in (1, 2) or not summary.get('episodes'):
        return stage
    baseline = baseline or scripted_baseline(stage)
    if stage == 1:
        if (summary['catches_per_minute'] > baseline['catches_per_minute']
                and summary['stuck_fraction'] < baseline['stuck_fraction']):
            return 2
        return stage
    if (summary['angular_spread'] > baseline['angular_spread']
            and summary['team_catches_per_minute'] > baseline['team_catches_per_minute']):
        return 3
    return stage


def compare(a, b):
    pairs = list(zip(a['episodes'], b['episodes']))
    values = np.array([x['catches']-y['catches'] for x, y in pairs], float)
    stuck = np.array([x['stuck_seconds']-y['stuck_seconds'] for x, y in pairs], float)
    # Stage 3 (reported; selection_score weighs them): extra hits on the ship and extra Jelly deaths per episode.
    hits = np.array([x.get('ship_hits', 0)-y.get('ship_hits', 0) for x, y in pairs], float)
    deaths = np.array([x.get('jelly_deaths', 0)-y.get('jelly_deaths', 0) for x, y in pairs], float)
    if not len(values):
        return dict(mean_extra_catches=0., lower_95=0., upper_95=0., mean_extra_stuck_seconds=0.,
                    mean_extra_ship_hits=0., mean_extra_jelly_deaths=0.)
    samples = np.random.default_rng(0).choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return dict(mean_extra_catches=float(values.mean()), lower_95=float(np.quantile(samples, .025)),
                upper_95=float(np.quantile(samples, .975)), mean_extra_stuck_seconds=float(stuck.mean()),
                mean_extra_ship_hits=float(hits.mean()), mean_extra_jelly_deaths=float(deaths.mean()))


_ship_pilot = []


def ship_pilot():
    """The training ship's pilot policy: the frozen neural hunter, loaded once per process (None if it is missing)."""
    if not _ship_pilot:
        from hunter.neural.policy import NeuralPolicy
        _ship_pilot.append(NeuralPolicy.load(SHIP_PILOT) if SHIP_PILOT.is_file() else None)
    return _ship_pilot[0]


# Curriculum hooks consumed by learning.run: every Jelly run starts in the first navigation lesson.
CURRICULUM_METRIC = 'navigation_lesson'


def initial_curriculum(args):
    state = jelly_curriculum.initial(args.horizon, environment_options(metadata(args)))
    state['stage'] = min(int(getattr(args, 'jelly_start_lesson', 0)), len(jelly_curriculum.LESSONS)-1)
    return state


advance_curriculum = jelly_curriculum.advance


def curriculum_key(state):
    return jelly_curriculum.LESSONS[state['stage']]


def curriculum_validation_cases(state):
    """Checkpoint selection: solo Jelly on the plain target in the current lesson's arenas, held-out levels. Training
    is solo in every lesson, so team and ship episodes would only dilute the score."""
    return jelly_curriculum.cases([dict(level=n, seed=VALIDATION_SEED+2*j+k, team_size=1, stage=1)
                                   for j, n in enumerate(HOLDOUT_LEVELS) for k in range(2)], state)


def curriculum_evaluation_cases(state):
    return jelly_curriculum.cases([dict(level=n, seed=1000+4*j+k, team_size=1, stage=1)
                                   for j, n in enumerate(HOLDOUT_LEVELS) for k in range(4)], state)


def curriculum_benchmark_cases(state):
    """A separate cave navigation report, never mixed into the easier lesson's score."""
    return evaluation_cases(1) if state['stage'] < 2 else []


def curriculum_selection_score(summary, state):
    """Catches per minute, less idle time; no gate against the scripted Jelly, so the best checkpoint keeps improving
    (a gate nothing passes leaves best.npz at the first checkpoint of the lesson)."""
    return (summary['catches_per_minute'] * (1-summary.get('idle_fraction', 0.))
            * (1-summary.get('stuck_fraction', 0.)) if summary['catches'] > 0 else -math.inf)


def curriculum_promote(stage, summary, state):
    if state['stage'] < 2 or summary.get('navigation_lesson', 0) < 2:
        return stage
    if stage >= state['environment']['rewards']['max_stage']:
        return stage
    key = ('navigation_promotion', stage, state['horizon'], str(state['environment']))
    if key not in _baselines:
        _baselines[key] = scripted_metrics(evaluation_cases(stage), state['horizon'], **state['environment'])
    return promote(stage, summary, _baselines[key])


def teacher_weight(args):
    return args.jelly_dagger_weight
