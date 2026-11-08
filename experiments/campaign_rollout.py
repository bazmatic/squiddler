"""Flies the Campaign Hunter through one level headless, on the real ship, and records what PPO learns from.

Unlike the hunter harness, nothing is parked or switched off: the exit finishes the level, an Ally beacon summons the
allied hunter, and the draining score can run out. The episode is paid roughly as a good player would want: its own
kills, pickups and the beacon pay, wall crashes cost, and leaving through the exit pays most when no enemy is left.
"""
import math
import random
from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np

from experiments import harness
from experiments.harness import FPS
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
from hunter.maze_route import MazeRoutes
from learning.campaign_rewards import CampaignShaping

IDLE_SPEED = 1.0           # a ship slower than this and not thrusting is idling
FOLLOW_DEGREES = 45.0      # the ship's flight follows a compass when its velocity is this close to it
START_TIME = 1000.0        # the episode clock's first reading, in seconds

_INDEX = {name: i for i, name in enumerate(CAMPAIGN_FEATURE_NAMES)}
_EVENTS = ('own_kills', 'pickups', 'beacon', 'wall_crashes', 'following', 'idle', 'exit_progress')


@dataclass(frozen=True)
class CampaignResult:
    level: int
    seed: int
    end: str                  # 'exited', 'scored_out' or 'horizon'
    elapsed: float
    enemies_at_start: int
    own_kills: int
    ally_kills: int           # by the allied hunter
    cleared: bool             # no enemy was left when the episode ended
    pickups: int              # crystals the ship collected
    beacon: bool              # the ship fetched an Ally beacon
    max_no_kill_gap: float    # the longest spell, in episode seconds, with enemies left and no kill by ship or ally


def _following(senses, game):
    """Whether the ship flies within FOLLOW_DEGREES of the enemy compass, or of the exit's once no enemy is left."""
    ship = game.ship
    if math.hypot(ship.vx, ship.vy) < 1e-6:
        return False
    which = 'enemy' if game.roster.alive() else 'exit'
    if senses[_INDEX[f'{which}_compass_present']] != 1:
        return False
    # The compass is relative to the nose; so, once turned the same way, is the velocity.
    bearing = math.atan2(senses[_INDEX[f'{which}_compass_sin']], senses[_INDEX[f'{which}_compass_cos']])
    course = math.atan2(ship.vy, ship.vx) - math.radians(ship.angle)
    off = (course - bearing + math.pi) % math.tau - math.pi
    return abs(math.degrees(off)) < FOLLOW_DEGREES


class _Recorder:
    """Stands between the game and the pilot: one row per decision (the pilot reports its senses through
    on_observed), holding the controls it chose and the events that followed until the next."""

    def __init__(self, pilot, game):
        self.pilot, self.game = pilot, game
        self.senses, self.controls, self.temperatures, self.events = [], [], [], []
        self._observed = False
        self._routes, self._exit_distance = MazeRoutes(), None
        pilot.on_observed(self._observe)

    def _observe(self, senses, last_controls):
        self._observed = True
        self.senses.append(np.asarray(senses, dtype=np.float32))
        controller = getattr(self.pilot, 'controller', None)
        self.temperatures.append(getattr(controller, 'temperature', 1.0))
        self._pay_exit_progress()
        row = dict.fromkeys(_EVENTS, 0)
        row['following'] = int(_following(self.senses[-1], self.game))
        self.events.append(row)

    def _pay_exit_progress(self):
        """Once the exit is open, the last decision is paid for the ground it made toward the exit by route, in maze
        widths: what steering for the exit earns and circling does not."""
        game = self.game
        if not game.exit_opened:
            return
        maze, exit_pos, ship = game.maze, game.maze.exit.get_pos(), game.ship
        origin = (ship.x, ship.y)
        found = self._routes.route(maze, origin, [exit_pos])
        points = [origin] + (list(reversed(found[1])) if found else []) + [tuple(exit_pos)]
        distance = sum(math.dist(a, b) for a, b in zip(points, points[1:])) / max(maze.grid_width * maze.cell_size_x, 1.0)
        if self._exit_distance is not None:
            self.add('exit_progress', self._exit_distance - distance)
        self._exit_distance = distance

    def decide(self, game):
        self._observed = False
        command = self.pilot.decide(game)
        if self._observed:
            self.controls.append((command.turn + 1, int(command.thrust), int(command.fire)))
            ship = game.ship
            self.events[-1]['idle'] = int(not command.thrust and math.hypot(ship.vx, ship.vy) < IDLE_SPEED)
        return command

    def add(self, event, count):
        if count and self.events:
            self.events[-1][event] += count

    def arrays(self, shaping, terminal):
        n = min(len(self.senses), len(self.controls))
        senses = np.asarray(self.senses[:n], dtype=np.float32).reshape(n, len(CAMPAIGN_FEATURE_NAMES))
        controls = np.asarray(self.controls[:n], dtype=np.int8).reshape(n, 3)
        events = np.asarray([[row[e] for e in _EVENTS] for row in self.events[:n]], dtype=np.float64).reshape(n, len(_EVENTS))
        weights = np.array([shaping.own_kill, shaping.pickup, shaping.beacon, shaping.wall_crash,
                            shaping.compass_following, shaping.idle, shaping.exit_progress])
        rewards = events @ weights
        if n:
            rewards[-1] += terminal
        temperatures = np.asarray(self.temperatures[:n], dtype=np.float32)
        return senses, controls, rewards.astype(np.float32), temperatures


def _setup(game, level, seed, now):
    """Start the level as Demo does, on the episode clock, with none of the hunter harness's overrides left over."""
    import config
    from game_handlers.autoplay import AutoplayDirector
    from weapons.catalog import RULESET
    harness._clear_overrides(game)
    # The hunter harness parks the player on its levels: the exit and the draining score are live here.
    game.scoring.__dict__.pop('calculate_current_potential_score', None)
    game.clock = lambda: now[0]
    game.clock_ms = lambda: now[0] * 1000.0
    game.__dict__.pop('last_shot_time', None)        # the last episode's shot, on its clock: the gun starts ready
    # The level as it is played: the game's own rules, weapons, walls and weed.
    game.combat_ruleset, game.weapon_catalog = RULESET, None
    game.curved_walls, game.grows_flora = config.CAVE.enabled, config.FLORA.enabled
    # A director that never ends the level itself: the episode decides when it is over.
    game.autoplay = AutoplayDirector(level, level, math.inf, clock=game.clock)
    game.level, game.state = level, config.STATE_PLAYING
    game.start_level()
    random.seed(seed)
    game.ship.sound_manager = Mock()
    game.player_has_moved = True
    game.ship.game_started = True
    game.autoplay.start_level(len(game.roster.alive()))


def _restore(game):
    game._close_hunter()
    game._close_enemy_hunters()
    game.autoplay = game.ship_pilot = None
    for name in ('clock', 'clock_ms'):
        game.__dict__.pop(name, None)
    harness._clear_overrides(game)


def fly_campaign(level, seed, spec, horizon=180.0, shaping=CampaignShaping(), greedy=False, *, pilot=None,
                 policy=None, trace=None, low_graphics=True):
    """One episode on `level` flown by the campaign policy in `spec` (sampled from `spec.sample_seed`, unless it or
    `greedy` asks for its best choices). It ends at the exit, on scoring out, or after `horizon` seconds.

    Returns the per-decision `senses`, `controls`, `rewards` and `temperatures`, and the episode's `result`; the
    exit's or the score-out's reward lands on the last decision. `pilot` replaces the policy (for tests): anything
    with decide(game) and on_observed(listener). `policy`, a loaded policy, is flown in place of `spec.weights` (a
    replay's start policy, migrated in memory).

    For watching: `trace`, if given, is called each frame, after the game updates, with (seconds, game); and
    `low_graphics=False` keeps the tentacles' drag physics, which the episode does not depend on.

    Greedy wins if either `greedy` or `spec.greedy` says so, and StudentSpec defaults to greedy=True: a training job
    that wants sampled choices must pass a spec built with greedy=False, not just greedy=False here.
    """
    import config
    from hunter.campaign_pilot import CampaignPilot
    from hunter.neural.policy import NeuralPolicy
    if harness._game is None:
        harness._game = harness._make_game()
    game = harness._game
    if pilot is None:
        greedy = greedy or spec.greedy
        # Loaded afresh each episode: a trainer may rewrite the same weights file between iterations.
        if policy is None:
            policy = NeuralPolicy.load(spec.weights, CAMPAIGN_FEATURE_NAMES)
        pilot = CampaignPilot(policy, spec.decision_frames, greedy=greedy,
                              rng=None if greedy else np.random.default_rng(spec.sample_seed))
    now = [START_TIME]
    previous_graphics, previous_ally = config.SETTINGS.game.lowGraphics, config.SETTINGS.game.hunterPilot
    config.SETTINGS.game.lowGraphics = low_graphics  # tentacle physics: nothing the pilot senses reads it
    # A beacon summons the ally with the configured pilot; Jev's reads wall time and calls out, so it is pinned.
    config.SETTINGS.game.hunterPilot = 'neural'
    try:
        _setup(game, level, seed, now)
        recorder = _Recorder(pilot, game)
        if hasattr(pilot, 'reset'):
            pilot.reset()
        game.ship_pilot = recorder
        scoring, handler = game.scoring, game.collision_handler
        enemies_at_start = len(game.roster.alive())
        seen = dict(own=0, pickups=0, walls=0, kills=handler.level_kills)
        # Demo's no-kill skip, measured rather than applied: the clock waits while no enemy is left.
        last_kill, max_gap = now[0], 0.0
        flight, beacon, end, frames = game.hunter_flight, False, 'horizon', 0
        while frames < int(horizon * FPS):
            game.update(1.0)
            now[0] += 1 / FPS
            frames += 1
            recorder.add('own_kills', scoring.enemies_destroyed - seen['own'])
            recorder.add('pickups', scoring.powerup_crystals_collected - seen['pickups'])
            recorder.add('wall_crashes', scoring.wall_collisions - seen['walls'])
            seen.update(own=scoring.enemies_destroyed, pickups=scoring.powerup_crystals_collected,
                        walls=scoring.wall_collisions)
            # The allied hunter arrives only when the ship fetches a beacon.
            if game.hunter_flight is not None and game.hunter_flight is not flight and not beacon:
                beacon = True
                recorder.add('beacon', 1)
            flight = game.hunter_flight
            if handler.level_kills > seen['kills'] or not game.roster.alive():
                last_kill = now[0]
            seen['kills'] = handler.level_kills
            max_gap = max(max_gap, now[0] - last_kill)
            if trace is not None:
                trace(frames / FPS, game)
            if game.exit_explosion_active:
                end = 'exited'
                break
            if game.game_frozen:
                end = 'scored_out'
                break
        cleared = not game.roster.alive()
        terminal = {'exited': shaping.exit + (shaping.cleared_exit if cleared else 0.0),
                    'scored_out': shaping.score_out, 'horizon': 0.0}[end]
        senses, controls, rewards, temperatures = recorder.arrays(shaping, terminal)
        result = CampaignResult(level, seed, end, frames / FPS, enemies_at_start, scoring.enemies_destroyed,
                                handler.level_kills - scoring.enemies_destroyed, cleared,
                                scoring.powerup_crystals_collected, beacon, max_gap)
    finally:
        _restore(game)
        config.SETTINGS.game.lowGraphics = previous_graphics
        config.SETTINGS.game.hunterPilot = previous_ally
    return {'senses': senses, 'controls': controls, 'rewards': rewards, 'temperatures': temperatures,
            'result': result}
