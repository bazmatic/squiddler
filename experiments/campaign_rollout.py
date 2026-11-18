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
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES, CAMPAIGN_HIT_POINT_CELLS
from hunter.maze_route import MazeRoutes
from hunter.senses import MAX_ENEMIES
from learning.campaign_rewards import TABLE, CampaignShaping
from learning.shooting_rewards import unaimed_shots

PROGRESS_CAP = 0.2         # the most ground one decision can pay or cost, in maze widths
IDLE_SPEED = 1.0           # a ship slower than this and not thrusting is idling
FOLLOW_DEGREES = 45.0      # the ship's flight follows a compass when its velocity is this close to it
REVERSAL_WINDOW = 5        # a reversal this many decisions (about half a second) or fewer after the last one is a shake
START_TIME = 1000.0        # the episode clock's first reading, in seconds

_INDEX = {name: i for i, name in enumerate(CAMPAIGN_FEATURE_NAMES)}


@dataclass(frozen=True)
class CampaignResult:
    level: int
    seed: int
    end: str                  # normal: exited/scored_out/horizon; pursuit drills: contact/visible_start too
    elapsed: float
    enemies_at_start: int
    own_kills: int
    ally_kills: int           # by the allied hunter
    cleared: bool             # no enemy was left when the episode ended
    pickups: int              # crystals the ship collected
    beacon: bool              # the ship fetched an Ally beacon
    max_no_kill_gap: float    # the longest spell, in episode seconds, with enemies left and no kill by ship or ally
    wall_crashes: int = 0     # times the ship struck a wall
    shots: int = 0            # bullets the ship fired
    started_unseen: bool = False
    first_sighting_seconds: float | None = None
    unseen_seconds: float = 0.0
    search_horizon_seconds: float = 180.0
    damaging_hits: int = 0
    unaimed_shots: int = 0
    breach_blocks: int = 0
    wall_impulse: float = 0.0  # summed wall-induced velocity change, in ship top-speed units
    pursuit_lesson: bool = False
    missed_opportunities: float | None = None  # firing-interval equivalents; None when the rule was disabled
    graded_shooting: bool = False


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

    def __init__(self, pilot, game, missed_opportunities=False, graded_shooting=False, accuracy_power=2):
        self.pilot, self.game = pilot, game
        self.senses, self.controls, self.temperatures, self.events = [], [], [], []
        self._observed = False
        self._routes, self._progress = MazeRoutes(), None
        self._turn = None
        self._since_reversal = None       # decisions since the last reversal of the turn direction
        self._was_unseen = False
        self.started_unseen = False
        self.first_sighting_seconds = None
        self.unseen_seconds = 0.0
        self._observed_at = None
        self._count_missed = missed_opportunities
        self._graded_shooting, self._seen_shot_targets = graded_shooting, ()
        self._accuracy_power = accuracy_power
        self._rough_shots, self._raw_shots = [], None
        self._shot_targets, self._shot_interval, self._held_fire = (), 1.0, False
        from learning.campaign_breach import BreachPlanner
        self._breach_planner = BreachPlanner()
        pilot.on_observed(self._observe)

    def _observe(self, senses, last_controls):
        self._observed = True
        self.senses.append(np.asarray(senses, dtype=np.float32))
        controller = getattr(self.pilot, 'controller', None)
        self.temperatures.append(getattr(controller, 'temperature', 1.0))
        self._pay_progress(senses)
        self._observe_search(senses)
        row = TABLE.row()              # the counts each reward term pays on (learning.campaign_rewards)
        row['following'] = int(_following(self.senses[-1], self.game))
        self.events.append(row)
        if self._graded_shooting:
            from learning.campaign_shot_opportunity import observed_targets
            self._seen_shot_targets = observed_targets(self.game, senses, getattr(self.pilot, 'settings', None))
            self._rough_shots.append(0)

    def _pay_progress(self, senses):
        """Pay only the ship's movement toward the preceding goal's frozen position. Both distances use the current
        wall layout, so approaching enemies, destroyed blocks and goal switches cannot manufacture progress."""
        game = self.game
        alive = game.roster.alive()
        target = getattr(self.pilot, 'target_enemy', None)
        if alive:
            kind, goals = 'hunt_progress', [target] if any(target is e for e in alive) else alive
        elif game.exit_opened:
            kind, goals = 'exit_progress', [game.maze.exit]
        else:
            self._progress = None
            return
        origin = tuple(game.ship.get_pos())
        if self._progress is not None:
            previous_kind, goal, target, previous, unseen = self._progress
            if kind == previous_kind and any(goal is candidate for candidate in goals) and origin != previous:
                before = self._route_distance(previous, target)
                after = self._route_distance(origin, target)
                if before is not None and after is not None:
                    width = max(game.maze.grid_width * game.maze.cell_size_x, 1.0)
                    delta = max(-PROGRESS_CAP, min(PROGRESS_CAP, (before - after) / width))
                    self.add(kind, delta)
                    if kind == 'hunt_progress' and unseen:
                        self.add('unseen_progress', delta)
        found = self._routes.cheapest_route(game.maze, origin, [g.get_pos() for g in goals],
                                            CAMPAIGN_HIT_POINT_CELLS)
        self._progress = None
        if found is not None:
            goal = goals[found[0]]
            unseen = not any(senses[_INDEX[f'enemy{i}_present']] > 0 for i in range(MAX_ENEMIES))
            self._progress = (kind, goal, tuple(goal.get_pos()), origin, unseen)

    def _observe_search(self, senses):
        """Contact-time metrics use direct sight, independently of whether a route currently exists."""
        game = self.game
        now = game.clock()
        alive = game.roster.alive()
        visible = any(senses[_INDEX[f'enemy{i}_present']] > 0 for i in range(MAX_ENEMIES))
        if self._observed_at is None:
            self.started_unseen = bool(alive) and not visible
        elif self._was_unseen:
            self.unseen_seconds += max(0.0, now - self._observed_at)
        if visible and self.first_sighting_seconds is None:
            self.first_sighting_seconds = max(0.0, now - START_TIME)
        self._observed_at = now
        self._was_unseen = bool(alive) and not visible

    def _route_distance(self, origin, target):
        found = self._routes.cheapest_route(self.game.maze, origin, [target], CAMPAIGN_HIT_POINT_CELLS)
        if found is None:
            return None
        _, route, extra = found
        points = [origin] + list(reversed(route)) + [target]
        return sum(math.dist(a, b) for a, b in zip(points, points[1:])) + extra

    def decide(self, game):
        self._observed = False
        command = self.pilot.decide(game)
        self._held_fire = command.fire
        if self._observed:
            if self._count_missed:
                from learning.campaign_shot_opportunity import opportunity, rough_opportunity
                check = rough_opportunity if self._graded_shooting else opportunity
                self._shot_targets, self._shot_interval = check(game, self.senses[-1],
                                                                     getattr(self.pilot, 'settings', None),
                                                                     turn=command.turn)
            self.controls.append((command.turn + 1, int(command.thrust), int(command.fire)))
            ship = game.ship
            self.events[-1]['thrusting'] = int(command.thrust)
            if self._since_reversal is not None:
                self._since_reversal += 1
            reversed_ = command.turn != 0 and self._turn is not None and command.turn != self._turn
            self.events[-1]['turn_changes'] = int(reversed_ and self._since_reversal is not None
                                                 and self._since_reversal <= REVERSAL_WINDOW)
            if reversed_:
                self._since_reversal = 0
            if command.turn != 0:
                self._turn = command.turn
            self.events[-1]['idle'] = int(not command.thrust and math.hypot(ship.vx, ship.vy) < IDLE_SPEED)
        return command

    def count_missed_opportunity(self, seconds):
        """Prorate the held decision by actual elapsed time, including a truncated final decision."""
        if not self._held_fire and any(target.active for target in self._shot_targets):
            self.add('missed_opportunities', seconds / self._shot_interval)

    def fired(self, projectiles):
        if self._graded_shooting and self.events:
            from learning.campaign_shot_opportunity import fired_feedback
            feedback = fired_feedback(self.game, self._seen_shot_targets, projectiles,
                                       getattr(self.pilot, 'settings', None))
            self._rough_shots[-1] += int(feedback.rough)
            self.add('accurate_shots', feedback.accuracy ** self._accuracy_power)

    def add(self, event, count):
        if count and self.events:
            self.events[-1][event] += count

    def block_destroying(self, cell, source):
        if source == 'player':
            cells = self._breach_planner.blocking_cells(
                self.game.maze, self.game.ship.get_pos(), [e.get_pos() for e in self.game.roster.alive()])
            if cell in cells:
                self.add('breach_blocks', 1)

    def _complete(self):
        """Decisions with both senses and controls: the episode's steps."""
        return min(len(self.senses), len(self.controls))

    def finish(self, end, cleared):
        """Count how the episode ended on its last complete decision."""
        n = self._complete()
        if n:
            last = self.events[n - 1]
            last['exited'] += int(end == 'exited')
            last['cleared_exit'] += int(end == 'exited' and cleared)
            last['scored_out'] += int(end == 'scored_out')

    def arrays(self, shaping):
        n = self._complete()
        senses = np.asarray(self.senses[:n], dtype=np.float32).reshape(n, len(CAMPAIGN_FEATURE_NAMES))
        controls = np.asarray(self.controls[:n], dtype=np.int8).reshape(n, 3)
        if self._graded_shooting:
            if self._raw_shots is None:
                self._raw_shots = [row['shots'] for row in self.events[:n]]
            for row, raw, rough in zip(self.events[:n], self._raw_shots, self._rough_shots):
                row['shots'] = max(0, raw - rough)
                row['unaimed_shots'] = row['shots']
        else:
            counts = unaimed_shots(senses, [row['shots'] for row in self.events[:n]], CAMPAIGN_FEATURE_NAMES)
            for row, count in zip(self.events[:n], counts):
                row['unaimed_shots'] = count
        rewards = TABLE.pay(self.events[:n], vars(shaping))
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
                 policy=None, trace=None, low_graphics=True, pursuit_lesson=False, ship_hit_feedback=True,
                 graded_shooting=False, accuracy_power=2, sticky_targets=False, target_recovery=False):
    """One episode on `level` flown by the campaign policy in `spec` (sampled from `spec.sample_seed`, unless it or
    `greedy` asks for its best choices). It ends at the exit, on scoring out, or after `horizon` seconds.

    Returns the per-decision `senses`, `controls`, `rewards` and `temperatures`, and the episode's `result`; the
    exit's or the score-out's reward lands on the last decision. `pilot` replaces the policy (for tests): anything
    with decide(game) and on_observed(listener). `policy`, a loaded policy, is flown in place of `spec.weights` (a
    replay's start policy, migrated in memory).

    For watching: `trace`, if given, is called each frame, after the game updates, with (seconds, game); and
    `low_graphics=False` keeps the tentacles' drag physics, which the episode does not depend on.

    With `pursuit_lesson`, stop at first sight of an enemy and pay search_contact once. A visible start ends
    immediately without that bonus; it cannot masquerade as successful searching. The real maze and ship physics
    remain, with one stationary hidden enemy on an open route and no pickups. The trainer records the option.

    `graded_shooting` waives shooting costs for rough aim and grades emitted volleys toward a clear intercept.
    False retains historical shooting rules. The trainer records this as a lifetime rule for faithful replays.

    `sticky_targets` keeps the enemy compass, first visible contact and pursuit reward on a persistent target.
    False preserves historical nearest-enemy observations; new runs record True as a lifetime rule.

    `target_recovery` opts persistent targeting into stalled-chase recovery and switch-back cooldowns.
    Historical episodes default to False; new training runs record this rule explicitly.

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
                              rng=None if greedy else np.random.default_rng(spec.sample_seed),
                              hit_feedback=ship_hit_feedback, sticky_targets=sticky_targets, target_recovery=target_recovery)
    now = [START_TIME]
    previous_graphics, previous_ally = config.SETTINGS.game.lowGraphics, config.SETTINGS.game.hunterPilot
    config.SETTINGS.game.lowGraphics = low_graphics  # tentacle physics: nothing the pilot senses reads it
    # A beacon summons the ally with the configured pilot; Jev's reads wall time and calls out, so it is pinned.
    config.SETTINGS.game.hunterPilot = 'neural'
    episode_encounter, previous_damage_callback = None, None
    episode_maze, previous_block_callback = None, None
    episode_ship, previous_fire, previous_fire_override = None, None, None
    try:
        from learning.campaign_training_levels import EXIT_WITH_CRAWLER, exit_with_crawler_layout
        if level == EXIT_WITH_CRAWLER:
            # Keep the real level's physics, strength, maze and exit rules. This
            # temporary layout override is confined to this training episode.
            import level_config
            from unittest.mock import patch
            original_layout = level_config.get_level_layout
            def practice_layout(number):
                layout = original_layout(number)
                return exit_with_crawler_layout(layout) if number == 32 else layout
            with patch.object(level_config, 'get_level_layout', side_effect=practice_layout):
                _setup(game, 32, seed, now)
        else:
            _setup(game, level, seed, now)
        if pursuit_lesson:
            from learning.campaign_pursuit import setup as setup_pursuit
            setup_pursuit(game, seed)
        recorder = _Recorder(pilot, game, missed_opportunities=bool(shaping.missed_opportunity),
                              graded_shooting=graded_shooting, accuracy_power=accuracy_power)
        if graded_shooting:
            episode_ship, previous_fire = game.ship, game.ship.fire
            previous_fire_override = game.ship.__dict__.get('fire')

            def fire():
                shots = previous_fire()
                if shots:
                    recorder.fired(shots)
                return shots

            episode_ship.fire = fire
        episode_maze = game.maze
        previous_block_callback = episode_maze.on_block_destroying

        def block_destroying(cell, source):
            if previous_block_callback is not None:
                previous_block_callback(cell, source)
            recorder.block_destroying(cell, source)

        episode_maze.on_block_destroying = block_destroying
        episode_encounter = game.encounter
        previous_damage_callback = episode_encounter.on_projectile_damage

        def damaged_enemy(creature, projectile):
            if previous_damage_callback is not None:
                previous_damage_callback(creature, projectile)
            if creature.allegiance == 'enemy' and projectile.source == 'player':
                recorder.add('enemy_hits', 1)

        episode_encounter.on_projectile_damage = damaged_enemy
        if hasattr(pilot, 'reset'):
            pilot.reset()
        game.ship_pilot = recorder
        scoring, handler = game.scoring, game.collision_handler
        enemies_at_start = len(game.roster.alive())
        seen = dict(own=0, pickups=0, walls=0, shots=0, kills=handler.level_kills)
        # Demo's no-kill skip, measured rather than applied: the clock waits while no enemy is left.
        last_kill, max_gap = now[0], 0.0
        flight, beacon, end, frames = game.hunter_flight, False, 'horizon', 0
        while frames < int(horizon * FPS):
            game.update(1.0)
            now[0] += 1 / FPS
            frames += 1
            recorder.count_missed_opportunity(1 / FPS)
            recorder.add('own_kills', scoring.enemies_destroyed - seen['own'])
            recorder.add('pickups', scoring.powerup_crystals_collected - seen['pickups'])
            if scoring.wall_collisions > seen['walls']:
                # Match the allied hunter's impact measure. Shielded contacts still bounce but do not incur
                # the scoring penalty, so they also stay free of this training penalty.
                recorder.add('wall_impulse', math.hypot(*game.ship.wall_impulse) / max(game.ship.max_speed, 1e-6))
            recorder.add('shots', scoring.shots_fired - seen['shots'])
            seen.update(shots=scoring.shots_fired, own=scoring.enemies_destroyed, pickups=scoring.powerup_crystals_collected,
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
            if pursuit_lesson and recorder._observed_at is not None:
                if not recorder.started_unseen:
                    end = 'visible_start'
                    break
                if recorder.first_sighting_seconds is not None:
                    end = 'contact'
                    recorder.add('search_contacts', 1)
                    break
            if game.exit_explosion_active:
                end = 'exited'
                break
            if game.game_frozen:
                end = 'scored_out'
                break
        cleared = not game.roster.alive()
        recorder.finish(end, cleared)
        senses, controls, rewards, temperatures = recorder.arrays(shaping)
        result = CampaignResult(level, seed, end, frames / FPS, enemies_at_start, scoring.enemies_destroyed,
                                handler.level_kills - scoring.enemies_destroyed, cleared,
                                scoring.powerup_crystals_collected, beacon, max_gap, scoring.wall_collisions, scoring.shots_fired,
                                started_unseen=recorder.started_unseen,
                                first_sighting_seconds=recorder.first_sighting_seconds,
                                unseen_seconds=recorder.unseen_seconds, search_horizon_seconds=horizon,
                                damaging_hits=int(sum(row['enemy_hits'] for row in recorder.events[:len(senses)])),
                                unaimed_shots=int(sum(row['unaimed_shots'] for row in recorder.events[:len(senses)])),
                                breach_blocks=int(sum(row['breach_blocks'] for row in recorder.events[:len(senses)])),
                                wall_impulse=float(sum(row['wall_impulse'] for row in recorder.events[:len(senses)])),
                                pursuit_lesson=pursuit_lesson,
                                graded_shooting=graded_shooting,
                                missed_opportunities=(float(sum(row['missed_opportunities']
                                                               for row in recorder.events[:len(senses)]))
                                                      if shaping.missed_opportunity else None))
    finally:
        if episode_ship is not None:
            if previous_fire_override is None:
                episode_ship.__dict__.pop('fire', None)
            else:
                episode_ship.fire = previous_fire_override
        if episode_encounter is not None:
            episode_encounter.on_projectile_damage = previous_damage_callback
        if episode_maze is not None:
            episode_maze.on_block_destroying = previous_block_callback
        _restore(game)
        config.SETTINGS.game.lowGraphics = previous_graphics
        config.SETTINGS.game.hunterPilot = previous_ally
    totals = {term.key: float(sum(row[term.event] for row in recorder.events[:len(senses)])
                              * TABLE.weight(term, vars(shaping))) for term in TABLE.paying}
    return {'senses': senses, 'controls': controls, 'rewards': rewards, 'temperatures': temperatures,
            'result': result, 'reward_totals': totals}
