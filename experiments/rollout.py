"""Flies the neural student with sampled controls and records what it needs to learn from the episode."""
import math
import numpy as np
from experiments.harness import FPS, run_episode
from experiments.score import Weights
from hunter.rule_params import RULES
from hunter.senses import FEATURE_NAMES, HIT_POINT_CELLS, SENSE_SIZE, blocked, ray_hits

KILL_REWARD_TOTAL = 100.0       # shared across every enemy that existed in the episode
BOREDOM_GRACE = 1.0             # seconds with nothing new in view before the hunter is bored
VIEW_CELLS = 3.0                # walls further than this many cells do not count as in view for boredom: far off, a
                                # slight move makes the rays land on different blocks, which would look new
AIM_DEGREES = 20.0              # a shot this close to an enemy in sight is aimed, and free
FOLLOW_DEGREES = 45.0           # the compass counts as followed when it is this close to the nose
PANIC_WINDOW = 5.0              # seconds of score that decide whether the hunter panics

_SEEN = FEATURE_NAMES.index('enemy0_present')
_ENEMY = [tuple(FEATURE_NAMES.index(f'enemy{i}_{part}') for part in ('present', 'sin', 'cos')) for i in range(6)]
_ALIVE = FEATURE_NAMES.index('enemies_present')
_COMPASS = {part: FEATURE_NAMES.index(f'enemy_compass_{part}') for part in ('present', 'sin', 'cos', 'dist')}


def compass_following(senses):
    """(blind steps, steps of those heading within FOLLOW_DEGREES of the enemy compass). Blind: enemies are left
    but none is in sight."""
    senses = np.asarray(senses).reshape(-1, SENSE_SIZE)
    blind = (senses[:, _ALIVE] == 1) & (senses[:, _SEEN] == 0) & (senses[:, _COMPASS['present']] == 1)
    off = np.degrees(np.abs(np.arctan2(senses[:, _COMPASS['sin']], senses[:, _COMPASS['cos']])))
    return int(blind.sum()), int((blind & (off < FOLLOW_DEGREES)).sum())


class RolloutRecorder:
    """Per decision step: the senses, the controls the policy chose, and the events that followed."""

    def __init__(self, decision_frames, wall_scale=1.0, hit_scale=1.0, still_penalty=0.0, still_speed=0.15,
                 turn_change_penalty=0.0, speed_penalty=0.0, powerup_reward=0.0, upgrade_bonus=0.0,
                 reversal_penalty=0.0, death_penalty=0.0, rules_bounty=0.0, compass_reward=0.0,
                 boredom_penalty=0.0, boredom_window=10.0, shot_price=0.0, turning_penalty=0.0, hit_reward=0.0,
                 panic_score=None, panic_temperature=1.0):
        self.decision_frames = decision_frames
        # Panic: once the last PANIC_WINDOW seconds score below panic_score, choices are sampled at
        # panic_temperature until that score is back above zero. None leaves it off.
        self.panic_score, self.panic_temperature = panic_score, panic_temperature
        self._panicking = False
        self._temperatures = []                   # per step: the temperature its choice was sampled at
        self._stale = []                          # per step: nothing new in view for BOREDOM_GRACE (blind or not)
        self._view_memory = ({}, None)            # thing -> step last seen, and the step something was last new
        self.boredom_penalty = boredom_penalty    # shaping: points lost per second blind with nothing new in view
        self.boredom_window = boredom_window      # seconds before a thing seen counts as new again
        self._views = []                          # per step: the wall blocks and crystals in the hunter's view
        self.shot_price = shot_price              # shaping: points lost per shot fired with no enemy in sight
        self.turning_penalty = turning_penalty    # shaping: points lost per second spent turning
        self.hit_reward = hit_reward              # shaping: points gained per hunter shot that reaches an enemy
        self._hits = []                           # per step: the hunter's shots that reached an enemy
        self._hits_before = 0
        self._shots = []                          # per step: shots the hunter fired
        self._shots_before = 0
        self.compass_reward = compass_reward    # shaping: points per maze width the compass route shortens, while blind
        self.compass_step_cap = 0.05    # a larger change in one step is the nearest enemy changing, not the hunter moving
        self.compass_hit_cap = 0.05     # ...except a drop of up to one hit on a wall block on the compass route
        self.wall_scale = wall_scale      # shaping: multiplies the wall-hit penalty in the reward only
        self.hit_scale = hit_scale        # likewise for the enemy-hit penalty
        self.still_penalty = still_penalty    # shaping: points lost per second spent slower than still_speed
        self.still_speed = still_speed        # as a fraction of the hunter's top speed
        self.turn_change_penalty = turn_change_penalty    # points lost each time a turn starts or reverses
        self.speed_penalty = speed_penalty    # shaping: points lost per second at standstill, scaled by the shortfall from top speed
        self.powerup_reward = powerup_reward    # shaping: points gained per powerup crystal the hunter collects
        self.death_penalty = death_penalty    # shaping: points lost when a mortal hunter is destroyed
        self.reversal_penalty = reversal_penalty    # shaping: extra points lost when the turn flips straight from left to right
        self.rules_bounty = rules_bounty    # shaping: points gained when the last Rules hunter is destroyed
        self._rules_paid = False
        self.upgrade_bonus = upgrade_bonus    # shaping: points gained per second the hunter's guns are upgraded
        self.upgraded_frames = 0
        self._weapon_kills = {}
        self._weapon_pickups = {}
        self._died = False
        self.senses, self.controls = [], []
        self.speed_sum, self.frames = 0.0, 0     # mean speed as a share of top speed, for the dashboard
        self._events = []                 # per step: kills, wall impulse, enemy hits, idle seconds, near-still seconds, top-speed shortfall seconds, powerups, upgraded seconds, death, rules bounty
        self._pending = False
        self._controller = None
        self._counter = None
        self._start_enemies = 0
        self._previous = (0, 0, 0, 0.0)
        self._previous_powerups = 0

    # --- called by the harness ---
    def begin(self, game, hunter, settings):
        self._start_enemies = len(game.roster.alive())
        handler = getattr(game, 'collision_handler', None)
        self._weapon_kills = getattr(handler, 'weapon_kills', {})
        self._weapon_pickups = getattr(handler, 'weapon_pickups', {})
        width = game.maze.grid_width * game.maze.cell_size_x
        self.compass_step_cap = 1.5 * hunter.max_speed * self.decision_frames / max(width, 1.0)
        cell = (game.maze.cell_size_x + game.maze.cell_size_y) / 2
        self.compass_hit_cap = 1.1 * HIT_POINT_CELLS * cell / max(width, 1.0)
        # A worker reuses its game from one episode to the next: count only what happens from here on.
        encounter = getattr(game, 'encounter', None)
        self._hits_before = encounter.hits['hunter'] if encounter is not None else 0
        self._shots_before = getattr(hunter, 'shots_fired', 0)
        game.hunter_flight.on_observed(self._observed)

    def _observed(self, sense):
        if self._pending:
            if self.panic_score is not None and self._controller is not None:
                self._controller.temperature = self._panic(sense)
            self._temperatures.append(getattr(self._controller, 'temperature', 1.0))
            self.senses.append(sense)

    def _panic(self, sense):
        """The temperature for the step about to be chosen: the score of the last PANIC_WINDOW seconds of complete
        steps decides whether the hunter panics (below panic_score) or calms down (above zero)."""
        k = len(self.senses)
        if k == 0:
            return 1.0
        lo = max(0, k - round(PANIC_WINDOW * FPS / self.decision_frames))
        senses = np.asarray(self.senses[lo:k] + [sense], dtype=np.float32).reshape(-1, SENSE_SIZE)
        stale = np.zeros(k - lo, dtype=bool)
        known = np.asarray(self._stale[lo:k], dtype=bool)
        stale[:len(known)] = known
        bored = stale & (senses[:-1, _ALIVE] == 1) & (senses[:-1, _SEEN] == 0)
        score = float(self._step_rewards(lo, k, senses, bored).sum())
        if score < self.panic_score:
            self._panicking = True
        elif score > 0:
            self._panicking = False
        return self.panic_temperature if self._panicking else 1.0

    def before_frame(self, game, hunter):
        self._controller = game.hunter_flight.controller
        if self._controller.frames % self.decision_frames == 0:
            self._pending = True
            self._events.append([0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0])
            self._shots.append(0)
            self._hits.append(0)
            maze = getattr(game, 'maze', None)
            if maze is not None:
                self._views.append(self._view(game, hunter, maze))
                self._stale.append(self._remember_view(self._views[-1]))
        encounter = getattr(game, 'encounter', None)
        if self._hits and encounter is not None:
            landed = encounter.hits['hunter']
            self._hits[-1] += max(0, landed - self._hits_before)
            self._hits_before = landed
        if self._shots:
            fired = getattr(hunter, 'shots_fired', 0)
            self._shots[-1] += fired - self._shots_before
            self._shots_before = fired
        if self._events and math.hypot(hunter.vx, hunter.vy) < self.still_speed * hunter.max_speed:
            self._events[-1][4] += 1.0 / FPS
        if self._events:
            share = math.hypot(hunter.vx, hunter.vy) / hunter.max_speed
            self.speed_sum += min(share, 1.0)
            self.frames += 1
            from weapons.state import held_weapon  # here, not at the top: the weapons package loads pygame
            if getattr(hunter, 'gun_upgrade_level', 0) or held_weapon(hunter) is not None:
                self.upgraded_frames += 1
                self._events[-1][7] += 1.0 / FPS
            shortfall = 1.0 - share
            self._events[-1][5] += min(max(shortfall, 0.0), 1.0) / FPS

    def add_frame(self, rotation, engine, trigger):
        pass

    def after_frame(self):
        if self._pending:
            self.controls.append(self._controller.last_controls)
            self._pending = False

    def counted(self, counter, elapsed):
        current = (counter.kills, counter.wall_impulse, counter.hits_taken, counter.idle_seconds)
        for i, (now, before) in enumerate(zip(current, self._previous)):
            self._events[-1][i] += now - before
        self._previous = current
        self._events[-1][6] += counter.powerups - self._previous_powerups
        self._previous_powerups = counter.powerups
        if counter.died and not self._died:
            self._events[-1][8] = 1        # the step in which the hunter was destroyed
            self._died = True
        if getattr(counter, 'rules_beaten', False) and not self._rules_paid:
            self._events[-1][9] = 1        # the step in which the last Rules hunter was destroyed
            self._rules_paid = True
        self._counter = counter

    # --- result ---
    def arrays(self):
        n = min(len(self.senses), len(self.controls), len(self._events))
        senses = np.asarray(self.senses[:n], dtype=np.float32).reshape(n, SENSE_SIZE)
        controls = np.asarray(self.controls[:n], dtype=np.int8).reshape(n, 3)
        bored = self._bored(senses, self._views[:n]) if self.boredom_penalty else np.zeros(n, dtype=bool)
        return senses, controls, self._step_rewards(0, n, senses, bored).astype(np.float32)

    def temperatures(self, n):
        """The temperature each of the first n steps' choices was sampled at."""
        t = np.ones(n, dtype=np.float32)
        known = np.asarray(self._temperatures[:n], dtype=np.float32)
        t[:len(known)] = known
        return t

    def _step_rewards(self, lo, hi, senses, bored):
        """The rewards of steps lo..hi-1. `senses` holds those steps' senses and, when it is known, step hi's too (the
        compass progress of the last step needs it); `bored` holds whether each of them was bored. The whole episode
        is scored this way at the end, and the last few seconds as it goes for panic."""
        m = hi - lo
        spawned = self._counter.spawned if self._counter is not None else 0
        total = max(self._start_enemies + spawned, 1)
        weights = Weights()
        events = np.asarray(self._events[lo:hi], dtype=np.float64).reshape(m, 10)
        rewards = (KILL_REWARD_TOTAL / total * events[:, 0] - weights.quotient * self.wall_scale * events[:, 1]
                   - weights.hit * self.hit_scale * events[:, 2] - weights.idle / 60.0 * events[:, 3]
                   - self.still_penalty * events[:, 4] - self.speed_penalty * events[:, 5]
                   + self.powerup_reward * events[:, 6] + self.upgrade_bonus * events[:, 7]
                   - self.death_penalty * events[:, 8] + self.rules_bounty * events[:, 9])
        controls = np.asarray(self.controls[lo:hi], dtype=np.int8).reshape(m, 3)
        # Turn control: 0 left, 1 none, 2 right. A turn starts (from none) or reverses when it differs from the
        # previous decision's and is not 'none'; holding a turn, or stopping one, is free.
        before = self.controls[lo - 1][0] if lo > 0 else 1
        previous = np.concatenate([[before], controls[:-1, 0]]) if m else controls[:, 0]
        changes = (controls[:, 0] != 1) & (controls[:, 0] != previous)
        rewards = rewards - self.turn_change_penalty * changes
        reversals = changes & (previous != 1)             # the turn flipped straight over, with no pause between
        rewards = rewards - self.reversal_penalty * reversals
        # Charged for the time spent turning, so circling costs every second while a quick correction is cheap.
        rewards = rewards - self.turning_penalty * self.decision_frames / FPS * (controls[:, 0] != 1)
        if self.compass_reward and len(senses) > 1:
            rewards = rewards + self.compass_reward * self._compass_progress(senses)[:m]
        if self.hit_reward:
            hits = np.zeros(m)
            counted = np.asarray(self._hits[lo:hi], dtype=np.float64)
            hits[:len(counted)] = counted
            rewards = rewards + self.hit_reward * hits
        if self.shot_price:
            rewards = rewards - self.shot_price * self.unaimed_shots(senses[:m], lo)
        if self.boredom_penalty:
            rewards = rewards - self.boredom_penalty * self.decision_frames / FPS * np.asarray(bored[:m], dtype=bool)
        return rewards

    def _remember_view(self, view):
        """Note what is in view this step; whether nothing new has come into view for BOREDOM_GRACE (see _bored)."""
        step = self.decision_frames / FPS
        grace, window = round(BOREDOM_GRACE / step), round(self.boredom_window / step)
        last_seen, last_new = self._view_memory
        i = len(self._stale)
        if any(i - last_seen.get(thing, -10 ** 9) > window for thing in view):
            last_new = i
        for thing in view:
            last_seen[thing] = i
        self._view_memory = (last_seen, last_new)
        return last_new is None or i - last_new >= grace

    @staticmethod
    def _view(game, hunter, maze):
        """The things in the hunter's view: the wall block each sight ray meets first within VIEW_CELLS (a loose
        segment or a gate by itself), and the crystals in plain sight."""
        origin = (hunter.x, hunter.y)
        reach = hunter.settings.sensor_range(maze)
        walls = [w for w in list(maze.spatial_grid.get_nearby_walls(origin, reach))
                 + list(getattr(maze, 'gate_segments', ())) if w.active]
        near = VIEW_CELLS * (maze.cell_size_x + maze.cell_size_y) / 2
        things = {w.cell if getattr(w, 'cell', None) is not None else ('segment', id(w))
                  for w in ray_hits(origin, hunter.angle, walls, near) if w is not None}
        flight = getattr(game, 'hunter_flight', None)
        crystals = [c for c in (flight.surroundings.pickups() if flight is not None else ()) if c.active
                    and math.dist(origin, (c.x, c.y)) <= reach]
        if crystals and walls:
            starts = np.array([w.start for w in walls], dtype=float)
            ends = np.array([w.end for w in walls], dtype=float)
            hidden = blocked(origin, [(c.x, c.y) for c in crystals], starts, ends)
            crystals = [c for c, hid in zip(crystals, hidden) if not hid]
        things.update(('crystal', id(c)) for c in crystals)
        return frozenset(things)

    def _bored(self, senses, views):
        """Per step, whether the hunter is blind (enemies left, none in sight) and nothing new has come into its view
        for BOREDOM_GRACE, new meaning not seen in the last boredom_window. Spinning, circling or dithering keeps
        showing it the same walls; flying on shows it new ones."""
        step = self.decision_frames / FPS
        grace, window = round(BOREDOM_GRACE / step), round(self.boredom_window / step)
        last_seen, last_new, bored = {}, None, np.zeros(len(views), dtype=bool)
        for i, view in enumerate(views):
            if any(i - last_seen.get(thing, -10 ** 9) > window for thing in view):
                last_new = i
            for thing in view:
                last_seen[thing] = i
            bored[i] = last_new is None or i - last_new >= grace
        n = min(len(senses), len(views))
        blind = np.zeros(len(views), dtype=bool)
        blind[:n] = (senses[:n, _ALIVE] == 1) & (senses[:n, _SEEN] == 0)
        return bored & blind

    def unaimed_shots(self, senses, lo=0):
        """Per step from step `lo`, the shots fired with no enemy in sight within AIM_DEGREES of the nose (at walls,
        at nothing, or well wide)."""
        n = len(senses)
        shots = np.zeros(n)
        counted = np.asarray(self._shots[lo:lo + n], dtype=np.float64)
        shots[:len(counted)] = counted
        aimed = np.zeros(n, dtype=bool)
        for present, sin, cos in _ENEMY:
            off = np.degrees(np.abs(np.arctan2(senses[:, sin], senses[:, cos])))
            aimed |= (senses[:, present] == 1) & (off <= AIM_DEGREES)
        return np.where(aimed, 0.0, shots)

    def bored_steps(self, senses, views):
        """(blind steps, bored steps of those) for the dashboard."""
        senses = np.asarray(senses).reshape(-1, SENSE_SIZE)
        blind = (senses[:, _ALIVE] == 1) & (senses[:, _SEEN] == 0)
        return int(blind.sum()), int(self._bored(senses, list(views)[:len(senses)]).sum())

    def _compass_progress(self, senses):
        """Per step, how far (in maze widths) the compass route to the nearest enemy shortened by the next step,
        counted only while no enemy is in sight and when the change is one the hunter's own flight could make."""
        on = senses[:, _COMPASS['present']] == 1
        dist = senses[:, _COMPASS['dist']].astype(np.float64)
        progress = np.zeros(len(senses))
        delta = dist[:-1] - dist[1:]
        valid = (on[:-1] & on[1:] & (senses[:-1, _SEEN] == 0) & (delta >= -self.compass_step_cap)
                 & (delta <= max(self.compass_step_cap, self.compass_hit_cap)))
        progress[:-1] = np.where(valid, delta, 0.0)
        return progress


def fly_rollout(level, seed, spec, horizon=90.0, wall_scale=1.0, hit_scale=1.0,
                still_penalty=0.0, still_speed=0.15, random_start=False, turn_change_penalty=0.0,
                speed_penalty=0.0, powerup_reward=0.0, after_clear=0.0, upgrade_bonus=0.0, pursue_hunter=True, reversal_penalty=0.0,
                health=0, death_penalty=0.0, rules_bounty=0.0, compass_reward=0.0, boredom_penalty=0.0,
                boredom_window=10.0, shot_price=0.0, turning_penalty=0.0, hit_reward=0.0, panic_score=None,
                panic_temperature=1.0, duel_separation=False, trace=None, low_graphics=True, random_maze=False, combat_ruleset="legacy", weapon_tuning=None, curved_walls=False):
    recorder = RolloutRecorder(spec.decision_frames, wall_scale, hit_scale, still_penalty, still_speed,
                               turn_change_penalty, speed_penalty, powerup_reward, upgrade_bonus, reversal_penalty, death_penalty,
                               rules_bounty, compass_reward, boredom_penalty, boredom_window, shot_price,
                               turning_penalty, hit_reward, panic_score, panic_temperature)
    result = run_episode(level, seed, RULES, horizon, student=spec, recorder=recorder, random_start=random_start,
                         after_clear=after_clear, pursue_hunter=pursue_hunter, health=health, trace=trace,
                         duel_separation=duel_separation, low_graphics=low_graphics, random_maze=random_maze,
                         combat_ruleset=combat_ruleset, weapon_tuning=weapon_tuning, curved_walls=curved_walls)
    senses, controls, rewards = recorder.arrays()
    blind, following = compass_following(senses)
    unseen, bored = recorder.bored_steps(senses, recorder._views)
    return {'blind_steps': blind, 'following_steps': following, 'unseen_steps': unseen, 'bored_steps': bored,
            'unaimed_shots': int(recorder.unaimed_shots(senses).sum()),
            'hits': int(sum(recorder._hits[:len(senses)])),
            'temperatures': recorder.temperatures(len(senses)),
            'senses': senses, 'controls': controls, 'rewards': rewards, 'result': result,
            'speed_fraction': recorder.speed_sum / recorder.frames if recorder.frames else 0.0,
            'upgraded_fraction': recorder.upgraded_frames / recorder.frames if recorder.frames else 0.0,
            'weapon_kills': dict(recorder._weapon_kills), 'weapon_pickups': dict(recorder._weapon_pickups),
            'powerups': recorder._counter.powerups if recorder._counter is not None else 0,
            'died': result.end == 'died',
            'rules_arena': result.rules_hunters > 0, 'rules_beaten': result.rules_beaten}
