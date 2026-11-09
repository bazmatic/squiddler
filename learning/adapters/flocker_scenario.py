"""Real-game Flocker team drills, staged recurrent PPO and measured volley credit."""
from collections import Counter, defaultdict, deque
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np

from creatures.flocker_team import DECISION_FRAMES, get_flocker_team, muzzle, visible
from creatures.traits.rooted import wall_between
from learning.environment import Snapshot, Transition
from learning.adapters.flocker import SPEC, CRITIC_SIZE, observe, critic, intention, enabled, load_transfer_policy
from learning.adapters.crawler_scenario import TRAIN_LEVELS, HOLDOUT_LEVELS, SilentSound
from learning import flocker_rewards
from learning.flocker_rewards import TABLE

INITIAL_STAGE = 1
IMITATION_HEADS = ()
SHIP_PILOT = Path(__file__).resolve().parents[2]/'assets/neural/hunter.npz'
STAGES = ('pursuit', 'perching', 'aiming', 'volleys', 'combat', 'hunter')
# Stage 1 only: a fixed bonus each time a bird comes within 100 px of a visible target. It is not a run setting (one
# would change every recorded manifest), so it is paid beside the reward table rather than from it.
APPROACH_REWARD = .5


class VolleyTracker:
    """Launch groups are fixed six-frame windows, never a sliding chain of shots."""
    def __init__(self):
        self.groups = []
        self.by_target = {}

    def launch(self, shot, target, shooter, frame, eligible):
        generation = getattr(target,'generation',0)
        key = (id(target),generation)
        group = self.by_target.get(key)
        if group is None or frame-group['frame'] > 6:
            group = dict(target=target, generation=generation, frame=frame, last=frame, shooters=set(), eligible=set(eligible),
                         hit=False, paid=False, measured=False)
            self.groups.append(group)
            self.by_target[key] = group
        group['shooters'].add(shooter)
        group['last'] = frame
        shot.flocker_volley = group

    def hit(self, shot, target):
        group = getattr(shot, 'flocker_volley', None)
        if group is not None and group['target'] is target and group['generation']==getattr(target,'generation',0):
            group['hit'] = True

    def settle(self, frame, rewards, weight):
        for group in self.groups:
            if frame-group['frame'] <= 6 or len(group['shooters']) < 2 or not group['hit'] or group['paid']:
                continue
            group['paid'] = True
            participation = len(group['shooters'] & group['eligible'])/max(1,len(group['eligible']))
            bonus = weight*min(1.,participation)/len(group['shooters'])
            for i in group['shooters']:
                rewards[i] += bonus

    def metrics(self):
        volleys = [g for g in self.groups if len(g['shooters']) >= 2]
        return dict(volleys=len(volleys), useful_volleys=sum(g['hit'] for g in volleys),
                    volley_spreads=[g['last']-g['frame'] for g in volleys],
                    participation_sum=sum(len(g['shooters'] & g['eligible'])/max(1,len(g['eligible'])) for g in volleys))


class FlockerEnvironment:
    spec = SPEC
    adapter = sys.modules[__name__]

    def __init__(self, horizon=60., arenas=None, scripted=False, muted=False, rewards=None,
                 opponent=None, legacy=False, flocker_radius=None, flocker_propulsion=None,opponent_controller='neural'):
        self.horizon, self.arenas = horizon, arenas
        self.scripted, self.muted, self.legacy = scripted, muted, legacy
        self.rewards = flocker_rewards.validate(rewards or {})
        self.flocker_propulsion = flocker_propulsion
        self.flocker_radius = flocker_radius
        self.opponent = str(opponent or SHIP_PILOT)
        if opponent_controller!='neural':
            raise ValueError('unknown Flocker opponent controller')
        self.opponent_controller = opponent_controller
        self.world = None
        self.render_frames = None

    def reset(self, seed, level=11, team_size=2, stage=1, moving=True, extra_targets=0, reinforcements=0, split_group=False, regroup=False, compact_group=False,target_speed=None,visible_start=False,hunter_strength=None,hunter_health=None):
        import pygame
        import config
        import creatures
        import level_config
        from level_layout import parse_layout
        from maze.config import MazeComplexity
        from maze.generator import Maze
        from physics.obstacle_world import ObstacleWorld
        from game_handlers.enemy_roster import EnemyRoster
        from game_handlers.collision_handler import CollisionHandler
        from entities.command_recorder import CommandRecorder
        from scoring.system import ScoringSystem
        from creatures.encounter import Encounter
        from learning.adapters.practice_targets import PracticeTarget
        if not 1 <= stage <= 6 or not 1 <= team_size <= 8 or team_size+reinforcements > 8:
            raise ValueError('Flocker stage must be 1..6 and total team size 1..8')
        self.close()
        random.seed(seed)
        self.rng = rng = random.Random(seed)
        pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1352,878))
        self._screen_settings = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT = 1352,878
        arena = self.arenas[str(level)] if self.arenas else level_config.load_level_config(level)
        layout = parse_layout(arena['layout'], arena['maze']['grid_size']) if 'layout' in arena else None
        if layout is not None:
            layout = replace(layout, enemies=(), beacon=None)
        maze = Maze(level, MazeComplexity[arena['maze']['complexity'].upper()], arena['maze']['grid_size'],
                    layout=layout, seed=seed, curved=True)
        self.obstacles = ObstacleWorld(maze,layout.gates) if layout is not None and layout.gates else None
        maze.gate_segments = self.obstacles.segments() if self.obstacles else []
        cells = [maze.position_calculator.grid_center_to_screen(c,r)
                 for r in range(1,maze.grid_height-1) for c in range(1,maze.grid_width-1) if not maze.grid[r][c]]
        rng.shuffle(cells)
        from creatures.wallcrawler_team import capsule_hits
        self.cells = [p for p in cells if maze.is_open(p,15) and not any(
            w.active and capsule_hits(p,p,15,w.start,w.end) for w in maze.gate_segments)]
        from hunter.maze_route import MazeRoutes
        routes = MazeRoutes()
        remaining, largest = set(range(len(self.cells))), set()
        while remaining:
            first = min(remaining)
            component = set(routes.reachable_targets(maze,self.cells[first],self.cells))
            if not component:
                component = {first}
            if len(component) > len(largest):
                largest = component
            remaining -= component
        self.cells = [p for i,p in enumerate(self.cells) if i in largest]
        if len(self.cells) < team_size+1:
            raise ValueError('arena lacks connected Flocker starts')
        scoring = ScoringSystem()
        scoring.power_drain_enabled = False
        handler = CollisionHandler(SilentSound(),scoring,CommandRecorder())
        handler.drop_pickup = lambda *a,**k: None
        pilot = None
        if Path(self.opponent).is_file():
            from hunter.neural.policy import NeuralPolicy
            pilot = NeuralPolicy.load(self.opponent)
        self.world = SimpleNamespace(ship=None,hunter=None,maze=maze,projectiles=[],roster=EnemyRoster(),
            scoring=scoring,sound_manager=SilentSound(),command_recorder=handler.command_recorder,
            collision_handler=handler,powerup_crystals=[],jev_beacon_allowed=lambda:False,combat_ruleset='legacy',
            additional_targets=[],flocker_policy=None,enemy_policy=lambda:pilot,_hunter_sees_player=lambda:False)
        chosen = []
        if compact_group and not split_group:
            # Use local free space rather than widely spaced maze-cell centres.
            offsets = sorted(((20*x,20*y) for x in range(-3,4) for y in range(-3,4)),
                             key=lambda p:(math.hypot(*p),p))
            for anchor in self.cells:
                group = []
                for dx,dy in offsets:
                    p = (anchor[0]+dx,anchor[1]+dy)
                    if (maze.is_open(p,15) and not wall_between(anchor,p,maze)
                        and all(math.dist(p,q)<=120 and not wall_between(p,q,maze) for q in group)
                        and not any(w.active and capsule_hits(p,p,15,w.start,w.end) for w in maze.gate_segments)):
                        group.append(p)
                        if len(group)==team_size:
                            break
                if len(group)==team_size:
                    chosen = group
                    break
            if not chosen:
                raise ValueError('arena lacks space for a compact Flocker start')
        for i in range(team_size):
            pos = chosen[i] if compact_group and not split_group else next((p for p in self.cells if all(math.dist(p,q)>=45 for q in chosen)
                        and (split_group or not chosen or math.dist(p,chosen[0])<280)), None)
            if pos is None:
                pos = next((p for p in self.cells if all(math.dist(p,q)>=30 for q in chosen)), None)
            if pos is None:
                raise ValueError('arena lacks separated Flocker starts')
            if not (compact_group and not split_group):
                chosen.append(pos)
            c = self.build_flocker(pos,level,angle=rng.uniform(0,360))
            c.agent_id = i
            self.world.roster.add(c)
        self.enc = Encounter(self.world.roster,self.world)
        self.enc.ready()
        self.team = get_flocker_team(self.enc)
        self.team.external, self.team.muted, self.team.legacy = not self.scripted,self.muted,self.legacy
        self.team._next_id = team_size
        self.targets = []
        self.ships = []
        self.level, self.seed, self.team_size, self.stage = level,seed,team_size,stage
        self.regroup = regroup
        self.visible_start = visible_start
        self.agent_ids = tuple(range(team_size))
        self.initial_size = team_size
        self.reinforcements = reinforcements
        self.hunter_strength = .5 if hunter_strength is None else hunter_strength
        self.hp = {}
        if stage >= 5:
            if pilot is None:
                raise ValueError('combat training requires a frozen hunter policy')
            for _ in range(1+extra_targets):
                from hunter.build import PilotSetup
                from hunter.flight import NeuralPilot
                from hunter.settings_for import hunter_settings_for
                from hunter.surroundings import RosterSurroundings
                from creatures.traits.combat import Gun
                setup = PilotSetup(hunter_settings_for('neural'),
                                   NeuralPilot(pilot,config.SETTINGS.game.neuralDecisionFrames),None)
                ship = creatures.build('allied_hunter',self.free_cell(200),level,pilot_setup=setup)
                ship.pilot_setup = replace(setup,surroundings=RosterSurroundings(self.enc,ship))
                strength = self.hunter_strength if stage==5 else 1.
                traits = tuple((name,dict(params,shotStrength=strength) if name=='gun' else params)
                               for name,params in ship.kind.traits)
                ship.kind = replace(ship.kind,traits=traits)
                gun_params = dict(traits)['gun']
                ship.traits = tuple(Gun.from_params(gun_params) if t.NAME=='gun' else t for t in ship.traits)
                self.world.roster.add(ship)
                self.enc.ready()
                if hunter_health is not None:
                    ship.health = hunter_health
                self.ships.append(ship)
        else:
            for _ in range(1+extra_targets):
                speed = (.7 if moving else 0.) if target_speed is None else target_speed
                if not math.isfinite(speed) or speed<0:
                    raise ValueError('practice target speed must be finite and nonnegative')
                t = PracticeTarget(self.practice_position(),rng.uniform(0,math.tau),speed,firing_enabled=False)
                t.health = 8
                t.generation = 0
                t.active = stage != 2 and not regroup
                self.targets.append(t)
                self.world.additional_targets.append(t)
        if stage == 2:
            from creatures.traits.wallcrawl import contact, clear
            crew = self.team.members()
            for c in crew:
                surface = contact(maze,c.get_pos())
                if surface is None:
                    continue
                _,foot,normal,_ = surface
                pos = (foot[0]+normal[0]*(c.radius+3),foot[1]+normal[1]*(c.radius+3))
                if clear(maze,pos,c.radius) and all(o is c or math.dist(pos,o.get_pos())>c.radius+o.radius+2 for o in crew):
                    c.x,c.y = pos
                    c.prev_x,c.prev_y = pos
                    c.angle = math.degrees(math.atan2(normal[1],normal[0]))
        for c in self.team.members():
            c.gun_cooldown = rng.uniform(0,c.gun_interval)
        self.frames = 0
        self.finished = False
        self.counts = Counter()
        self.agent_returns = Counter()
        self.agent_wall_collision_costs = Counter()
        self.agent_hits = Counter()
        self.living = set(self.agent_ids)
        self.trails = {c.agent_id:deque(maxlen=11) for c in self.team.members()}
        self.pushes = {c.agent_id:deque(maxlen=10) for c in self.team.members()}
        self.volley = VolleyTracker()
        self._known_shots = set()
        self.action_trace = []
        self._perched = set()
        self._reacquire_at = None
        self.reacquire_times = []
        self.contact_frame = None
        return self.snapshot()

    def build_flocker(self, position, level, angle=None):
        """Historical runs keep their recorded body size, including new arrivals."""
        import creatures
        c = creatures.build('flocker',position,level,angle=angle)
        if self.flocker_radius is not None:
            c.radius = float(self.flocker_radius)
            c.kind = replace(c.kind,tuning=dict(c.kind.tuning,size=c.radius))
        propulsion = self.flocker_propulsion
        if propulsion is None and self.legacy:
            propulsion = dict(speedShare=.4)  # The unchanged space-flight scripted benchmark.
        if propulsion is not None:
            from creatures.traits.propulsion import Thrusters
            c.traits = tuple(Thrusters.from_params(propulsion) if t.NAME=='thrusters' else t for t in c.traits)
            c.kind = replace(c.kind,traits=tuple((n,propulsion if n=='thrusters' else p) for n,p in c.kind.traits))
            from creatures.trait import Trait
            c.after_frame_traits = tuple(t for t in c.traits if type(t).on_after_frame is not Trait.on_after_frame)
        return c

    def free_cell(self, distance):
        members = self.team.members()
        clearance = lambda p: min((math.dist(p,c.get_pos()) for c in members), default=1000)
        return next((p for p in self.cells if clearance(p)>=distance),max(self.cells,key=clearance))

    def practice_position(self):
        if not self.visible_start:
            return self.free_cell(180)
        from creatures.wallcrawler_team import capsule_hits
        members = self.team.members()
        angle = self.rng.uniform(0,math.tau)
        for distance in (180.,120.,80.,60.,40.):
            for c in members:
                for j in range(16):
                    heading = angle+j*math.tau/16
                    point = c.x+distance*math.cos(heading),c.y+distance*math.sin(heading)
                    maze = self.world.maze
                    if (maze.is_open(point,15) and not wall_between(c.get_pos(),point,maze)
                        and all(math.dist(point,o.get_pos())>=30 for o in members)
                        and not any(w.active and capsule_hits(point,point,15,w.start,w.end)
                                    for w in maze.gate_segments)):
                        return point
        raise ValueError('arena lacks a visible aiming-practice target position')

    def snapshot(self):
        self.team.prepare_decision()
        batch = {c.agent_id:observe(self.team,c) for c in self.team.members()}
        return Snapshot({i:v[0] for i,v in batch.items()},{i:v[1] for i,v in batch.items()},critic(self.team),
                        {i:critic(self.team,i) for i in batch})

    def terminal(self):
        return not self.team.members() or bool(self.ships) and not any(s.active for s in self.ships)

    def fly_shots(self, events):
        for shot in self.world.projectiles:
            shot.update(1)
            if not shot.active:
                continue
            if shot.check_wall_collision(self.world.maze.walls,self.world.maze.spatial_grid) or (
                    self.obstacles and self.obstacles.bullet_hit(shot)):
                shot.active = False
                continue
            if not shot.is_enemy:
                if self.stage==5 and shot.source=='hunter':
                    # Creature hit points use creature_damage, independently of
                    # the projectile strength used for player damage and push.
                    shot.creature_damage = self.hunter_strength
                if self.enc.shoot(shot):
                    shot.active = False
                continue
            for t in self.targets+self.ships:
                if not t.active:
                    continue
                accepted = False
                landed = False
                if any(t is ship for ship in self.ships):
                    before = t.health
                    landed = self.world.collision_handler.handle_projectile_hunter_collision(shot,t)
                    accepted = t.health < before
                elif shot.check_circle_collision(t.get_pos(),t.radius):
                    landed = True
                    shot.active = False
                    accepted = True
                    if self.stage != 3:       # the aiming drill's targets last, so every bird keeps shooting at one
                        t.health -= 1
                        if t.health <= 0:
                            t.active = False
                if landed:
                    self.counts['landed_hits'] += 1
                    self.counts['non_damaging_hits'] += not accepted
                if accepted:
                    i = shot.actor_id
                    self.counts['hits'] += 1
                    if self.contact_frame is None:
                        self.contact_frame = self.frames
                    self.agent_hits[i] += 1
                    events[i]['hits'] += 1
                    self.volley.hit(shot,t)
                    if not t.active:
                        self.counts['kills'] += 1
                        events[i]['kills'] += 1
                        if any(t is target for target in self.targets):
                            t.health, t.active = 8, True
                            t.generation += 1
                            t.x,t.y = self.practice_position()
                if not shot.active:
                    break
        self.world.projectiles = [s for s in self.world.projectiles if s.active]

    def register_shots(self, events):
        members = {c.agent_id:c for c in self.team.members()}
        # The set retains only live shots: object identity is never serialized or allowed to be reused while tracked.
        self._known_shots.intersection_update(self.world.projectiles)
        for shot in self.world.projectiles:
            if not shot.is_enemy or shot in self._known_shots:
                continue
            self._known_shots.add(shot)
            c = members.get(getattr(shot,'actor_id',None))
            if c is None:
                continue
            self.counts['shots'] += 1
            events[c.agent_id]['shots'] += 1
            t = getattr(shot,'flocker_target',None) or self.enc.target_entity_for(c)
            if t is None:
                continue
            eligible = {o.agent_id for o in members.values() if (o is c or o.agent_id in self._ready_at_launch)
                        and math.dist(o.get_pos(),t.get_pos())<=o.gun_range
                        and not wall_between(muzzle(o)[0],t.get_pos(),self.world.maze)}
            self.volley.launch(shot,t,c.agent_id,self.frames,eligible)

    def step(self, actions):
        if self.finished:
            raise RuntimeError('step after episode ended')
        # Per bird, the counts each reward term pays on (learning.flocker_rewards); paid once the decision ends. A bird
        # that joins mid-decision gets its row when it first counts something.
        events = self.event_rows(self.agent_ids)
        approaches = Counter()
        members = self.team.members()
        before_wall_hits = {c.agent_id:c.flocker_hard_landings for c in self.enc.creatures()
                            if hasattr(c,'flocker_hard_landings') and hasattr(c,'agent_id')}
        before = {}
        aim_before = {}
        for c in members:
            seen = visible(self.enc,c)
            target = min(seen,key=lambda t:(t is not c.flocker_target,math.dist(c.get_pos(),t.get_pos())),default=None)
            report = self.team.target_report(c) if target is None else None
            point = tuple(target.get_pos()) if target is not None else report['position'] if report else None
            if point is not None:
                before[c.agent_id] = (point,math.dist(c.get_pos(),point))
            if self.rewards['aim_progress_reward'] and target is not None and getattr(target,'immunity_remaining',0.)<=0:
                from learning.flocker_aim import intercept_point,normalized_error
                aim_point = intercept_point(c,target,self.world.maze)
                if aim_point is not None:
                    aim_before[c.agent_id] = (aim_point,normalized_error(c,aim_point))
        if self.rewards['threat_penalty']:
            from learning.flocker_threat import danger,hostile_shots
            shots = hostile_shots(self.enc)
            for c in members:
                if any(danger(c,shot) for shot in shots):
                    events[c.agent_id]['threatened_decisions'] += 1
                    self.counts['threat_decisions'] += 1
        if not self.scripted:
            self.team.commit({c.agent_id:intention(actions[c.agent_id],c.flocker_perched)
                              for c in members if c.agent_id in actions})
        self.action_trace.append({str(i):list(map(float,a)) for i,a in actions.items()})
        for _ in range(DECISION_FRAMES):
            if self.frames >= self.horizon*60 or self.terminal():
                break
            self._ready_at_launch = {c.agent_id for c in self.team.members() if c.gun_cooldown<=1}
            self.enc.step(1.)
            for t in self.targets:
                if t.active:
                    t.step(self.world.maze)
            self.register_shots(events)
            self.fly_shots(events)
            for c in members:
                lost = self.hp.setdefault(c.agent_id,c.hit_points)-c.hit_points
                if lost > 0:
                    events[c.agent_id]['hit_points_lost'] += lost
                    self.counts['damage_taken'] += lost
                self.hp[c.agent_id] = c.hit_points
            if self.obstacles:
                self.obstacles.step(1/60,self.world.roster.alive())
                self.world.maze.gate_segments = self.obstacles.segments()
            live = {c.agent_id for c in self.team.members()}
            for i in self.living-live:
                events[i]['deaths'] += 1
                self.counts['deaths'] += 1
            self.living = live
            self.frames += 1
            for t in self.targets:
                t.active = not self.regroup and (self.stage != 2 or self.frames % 600 >= 300)
            if self.stage==2 and self.frames%600==300:
                self._reacquire_at = self.frames
            self._settle_volleys(self.frames,events)
            self.world.roster.update_dying(1)
            if self.render_frames is not None:
                self.render_frames(self)
            self.sample_motion(events,approaches)
            if self.reinforcements and self.frames >= 120:
                self.add_reinforcements()
        for c in self.team.members():
            old = before.get(c.agent_id)
            if old:
                # Hold the observed position fixed: only the bird's own progress
                # counts, never target motion, waypoint teleportation or hidden tracking.
                closing = max(-20.,min(20.,old[1]-math.dist(c.get_pos(),old[0])))
                events[c.agent_id]['closing_pixels'] += closing
                self.counts['pursuit_progress_reward'] += self.rewards['progress_reward']*closing
            if c.agent_id in aim_before:
                from learning.flocker_aim import normalized_error
                point,error = aim_before[c.agent_id]
                improvement = error-normalized_error(c,point)
                events[c.agent_id]['aim_improvement'] += improvement
                self.counts['aim_progress_reward'] += self.rewards['aim_progress_reward']*improvement
            trail, pushes = self.trails[c.agent_id],self.pushes[c.agent_id]
            trail.append(c.get_pos())
            pushes.append(c.flocker_throttle>.1 and not c.flocker_perched)
            if len(trail)==11 and all(pushes) and math.dist(trail[0],trail[-1])<10:
                events[c.agent_id]['stuck_decisions'] += 1
                self.counts['stuck_agent_frames'] += DECISION_FRAMES
        for c in self.enc.creatures():
            if not hasattr(c,'flocker_hard_landings') or not hasattr(c,'agent_id'):
                continue
            hits = c.flocker_hard_landings-before_wall_hits.get(c.agent_id,0)
            cost = max(0,hits)*self.rewards['wall_collision_penalty']
            if cost:
                events[c.agent_id]['wall_impacts'] += max(0,hits)
                self.agent_wall_collision_costs[c.agent_id] += cost
        rewards = self.paid(events,approaches)
        for i,r in rewards.items():
            self.agent_returns[i] += r
        terminated = self.terminal()
        truncated = self.frames >= self.horizon*60 and not terminated
        self.finished = terminated or truncated
        if self.finished:
            final_events = self.event_rows()
            self._settle_volleys(self.frames+7,final_events)
            for i,value in self.paid(final_events).items():
                rewards[i] += value
                self.agent_returns[i] += value
        return Transition(self.snapshot(),sum(rewards.values()),terminated,truncated,self.events(),dict(rewards))

    def _settle_volleys(self, frame, events):
        """Count each shooter's share of the volleys that are settled by `frame`."""
        credit = Counter()
        self.volley.settle(frame,credit,1.)
        for i,share in credit.items():
            events[i]['volley_credit'] += share

    @staticmethod
    def event_rows(birds=()):
        """Per bird, a decision's event counts (learning.flocker_rewards), all 0; a bird not listed gets its row when it
        first counts something."""
        return defaultdict(TABLE.row, {i:TABLE.row() for i in birds})

    def paid(self, events, approaches=None):
        """{bird: what its counted events pay at this run's reward settings, plus its stage-1 approach bonuses}."""
        ids = list(events)
        paid = TABLE.pay([events[i] for i in ids],self.rewards)
        rewards = Counter(dict(zip(ids,map(float,paid))))
        for i,n in (approaches or {}).items():
            rewards[i] += APPROACH_REWARD*n
        return rewards

    def sample_motion(self, events, approaches=None):
        approaches = Counter() if approaches is None else approaches
        members = self.team.members()
        if self._reacquire_at is not None and any(visible(self.enc,c) for c in members):
            self.reacquire_times.append((self.frames-self._reacquire_at)/60)
            self._reacquire_at = None
        for c in members:
            i = c.agent_id
            self.counts['agent_frames'] += 1
            self.counts['throttle_sum'] += c.flocker_throttle
            self.counts['fractional_throttle_frames'] += .05<c.flocker_throttle<.95
            near = [o for o in members if o is not c and math.dist(c.get_pos(),o.get_pos())<100]
            if self.rewards['contact_only']:
                bunched = any(o is not c and math.dist(c.get_pos(),o.get_pos())<=c.radius+o.radius
                              for o in members)
            else:
                bunched = any(math.dist(c.get_pos(),o.get_pos())<max(self.rewards['bunch_distance'],c.radius+o.radius+2)
                              for o in near)
            self.counts['bunched_frames'] += bunched
            self.counts['cohesive_frames'] += bool(near)
            if bunched:
                events[i]['bunched_decisions'] += 1/6
            else:
                # Credit the bird's own movement towards a locally sensed neighbour.
                # A target is unnecessary; passive convergence earns no progress credit.
                neighbours = [(math.dist(c.get_pos(),o.get_pos()),o) for o in members if o is not c
                              and math.dist(c.get_pos(),o.get_pos())<=550.
                              and not wall_between(c.get_pos(),o.get_pos(),self.world.maze)]
                if neighbours:
                    distance, neighbour = min(neighbours,key=lambda item:item[0])
                    if distance>=100:
                        closing = ((neighbour.x-c.x)*c.vx+(neighbour.y-c.y)*c.vy)/max(distance,.001)
                        events[i]['cohesion'] += max(-1.,min(1.,closing/max(c.max_speed,.01)))/6
                    else:
                        close = [o for d,o in neighbours if d<100]
                        speed = math.hypot(c.vx,c.vy)
                        alignment = 0.
                        if speed>.5:
                            agreement = sum((c.vx*o.vx+c.vy*o.vy)/(speed*math.hypot(o.vx,o.vy))
                                            for o in close if math.hypot(o.vx,o.vy)>.5)
                            alignment = max(0.,min(1.,agreement/len(close)))
                        self.counts['alignment_sum'] += alignment
                        events[i]['cohesion'] += alignment/6
            if c.flocker_perched:
                self.counts['perched_frames'] += 1
                if i not in self._perched:
                    self.counts['perches'] += 1
                    if not c.flocker_perch_paid:
                        events[i]['perches'] += 1
                        c.flocker_perch_paid = True
            self._perched = (self._perched | {i}) if c.flocker_perched else (self._perched-{i})
            if self.stage == 1:
                ts = visible(self.enc,c)
                t = min(ts,key=lambda t:math.dist(c.get_pos(),t.get_pos()),default=None)
                if t is not None and math.dist(c.get_pos(),t.get_pos())<100:
                    self.counts['approaches'] += 1
                    if self.contact_frame is None:
                        self.contact_frame = self.frames
                    approaches[i] += 1
                    t.x,t.y = self.free_cell(250)

    def add_reinforcements(self):
        import creatures
        for _ in range(self.reinforcements):
            i = self.team._next_id
            c = self.build_flocker(self.free_cell(60),self.level)
            c.agent_id = i
            self.world.roster.add(c)
            self.enc.ready()
            self.team.members()
            self.agent_ids += (i,)
            self.living.add(i)
            self.trails[i],self.pushes[i] = deque(maxlen=11),deque(maxlen=10)
            self.team_size += 1
        self.reinforcements = 0

    def events(self):
        n = max(1,self.counts['agent_frames'])
        v = self.volley.metrics()
        all_members = [c for c in self.enc.creatures() if hasattr(c,'flocker_perch_attempts')]
        attempts = sum(c.flocker_perch_attempts for c in all_members)
        collision_credit = (dict(wall_collision_cost=sum(self.agent_wall_collision_costs.values()),
                                 agent_wall_collision_costs=dict(self.agent_wall_collision_costs))
                            if self.rewards['wall_collision_penalty'] else {})
        return dict(self.counts,**v,**collision_credit,perch_attempts=attempts,
                    landed_hits=self.counts['landed_hits'],non_damaging_hits=self.counts['non_damaging_hits'],
                    perch_success=self.counts['perches']/max(1,attempts),
                    hard_landings=sum(c.flocker_hard_landings for c in all_members),
                    max_wake_latency=max((c.flocker_wake_latency for c in all_members),default=0),
                    reacquire_times=self.reacquire_times+([(self.frames-self._reacquire_at)/60] if self._reacquire_at is not None else []),
                    time_to_contact=(self.contact_frame if self.contact_frame is not None else self.frames)/60,
                    seconds=self.frames/60,team_size=self.team_size,stage=self.stage,
                    alive=len(self.team.members()),agent_returns=dict(self.agent_returns),
                    survival_fraction=len(self.team.members())/self.team_size,
                    cohesion=self.counts['cohesive_frames']/n,alignment=self.counts['alignment_sum']/n,
                    bunched_fraction=self.counts['bunched_frames']/n,
                    stuck_fraction=self.counts['stuck_agent_frames']/n,
                    mean_throttle=self.counts['throttle_sum']/n,perched_fraction=self.counts['perched_frames']/n,
                    fractional_throttle_fraction=self.counts['fractional_throttle_frames']/n)

    def close(self):
        if self.world is not None:
            self.enc.close()
            self.world = None
        if hasattr(self,'_screen_settings'):
            import config
            config.SCREEN_WIDTH,config.SCREEN_HEIGHT = self._screen_settings
            del self._screen_settings


def environment(**options):
    return FlockerEnvironment(**options)


def environment_options(manifest):
    return dict(rewards=flocker_rewards.from_manifest(manifest),
                opponent_controller=manifest.get('opponent_controller','neural'),
                flocker_radius=manifest.get('creature',{}).get('tuning',{}).get('size'),
                flocker_propulsion=dict(manifest.get('creature',{}).get('traits',())).get('thrusters'))


def metadata(args):
    if getattr(args, 'bootstrap_episodes', 0):
        raise ValueError('Flocker training starts without imitation; use --bootstrap-episodes 0')
    import config
    import enemy_kinds
    import level_config
    opponent = Path(getattr(args,'opponent',SHIP_PILOT))
    from hunter.settings_for import hunter_settings_for
    return dict(task=SPEC.task,scenario_version=13,spec=SPEC.metadata(),horizon=args.horizon,
                aiming_practice=dict(speeds=[0.,1.,2.,3.,4.],visible_starts=True,durable_targets=True),
                combat_ramp=[dict(strength=a,health=b) for a,b in HUNTER_RAMP],
                opponent_controller='neural',opponent_settings=asdict(hunter_settings_for('neural')),
                rewards=flocker_rewards.from_args(args),decision_frames=6,
                creature=asdict(enemy_kinds.kind('flocker')),ship=asdict(enemy_kinds.kind('allied_hunter')),
                training_shot_strength=.5,full_shot_strength=1.,
                physics={n:asdict(getattr(config.SETTINGS,n)) for n in ('maze','obstacles','ship','hunterShip','projectile','powerups')},
                controls=dict(head_rate=180,head_limit=90,throttle='unit_interval',sight=550,radio=300,
                              alert_frames=60,perch_quiet_frames=60,volley_frames=6),
                arenas={str(n):level_config.load_level_config(n) for n in TRAIN_LEVELS+HOLDOUT_LEVELS},
                opponent_sha256=hashlib.sha256(opponent.read_bytes()).hexdigest() if opponent.is_file() else None)


# Combat opens against a weakened hunter and strengthens: damage to a bird per hit, and the hunter's hit points.
# Held-out validation and evaluation always meet the full-strength hunter.
HUNTER_RAMP = ((.125,4),(.25,7),(.5,10))


def training_cases(iteration,count,stage,bootstrap=False,curriculum=None):
    stage = curriculum['stage'] if curriculum else stage
    cases = []
    for j in range(count):
        i = iteration*count+j
        s = 1+(i//4)%(stage-1) if stage>1 and i%4==0 else stage
        pursuit_index = i if stage==1 else (i//4)//(stage-1)
        aim_index = i if stage==3 else (i//4)//max(1,stage-1)
        target_speed = (0.,1.,2.,3.,4.)[aim_index%5] if s==3 else None
        cases.append(dict(level=TRAIN_LEVELS[i%len(TRAIN_LEVELS)],seed=(100 if bootstrap else 300)+i,
                          stage=s,team_size=(4,6,8)[i%3],moving=target_speed>0 if s==3 else bool(i%2),
                          target_speed=target_speed,visible_start=s==3,
                          regroup=s==1 and bool(pursuit_index%2),
                          split_group=s==1 and pursuit_index%4==3,
                          compact_group=s==1 and pursuit_index%4!=3,
                          **(dict(zip(('hunter_strength','hunter_health'),HUNTER_RAMP[(i//3)%3])) if s==5 else {})))
    return cases


def validation_cases():
    return [dict(level=n,seed=900+j,stage=s,team_size=(2,4,6)[j%3],moving=s==3 or bool(j%2),regroup=s==1 and j==1,
                 target_speed=(1.,3.,4.)[j] if s==3 else None,visible_start=s==3)
            for s in range(1,7) for j,n in enumerate(HOLDOUT_LEVELS)]


def evaluation_cases(stage=None):
    return [dict(level=n,seed=seed,stage=s,team_size=size,moving=True,
                 extra_targets=int(s>=4 and seed==1001),reinforcements=2 if seed==1001 and size<=6 else 0,
                 split_group=seed==1001)
            for s in ([stage] if stage else range(1,7)) for n in HOLDOUT_LEVELS
            for seed in (1000,1001) for size in (2,4,6,8)]


def metrics(rows):
    seconds = sum(r.get('seconds',0) for r in rows)
    shots,hits = (sum(r.get(k,0) for r in rows) for k in ('shots','hits'))
    volleys = sum(r.get('volleys',0) for r in rows)
    spreads = [v for r in rows for v in r.get('volley_spreads',[])]
    result = dict(episodes=len(rows),seconds=seconds,hits=hits,shots=shots,accuracy=hits/max(1,shots),
                  aim_accuracy=sum(r.get('landed_hits',r.get('hits',0)) for r in rows)/max(1,shots),
                  hits_per_minute=hits*60/max(.001,seconds),
                  approaches_per_minute=sum(r.get('approaches',0) for r in rows)*60/max(.001,seconds),
                  useful_volleys_per_minute=sum(r.get('useful_volleys',0) for r in rows)*60/max(.001,seconds),
                  participation=sum(r.get('participation_sum',0) for r in rows)/max(1,volleys),
                  median_launch_spread=float(np.median(spreads)) if spreads else 7.,volleys=volleys,
                  perches=sum(r.get('perches',0) for r in rows),kills=sum(r.get('kills',0) for r in rows))
    result['perch_success'] = result['perches']/max(1,sum(r.get('perch_attempts',0) for r in rows))
    result['max_wake_latency'] = max((r.get('max_wake_latency',0) for r in rows),default=0)
    times = [t for r in rows for t in r.get('reacquire_times',[])]
    result['reacquisition_seconds'] = float(np.mean(times)) if times else 0.
    result['hard_landings'] = sum(r.get('hard_landings',0) for r in rows)
    result['wall_collision_cost'] = sum(r.get('wall_collision_cost',0) for r in rows)
    for k in ('cohesion','alignment','bunched_fraction','stuck_fraction','survival_fraction','mean_throttle',
              'fractional_throttle_fraction','perched_fraction'):
        result[k] = float(np.mean([r.get(k,0) for r in rows])) if rows else 0.
    for s in range(1,7):
        part = [r for r in rows if r.get('stage')==s]
        if part:
            result[f'stage{s}_episodes'] = len(part)
    return result


def selection_score(summary):
    # Selection tracks progress; deployment has a separate hard gate against both baselines.
    return summary['hits_per_minute']+summary['useful_volleys_per_minute']+summary['approaches_per_minute']*.1


def compare(a,b):
    rows = [(x,y) for x,y in zip(a['episodes'],b['episodes'])]
    result = {}
    for key in ('hits','hits_per_minute','useful_volleys','useful_volleys_per_minute','survival_fraction','cohesion','reacquisition_seconds'):
        values = np.array([(float(np.mean(x.get('reacquire_times',[]) or [0]))-float(np.mean(y.get('reacquire_times',[]) or [0])))
                           if key=='reacquisition_seconds' else
                           x.get('hits',0)*60/max(.001,x['seconds'])-y.get('hits',0)*60/max(.001,y['seconds'])
                           if key=='hits_per_minute' else
                           x.get('useful_volleys',0)*60/max(.001,x['seconds'])-y.get('useful_volleys',0)*60/max(.001,y['seconds'])
                           if key=='useful_volleys_per_minute' else x.get(key,0)-y.get(key,0) for x,y in rows])
        samples = np.random.default_rng(0).choice(values,(2000,len(values)),replace=True).mean(axis=1) if len(values) else np.zeros(1)
        result[key] = dict(mean=float(values.mean()) if len(values) else 0.,lower_95=float(np.quantile(samples,.025)),
                           upper_95=float(np.quantile(samples,.975)))
    return result


def promote(stage, summary):
    return stage


def initial_curriculum(args):
    return dict(stage=1,passing_blocks=0)


def advance_curriculum(state, summary):
    state = dict(state)
    s = state['stage']
    # Fixed held-out baseline metrics are supplied by the validation callback below.
    if 'flocker_validation_passed' not in summary:
        return state
    passed = bool(summary['flocker_validation_passed'])
    state['passing_blocks'] = state['passing_blocks']+1 if passed else 0
    if state['passing_blocks']>=2 and s<6:
        state.update(stage=s+1,passing_blocks=0)
    return state


def curriculum_key(state):
    return STAGES[state['stage']-1]


def curriculum_validation_cases(state):
    return [c for c in validation_cases() if c['stage']==state['stage']]


def curriculum_evaluation_cases(state):
    return evaluation_cases()


def curriculum_selection_score(summary,state):
    s = state['stage']
    # Pursuit alone can select isolated hunters. Bound its contribution so a
    # cohesive flock remains the primary criterion for the first lesson.
    score = (summary['cohesion']*(1.+min(10.,summary['approaches_per_minute'])/10.) if s==1 else
             summary['perches']*summary['perch_success'] if s==2 else
             summary['aim_accuracy']*summary['hits_per_minute'] if s==3 else selection_score(summary))
    return score if score>0 else -math.inf

_baselines = {}


def scripted_report(cases,horizon,opponent=None,legacy=False,rewards=None,arenas=None,deadline=math.inf):
    env = environment(horizon=horizon,opponent=opponent,scripted=True,legacy=legacy,rewards=rewards,arenas=arenas)
    rows = []
    try:
        for case in cases:
            case = dict(case)
            seed = case.pop('seed')
            env.reset(seed,**case)
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Flocker baseline exhausted the run budget')
                result = env.step({})
                if result.terminated or result.truncated:
                    break
            rows.append(dict(result.events,**case))
    finally:
        env.close()
    return dict(summary=metrics(rows),episodes=rows)


def validation_evidence(summary,state,args,deadline=math.inf):
    cases = curriculum_validation_cases(state)
    key = (state['stage'],args.horizon,str(args.opponent))
    if key not in _baselines:
        _baselines[key] = scripted_report(cases,args.horizon,args.opponent,arenas=run_arenas(args),deadline=deadline)['summary']
    baseline = _baselines[key]
    s = state['stage']
    if s == 1:
        passed = (summary['approaches_per_minute']>0 and summary['approaches_per_minute']>=baseline['approaches_per_minute']
                  and summary['cohesion']>=baseline['cohesion']-.05
                  and summary['bunched_fraction']<=baseline['bunched_fraction']+.02)
    elif s == 2:
        passed = summary.get('perch_success',0)>=.9 and summary['perches']>0 and summary.get('max_wake_latency',7)<=6
    elif s == 3:
        required_accuracy = max(baseline['accuracy'],state.get('minimum_aim_accuracy',0.))
        passed = summary['hits']>0 and summary['accuracy']>=required_accuracy and summary['hits_per_minute']>=baseline['hits_per_minute']
    elif s == 4:
        passed = (summary['volleys']>0 and summary['useful_volleys_per_minute']>0
                  and summary['participation']>=.6 and summary['median_launch_spread']<=6)
    else:
        passed = (summary['hits']>0 and summary['hits_per_minute']>=baseline['hits_per_minute']
                  and summary['survival_fraction']>=max(baseline['survival_fraction'],state.get('minimum_survival',0.)))
    return dict(flocker_validation_passed=passed)


def deployment_report(reports,args,deadline=math.inf):
    legacy = scripted_report(evaluation_cases(),args.horizon,args.opponent,legacy=True,arenas=run_arenas(args),deadline=deadline)
    comparisons = dict(bounded=compare(reports['trained'],reports['scripted']),legacy=compare(reports['trained'],legacy))
    # A confidence interval crossing the required margin is inconclusive, never promotion.
    passed = all(c['useful_volleys_per_minute']['lower_95']>0 and c['hits_per_minute']['lower_95']>=0
                 and c['survival_fraction']['lower_95']>=-.05 and c['cohesion']['lower_95']>=-.05 and c['reacquisition_seconds']['upper_95']<=1.
                 for c in comparisons.values())
    return dict(passed=passed,comparisons=comparisons,legacy=legacy,
                message='Eligible for explicit packaging.' if passed else 'Keep scripted campaign default.')


def run_arenas(args):
    """Baselines use the same immutable arena snapshots as rollout workers."""
    path = Path(args.out)/'config.json'
    return json.loads(path.read_text())['manifest']['arenas'] if path.is_file() else None
