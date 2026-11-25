"""Single-worm physical drills and a gated recurrent PPO curriculum."""
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np

from learning.environment import Snapshot, Transition
from learning.adapters.crawler_scenario import SilentSound
from learning.adapters.inchworm import (load_transfer_policy, SPEC, CRITIC_SIZE, DECISION_FRAMES, INFERENCE_ACTION_MODE, observe, intention, bend_signs,
                                        action_for, enabled, load_policy)
from learning import inchworm_rewards
from learning.inchworm_rewards import TABLE
from physics.inchworm import MAX_BEND

INITIAL_STAGE = 1
USES_OPPONENT = False
EVALUATION_MODES = ('trained', 'scripted')
EVALUATION_ACTION_MODE = INFERENCE_ACTION_MODE
IMITATION_HEADS = tuple(n for n,_ in SPEC.heads)
IMITATION_EPOCHS = 4
# The scripted demonstrations are rehearsed during PPO, not only cloned at the start: About-face is reached hundreds
# of iterations after the cloning, and a policy that has forgotten how to release a grip freezes there.
IMITATION_REHEARSAL_WEIGHT = .2
IMITATION_CLONE_OVER_START = False
STAGES = ('crawl','turn','reach','feed','recover','rear','curl','about-face','gap','pursue','leap')
# Preserve existing stage identities in saved episode records.
CURRICULUM = (1,6,7,2,8,9,3,4,5,11,10)       # Leap (11) comes before Pursue (10)
DEMONSTRATED = tuple(s for s in CURRICULUM if s not in (9,10))+(11,11)   # the scripted stride cannot cross a gap or chase across a cave; the whiplash leaps, and its short demonstrations come up three times as often
# Pixels from the first joint, which the head sweeps about 49 px around: far targets are
# the longest a pure turn reaches (a flat target past about 58 px is out of reach).
TURN_REACH = (12., 58.)
TURN_MAX_ANGLE = 80.
SUCCESS_GATES = {1:.95,6:.8,7:.8,2:.8,8:.8,9:.7,3:.8,4:.75,5:.65,10:.6,11:.7}
PASSES_REQUIRED = {1:5}      # Crawl is practised longer: every later lesson builds on its gait
CRAWL_MAX_SECONDS = 80.      # mean time to the goal; the scripted crawl now averages about 10 s over the held-out goals
CRAWL_MAX_AIRBORNE = .5      # leaping is allowed; a crawl that is mostly airborne is not
PITCH_SLACK = math.radians(15.)    # a head or tail link this far off parallel to the ground is not penalised
GAP_EDGE = (25.,60.)       # Gap lesson: the first island ends this far ahead of the arena origin (the worm starts 40-70 behind it)
GAP_WIDTH = (14.,42.)      # the empty space between the islands; a worm reaches about 50 px with its front foot
GAP_RISE = (-12.,12.)      # the second island sits this much lower or higher
GAP_GOAL_BEYOND = (55.,110.)   # the marker lies this far past the second island's edge
GAP_BLOCK = 43.            # islands are solid blocks this thick (a game cell), with an underside and an end face each
ROUGH_STAGES = (1,)       # lessons whose floor can be uneven; the goal is placed on the surface
ROUGH_SHARE = .6          # fraction of such episodes that get uneven ground (seeded, so held-out ones do too)
ROUGH_FLAT_RUN = 70.      # the worm starts on level ground, then the terrain begins
ROUGH_NODE = 35.          # spacing of the height points the floor is drawn between
ROUGH_STEP = 5.           # typical height change per point
ROUGH_RELIEF = 16.        # the floor stays within this of level
ROUGH_SLOPE = math.tan(math.radians(28.))   # and no segment is steeper than this
# Pursue: the worm chases a drifting target through the real cave mazes of the levels it hunts on.
PURSUE_TRAIN_LEVELS = (13,19)       # level 16 is left out: its sealed-off areas can be neither entered nor escaped
PURSUE_VALIDATION_LEVELS = (22,26)
PURSUE_HOLDOUT_LEVELS = (29,)
PURSUE_LEVELS = PURSUE_TRAIN_LEVELS+PURSUE_VALIDATION_LEVELS+PURSUE_HOLDOUT_LEVELS
LEAP_ALONG = (15.,70.)           # Leap: the target floats this far ahead along the surface of the worm...
LEAP_OUTWARD = (80.,140.)        # ...and this far off it: a worm that only reaches catches nothing past about 60 px, so only a leap succeeds
LEAP_PROTECTED = (8,9)            # while Leap trains, an eighth of every batch goes to About-face and Gap, which slip the most
LEAP_REHEARSAL = (7,9,3,8)        # while Leap trains, the rotation through earlier lessons repeats the ones it was forgetting: Curl, Gap, Reach and About-face
LEAP_AIM_TOLERANCE = math.radians(30.)   # the aim reward falls to nothing this far off the marker (about the catch radius at the far end)
LEAP_BODY_SLACK = 13.            # a body part must be this much nearer than the mouth to count: the marker's own size
LEAP_CATCH = 25.                 # the mouth is this close to the marker; generous while the lesson is young, so catches are common
LEAP_SECONDS = 20.               # the time allowed to launch at all
LEAP_MIN_TRAVEL = 40.            # a flight that carries the body this far from where it let go is a leap; touching the wall again before that is a step, not a landing
LEAP_RECEDE = 40.                # flying this far beyond the closest approach so far is a miss too
PURSUE_DROPS = (5,)              # once Pursue starts, practice stops rehearsing Recover
PURSUE_DISTANCE = (90.,240.)     # the target starts this far from the worm, in plain sight
PURSUE_SPEED = (.15,.45)         # px per frame; slower than a walking worm, so a chase can be won
PURSUE_CATCH = 10.               # the mouth is this close to the target's edge
TRAIN_LEVELS = (0,1,2,3)
VALIDATION_LEVELS = (4,5,6,7)
HOLDOUT_LEVELS = (8,9,10,11)
ARENAS = {str(i):dict(angle=angle,half_length=350.) for i,angle in enumerate(
    (0.,90.,180.,270.,22.5,112.5,202.5,292.5,45.,135.,225.,315.))}


class InchwormEnvironment:
    spec = SPEC
    adapter = sys.modules[__name__]

    def __init__(self,horizon=120.,arenas=None,scripted=False,rewards=None,opponent=None,muted=False):
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError('inchworm horizon must be finite and positive')
        self.horizon, self.arenas = horizon, arenas or ARENAS
        self.scripted = scripted
        self.rewards = inchworm_rewards.validate(rewards or {})
        self.world = None
        self.render_frames = None
        self.enc = self.pursuit_target = self.obstacles = self._screen_settings = None

    @staticmethod
    def terrain(seed,stage,half):
        """Floor profile as (along, height) points. Level ground, or in some episodes a bumpy floor after a flat run-up."""
        flat = [(-half,0.),(half,0.)]
        rng = random.Random(seed*7919+13)      # its own stream: the rest of the episode draws exactly as before
        if stage not in ROUGH_STAGES or rng.random() >= ROUGH_SHARE:
            return flat
        along = [-half]+[x for x in np.arange(ROUGH_FLAT_RUN,half,ROUGH_NODE)]+[half]
        points,height = [(-half,0.),(ROUGH_FLAT_RUN-ROUGH_NODE,0.)],0.
        for x in along[1:]:
            if x < ROUGH_FLAT_RUN:
                continue
            gap = x-points[-1][0]
            height += rng.gauss(0.,ROUGH_STEP)
            limit = min(ROUGH_RELIEF,abs(ROUGH_SLOPE)*gap)
            height = max(points[-1][1]-limit,min(points[-1][1]+limit,height))
            height = max(-ROUGH_RELIEF,min(ROUGH_RELIEF,height))
            points.append((float(x),height))
        return points

    @staticmethod
    def gap_course(seed):
        """The Gap lesson's islands: where the first ends, how wide the gap is, and how far the second is raised."""
        rng = random.Random(seed*104729+7)
        return dict(edge=rng.uniform(*GAP_EDGE),width=rng.uniform(*GAP_WIDTH),rise=rng.uniform(*GAP_RISE))

    @classmethod
    def islands(cls,seed,stage,half):
        """The floor as separate runs of (along, height) points: one, or for the Gap lesson two islands."""
        if stage != 9:
            return [cls.terrain(seed,stage,half)]
        g = cls.gap_course(seed)
        return [[(-half,0.),(g['edge'],0.)],[(g['edge']+g['width'],g['rise']),(half,g['rise'])]]

    def surface_height(self,along):
        for run in self.islands_built:
            if run[0][0] <= along <= run[-1][0]:
                xs,ys = zip(*run)
                return float(np.interp(along,xs,ys))
        return 0.

    def reset(self,seed,level=0,stage=1):
        import creatures
        from creatures.encounter import Encounter
        from creatures.inchworm_control import InchwormController
        from creatures.traits.combat import TakesHits
        from game_handlers.enemy_roster import EnemyRoster
        from maze.wall_segment import WallSegment
        self.close()
        if stage not in CURRICULUM:
            raise ValueError('inchworm stage must be one of the curriculum lessons')
        self.seed, self.level, self.stage = seed,level,stage
        self.rng = rng = random.Random(seed)
        if stage == 10:
            return self.reset_pursue(seed,level)
        angle = math.radians(self.arenas[str(level)]['angle'])
        self.forward, self.normal = (math.cos(angle),math.sin(angle)),(math.sin(angle),-math.cos(angle))
        self.origin = (650.,430.)
        def point(along,out):
            return (self.origin[0]+self.forward[0]*along+self.normal[0]*out,
                    self.origin[1]+self.forward[1]*along+self.normal[1]*out)
        half = self.arenas[str(level)]['half_length']
        self.islands_built = self.islands(seed,stage,half)
        self.heights = self.islands_built[0]
        if stage == 9:
            outlines = []
            for (x0,top),(x1,_) in (run[0::len(run)-1] for run in self.islands_built):
                corners = [(x0,top),(x1,top),(x1,top-GAP_BLOCK),(x0,top-GAP_BLOCK)]
                outlines += list(zip(corners,corners[1:]+corners[:1]))      # top, far face, underside, near face
        else:
            outlines = [(a,b) for run in self.islands_built for a,b in zip(run,run[1:])]
        walls = [WallSegment(point(*a),point(*b),1,destructible=False) for a,b in outlines]
        maze = SimpleNamespace(walls=walls,gate_segments=[],spatial_grid=None)
        self.world = SimpleNamespace(ship=None,hunter=None,maze=maze,roster=EnemyRoster(),
            inchworm_policy=None,inchworm_posture=(0.,)*5,additional_targets=[],projectiles=[],worm_goal=None,
            sound_manager=SimpleNamespace(play_enemy_destroy=lambda:None),
            collision_handler=SimpleNamespace(drop_pickup=lambda *a:None),powerup_crystals=[],
            jev_beacon_allowed=lambda:False,command_recorder=None)
        self.worm = c = creatures.build('inchworm',point(rng.uniform(-70,-40),14.),angle=math.degrees(angle))
        c.agent_id = 0
        self.world.roster.add(c)
        self.enc = Encounter(self.world.roster,self.world)
        self.enc.ready()
        self.curl_anchor = None
        self.curl_anchor_lost = False
        if stage == 7:
            from physics.inchworm import InchwormCommand
            # Start this supported bend with an ordinary, breakable foot grip.
            # One normal physics frame plants it; no body positions are assigned.
            c.worm_command = InchwormCommand(rear_grip=True)
            self.enc.step(1.)
            c.worm_command = None
            self.curl_anchor = c.worm_physics.grips[0]
            if self.curl_anchor is None:
                raise RuntimeError('curl drill could not plant its starting rear grip')
        self.start = c.get_pos()
        self.goal_direction = -1 if stage in (2,8) else 1
        distance = rng.uniform(60.,90.)
        if stage == 1:
            # Short drills retain the basic gait; every held-out crawl is sustained.
            short = level in TRAIN_LEVELS and rng.random() < .25
            if not short:
                distance = rng.uniform(180.,300.)
        if stage == 8:
            # Turn about the first joint, then walk head-first. A goal within about 65 px is reached by curling back
            # alone (the head arrives with the rear still planted), which teaches nothing about releasing the rear
            # and unfolding; from 90 px the best a curl does is about 36 px short.
            distance = rng.uniform(90.,200.)
        self.world.worm_goal = (c.x+self.forward[0]*distance*self.goal_direction,
                                c.y+self.forward[1]*distance*self.goal_direction)
        if stage in ROUGH_STAGES:
            # The marker rides on the (possibly uneven) surface, as high above it as a worm resting on it.
            along = (c.x-self.origin[0])*self.forward[0]+(c.y-self.origin[1])*self.forward[1]+distance*self.goal_direction
            self.world.worm_goal = point(along,self.surface_height(along)+14.)
        if stage == 9:
            # Across the gap: the marker sits on the second island, past its edge.
            g = self.gap_course(seed)
            along = g['edge']+g['width']+rng.uniform(*GAP_GOAL_BEYOND)
            self.world.worm_goal = point(along,self.surface_height(along)+14.)
        if stage == 2:
            # The worm turns about its first joint. Targets sit within the reach of
            # that pivot (TURN_REACH from it, near to far) on an arc from straight
            # behind up to TURN_MAX_ANGLE above the surface. Near targets reward
            # curling round; squaring the angle draw favours low targets.
            links = c.worm_physics.links
            pivot = tuple((links[0].position[i]+links[1].position[i])/2 for i in range(2))
            radius = rng.uniform(*TURN_REACH)
            angle = math.radians(TURN_MAX_ANGLE)*rng.random()**2
            self.world.worm_goal = tuple(pivot[i]-self.forward[i]*radius*math.cos(angle)
                                         +self.normal[i]*radius*math.sin(angle) for i in range(2))
        self.prey = None
        self.pursuit_target = None
        if stage == 11:
            # Out of reach from the wall: the only way to the marker is to let go and leap.
            along = rng.uniform(*LEAP_ALONG)
            outward = rng.uniform(*LEAP_OUTWARD)
            self.world.worm_goal = (c.x+self.forward[0]*along+self.normal[0]*outward,
                                    c.y+self.forward[1]*along+self.normal[1]*outward)
        if stage in (6,7):
            along = rng.uniform(22.,28.) if stage == 6 else rng.uniform(-10.,-4.)
            outward = rng.uniform(28.,36.) if stage == 6 else rng.uniform(20.,28.)
            tail = c.worm_physics.tail
            self.world.worm_goal = tuple(tail[i]+self.forward[i]*along+self.normal[i]*outward for i in range(2))
        if stage in (3,4,5):
            along = rng.uniform(15.,30.)
            outward = rng.uniform(28.,40.) if stage == 3 else rng.uniform(12.,26.)
            pos = (c.x+self.forward[0]*along+self.normal[0]*outward,
                   c.y+self.forward[1]*along+self.normal[1]*outward)
            self.prey = creatures.build('static',pos,angle=0.)
            self.prey.allegiance = 'friend'
            self.prey.traits = (TakesHits(6),)
            self.prey.after_frame_traits = ()
            self.world.roster.add(self.prey)
            self.enc.ready()
            self.world.worm_goal = None
            self.prey_anchor = pos
        return self.finish_reset()

    def finish_reset(self):
        self.limit_frames = math.ceil((min(self.horizon,LEAP_SECONDS) if self.stage == 11 else self.horizon)*60)
        self.start_distance = self.distance()                   # reported with every episode, so results can be read by how far the target was
        self.closest = self.distance() if self.stage == 11 else 0.     # Leap pays only for a new closest approach
        self.takeoff,self.leaping,self.missed,self.settled = None,False,False,False     # settled: it has first planted a foot
        self.aim_paid = False
        self.whip_sign,self.whipping,self.front_held = None,False,False       # Leap's expert: which way it whips, and whether the whip has begun
        from creatures.inchworm_control import InchwormController
        c,rng,stage = self.worm,self.rng,self.stage
        self.teacher = InchwormController()
        self.label_crawler = None
        self.rear_landed = self.rear_freed = self.flipped = False
        if stage in (2,8):
            self.teacher.request_turn(c.worm_physics)
        self.frames = self.bites = self.contact_frames = self.recovered_frames = 0
        self.shove_at = rng.randrange(180,241)
        self.shoved = False
        self.bites_at_shove = 0
        self.finished = self.success = False
        self.effort_sum = self.airborne_frames = self.return_total = 0.
        self.backward_distance = 0.
        # Allow the first gait contractions before forward travel can replenish credit.
        # Only episode reset grants this allowance; turning/regripping never renews it.
        self.backward_owed = -BACKWARD_SLACK
        self.previous = np.zeros(8,np.float32)
        self.action_trace = []
        return self.snapshot()

    def reset_pursue(self,seed,level):
        """The Pursue lesson: a real level's cave maze, a worm planted on one wall and a target drifting in view."""
        import pygame
        import config
        import creatures
        import level_config
        from creatures.encounter import Encounter
        from creatures.traits.rooted import wall_between
        from creatures.traits.wallcrawl import contact, clear, nearby
        from entities.command_recorder import CommandRecorder
        from game_handlers.enemy_roster import EnemyRoster
        from level_layout import parse_layout
        from maze.config import MazeComplexity
        from maze.generator import Maze
        from physics.obstacle_world import ObstacleWorld
        from hunter.maze_route import MazeRoutes
        from learning.adapters.practice_targets import PracticeTarget, TARGET_RADIUS, spacious_start
        if level not in PURSUE_LEVELS:
            raise ValueError(f'the pursue lesson runs on levels {PURSUE_LEVELS}, not {level}')
        self.seed, self.level, self.stage = seed,level,10
        self.rng = rng = random.Random(seed)
        pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1352,878))
        self._screen_settings = config.SCREEN_WIDTH,config.SCREEN_HEIGHT
        config.SCREEN_WIDTH,config.SCREEN_HEIGHT = 1352,878
        arena = level_config.load_level_config(level)
        grid_size = arena['maze']['grid_size']
        layout = parse_layout(arena['layout'],grid_size)
        layout = replace(layout,enemies=(),beacon=None)
        maze = Maze(level,MazeComplexity[arena['maze']['complexity'].upper()],grid_size,layout=layout,seed=seed,curved=True)
        self.obstacles = ObstacleWorld(maze,layout.gates) if layout.gates else None
        maze.gate_segments = self.obstacles.segments() if self.obstacles else []
        self.forward, self.normal = (1.,0.),(0.,-1.)       # only the Crawl-style teacher and rewards read these
        self.origin = maze.start_pos if getattr(maze,'start_pos',None) else (650.,430.)
        self.islands_built = self.heights = []
        self.world = SimpleNamespace(ship=None,hunter=None,maze=maze,roster=EnemyRoster(),
            inchworm_policy=None,inchworm_posture=(0.,)*5,additional_targets=[],projectiles=[],worm_goal=None,
            sound_manager=SilentSound(),command_recorder=CommandRecorder(),combat_ruleset='legacy',
            collision_handler=SimpleNamespace(drop_pickup=lambda *a,**k:None),powerup_crystals=[],
            jev_beacon_allowed=lambda:False,_hunter_sees_player=lambda:False,enemy_policy=None)
        cells = [maze.position_calculator.grid_center_to_screen(c,r)
                 for r in range(1,maze.grid_height-1) for c in range(1,maze.grid_width-1) if not maze.grid[r][c]]
        rng.shuffle(cells)
        worm = creatures.build('inchworm',cells[0],angle=0.)
        reach = max(maze.cell_size_x,maze.cell_size_y)*2
        routes = MazeRoutes()
        def find_start(open_floor):
            """A planted start: the clear spot beside a wall, with room for a target in sight of it. Open floor
            matters: a worm cornered in a narrow dead end jams its arch and breaks the physics solver."""
            for cell in cells:
                surfaces = sorted((s for s in (contact(maze,cell,[w]) for w in nearby(maze,cell,reach)) if s is not None),
                                  key=lambda s:s[3])
                if not surfaces or surfaces[0][3] > reach:
                    continue
                _,foot,(nx,ny),_ = surfaces[0]
                start = (foot[0]+nx*(worm.radius+2.),foot[1]+ny*(worm.radius+2.))
                if (not clear(maze,start,worm.radius) or wall_between(cell,start,maze)
                        or open_floor and not spacious_start(maze,start,(nx,ny))):
                    continue
                # The target must be somewhere the worm can walk to: never in a sealed-off pocket.
                linked = set(routes.reachable_targets(maze,start,cells))
                seen = [p for i,p in enumerate(cells) if i in linked
                        and PURSUE_DISTANCE[0] <= math.dist(p,start) <= PURSUE_DISTANCE[1]
                        and maze.is_open(p,TARGET_RADIUS+2) and not wall_between(start,p,maze)]
                if seen:
                    return start,(nx,ny),seen
            return None
        found = find_start(True) or find_start(False)
        if found is None:
            raise ValueError(f'level {level} has no worm start with a target in view')
        start,(nx,ny),seen = found
        worm.x,worm.y = start
        worm.angle = math.degrees(math.atan2(ny,nx))-90.
        worm.agent_id = 0
        self.worm = c = worm
        self.world.roster.add(c)
        target = PracticeTarget(rng.choice(seen),0.,0.,firing_enabled=False)
        heading,speed = rng.uniform(0,2*math.pi),rng.uniform(*PURSUE_SPEED)
        target.vx,target.vy = math.cos(heading)*speed,math.sin(heading)*speed
        self.pursuit_target = target
        self.world.additional_targets.append(target)
        self.enc = Encounter(self.world.roster,self.world)
        self.enc.ready()
        self.curl_anchor = None
        self.curl_anchor_lost = False
        self.start = c.get_pos()
        self.goal_direction = 1
        self.prey = None
        return self.finish_reset()

    def snapshot(self):
        x,masks,_ = observe(self.worm,self.enc)
        live = self.worm.active
        return Snapshot({0:x} if live else {},{0:masks} if live else {},x,{0:x})

    @property
    def teacher_weight(self):
        """DAgger: in About-face the scripted controller is asked what it would do in every state the policy
        reaches, and the policy is pulled toward those answers with this weight (0: no teacher)."""
        return self.rewards['dagger_weight']

    def leap_distance(self):
        """How near the leap is to the marker: the mouth, or any part of the body (a leaping worm spins, and a body that
        strikes the marker has caught it as surely as the mouth), counted as LEAP_BODY_SLACK further off than the mouth."""
        goal = self.world.worm_goal
        best = math.dist(self.worm.worm_front,goal)
        half = self.worm.worm_physics.length/2
        for link in self.worm.worm_physics.links:
            a,b = link.local_to_world((-half,0.)),link.local_to_world((half,0.))
            ax,ay,dx,dy = a[0],a[1],b[0]-a[0],b[1]-a[1]
            t = max(0.,min(1.,((goal[0]-ax)*dx+(goal[1]-ay)*dy)/max(1e-9,dx*dx+dy*dy)))
            best = min(best,math.hypot(goal[0]-ax-t*dx,goal[1]-ay-t*dy)+LEAP_BODY_SLACK)
        return best

    def distance(self):
        if self.pursuit_target is not None:
            return max(0.,math.dist(self.worm.worm_front,self.pursuit_target.get_pos())
                       -self.pursuit_target.radius-self.worm.worm_skin)
        if self.stage == 11:
            return self.leap_distance()
        if self.stage in (6,7,8):
            return math.dist(self.worm.worm_front,self.world.worm_goal)
        if self.prey is not None:
            return max(0.,math.dist(self.worm.worm_front,self.prey.get_pos())-self.prey.radius-self.worm.worm_skin)
        return math.dist(self.worm.get_pos(),self.world.worm_goal)

    def end_pitch_excess(self):
        """How far the tail and head links are tilted off parallel to the nearest ground, beyond PITCH_SLACK, as a
        fraction of what is left up to a right angle (0: both parallel enough, 1: both upright), averaged over the two."""
        from utils import get_closest_point_on_line
        links = self.worm.worm_physics.links
        total = 0.
        for body in (links[0],links[-1]):
            nearest = min((w for w in self.world.maze.walls if w.active),
                          key=lambda w:math.dist(body.position,get_closest_point_on_line(body.position,w.start,w.end)))
            ground = math.atan2(nearest.end[1]-nearest.start[1],nearest.end[0]-nearest.start[0])
            tilt = abs((body.angle-ground+math.pi/2) % math.pi-math.pi/2)        # a link is an undirected line: 0..pi/2
            total += max(0.,tilt-PITCH_SLACK)/(math.pi/2-PITCH_SLACK)
        return total/2

    def goal_facing(self):
        """Cosine between the head direction and the goal: 1 faces it, -1 faces away. Unlike alignment it is signed."""
        origin = self.worm.get_pos()
        goal = self.pursuit_target.get_pos() if self.pursuit_target is not None else self.world.worm_goal
        dx,dy = goal[0]-origin[0],goal[1]-origin[1]
        distance = math.hypot(dx,dy)
        if distance < 1e-9:
            return 0.
        heading = self.worm.worm_physics.heading
        return (math.cos(heading)*dx+math.sin(heading)*dy)/distance

    def turn_potential(self):
        """Facing the goal times how stretched out the body is: a body folded double over the goal has not turned."""
        p = self.worm.worm_physics
        return max(0.,self.goal_facing())*min(1.,math.dist(p.head,p.tail)/(6*p.length))

    def target_alignment(self):
        """Positive alignment means the head end leads toward the target."""
        target = self.pursuit_target.get_pos() if self.pursuit_target is not None else (
            self.prey.get_pos() if self.prey is not None else self.world.worm_goal)
        origin = (self.worm.worm_front if self.prey is not None or self.pursuit_target is not None or self.stage == 8
                  else self.worm.get_pos())
        dx, dy = target[0]-origin[0], target[1]-origin[1]
        distance = math.hypot(dx,dy)
        if distance < 1e-9:
            return 0.
        heading = self.worm.worm_physics.heading
        return max(0.,min(1.,(math.cos(heading)*dx+math.sin(heading)*dy)/distance))

    def step(self,actions):
        if self.finished:
            raise RuntimeError('step after inchworm episode ended')
        if self.scripted:
            action = teacher_action(self,0,None)
        else:
            if set(actions) != {0}:
                raise ValueError('inchworm episode expects action for agent 0')
            action = tuple(actions[0])
        command = intention(action,bend_signs(self.worm.worm_physics))
        self.action_trace.append({0:tuple(float(v) for v in action)})
        self.worm.worm_command = command
        old_bites = self.bites
        applied = np.asarray([*command.muscles,command.rear_grip,command.front_grip,command.bite])
        # The counts each reward term pays on (learning.inchworm_rewards); paid together once the decision ends.
        events = TABLE.row()
        events['action_change'] = float(np.mean((applied-self.previous)**2))
        self.previous = applied
        count = min(DECISION_FRAMES, max(0,self.limit_frames-self.frames))
        backward_net = 0.
        for _ in range(count):
            start_distance, alignment = self.distance(), self.target_alignment()
            start_turn = self.turn_potential() if self.stage in (8,10) else 0.
            p = self.worm.worm_physics
            rear_grip, centre_before = p.grips[0], tuple(p.centre)
            # Pursue has no fixed 'forward' along a cave, so backward travel is judged against the heading itself.
            forward = (math.cos(p.heading),math.sin(p.heading)) if self.pursuit_target is not None else self.forward
            facing = math.cos(p.heading)*forward[0]+math.sin(p.heading)*forward[1]
            was_stunned = self.worm.worm_stun > 0
            if self.stage == 5 and self.frames == self.shove_at:
                self.worm.apply_momentum(self.normal[0]*.15+self.forward[0]*.25,
                                         self.normal[1]*.15+self.forward[1]*.25)
                self.shoved, self.bites_at_shove = True,self.bites
            before = self.prey.hit_points if self.prey is not None else 0
            self.enc.step(1.)
            if self.pursuit_target is not None:
                # Reward the worm's own movement only: measured against where the target is now, before it drifts
                # on. Otherwise a target walking away charges a worm that stands still, and one walking in pays it.
                own_progress,own_turn = start_distance-self.distance(),self.turn_potential()-start_turn
                self.pursuit_target.step(self.world.maze)
                if self.obstacles is not None:
                    self.world.maze.gate_segments = self.obstacles.segments()
            p = self.worm.worm_physics
            if self.stage == 7 and p.grips[0] is not self.curl_anchor:
                self.curl_anchor_lost = True
            # A retained foot anchor is articulation, not backward walking.
            # Ignore forced stun motion and near-vertical, ambiguous facing.
            flying = not (self.worm.worm_front_grip or self.worm.worm_rear_grip) and self.worm.worm_stun == 0
            if (not (rear_grip is not None and p.grips[0] is rear_grip)
                    and not (flying and (self.pursuit_target is not None or self.stage == 11))
                    and not was_stunned and self.worm.worm_stun == 0 and abs(facing) > .2):
                travel = sum((p.centre[i]-centre_before[i])*forward[i] for i in range(2))
                backward_net -= travel*math.copysign(1.,facing)
            self.bites += int(before-(self.prey.hit_points if self.prey else 0))
            if self.prey is not None and self.prey.active:
                # The solver supplies grasp impulses. Only stage 5 adds a modest
                # physical evasive acceleration; the prey is never repositioned.
                if self.stage == 5:
                    for axis,i in (('vx',0),('vy',1)):
                        restore = (self.prey_anchor[i]-self.prey.get_pos()[i])*.001
                        setattr(self.prey,axis,getattr(self.prey,axis)+restore+self.forward[i]*.008*math.sin(self.frames/90.))
                self.prey.vx *= .98
                self.prey.vy *= .98
                self.prey.prev_x,self.prey.prev_y = self.prey.get_pos()
                self.prey.x += self.prey.vx
                self.prey.y += self.prey.vy
                from physics.thrust import sweep_walls
                sweep_walls(self.prey,self.world.maze.walls)
            progress = own_progress if self.pursuit_target is not None else start_distance-self.distance()
            # Supported reaching teaches articulation before locomotion. Its
            # mouth can approach an overhead target while the body faces across
            # it; head-first gating would suppress that useful early movement.
            scale = 1. if self.stage in (6,7) else alignment
            if (self.pursuit_target is not None or self.stage == 11) and flying:
                scale = 1.      # a leap or drift through open space pays for closing on the target however the body faces
            if self.curl_anchor_lost:
                scale = 0.
            if self.stage == 11:
                # A leap that misses flies off in the wrong direction; charging that retreat at full rate made every
                # attempt cost more than lying still, and the worm stopped trying. Pay each new closest approach once.
                events['progress_pixels'] += max(0.,self.closest-self.distance())
                self.closest = min(self.closest,self.distance())
            else:
                events['progress_pixels'] += progress*scale if progress > 0 else progress
            # About-face starts facing away, where head-first progress pays nothing and crawling pays a penalty, so
            # turning needs its own reward: it is potential-based, so swinging back and forth earns nothing, and it needs the
            # body stretched out as well, since a body folded double over the goal is only half way through the turn.
            if self.stage == 8:
                events['turn_progress'] += self.turn_potential()-start_turn
            elif self.stage == 10:
                events['turn_progress'] += own_turn      # a target that slipped behind: turn round and face it again
            touching = p.mouth_contact and self.worm.worm_rear_grip
            if self.stage == 6:
                touching = self.distance() <= 12.
            elif self.stage == 7:
                touching = self.distance() <= 12. and not self.curl_anchor_lost
            self.contact_frames = self.contact_frames+1 if touching else 0
            grounded = self.worm.worm_front_grip or self.worm.worm_rear_grip
            if self.shoved and grounded and self.worm.worm_stun == 0:
                self.recovered_frames += 1
            self.effort_sum += p.effort
            self.airborne_frames += not grounded
            if self.stage == 1 or self.stage == 8 and self.goal_facing() > .3:
                events['end_pitch_seconds'] += self.end_pitch_excess()/60.
            events['effort_seconds'] += p.effort/60.
            events['seconds'] += 1/60.
            events['crawl_seconds'] += (self.stage in (1,9,10,11))/60.
            events['about_face_seconds'] += (self.stage == 8)/60.
            events['airborne_seconds'] += float(not grounded and self.worm.worm_stun == 0)/60.
            self.frames += 1
            if self.stage in (1,8,9):
                self.success = self.distance() < 14. and grounded
            elif self.stage == 2:
                self.success = math.dist(p.head,self.world.worm_goal) < 25.
            elif self.stage in (6,7):
                self.success = touching
            elif self.stage == 3:
                self.success = self.contact_frames >= 12
            elif self.stage == 10:
                self.success = self.distance() <= PURSUE_CATCH
            elif self.stage == 11:
                self.success = self.distance() <= LEAP_CATCH
            elif self.stage == 4:
                self.success = self.bites >= 2
            else:
                self.success = self.shoved and self.recovered_frames >= 30 and self.bites-self.bites_at_shove >= 2
            if self.render_frames is not None:
                self.render_frames(self)
            if self.stage == 11:
                # The release itself, the decision that aims the leap: paid at the frame the front foot lets go while it
                # is the only foot down, by how squarely the body is then moving at the marker (and only if it is moving).
                held = p.grips[1] is not None
                if self.front_held and not held and p.grips[0] is None:
                    vx,vy = p.velocity; goal = self.world.worm_goal
                    off = abs((math.atan2(vy,vx)-math.atan2(goal[1]-p.centre[1],goal[0]-p.centre[0])+math.pi)%(2*math.pi)-math.pi)
                    events['release_aim'] += release_score(off,math.hypot(vx,vy))
                self.front_held = held
            if self.stage == 11 and not self.success:
                # Leap trains accuracy, not recovery: a leap that lands without a catch, or flies off, ends the trial.
                if not flying:
                    self.missed = self.leaping
                    self.settled,self.takeoff,self.leaping,self.aim_paid = True,None,False,False
                elif self.settled:
                    if self.takeoff is None:
                        self.takeoff = tuple(p.centre)
                    self.leaping = self.leaping or math.dist(self.takeoff,p.centre) >= LEAP_MIN_TRAVEL
                    if self.leaping and not self.aim_paid:
                        # Paid as the leap leaves: how squarely its direction points at the marker.
                        goal = self.world.worm_goal
                        gx,gy,fx,fy = goal[0]-self.takeoff[0],goal[1]-self.takeoff[1],p.centre[0]-self.takeoff[0],p.centre[1]-self.takeoff[1]
                        cosine = (gx*fx+gy*fy)/max(1e-9,math.hypot(gx,gy)*math.hypot(fx,fy))
                        events['leap_aim'] += max(0.,1.-math.acos(max(-1.,min(1.,cosine)))/LEAP_AIM_TOLERANCE)
                        self.aim_paid = True
                    self.missed = self.leaping and self.distance() > self.closest+LEAP_RECEDE
            if self.stage == 1 and not self.success:
                # A crawl may hop towards its marker, but with no gravity a hop that carries it away from the marker never
                # comes back: it ends the trial, so the worm learns to walk rather than throw itself about.
                if not flying:
                    self.takeoff,self.leaping = None,False
                else:
                    if self.takeoff is None:
                        self.takeoff,self.takeoff_distance = tuple(p.centre),self.distance()
                    self.leaping = self.leaping or math.dist(self.takeoff,p.centre) >= LEAP_MIN_TRAVEL
                    self.missed = self.leaping and self.distance() > self.takeoff_distance+LEAP_RECEDE
            if self.success or not self.worm.active or self.curl_anchor_lost or self.missed:
                break
        # Backward walking is the body's net travel over the decision. Summing only the frames it moved
        # backward charged every stride of a normal gait, whose tail swings back each step, for walking forwards.
        # Forward travel earns up to BACKWARD_SLACK of credit against later retreat, so the swing-back inside a
        # stride of a gait that is advancing is free; walking tail-first uses the credit up and is charged.
        owed = max(-BACKWARD_SLACK,self.backward_owed+backward_net)
        backward = max(0.,max(0.,owed)-max(0.,self.backward_owed))
        self.backward_owed = owed
        self.backward_distance += backward
        events['backward_pixels'] = backward
        # Positive progress is gated per physics frame by pre-movement facing.
        # Retreat always pays the full penalty, so a closed path cannot earn
        # positive progress reward by changing orientation on the return trip.
        events['bites'] = self.bites-old_bites
        if self.missed:
            events['leap_misses'] = 1
        if self.success:
            events['successes'] = 1
            if self.stage in (1,9,10,11):
                events['time_left_on_arrival'] = max(0.,1.-self.frames/self.limit_frames)
        if not self.worm.active:
            events['deaths'] = 1
        reward = float(TABLE.pay([events],self.rewards)[0])
        self.return_total += reward
        terminated = self.success or not self.worm.active or self.curl_anchor_lost or self.missed
        truncated = not terminated and self.frames >= self.limit_frames
        self.finished = terminated or truncated
        return Transition(self.snapshot(),reward,terminated,truncated,self.events(),{0:reward})

    def events(self):
        return dict(seconds=self.frames/60.,success=float(self.success),bites=self.bites,
                    distance=self.distance(),effort=self.effort_sum/max(1,self.frames),
                    airborne_fraction=self.airborne_frames/max(1,self.frames),
                    survived=float(self.worm.active),recovered=float(self.recovered_frames>=30),
                    return_total=self.return_total,stage=self.stage,curl_anchor_lost=self.curl_anchor_lost,
                    backward_distance=self.backward_distance,start_distance=self.start_distance)

    def close(self):
        if self.world is not None:
            if self.enc is not None:
                self.enc.close()
            self.enc = self.world = None
        if self._screen_settings is not None:
            import config
            config.SCREEN_WIDTH,config.SCREEN_HEIGHT = self._screen_settings
            self._screen_settings = None


def teacher_weight(args):
    return args.inchworm_dagger_weight


BACKWARD_SLACK = 30.   # pixels of forward travel that can offset later retreat
TURN_LOOP = tuple(v/MAX_BEND for v in (-1.4, -1.4, -1.4, -.34, 1.3984))


PHASES = ('plant','curl','land','release','crawl','cock','whip','let-go')


def posture_expert(env):
    """The next command in About-face; see `posture_phase`."""
    return posture_phase(env)[1]


def posture_phase(env):
    """(phase, command): what to do next in About-face, decided from the worm's posture and grips alone.

    The scripted controller walks a timed sequence, so it goes on to the next step even when the policy being
    labelled has not done the last one. This reads the state instead, which makes it a valid answer for any state
    the policy reaches: plant, curl with the front free, land the head, release the rear and unfold, then crawl.
    """
    from creatures.inchworm_control import InchwormController
    from physics.inchworm import InchwormCommand
    p = env.worm.worm_physics
    facing, extension = env.goal_facing(), min(1.,math.dist(p.head,p.tail)/(6*p.length))
    # Once the worm has flipped, a deep arch in its stride (stretch below .6) is still walking, not an unfinished
    # turn; the flip stays done until the worm faces away from the goal again.
    if facing > .3 and extension > .6 or env.flipped and facing > 0.:
        env.flipped = True
        if env.label_crawler is None:
            env.label_crawler = InchwormController()
        return 'crawl',env.label_crawler.command(p,DECISION_FRAMES,None)
    env.flipped = False
    env.label_crawler = None
    rear, front = p.grips[0] is not None, p.grips[1] is not None
    curled = max(abs(a-b) for a,b in zip(p.observe()['joint_angles'],TURN_LOOP)) < .35
    if facing > .3 and front:
        return 'release',InchwormCommand((0.,)*5,False,True)         # head planted over the goal: let the tail go, unfold
    if curled and not front:
        return 'land',InchwormCommand(TURN_LOOP,True,True)        # land the head
    if rear:
        return 'curl',InchwormCommand(TURN_LOOP,True,False)       # curl with the front free
    return 'plant',InchwormCommand(tuple(p.activation),True,True)  # plant the rear first


WHIP_TOLERANCE = math.radians(10.)   # the front foot lets go once the body's motion points this near the marker
RELEASE_MISS = .5                    # what a let-go far off the marker costs, against the 1 a good one earns
WHIP_SPEED = .3                      # ...and is moving at least this fast


def release_score(off, speed):
    """What letting go of the wall is worth: +1 when the body is moving within WHIP_TOLERANCE of the marker, falling
    straight to -RELEASE_MISS at LEAP_AIM_TOLERANCE off (and staying there), and the same for a let-go with no speed.
    Waiting for the swing to come round has to beat letting go early."""
    if speed <= WHIP_SPEED:
        return -RELEASE_MISS
    fall = min(1.,max(0.,(off-WHIP_TOLERANCE)/(LEAP_AIM_TOLERANCE-WHIP_TOLERANCE)))
    return 1.-(1.+RELEASE_MISS)*fall


def whip_phase(env):
    """(phase, command): what a worm does to leap with a whiplash, decided from its posture, grips and motion alone.

    Cock: plant the front foot, free the rear, and hold the body relaxed and straight (the successful leaps of a
    trained worm cock no deeper than that, and a deeper arch is one it never reaches). Whip: snap the body into a curl
    around the planted front foot, so it swings. Let go: release the front foot when the swinging body is moving
    towards the marker, which is what aims the leap. The phase is None once the worm is in the air; nothing is taught
    there."""
    from physics.inchworm import InchwormCommand
    p = env.worm.worm_physics
    centre,goal = p.centre,env.world.worm_goal
    bearing = math.atan2(goal[1]-centre[1],goal[0]-centre[0])
    if env.whip_sign is None:         # the whip curls towards the marker's side of the body
        axis = math.atan2(p.head[1]-p.tail[1],p.head[0]-p.tail[0])
        env.whip_sign = 1. if (bearing-axis+math.pi)%(2*math.pi)-math.pi > 0 else -1.
    cock,whip = (0.,)*5,tuple(v*env.whip_sign for v in TURN_LOOP)
    if p.grips[1] is None:
        return (None,InchwormCommand(whip,False,False)) if env.whipping else ('cock',InchwormCommand(cock,False,True))
    if not env.whipping:
        env.whipping = p.grips[0] is None            # cocked: held by the front foot alone
        if not env.whipping:
            return 'cock',InchwormCommand(cock,False,True)
    vx,vy = p.velocity
    off = abs((math.atan2(vy,vx)-bearing+math.pi)%(2*math.pi)-math.pi)
    if math.hypot(vx,vy) > WHIP_SPEED and off < WHIP_TOLERANCE:
        return 'let-go',InchwormCommand(whip,False,False)
    return 'whip',InchwormCommand(whip,False,True)


def label_action(env, agent, action):
    """What the expert would do in the states a policy visits. About-face: up to the flip, once the worm faces the
    goal stretched out the expert stops teaching. Leap: the whiplash, until the worm is in the air."""
    if env.stage == 11:
        phase,command = whip_phase(env)
        if phase is None:
            return None
        return (*action_for(command,bend_signs(env.worm.worm_physics)),float(PHASES.index(phase)))
    if env.stage != 8 or env.rewards['about_face_dagger'] <= 0:
        return None
    phase,command = posture_phase(env)
    if phase in ('release','crawl') and env.worm.worm_physics.grips[0] is None:
        env.rear_freed = True
    if phase == 'crawl':
        # The flip is done and walking on is the policy's own. The one thing taught is to put the rear foot down
        # after releasing it, once: nothing else says so, and the release label alone teaches it to stay up.
        p = env.worm.worm_physics
        if env.rear_landed or p.grips[0] is not None:
            env.rear_landed = env.rear_landed or env.rear_freed     # landing counts only after the rear has been let go
            return None
        rear = list(dict(SPEC.heads)).index('rear_grip')
        label = [math.nan]*len(SPEC.heads)
        label[rear] = 1.
        return (*label,float(PHASES.index(phase)))
    # The phase rides along as a ninth value so the update can weight each phase equally: the rare ones
    # (curl, land, release) are brief but decide whether the turn happens, and a plain average drowns them.
    return (*action_for(command,bend_signs(env.worm.worm_physics)),float(PHASES.index(phase)))


def expert_leads(env):
    """Scheduled sampling: only Leap has an expert to lead its episodes (the others rely on their DAgger labels)."""
    return env.stage == 11


def handover(env, agent):
    """...and it leads only up to the cock: the worm then whips and lets go itself, so PPO scores the part that decides the aim."""
    return env.whipping


def expert_assist(env, agent, action):
    """After the handover the expert still swings the body (the bends and the rear grip) until the worm is in the air;
    the policy chooses only the front grip, the release that aims the leap, and PPO scores just that."""
    phase,command = whip_phase(env)
    if phase is None:
        return None
    driven = action_for(command,bend_signs(env.worm.worm_physics))
    front = list(dict(SPEC.heads)).index('front_grip')
    return tuple(action[j] if j == front else v for j,v in enumerate(driven)),{front}


def teacher_action(env, agent, action):
    if env.stage == 11:
        return action_for(whip_phase(env)[1],bend_signs(env.worm.worm_physics))
    if env.stage in (6,7):
        from creatures.inchworm_control import reach_angles
        from physics.inchworm import InchwormCommand
        return action_for(InchwormCommand(reach_angles(env.worm.worm_physics,env.world.worm_goal),True,False),
                         bend_signs(env.worm.worm_physics))
    target = env.prey if env.prey is not None and env.prey.active else None
    return action_for(env.teacher.command(env.worm.worm_physics,DECISION_FRAMES,target),bend_signs(env.worm.worm_physics))


def environment(**options):
    return InchwormEnvironment(**options)


def environment_options(manifest):
    return dict(rewards=inchworm_rewards.validate(manifest.get('rewards',{})))


def metadata(args):
    import enemy_kinds
    import pymunk
    root = Path(__file__).resolve().parents[2]
    files = ('physics/inchworm.py','creatures/inchworm_control.py','creatures/traits/inchworm.py',
             'learning/adapters/inchworm.py','learning/adapters/inchworm_scenario.py')
    levels = {str(n):hashlib.sha256((root/f'levels/{n}.json').read_bytes()).hexdigest() for n in PURSUE_LEVELS}
    return dict(task=SPEC.task,scenario_version=74,spec=SPEC.metadata(),horizon=args.horizon,
                about_face_success=dict(distance_origin='head',distance_less_than=14.,
                                        planted_grip_required=True,hold_frames=1),
                front_grip=dict(attachment='automatic_on_reentry_into_reach',
                                release='front_request_falling_edge',rearm='head_clear_of_surfaces_and_prey',
                                initially_armed=False,disabled_when_stunned=True),
                motion_shaping=dict(backward='body_centre_net_surface_travel_against_pre_frame_facing',
                                    backward_slack=BACKWARD_SLACK,initial_backward_credit=BACKWARD_SLACK,
                                    exempt='retained_rear_anchor_or_stun',minimum_facing_projection=.2),
                curl_success=dict(mouth_distance=12.,hold_frames=1,retain_starting_rear_grip=True,
                                  end_on_anchor_loss=True,setup_physics_frames=1),
                action_selection=dict(training='sample',validation=EVALUATION_ACTION_MODE,gameplay=INFERENCE_ACTION_MODE,
                                      validation_rng='SeedSequence([episode_seed, agent_id])'),
                rear_success=dict(mouth_distance=12.,hold_frames=1,rear_grip_required=False),
                progress_rule='head_first_locomotion_signed_mouth_progress_for_rear_and_curl',
                curriculum_order=CURRICULUM,leap=dict(along=LEAP_ALONG,outward=LEAP_OUTWARD,catch=LEAP_CATCH,seconds=LEAP_SECONDS,
                                                  progress='new_closest_approach_only',aim='angle_falloff_to_30_degrees_once_per_leap_far_markers_only',miss='lands_after_a_leap_or_recedes',min_travel=LEAP_MIN_TRAVEL,recede=LEAP_RECEDE),pursue=dict(levels=levels,distance=PURSUE_DISTANCE,speed=PURSUE_SPEED,catch=PURSUE_CATCH),success_gates=SUCCESS_GATES,
                crawl_distance=dict(short=[60.,90.],long=[180.,300.],short_training_fraction=.25,
                                    validation='long_only'),
                decision_frames=DECISION_FRAMES,opponent_sha256=None,arenas=ARENAS,
                creature=asdict(enemy_kinds.kind('inchworm')),rewards=inchworm_rewards.from_args(args),
                pymunk=pymunk.version,physics_fps=60,twist='automatic_bounded_reflex',
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


def training_cases(iteration,count,stage,bootstrap=False,curriculum=None):
    current = curriculum['stage'] if curriculum else stage
    earlier = CURRICULUM[:CURRICULUM.index(current)]
    if current == 10:
        earlier = tuple(s for s in earlier if s not in PURSUE_DROPS)
    if current == 11:
        earlier = earlier+LEAP_REHEARSAL*2

    def practice_stage(j):
        """While Leap trains, Crawl is what production needs most, so it gets a quarter of every batch ; another quarter rotates through the earlier lessons; Leap, the lesson being learned, has the rest.
        Otherwise a quarter is Crawl (every later lesson builds on its gait), a quarter rotates and the rest is the current lesson."""
        if current == 11:
            slot = j%8
            if slot in (2,6):
                return 1
            if slot == 5:
                return LEAP_PROTECTED[(iteration+j//8)%len(LEAP_PROTECTED)]
            return earlier[(iteration+j//4)%len(earlier)] if slot in (0,4) else current
        if j%4 == 2 and 1 in earlier:
            return 1
        return earlier[(iteration+j//4)%len(earlier)] if earlier and j%4 == 0 else current
    # A run that starts at a later lesson has already learned the earlier ones; scripted demonstrations of those would
    # only pull it toward the scripted gait, so it is shown its own lesson alone.
    shown = (current,) if bootstrap and current in DEMONSTRATED and current != CURRICULUM[0] else DEMONSTRATED
    cases = [dict(level=TRAIN_LEVELS[j%4],seed=10000+iteration*count+j+(500000000 if bootstrap else 0),
                  stage=shown[j%len(shown)] if bootstrap else practice_stage(j)) for j in range(count)]
    for case in cases:
        if case['stage'] == 10:
            case['level'] = PURSUE_TRAIN_LEVELS[case['seed']%len(PURSUE_TRAIN_LEVELS)]
    return cases


def cases_for(stage,levels,seed_base):
    """Twelve held-out cases per lesson; Pursue's come from real levels, the others from the angled arenas."""
    if stage == 10:
        return [dict(level=levels[1][k%len(levels[1])],seed=seed_base+stage*100+k,stage=stage) for k in range(12)]
    return [dict(level=level,seed=seed_base+stage*100+j*4+k,stage=stage)
            for j,level in enumerate(levels[0]) for k in range(3)]


def validation_cases():
    return [c for stage in CURRICULUM
            for c in cases_for(stage,(VALIDATION_LEVELS,PURSUE_VALIDATION_LEVELS),2000000000)]


def evaluation_cases():
    return [c for stage in CURRICULUM
            for c in cases_for(stage,(HOLDOUT_LEVELS,PURSUE_HOLDOUT_LEVELS),3000000000)]


def metrics(rows):
    return {key:float(np.mean([r.get(key,0.) for r in rows])) if rows else 0.
            for key in ('success','bites','distance','effort','airborne_fraction','survived','recovered','return_total',
                        'backward_distance','seconds')}


def selection_score(summary):
    return summary['success']*10+summary['bites']-summary['distance']*.01-summary['effort']*.1


def compare(a,b):
    if len(a['episodes']) != len(b['episodes']):
        raise ValueError('inchworm comparison requires paired cases')
    values = []
    for x,y in zip(a['episodes'],b['episodes']):
        if any(x[k] != y[k] for k in ('level','seed','stage')):
            raise ValueError('inchworm comparison cases do not match')
        values.append(x['success']-y['success'])
    if not values:
        return dict(mean=0.,lower_95=-1.,upper_95=1.)
    samples = np.random.default_rng(0).choice(values,(2000,len(values))).mean(axis=1)
    return dict(mean=float(np.mean(values)),lower_95=float(np.quantile(samples,.025)),upper_95=float(np.quantile(samples,.975)))


def promote(stage,summary):
    return stage


def widen_policy(policy,args):
    """A new run's starting policy, given more memory when the run asks for it (embedding half as wide as the memory)."""
    hidden = int(getattr(args,'inchworm_hidden_units',0))
    if hidden <= policy.hidden:
        return policy
    from learning.widen import widen
    return widen(policy,hidden,max(policy.params['embed_w'].shape[0],hidden//2))


def initial_curriculum(args):
    stage = int(getattr(args,'inchworm_start_stage',0)) or CURRICULUM[0]
    if stage not in CURRICULUM:
        raise ValueError(f'inchworm start lesson must be one of {CURRICULUM}, not {stage}')
    return dict(stage=stage,passing_blocks=0)


def validation_evidence(summary,state,args,deadline=math.inf):
    threshold = SUCCESS_GATES[state['stage']]
    passed = summary['success'] >= threshold and summary['survived'] == 1.
    if state['stage'] == 1:
        passed = (passed and summary.get('airborne_fraction',0.) <= CRAWL_MAX_AIRBORNE
                  and summary.get('seconds',0.) <= CRAWL_MAX_SECONDS)
    return dict(inchworm_validation_passed=passed)


def advance_curriculum(state,summary):
    state = dict(state)
    if 'inchworm_validation_passed' not in summary:
        return state
    state['passing_blocks'] = state['passing_blocks']+1 if summary['inchworm_validation_passed'] else 0
    index = CURRICULUM.index(state['stage'])
    if state['passing_blocks'] >= PASSES_REQUIRED.get(state['stage'],2) and index+1 < len(CURRICULUM):
        state.update(stage=CURRICULUM[index+1],passing_blocks=0)
    return state


def curriculum_key(state):
    return STAGES[state['stage']-1]


def curriculum_validation_cases(state):
    return [c for c in validation_cases() if c['stage']==state['stage']]


def curriculum_evaluation_cases(state):
    return evaluation_cases()


def curriculum_selection_score(summary,state):
    return selection_score(summary)


def deployment_report(reports,args,deadline=math.inf):
    rows = reports['trained']['episodes']
    coverage = {r['stage'] for r in rows} == set(CURRICULUM)
    passed = coverage and all(metrics([r for r in rows if r['stage']==s])['success'] >= SUCCESS_GATES[s]
                              for s in CURRICULUM)
    paired = compare(reports['trained'],reports['scripted'])
    return dict(passed=bool(passed and paired['lower_95']>=-.05),versus_reflex=paired,
                automatic_install=False)
