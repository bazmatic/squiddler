"""Single-worm physical drills and a gated recurrent PPO curriculum."""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np

from learning.environment import Snapshot, Transition
from learning.adapters.inchworm import (SPEC, CRITIC_SIZE, DECISION_FRAMES, INFERENCE_ACTION_MODE, observe, intention, bend_signs,
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
STAGES = ('crawl','turn','reach','feed','recover','rear','curl','about-face','gap')
# Preserve existing stage identities in saved episode records.
CURRICULUM = (1,6,7,2,8,9,3,4,5)
DEMONSTRATED = tuple(s for s in CURRICULUM if s != 9)   # the scripted stride cannot cross a gap, so it demonstrates none
# Pixels from the first joint, which the head sweeps about 49 px around: far targets are
# the longest a pure turn reaches (a flat target past about 58 px is out of reach).
TURN_REACH = (12., 58.)
TURN_MAX_ANGLE = 80.
SUCCESS_GATES = {1:.95,6:.8,7:.8,2:.8,8:.8,9:.7,3:.8,4:.75,5:.65}
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
            raise ValueError('inchworm stage must be 1..7')
        self.seed, self.level, self.stage = seed,level,stage
        self.rng = rng = random.Random(seed)
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

    def snapshot(self):
        x,masks,_ = observe(self.worm,self.enc)
        live = self.worm.active
        return Snapshot({0:x} if live else {},{0:masks} if live else {},x,{0:x})

    @property
    def teacher_weight(self):
        """DAgger: in About-face the scripted controller is asked what it would do in every state the policy
        reaches, and the policy is pulled toward those answers with this weight (0: no teacher)."""
        return self.rewards['dagger_weight']

    def distance(self):
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
        origin,goal = self.worm.get_pos(),self.world.worm_goal
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
        target = self.prey.get_pos() if self.prey is not None else self.world.worm_goal
        origin = self.worm.worm_front if self.prey is not None or self.stage == 8 else self.worm.get_pos()
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
        count = min(DECISION_FRAMES, max(0,math.ceil(self.horizon*60)-self.frames))
        backward_net = 0.
        for _ in range(count):
            start_distance, alignment = self.distance(), self.target_alignment()
            start_turn = self.turn_potential() if self.stage == 8 else 0.
            p = self.worm.worm_physics
            rear_grip, centre_before = p.grips[0], tuple(p.centre)
            facing = math.cos(p.heading)*self.forward[0]+math.sin(p.heading)*self.forward[1]
            was_stunned = self.worm.worm_stun > 0
            if self.stage == 5 and self.frames == self.shove_at:
                self.worm.apply_momentum(self.normal[0]*.15+self.forward[0]*.25,
                                         self.normal[1]*.15+self.forward[1]*.25)
                self.shoved, self.bites_at_shove = True,self.bites
            before = self.prey.hit_points if self.prey is not None else 0
            self.enc.step(1.)
            p = self.worm.worm_physics
            if self.stage == 7 and p.grips[0] is not self.curl_anchor:
                self.curl_anchor_lost = True
            # A retained foot anchor is articulation, not backward walking.
            # Ignore forced stun motion and near-vertical, ambiguous facing.
            if (not (rear_grip is not None and p.grips[0] is rear_grip)
                    and not was_stunned and self.worm.worm_stun == 0 and abs(facing) > .2):
                travel = sum((p.centre[i]-centre_before[i])*self.forward[i] for i in range(2))
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
            progress = start_distance-self.distance()
            # Supported reaching teaches articulation before locomotion. Its
            # mouth can approach an overhead target while the body faces across
            # it; head-first gating would suppress that useful early movement.
            scale = 1. if self.stage in (6,7) else alignment
            if self.curl_anchor_lost:
                scale = 0.
            events['progress_pixels'] += progress*scale if progress > 0 else progress
            # About-face starts facing away, where head-first progress pays nothing and crawling pays a penalty, so
            # turning needs its own reward: it is potential-based, so swinging back and forth earns nothing, and it needs the
            # body stretched out as well, since a body folded double over the goal is only half way through the turn.
            if self.stage == 8:
                events['turn_progress'] += self.turn_potential()-start_turn
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
            events['crawl_seconds'] += (self.stage in (1,9))/60.
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
            elif self.stage == 4:
                self.success = self.bites >= 2
            else:
                self.success = self.shoved and self.recovered_frames >= 30 and self.bites-self.bites_at_shove >= 2
            if self.render_frames is not None:
                self.render_frames(self)
            if self.success or not self.worm.active or self.curl_anchor_lost:
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
        if self.success:
            events['successes'] = 1
            if self.stage in (1,9):
                events['time_left_on_arrival'] = max(0.,1.-self.frames/math.ceil(self.horizon*60))
        if not self.worm.active:
            events['deaths'] = 1
        reward = float(TABLE.pay([events],self.rewards)[0])
        self.return_total += reward
        terminated = self.success or not self.worm.active or self.curl_anchor_lost
        truncated = not terminated and self.frames >= math.ceil(self.horizon*60)
        self.finished = terminated or truncated
        return Transition(self.snapshot(),reward,terminated,truncated,self.events(),{0:reward})

    def events(self):
        return dict(seconds=self.frames/60.,success=float(self.success),bites=self.bites,
                    distance=self.distance(),effort=self.effort_sum/max(1,self.frames),
                    airborne_fraction=self.airborne_frames/max(1,self.frames),
                    survived=float(self.worm.active),recovered=float(self.recovered_frames>=30),
                    return_total=self.return_total,stage=self.stage,curl_anchor_lost=self.curl_anchor_lost,
                    backward_distance=self.backward_distance)

    def close(self):
        if self.world is not None:
            self.enc.close()
            self.world = None


def teacher_weight(args):
    return args.inchworm_dagger_weight


BACKWARD_SLACK = 30.   # pixels of forward travel that can offset later retreat
TURN_LOOP = tuple(v/MAX_BEND for v in (-1.4, -1.4, -1.4, -.34, 1.3984))


PHASES = ('plant','curl','land','release','crawl')


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


def label_action(env, agent, action):
    """What the expert would do in the states a policy visits; only About-face, and only up to the flip: once the worm
    faces the goal stretched out, the expert stops teaching."""
    if env.stage != 8:
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


def teacher_action(env, agent, action):
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
    return dict(task=SPEC.task,scenario_version=45,spec=SPEC.metadata(),horizon=args.horizon,
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
                curriculum_order=CURRICULUM,success_gates=SUCCESS_GATES,
                crawl_distance=dict(short=[60.,90.],long=[180.,300.],short_training_fraction=.25,
                                    validation='long_only'),
                decision_frames=DECISION_FRAMES,opponent_sha256=None,arenas=ARENAS,
                creature=asdict(enemy_kinds.kind('inchworm')),rewards=inchworm_rewards.from_args(args),
                pymunk=pymunk.version,physics_fps=60,twist='automatic_bounded_reflex',
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


def training_cases(iteration,count,stage,bootstrap=False,curriculum=None):
    current = curriculum['stage'] if curriculum else stage
    earlier = CURRICULUM[:CURRICULUM.index(current)]

    def practice_stage(j):
        """A quarter of every batch is Crawl (every later lesson builds on its gait); another quarter rotates through
        the earlier lessons, Crawl included; the rest is the current lesson."""
        if j%4 == 2 and 1 in earlier:
            return 1
        return earlier[(iteration+j//4)%len(earlier)] if earlier and j%4 == 0 else current
    return [dict(level=TRAIN_LEVELS[j%4],seed=10000+iteration*count+j+(500000000 if bootstrap else 0),
                 stage=DEMONSTRATED[j%len(DEMONSTRATED)] if bootstrap else practice_stage(j)) for j in range(count)]


def validation_cases():
    return [dict(level=level,seed=2000000000+stage*100+j*4+k,stage=stage)
            for stage in CURRICULUM for j,level in enumerate(VALIDATION_LEVELS) for k in range(3)]


def evaluation_cases():
    return [dict(level=level,seed=3000000000+stage*100+j*4+k,stage=stage)
            for stage in CURRICULUM for j,level in enumerate(HOLDOUT_LEVELS) for k in range(3)]


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


def initial_curriculum(args):
    return dict(stage=1,passing_blocks=0)


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
