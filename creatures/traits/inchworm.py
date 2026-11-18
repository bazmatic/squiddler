"""World adapter for articulated locomotion, grip interruption and feeding."""
import logging
import math

from creatures.trait import Trait
from creatures.inchworm_control import REST_POSE, InchwormController
from creatures.traits.rooted import wall_between
from creatures.traits.wallcrawl import contact, nearby
from physics.inchworm import InchwormCommand, InchwormPhysics


def snapshot(c):
    """Keep rendering/collision data independent of live solver and target objects."""
    p = c.worm_physics
    c.x, c.y = p.centre
    c.vx, c.vy = p.velocity
    c.angle = math.degrees(p.heading)
    c.worm_pose = p.pose()
    c.worm_pose_centre = p.centre
    c.worm_skin = p.skin
    c.worm_roll = tuple(p.roll)
    c.worm_front, c.worm_rear = p.head, p.tail
    c.worm_muscles = tuple(p.activation)
    c.worm_front_grip = p.grips[1] is not None or p.grab is not None
    c.worm_rear_grip = p.grips[0] is not None
    c.worm_feet = tuple((g.foot, g.normal) if g else None for g in p.grips)
    c.worm_prey = p.prey if p.grab is not None else None
    c.worm_observation = p.observe()


class Inchworms(Trait):
    NAME = 'inchworms'
    LIVE = ('worm_prey', 'worm_physics', 'worm_driver')

    def __init__(self, strideFrames=160.0, biteFrames=60.0):
        if not all(math.isfinite(v) and v > 0 for v in (strideFrames, biteFrames)):
            raise ValueError('inchworms: frame intervals must be finite and positive')
        self.stride_frames, self.bite_frames = strideFrames, biteFrames

    def on_spawn(self, c, enc):
        surface = contact(enc.maze, c.get_pos())
        angle = math.radians(c.angle)
        if surface is not None and surface[3] <= c.radius+6:
            nx, ny = surface[2]
            angle = math.atan2(nx, -ny)
        # A world can name the posture worms appear in (training lessons start them straight).
        c.worm_posture = getattr(enc.world, 'inchworm_posture', REST_POSE)
        c.worm_physics = InchwormPhysics(c.get_pos(), c.radius, angle, (c.vx, c.vy), c.worm_posture)
        c.worm_controller = InchwormController(self.stride_frames)
        # None selects the reflex controller. Even an all-zero command overrides it.
        c.worm_command = None
        c.worm_bite = c.worm_stun = 0.
        c.worm_resets = 0
        c.worm_state = 'seeking'
        c.worm_bite_interval = self.bite_frames
        c.worm_last_command = InchwormCommand()
        from creatures.inchworm_neural import driver_for
        c.worm_driver = driver_for(enc)
        snapshot(c)

    def on_shot(self, c, shot, enc):
        p = c.worm_physics
        strength = max(0., getattr(shot, 'creature_damage', 1.))
        mass = shot.radius**2*strength
        p.impulse((shot.vx*mass, shot.vy*mass), shot.get_pos())
        snapshot(c)

    def on_event(self, c, name, enc):
        if name in ('hit', 'shoved'):
            p = c.worm_physics
            if name == 'shoved':
                p.impulse(((c.vx-p.velocity[0])*p.mass, (c.vy-p.velocity[1])*p.mass))
            c.worm_stun = 30.
            p.release()
            c.worm_bite = 0.
            snapshot(c)

    def on_death(self, c, enc):
        c.worm_physics.release()
        c.worm_bite = 0.
        snapshot(c)

    def on_close(self, c, enc):
        if c.worm_physics is not None:
            c.worm_physics.release()
        c.worm_physics = c.worm_prey = c.worm_driver = None

    def on_frame(self, c, enc, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('inchworm dt must be finite and nonnegative')
        p = c.worm_physics
        # Preserve impulses and corrections from the game's circle/obstacle systems.
        centre = p.centre
        p.translate((c.x-centre[0], c.y-centre[1]))
        velocity = p.velocity
        p.impulse(((c.vx-velocity[0])*p.mass, (c.vy-velocity[1])*p.mass))
        c.prev_x, c.prev_y = c.x, c.y
        remaining = dt
        while remaining > 1e-8:
            h = min(1., remaining)
            target = c.worm_prey if c.worm_prey is not None and c.worm_prey.active else self._prey(c, p, enc)
            valid = (target is not None and target.active
                     and not getattr(target, 'is_shield_active', lambda: False)()
                     and not wall_between(p.head, target.get_pos(), enc.maze))
            # Only a nearby adversary is mirrored; distant bodies cannot touch us.
            if target is not None and math.dist(p.centre, target.get_pos()) > p.length*7+target.radius:
                target = None
                valid = False
            walls = list(nearby(enc.maze, p.centre, p.length*7+p.reach+math.hypot(*p.velocity)*h))
            neural = (c.worm_driver.advance(c, enc, h)
                      if c.worm_command is None and c.worm_driver is not None else None)
            stunned = c.worm_stun > 0
            if stunned:
                command = InchwormCommand()
                c.worm_stun = max(0., c.worm_stun-h)
            else:
                command = c.worm_command
                if command is None:
                    command = neural if neural is not None else c.worm_controller.command(p, h, target if valid else None)
            c.worm_last_command = command
            try:
                p.step(command, h, walls, getattr(enc.maze, 'gate_segments', ()), target,
                       can_grab=valid, powered=not stunned)
            except RuntimeError:
                # The solver blew up (a body jammed in rock too tight for it): stand the worm up again where it is, and
                # if that keeps happening let it die rather than take the game down.
                logging.getLogger(__name__).warning('Inchworm physics unstable at %s; resetting it', c.get_pos())
                c.worm_resets += 1
                if c.worm_resets > 3:
                    enc.kill(c)
                    return
                p.release()
                c.worm_physics = p = InchwormPhysics(c.get_pos(), c.radius, math.radians(c.angle), posture=c.worm_posture)
                c.worm_controller = InchwormController(self.stride_frames)
                c.worm_bite = 0.
            snapshot(c)
            if command.bite and valid and p.mouth_contact:
                c.worm_bite += h
                c.worm_state = 'eating'
                if c.worm_bite >= self.bite_frames:
                    c.worm_bite -= self.bite_frames
                    self._bite(target, enc)
                    if not target.active:
                        p.release()
                        c.worm_bite = 0.
                        snapshot(c)
            else:
                c.worm_bite = 0.
                c.worm_state = ('grabbing' if p.grab else 'rearing' if c.worm_controller.attacking
                                else 'walking' if any(p.grips) else 'drifting')
            remaining -= h

    @staticmethod
    def _prey(c, p, enc):
        """An inchworm eats everything alive except other inchworms, the player ship and hunters included: the
        nearest of them that it can reach (in range, no wall between), else None."""
        from game_handlers.combat_targets import nearest_friendly
        candidates = [t for t in getattr(enc.world, 'additional_targets', ()) if t.active]
        candidates += [o for o in enc.roster.alive() if o is not c and o.type != 'inchworm']
        usual = nearest_friendly(c, enc.world.ship, enc.world.hunter)
        if usual is not None:
            candidates.append(usual)
        centre, head = p.centre, p.head
        for t in sorted(candidates, key=lambda t: (t.x-c.x)**2+(t.y-c.y)**2):
            if math.dist(centre, t.get_pos()) <= p.length*7+t.radius and not wall_between(head, t.get_pos(), enc.maze):
                return t
        return None

    @staticmethod
    def _bite(target, enc):
        if target is enc.ship:
            target.record_hit(0.0, 0.0)  # a bite hurts even when it imparts no momentum
            enc.scoring.record_enemy_collision()
        elif hasattr(target, 'health'):
            from creatures.traits.energy import damage
            damage(target, enc)
        elif hasattr(target, 'hit_points'):
            home = getattr(target, 'encounter', None) or enc    # prey from another encounter (a team battle) dies in its own
            target.hit_points -= 1
            target.emit('hit', home)
            if target.hit_points <= 0:
                home.kill(target)
