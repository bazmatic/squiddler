"""World adapter for the free-swimming sea snake, and its sting."""
import logging
import math

import config
from creatures.trait import Trait
from creatures.sea_snake_control import SeaSnakeController
from creatures.sea_snake_neural import driver_for
from creatures.traits.wallcrawl import nearby
from physics.sea_snake import LINKS, SeaSnakePhysics


WHIP = 0.15    # the share of a shot's knock that lands on the struck link alone
SHOVE = 0.1    # the share spread over the whole body: it slides the snake back, so keep it modest


RESET_FORGIVENESS = 600.    # frames without a solver reset after which the count of them starts again


def born(c):
    """A body for the creature where it is: lying in an S-curve, not stretched out like a rod. The curve's phase
    comes from where it is, so snakes differ but the same one is always born the same way."""
    k = c.snake_speed
    p = SeaSnakePhysics(c.get_pos(), c.radius, math.radians(c.angle), (c.vx/k, c.vy/k))
    p.curl(phase=(c.x*.0137+c.y*.0191) % math.tau)
    return p


def _closest(a0, a1, b0, b1):
    """The closest pair of points on two segments, as (point on a, point on b)."""
    ux, uy, vx, vy = a1[0]-a0[0], a1[1]-a0[1], b1[0]-b0[0], b1[1]-b0[1]
    wx, wy = a0[0]-b0[0], a0[1]-b0[1]
    a, b, c_, d, e = ux*ux+uy*uy, ux*vx+uy*vy, vx*vx+vy*vy, ux*wx+uy*wy, vx*wx+vy*wy
    det = a*c_-b*b
    s = max(0., min(1., (b*e-c_*d)/det)) if det > 1e-9 else 0.
    t = (b*s+e)/c_ if c_ > 1e-9 else 0.
    if t < 0.:
        t, s = 0., max(0., min(1., -d/a)) if a > 1e-9 else 0.
    elif t > 1.:
        t, s = 1., max(0., min(1., (b-d)/a)) if a > 1e-9 else 0.
    return (a0[0]+ux*s, a0[1]+uy*s), (b0[0]+vx*t, b0[1]+vy*t)


SOLID = .5      # the share of an overlap pushed out each frame, half by moving the links and half by shoving them
PADDING = 1.6   # they are held a little apart (a fraction of both skins more), so a fast closing never gets them deep in
GRIP = 1.       # the share of the speed two bodies press together at that is lost on touching


def keep_solid(snakes):
    """Snakes are solid to one another: where two bodies overlap, the links are pushed apart and stop pressing together
    (a shove, not a jam, so the solver is never given two moving bodies to crush)."""
    live = [c for c in snakes if c.active and c.snake_physics is not None and c.snake_pose is not None]
    touched = set()
    for i, a in enumerate(live):
        pa = a.snake_physics
        for b in live[i+1:]:
            pb = b.snake_physics
            reach = (pa.length+pb.length)*LINKS/2+pa.skin+pb.skin
            if math.dist(a.get_pos(), b.get_pos()) > reach:
                continue
            for ia, (a0, a1) in enumerate(a.snake_pose):
                for ib, (b0, b1) in enumerate(b.snake_pose):
                    pointa, pointb = _closest(a0, a1, b0, b1)
                    gap = math.dist(pointa, pointb)
                    depth = (pa.skin+pb.skin)*PADDING-gap
                    if depth <= 0.:
                        continue
                    if gap > 1e-6:
                        nx, ny = (pointa[0]-pointb[0])/gap, (pointa[1]-pointb[1])/gap
                    else:       # the bodies have crossed: part them the way their centres lie, or across link a
                        ca, cb = pa.links[ia].position, pb.links[ib].position
                        centres = math.dist(ca, cb)
                        if centres > 1e-6:
                            nx, ny = (ca[0]-cb[0])/centres, (ca[1]-cb[1])/centres
                        else:
                            nx, ny = a1[1]-a0[1], a0[0]-a1[0]
                            norm = math.hypot(nx, ny) or 1.
                            nx, ny = nx/norm, ny/norm
                    ma, mb = pa.mass/LINKS, pb.mass/LINKS
                    ka, kb = a.snake_speed, b.snake_speed
                    va, vb = pa.links[ia].velocity, pb.links[ib].velocity
                    closing = -((va[0]*ka-vb[0]*kb)*nx+(va[1]*ka-vb[1]*kb)*ny)     # in game units
                    push = depth*SOLID*.5+max(0., closing)*GRIP
                    sa, sb = depth*SOLID*.5*mb/(ma+mb), depth*SOLID*.5*ma/(ma+mb)
                    for snake, link, share, sign in ((pa, ia, sa, 1.), (pb, ib, sb, -1.)):
                        for neighbour, weight in ((link, 1.), (link-1, .5), (link+1, .5)):      # the body follows the link
                            if 0 <= neighbour < LINKS:
                                snake.nudge_link(neighbour, (sign*nx*share*weight, sign*ny*share*weight))
                    j = push*ma*mb/(ma+mb)
                    pa.push_link(ia, (nx*j/ka, ny*j/ka))
                    pb.push_link(ib, (-nx*j/kb, -ny*j/kb))
                    touched.update((id(a), id(b)))
    for c in live:
        if id(c) in touched:
            snapshot(c)


def snapshot(c):
    """Keep rendering/collision data independent of the live solver."""
    p = c.snake_physics
    if not all(math.isfinite(v) for v in (*p.centre, *p.velocity)):
        return      # a blown-up solver must not overwrite the last good pose (drawing and drops would go to NaN)
    c.x, c.y = p.centre
    c.vx, c.vy = p.velocity[0]*c.snake_speed, p.velocity[1]*c.snake_speed      # game units: a quick snake's own clock runs fast
    c.angle = math.degrees(p.heading)
    c.snake_pose = p.pose()
    c.snake_pose_centre = p.centre
    c.snake_skin = p.skin
    c.snake_muscles = tuple(p.activation)
    c.snake_observation = p.observe()


class SeaSnakes(Trait):
    NAME = 'sea_snakes'
    CROWD = True
    LIVE = ('snake_physics', 'snake_driver')

    def __init__(self, wavelengthFrames=32.0, shotKnock=config.MOMENTUM_TRANSFER_FACTOR, speed=1.0):
        if not (math.isfinite(wavelengthFrames) and wavelengthFrames > 0):
            raise ValueError('sea_snakes: wavelengthFrames must be finite and positive')
        if not (math.isfinite(shotKnock) and shotKnock >= 0):
            raise ValueError('sea_snakes: shotKnock must be finite and nonnegative')
        if not (math.isfinite(speed) and speed > 0):
            raise ValueError('sea_snakes: speed must be finite and positive')
        self.wavelength_frames = wavelengthFrames
        self.speed = speed      # how fast its own clock runs: a small snake swims in lengths, so it is slow unless hurried
        self.shot_knock = shotKnock

    def crowd(self, c):
        return 'sea'        # every kind of snake is solid to every other

    def on_crowd(self, snakes, enc, dt):
        keep_solid(snakes)

    def on_spawn(self, c, enc):
        c.snake_speed = self.speed
        c.snake_physics = born(c)
        c.snake_controller = SeaSnakeController(self.wavelength_frames)
        c.snake_command = None      # an explicit command overrides every controller
        c.snake_driver = driver_for(enc)    # None unless a trained policy is installed
        c.snake_resets = 0
        c.snake_calm = 0.      # frames since its solver last had to be reset
        snapshot(c)

    def on_shot(self, c, shot, enc):
        # A shot passes on a share of its speed to the whole snake (as for every shoved creature), and a part of that
        # lands on the struck link, so the body also whips round where it was hit.
        p = c.snake_physics
        share = self.shot_knock*max(0., getattr(shot, 'creature_damage', 1.))*p.mass/self.speed
        p.impulse((shot.vx*share*SHOVE, shot.vy*share*SHOVE))
        p.impulse((shot.vx*share*WHIP, shot.vy*share*WHIP), shot.get_pos())
        snapshot(c)

    def on_death(self, c, enc):
        snapshot(c)

    def on_close(self, c, enc):
        c.snake_physics = c.snake_driver = None

    def on_frame(self, c, enc, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('sea snake dt must be finite and nonnegative')
        p = c.snake_physics
        # Preserve impulses and corrections from the game's circle/obstacle systems.
        centre = p.centre
        p.translate((c.x-centre[0], c.y-centre[1]))
        velocity = p.velocity
        k = self.speed
        p.impulse(((c.vx/k-velocity[0])*p.mass, (c.vy/k-velocity[1])*p.mass))
        c.prev_x, c.prev_y = c.x, c.y
        remaining = dt
        while remaining > 1e-8:
            h = min(1., remaining, 1./max(1., self.speed))      # never more than one of its own frames at a time, so the brain keeps its beat
            own = h*self.speed      # the time it lives through in that step
            prey = enc.target_entity_for(c)
            target = prey.get_pos() if prey is not None and prey.active else None
            walls = list(nearby(enc.maze, p.centre, p.length*8+math.hypot(*p.velocity)*own))
            command = c.snake_command
            if command is None and c.snake_driver is not None:
                command = c.snake_driver.advance(c, enc, own)
            if command is None:
                command = c.snake_controller.command(p, own, target)
            try:
                p.step(command, own, walls, getattr(enc.maze, 'gate_segments', ()))
            except RuntimeError:
                # The solver blew up (jammed in rock too tight for it): lay the snake out again where it is, and if
                # that keeps happening let it die rather than take the game down.
                logging.getLogger(__name__).warning('Sea snake physics unstable at %s; resetting it', c.get_pos())
                # Resets far apart are forgiven: only a snake that keeps jamming (in rock, or in a spinning gate) dies of it.
                c.snake_resets = 1 if c.snake_calm > RESET_FORGIVENESS else c.snake_resets+1
                c.snake_calm = 0.
                if c.snake_resets > 3:
                    enc.kill(c)
                    return
                c.snake_physics = p = born(c)
                c.snake_controller = SeaSnakeController(self.wavelength_frames)
            snapshot(c)
            c.snake_calm += h
            remaining -= h


class StingsShip(Trait):
    """Any part of the body touching the ship costs the player, unless the shield is up, at most once per cooldown."""
    NAME = 'stingsShip'

    def __init__(self, cooldownFrames=45.0, stingKnock=4.0):
        if not (math.isfinite(cooldownFrames) and cooldownFrames >= 0):
            raise ValueError('stingsShip: cooldownFrames must be finite and nonnegative')
        if not (math.isfinite(stingKnock) and stingKnock >= 0):
            raise ValueError('stingsShip: stingKnock must be finite and nonnegative')
        self.cooldown_frames = cooldownFrames
        self.sting_knock = stingKnock

    def on_spawn(self, c, enc):
        c.sting_cooldown = 0.

    def on_frame(self, c, enc, dt):
        from utils import get_closest_point_on_line
        c.sting_cooldown = max(0., c.sting_cooldown-dt)
        ship = enc.ship
        if (c.sting_cooldown > 0 or ship is None or not ship.active or ship.is_shield_active()):
            return
        pos = ship.get_pos()
        segments = getattr(c.body, 'collision_segments', None)
        for a, b, skin in (segments(c) if segments is not None else [(c.get_pos(), c.get_pos(), c.radius)]):
            touch = get_closest_point_on_line(pos, a, b)
            if math.dist(pos, touch) <= skin+ship.radius:
                enc.scoring.record_enemy_collision()
                c.emit('rammed', enc)
                c.sting_cooldown = self.cooldown_frames
                self._hurt(ship, touch)
                return

    def _hurt(self, ship, touch):
        """The ship feels it: it flinches and is thrown clear of the body (the player's ship only; stand-ins may lack both)."""
        flinch = getattr(ship, 'flinch', None)
        if flinch is not None:
            flinch()
        if hasattr(ship, 'vx'):
            dx, dy = ship.x-touch[0], ship.y-touch[1]
            length = math.hypot(dx, dy) or 1.
            ship.vx += dx/length*self.sting_knock
            ship.vy += dy/length*self.sting_knock
