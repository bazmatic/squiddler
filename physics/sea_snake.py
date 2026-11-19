"""Articulated sea snake mechanics, in pixels and game frames (60 frames/second).

Eight capsule links and seven torque-limited bend muscles, swimming in water. There are no grips and no bite:
propulsion comes only from anisotropic drag (each link resists sideways motion far more than lengthwise motion), so
a travelling wave of muscle commands pushes the body forward. The solver owns motion; controllers only supply a
SeaSnakeCommand.
"""
from dataclasses import dataclass
import math

import pymunk


LINKS = 8
MUSCLES = LINKS - 1
MAX_BEND = 0.9         # radians a joint can be commanded to bend; the hard stop sits .2 beyond
TANGENT_DRAG = 0.01    # lengthwise water drag, per frame (fraction of velocity shed)
NORMAL_DRAG = 1.5      # sideways drag is 150x lengthwise: this asymmetry turns a travelling wave into forward thrust
TURN_DRAG = 0.04       # angular water drag, per frame
SPINE_COUPLING = 0.2   # fraction of the muscle spring's stiffness that ties each joint to its neighbours


@dataclass(frozen=True)
class SeaSnakeCommand:
    """Signed joint setpoints (-1..1). Values outside the range are clipped; nonfinite values and wrong shapes fail."""
    muscles: tuple = (0.,) * MUSCLES

    def __post_init__(self):
        values = tuple(float(v) for v in self.muscles)
        if len(values) != MUSCLES or not all(math.isfinite(v) for v in values):
            raise ValueError('sea snake requires seven finite muscle commands')
        object.__setattr__(self, 'muscles', tuple(max(-1., min(1., v)) for v in values))


class SeaSnakePhysics:
    """A private articulated space with locally mirrored world walls.

    pose() and observe() are plain data, safe for rendering and snapshots. Wall objects need active/start/end.
    """
    def __init__(self, position, radius=10., angle=0., velocity=(0., 0.)):
        # Everything scales with radius (lengths by r, masses by r^2, moments and torques by r^4) and the drag is a
        # fraction per frame, so a bigger snake moves exactly like a scaled-up small one, in the same time.
        self.radius = radius
        self.length = radius * .7
        self.skin = radius * .3
        self.mass = radius * radius
        self.space = pymunk.Space()
        self.space.iterations = 30
        self.space.damping = .999
        self.space.collision_slop = radius*.005
        self.space.collision_bias = .1
        self.links, self.shapes = [], []
        self.activation = [0.] * MUSCLES
        self.surfaces = {}
        self.surface_contacts = [False] * LINKS
        mass = self.mass / LINKS
        direction = pymunk.Vec2d(math.cos(angle), math.sin(angle))
        for i in range(LINKS):
            a, b = (-self.length/2, 0.), (self.length/2, 0.)
            body = pymunk.Body(mass, pymunk.moment_for_segment(mass, a, b, self.skin))
            body.position = pymunk.Vec2d(*position)+direction*((i-(LINKS-1)/2)*self.length)
            body.velocity = velocity
            body.angle = angle
            shape = pymunk.Segment(body, a, b, self.skin)
            shape.friction, shape.elasticity = .3, .1
            shape.collision_type = 1
            self.space.add(body, shape)
            self.links.append(body)
            self.shapes.append(shape)
        for a, b in zip(self.links, self.links[1:]):
            joint = pymunk.PivotJoint(a, b, (self.length/2, 0.), (-self.length/2, 0.))
            joint.collide_bodies = False
            joint.error_bias = .01
            joint.max_bias = radius*.4
            limit = pymunk.RotaryLimitJoint(a, b, -MAX_BEND-.2, MAX_BEND+.2)
            limit.collide_bodies = False
            self.space.add(joint, limit)
        # Each muscle's spring (stiffness, damping, torque limit): the links never change, so these don't either.
        self.springs = []
        for a, b in zip(self.links, self.links[1:]):
            inertia = a.moment*b.moment/(a.moment+b.moment)
            stiffness = inertia*2.0
            self.springs.append((stiffness, 2*math.sqrt(stiffness*inertia)*.9, self.mass*self.length**2/28))
        self.space.on_collision(1, 2, post_solve=self._wall_contact)

    @property
    def centre(self):
        return self._mean('position')

    @property
    def velocity(self):
        return self._mean('velocity')

    def _mean(self, attribute):
        x = y = 0          # summed as plain floats: these are read several times a frame
        for body in self.links:
            vx, vy = getattr(body, attribute)
            x += vx
            y += vy
        return (x/LINKS, y/LINKS)

    @property
    def head(self):
        return tuple(self.links[-1].local_to_world((self.length/2, 0.)))

    @property
    def tail(self):
        return tuple(self.links[0].local_to_world((-self.length/2, 0.)))

    @property
    def heading(self):
        (hx, hy), (tx, ty) = self.head, self.tail
        return math.atan2(hy-ty, hx-tx)

    def pose(self):
        return [(tuple(b.local_to_world((-self.length/2, 0.))),
                 tuple(b.local_to_world((self.length/2, 0.)))) for b in self.links]

    def observe(self):
        """Body-relative proprioception; no privileged target/controller state."""
        heading = self.heading
        forward = pymunk.Vec2d(math.cos(heading), math.sin(heading))
        side = forward.perpendicular()
        velocity = pymunk.Vec2d(*self.velocity)
        return {
            'joint_angles': tuple((b.angle-a.angle)/MAX_BEND for a, b in zip(self.links, self.links[1:])),
            'joint_rates': tuple(b.angular_velocity-a.angular_velocity for a, b in zip(self.links, self.links[1:])),
            'activation': tuple(self.activation),
            'velocity': (velocity.dot(forward)/self.length, velocity.dot(side)/self.length),
            'contacts': tuple(self.surface_contacts),
        }

    def lay(self, pose, velocity=(0., 0.)):
        """Put the body in pose (one (tail end, head end) pair per link, end to end, a link's length apart), moving at
        velocity and not turning, its muscles already holding the bends between its links (so it stays as laid)."""
        angle = None
        for body, ((ax, ay), (bx, by)) in zip(self.links, pose):
            heading = math.atan2(by-ay, bx-ax)
            # unwound, as pymunk keeps them: each link's angle the turn nearest the one before
            angle = heading if angle is None else angle+math.atan2(math.sin(heading-angle), math.cos(heading-angle))
            body.position = ((ax+bx)/2, (ay+by)/2)
            body.angle = angle
            body.velocity = tuple(velocity)
            body.angular_velocity = 0.
            self.space.reindex_shapes_for_body(body)
        self.activation = [max(-1., min(1., (b.angle-a.angle)/MAX_BEND)) for a, b in zip(self.links, self.links[1:])]

    def curl(self, phase=0., bend=.65):
        """Lay the body out in a gentle S-curve about where it is, along its heading, as a resting snake lies: link
        headings swing by up to `bend` radians either side, one wave along the body, starting at `phase`."""
        heading, centre, step = self.heading, self.centre, self.length
        angles = [heading+bend*math.sin(phase+i*math.tau/LINKS) for i in range(LINKS)]
        points = [(0., 0.)]
        for angle in angles:
            points.append((points[-1][0]+step*math.cos(angle), points[-1][1]+step*math.sin(angle)))
        mx, my = sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points)
        shifted = [(x-mx+centre[0], y-my+centre[1]) for x, y in points]
        self.lay(list(zip(shifted, shifted[1:])), self.velocity)

    def translate(self, delta):
        """Reconcile an external collision correction."""
        if delta == (0., 0.):
            return
        for body in self.links:
            body.position += delta
            self.space.reindex_shapes_for_body(body)

    def impulse(self, impulse, point=None):
        """A point impulse spins the struck link; a bulk impulse shoves all links."""
        impulse = pymunk.Vec2d(*impulse)
        if point is None:
            for body in self.links:
                body.apply_impulse_at_world_point(impulse/LINKS, body.position)
        else:
            body = min(self.links, key=lambda b: b.position.get_distance(point))
            body.apply_impulse_at_world_point(impulse, point)

    def push_link(self, index, impulse):
        """Shove one link through its centre (no spin): how another snake it is pressed against pushes back."""
        body = self.links[index]
        body.apply_impulse_at_world_point(pymunk.Vec2d(*impulse), body.position)

    def nudge_link(self, index, delta):
        """Move one link bodily (its joints pull the rest of the body after it on the next solve)."""
        body = self.links[index]
        body.position += pymunk.Vec2d(*delta)
        self.space.reindex_shapes_for_body(body)

    def _wall_contact(self, arbiter, space, data):
        for shape in arbiter.shapes:
            if shape in self.shapes:
                self.surface_contacts[self.shapes.index(shape)] = True

    def _sync_surfaces(self, walls, gates):
        present = {}
        for surfaces in (walls, gates):
            for wall in surfaces:
                if not wall.active or wall.start == wall.end:
                    continue
                key = id(wall)
                geometry = (tuple(wall.start), tuple(wall.end))
                previous = self.surfaces.get(key)
                if previous is not None and previous[1] == geometry:
                    present[key] = previous
                    continue
                shape = pymunk.Segment(self.space.static_body, *geometry, 0.)
                shape.friction, shape.elasticity = .3, .1
                shape.collision_type = 2
                self.space.add(shape)
                present[key] = (wall, geometry, shape)
        for key, old in self.surfaces.items():
            if key not in present or present[key] is not old:
                self.space.remove(old[2])
        self.surfaces = present

    def _apply_drag(self, h):
        """Anisotropic water drag on every link: sideways resists 150x lengthwise, which is what makes swimming work."""
        for body in self.links:
            vx, vy = body.velocity
            c, s = body.rotation_vector
            along = vx*c+vy*s
            across = -vx*s+vy*c
            shed_along = TANGENT_DRAG*along
            shed_across = NORMAL_DRAG*across
            # Force that sheds this fraction of velocity per frame (mass * dv/dt), resolved back to world axes.
            fx = -(shed_along*c-shed_across*s)*body.mass
            fy = -(shed_along*s+shed_across*c)*body.mass
            body.apply_force_at_world_point((fx, fy), body.position)
            body.torque -= TURN_DRAG*body.angular_velocity*body.moment

    def step(self, command, dt, walls=(), gates=(), powered=True):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('sea snake dt must be finite and nonnegative')
        if dt == 0:
            return
        self._sync_surfaces(walls, gates)
        self.surface_contacts[:] = [False]*LINKS
        remaining = dt
        substeps = 0
        while remaining > 1e-9:
            # Bound travel per solve as well as duration: shots/blasts cannot tunnel
            # a thin link through a wall on a long game frame.
            speed = None
            for b in self.links:
                vx, vy = b.velocity
                v = math.sqrt(vx**2+vy**2)+abs(b.angular_velocity)*self.length
                if speed is None or v > speed:
                    speed = v
            substeps += 1
            if not math.isfinite(speed) or speed > self.skin*.4*4096 or substeps > math.ceil(dt)*4096:
                raise RuntimeError('sea snake physics exceeded its stable substep budget')
            desired = min(.25, self.skin*.4/max(1., speed))
            h = remaining / math.ceil(remaining / desired)
            angles = [b.angle for b in self.links]
            spins = [b.angular_velocity for b in self.links]
            torques = [0.]*LINKS
            bends = [b-a for a, b in zip(angles, angles[1:])]
            for i, (stiffness, damping, maximum) in enumerate(self.springs):
                self.activation[i] += max(-.15*h, min(.15*h, command.muscles[i]-self.activation[i]))
                error = self.activation[i]*MAX_BEND-bends[i]
                torque = max(-maximum, min(maximum, stiffness*error-damping*(spins[i+1]-spins[i]))) if powered else 0.
                # Neighbouring joints pull toward each other, so a bend spreads into an arc instead of a kink.
                neighbours = [bends[j] for j in (i-1, i+1) if 0 <= j < len(bends)]
                torque += SPINE_COUPLING*stiffness*(sum(neighbours)/len(neighbours)-bends[i])
                torques[i] -= torque
                torques[i+1] += torque
            for body, torque in zip(self.links, torques):
                body.torque = torque
            self._apply_drag(h)
            self.space.step(h)
            remaining -= h
