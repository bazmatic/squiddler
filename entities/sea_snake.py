"""The sea snake: a thick, tapering, banded body drawn along a smooth curve through the simulated links, with a slow
pulsing glow and two glowing eyes."""
import math
import random

import pygame

import config
from physics.sea_snake import LINKS, SeaSnakeCommand

FLANK = (186, 136, 228)     # pale belly edge
BODY = (124, 58, 168)      # bruised violet hide
BAND = (40, 12, 60)        # dark bands, one per link
DORSAL = (30, 10, 44)      # dark stripe down the back
EDGE = (70, 32, 98)        # outline
GLOW = (160, 80, 255)
SPOT = (110, 255, 215)     # photophores pulsing down the flanks
EYE = (225, 255, 120)
PUPIL = (24, 6, 20)
HEAD_WIDTH = 1.45   # half-width where the head is widest, in skins
BODY_WIDTH = 1.0    # half-width of the thickest part of the body, in skins
TAIL_WIDTH = .34    # half-width of the slim part of the tail, in skins
SNOUT = 1.5         # how far the snout reaches past the head's last point, in head half-widths
SMOOTH = 3          # curve points per span between spine points
SPOT_DIM = .2       # photophore brightness between pulses
GLOW_SHRINK = 3     # the halo is drawn this many times smaller, then smoothed up: soft and cheap

# The death, in frames after the kill: it thrashes, then its links burst one after another from the head, then the
# blast (the `explodes` trait's fuse, which should equal FINALE) goes off and embers drift down.
SPASM = 24          # how long it thrashes before the first link goes
BURST_GAP = 6       # frames between one link bursting and the next
FINALE = SPASM+BURST_GAP*LINKS+6
DEATH_FRAMES = FINALE+72
BURST_BITS = 14     # sparks from each bursting link
FINALE_BITS = 44    # embers from the last burst


def _default_pose(c):
    """Before its body is built: straight along its heading, centred on it."""
    length = c.kind.tuning['size']*.7
    dx, dy = math.cos(math.radians(c.angle))*length, math.sin(math.radians(c.angle))*length
    return tuple((((i-4)*dx+c.x, (i-4)*dy+c.y), ((i-3)*dx+c.x, (i-3)*dy+c.y)) for i in range(8))


def _curve(points):
    """Catmull-Rom through the points, SMOOTH samples per span, ending exactly on the last point."""
    out = []
    n = len(points)
    for i in range(n-1):
        p0, p1, p2, p3 = points[max(0, i-1)], points[i], points[i+1], points[min(n-1, i+2)]
        for k in range(SMOOTH):
            t = k/SMOOTH
            t2, t3 = t*t, t*t*t
            out.append(tuple(.5*(2*b+(c-a)*t+(2*a-5*b+4*c-d)*t2+(3*b-a-3*c+d)*t3)
                             for a, b, c, d in zip(p0, p1, p2, p3)))
    out.append(tuple(points[-1]))
    return out


def _normals(curve):
    """Unit left-hand normals; a zero-length step borrows its neighbour's, and a curve with none faces +y."""
    n = len(curve)
    normals = [None]*n
    for i in range(n):
        a, b = curve[max(0, i-1)], curve[min(n-1, i+1)]
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length > 1e-6:
            normals[i] = (-dy/length, dx/length)
    known = next((v for v in normals if v is not None), (0., 1.))
    for i in range(n):
        if normals[i] is None:
            normals[i] = known
        known = normals[i]
    return normals


def _half_width(t, skin):
    """From the tail (t=0): a rounded paddle on a slim tail, swelling to a body of even thickness, a pinched neck, then a
    head wider than the neck."""
    rise = min(1., t/.45)
    body = TAIL_WIDTH+(BODY_WIDTH-TAIL_WIDTH)*rise*rise*(3-2*rise)
    paddle = .2*math.exp(-((t-.05)/.045)**2)
    neck = -.28*math.exp(-((t-.86)/.03)**2)
    head = (HEAD_WIDTH-BODY_WIDTH)*math.exp(-((t-.95)/.04)**2)
    tip = math.sqrt(min(1., (t+.01)/.05))      # closes the tail's end to a round tip
    return skin*(body+paddle+neck+head)*tip


class _Shape:
    """The body outline at one moment: a smooth spine, its normals and its half-widths."""
    def __init__(self, spine, skin, width=None, headed=True):
        width = width or _half_width
        self.headed = headed
        self.curve = _curve(spine)
        self.normals = _normals(self.curve)
        count = len(self.curve)
        self.widths = [width(i/(count-1), skin) for i in range(count)]
        nx, ny = self.normals[-1]
        self.forward = (ny, -nx)     # at the head

    def sides(self, scale=1., extra=0., offset=(0., 0.), zoom=1.):
        """Left and right edges at scale times the half-width plus extra, moved by -offset, then zoomed."""
        ox, oy = offset
        left, right = [], []
        for (x, y), (nx, ny), w in zip(self.curve, self.normals, self.widths):
            d = w*scale+extra
            x, y, nx, ny = (x-ox)*zoom, (y-oy)*zoom, nx*d*zoom, ny*d*zoom
            left.append((x+nx, y+ny))
            right.append((x-nx, y-ny))
        return left, right

    def snout(self, radius, offset=(0., 0.), zoom=1.):
        """A rounded point ahead of the head (half an ellipse SNOUT times as long as it is wide), from its left side round
        to its right."""
        (x, y), (nx, ny), (fx, fy) = self.curve[-1], self.normals[-1], self.forward
        x, y = x-offset[0], y-offset[1]
        return [((x+radius*(math.cos(a)*nx+SNOUT*math.sin(a)*fx))*zoom, (y+radius*(math.cos(a)*ny+SNOUT*math.sin(a)*fy))*zoom)
                for a in (math.pi*k/8 for k in range(1, 8))]

    def folds(self, left, right):
        """Whether an outline runs backwards anywhere (the inside of a curl tighter than it is wide). A narrower
        outline folds only where a wider one does, so checking the widest of several serves them all."""
        curve = self.curve
        for edge in (left, right):
            for i in range(len(edge)-1):
                (ax, ay), (bx, by) = edge[i], edge[i+1]
                (cx, cy), (dx, dy) = curve[i], curve[i+1]
                if (bx-ax)*(dx-cx)+(by-ay)*(dy-cy) <= 0:
                    return True
        return False

    def fill(self, surface, rgba, scale=1., extra=0., offset=(0., 0.), zoom=1., first=0, last=None, folds=None):
        """Fill from curve point first to last (to the head adds the rounded snout): one polygon, or, if the outline
        folds (checked here unless the caller knows), one quad per step so a tight curl cannot leave holes.
        Returns the edges and whether they fold."""
        left, right = self.sides(scale, extra, offset, zoom)
        if folds is None:
            folds = self.folds(left, right)
        end = len(left)-1
        last = end if last is None else last
        snout = self.snout(self.widths[-1]*scale+extra, offset, zoom) if last == end and self.headed else []
        if not folds:
            pygame.draw.polygon(surface, rgba, left[first:last+1]+snout+right[first:last+1][::-1])
        else:
            for i in range(first, last):
                pygame.draw.polygon(surface, rgba, (left[i], left[i+1], right[i+1], right[i]))
            if snout:
                pygame.draw.polygon(surface, rgba, [left[-1]]+snout+[right[-1]])
        return left, right, folds

    def eyes(self, offset=(0., 0.)):
        (x, y), (nx, ny), (fx, fy) = self.curve[-1], self.normals[-1], self.forward
        w = self.widths[-1]
        return [(x+fx*w*.15+nx*w*.62*sign-offset[0], y+fy*w*.15+ny*w*.62*sign-offset[1]) for sign in (-1, 1)]


class SeaSnakeBody:
    FLANK, BODY, BAND, DORSAL, EDGE, GLOW, SPOT = FLANK, BODY, BAND, DORSAL, EDGE, GLOW, SPOT   # a subclass recolours
    EYES = True
    width = staticmethod(_half_width)

    DEATH_DURATION = float(DEATH_FRAMES)
    DEATH_FRICTION = .94

    def __init__(self, creature):
        self.phase = 0.
        self.gone = 0           # links burst so far, counting from the head
        self.finale = False
        self.bits = []          # sparks and embers: [x, y, vx, vy, age, life, size, colour]
        self.rings = []         # shock rings and flashes: [x, y, age, life, radius, colour, solid]

    def on_event(self, creature, name):
        pass

    def update(self, creature, dt, target=None):
        self.phase = (self.phase+dt*.04) % math.tau

    def update_death(self, c, dt):
        t = c.death_progress*self.DEATH_DURATION
        self._move_bits(dt)
        p = getattr(c, 'snake_physics', None)
        if p is not None:
            from creatures.traits.sea_snake import snapshot
            from creatures.traits.wallcrawl import nearby
            maze = c.encounter.maze
            walls = list(nearby(maze, p.centre, p.length*8+math.hypot(*p.velocity)*dt))     # a corpse still stops at rock
            # It thrashes while it dies, then goes slack.
            muscles = tuple(math.sin(t*.9+i*2.1)*(1.-t/SPASM) for i in range(LINKS-1)) if t < SPASM else ()
            try:
                p.step(SeaSnakeCommand(muscles) if muscles else SeaSnakeCommand(), dt, walls,
                       getattr(maze, 'gate_segments', ()), powered=t < SPASM)
            except RuntimeError:
                c.snake_physics = None   # hold the last pose and just fade
            else:
                snapshot(c)
        while self.gone < LINKS and t >= SPASM+BURST_GAP*self.gone:
            self._burst(c, LINKS-1-self.gone, self.gone)
            self.gone += 1
        if not self.finale and t >= FINALE:
            self.finale = True
            self._finale(c)

    def _sound(self, c, pitch):
        manager = getattr(getattr(getattr(c, 'encounter', None), 'world', None), 'sound_manager', None)
        if manager is not None:
            manager.play_enemy_destroy(pitch)

    def _burst(self, c, link, order):
        """One link goes up: a flash ring and a spray of sparks where it was."""
        (ax, ay), (bx, by), skin = self.collision_segments(c)[link]
        x, y = (ax+bx)/2, (ay+by)/2
        colours = (SPOT, EYE, GLOW, FLANK)
        for _ in range(BURST_BITS):
            angle, speed = random.uniform(0, math.tau), random.uniform(1., 4.5)
            self.bits.append([x, y, math.cos(angle)*speed, math.sin(angle)*speed, 0., random.uniform(26, 56),
                              skin*random.uniform(.18, .36), random.choice(colours)])
        self.rings.append([x, y, 0., 22., skin*4., SPOT, False])
        self.rings.append([x, y, 0., 9., skin*1.9, (255, 255, 235), True])
        self._sound(c, .8+.1*order)

    def _finale(self, c):
        """The last bang: embers thrown wide that drift down, and a big slow ring."""
        segments = self.collision_segments(c)
        x = sum(a[0]+b[0] for a, b, _ in segments)/(2*len(segments))
        y = sum(a[1]+b[1] for a, b, _ in segments)/(2*len(segments))
        skin = segments[0][2]
        for _ in range(FINALE_BITS):
            angle, speed = random.uniform(0, math.tau), random.uniform(1.5, 8.)
            self.bits.append([x, y, math.cos(angle)*speed, math.sin(angle)*speed, 0., random.uniform(40, 70),
                              skin*random.uniform(.2, .45), random.choice((GLOW, FLANK, SPOT, EYE, GLOW))])
        self.rings.append([x, y, 0., 34., skin*9., GLOW, False])
        self.rings.append([x, y, 0., 26., skin*6., EYE, False])
        self.rings.append([x, y, 0., 20., skin*3., SPOT, False])
        self.rings.append([x, y, 0., 14., skin*3.6, (255, 255, 235), True])

    def _move_bits(self, dt):
        drag = .96**dt
        for bit in self.bits:
            bit[0] += bit[2]*dt
            bit[1] += bit[3]*dt
            bit[2] *= drag
            bit[3] *= drag
            bit[4] += dt
        for ring in self.rings:
            ring[2] += dt
        self.bits = [b for b in self.bits if b[4] < b[5]]
        self.rings = [r for r in self.rings if r[2] < r[3]]

    def _draw_bits(self, screen):
        """Sparks streaking and cooling towards the dark, rings spreading and thinning, and flashes."""
        background = config.COLOR_BACKGROUND
        for x, y, vx, vy, age, life, size, rgb in self.bits:
            k = 1.-age/life
            colour = tuple(int(bg+(v-bg)*k) for v, bg in zip(rgb, background))
            width = max(1, round(size*(.4+.6*k)))
            pygame.draw.line(screen, colour, (x, y), (x-vx*2.5, y-vy*2.5), width)
            pygame.draw.circle(screen, colour, (x, y), width)
        for x, y, age, life, radius, rgb, solid in self.rings:
            k = age/life
            if solid:      # a flash: it shrinks and goes out fast
                colour = tuple(int(bg+(v-bg)*(1.-k)**3) for v, bg in zip(rgb, background))
                pygame.draw.circle(screen, colour, (x, y), max(1, radius*(1.-.6*k)))
                continue
            colour = tuple(int(bg+(v-bg)*(1.-k)) for v, bg in zip(rgb, background))
            reach = radius*(1.-(1.-k)**2)
            if reach > 2:
                pygame.draw.circle(screen, colour, (x, y), reach, max(1, round(7*(1.-k))))

    def collision_segments(self, c):
        pose = getattr(c, 'snake_pose', None) or _default_pose(c)
        centre = getattr(c, 'snake_pose_centre', c.get_pos())
        scale = c.inflation.scale

        def point(p):
            return (c.x+(p[0]-centre[0])*scale, c.y+(p[1]-centre[1])*scale)
        skin = getattr(c, 'snake_skin', c.kind.tuning['size']*.3)*scale
        return tuple((point(a), point(b), skin) for a, b in pose)

    def draw(self, screen, c):
        self._draw(screen, c, 1.)

    def draw_death(self, screen, c):
        t = c.death_progress*self.DEATH_DURATION
        # The links already burst are gone; while it thrashes the body flares.
        flare = (.4+.4*math.sin(t*1.4))*(1.-t/SPASM) if t < SPASM else 0.
        self._draw(screen, c, 1., LINKS-self.gone, flare)
        self._draw_bits(screen)

    def _draw(self, screen, c, fade, keep=LINKS, flare=0.):
        segments = self.collision_segments(c)
        skin = segments[0][2]
        segments = segments[:keep]
        headed = keep == LINKS      # once the head has burst, the stump is cut off square
        if not segments:
            return

        def color(rgb):
            rgb = tuple(v+(255-v)*flare for v in rgb)
            return tuple(int(bg+(v-bg)*fade) for v, bg in zip(rgb, config.COLOR_BACKGROUND))

        # The spine runs tail tip, link midpoints, head tip.
        spine = [segments[0][0]]+[((a[0]+b[0])/2, (a[1]+b[1])/2) for a, b, _ in segments]+[segments[-1][1]]
        if config.SETTINGS.game.lowGraphics:
            count = len(spine)
            for i in range(count-1):
                width = 2*self.width(i/(count-1), skin)
                pygame.draw.line(screen, color(self.BODY), spine[i], spine[i+1], max(1, round(width)))
            return

        # What is left keeps the width it had, not a new taper and head.
        width = self.width if headed else (lambda t, skin: self.width(t*keep/LINKS, skin))
        shape = _Shape(spine, skin, width, headed)
        count = len(shape.curve)
        pulse = max(.5+.5*math.sin(self.phase), flare)
        self._draw_glow(screen, shape, skin, pulse, fade)
        # Pale belly edge, the violet hide inside it, a dark stripe down the back, dark bands across it.
        left, right, folds = shape.fill(screen, color(self.FLANK))
        shape.fill(screen, color(self.BODY), .72, first=1, folds=folds)
        links = range(SMOOTH, count-SMOOTH, SMOOTH)
        stripe_left, stripe_right, _ = shape.fill(screen, color(self.DORSAL), .16, first=1, last=count-3,
                                                  folds=folds)
        pygame.draw.aalines(screen, color(self.DORSAL), False, stripe_left[1:-2])
        pygame.draw.aalines(screen, color(self.DORSAL), False, stripe_right[1:-2])
        band_left, band_right = shape.sides(.8)
        for i in links:
            pygame.draw.polygon(screen, color(self.BAND), band_left[i:i+2]+band_right[i:i+2][::-1])
        pygame.draw.aalines(screen, color(self.EDGE), True,
                            left+(shape.snout(shape.widths[-1]) if headed else [])+right[::-1])
        # Photophores along both flanks between the bands, a pulse running down from the head.
        spot_left, spot_right = shape.sides(.55)
        for k, i in enumerate(links):
            light = SPOT_DIM+(1-SPOT_DIM)*max(0., math.sin(self.phase*3+k*.8))**3
            rgb = tuple(int(b+(s-b)*light) for s, b in zip(self.SPOT, self.BODY))
            for edge in (spot_left, spot_right):
                (ax, ay), (bx, by) = edge[i+1], edge[i+2]     # between this band and the next one
                pygame.draw.circle(screen, color(rgb), ((ax+bx)/2, (ay+by)/2), max(1., skin*.12))
        # Two glowing eyes with slit pupils.
        if not self.EYES or not headed:
            return
        self._draw_eye_shine(screen, shape, skin, pulse, fade)
        fx, fy = shape.forward
        for ex, ey in shape.eyes():
            pygame.draw.circle(screen, color(EYE), (ex, ey), max(1., skin*.24))
            pygame.draw.line(screen, color(PUPIL), (ex-fx*skin*.17, ey-fy*skin*.17),
                             (ex+fx*skin*.17, ey+fy*skin*.17), max(1, round(skin*.09)))

    def _draw_glow(self, screen, shape, skin, pulse, fade):
        """A slow pulsing halo, alpha-blended through a layer only the size of the snake (drawn small, smoothed up)."""
        if fade <= 0:
            return
        margin = skin*2.5
        xs, ys = [p[0] for p in shape.curve], [p[1] for p in shape.curve]
        left, top = math.floor(min(xs)-margin), math.floor(min(ys)-margin)
        rect = pygame.Rect(left, top, math.ceil(max(xs)+margin)-left, math.ceil(max(ys)+margin)-top)
        if not rect.colliderect(screen.get_rect()):
            return
        zoom = 1/GLOW_SHRINK
        small = pygame.Surface((max(1, math.ceil(rect.width*zoom)), max(1, math.ceil(rect.height*zoom))),
                               pygame.SRCALPHA)
        offset = rect.topleft
        *_, folds = shape.fill(small, (*self.GLOW, int((20+25*pulse)*fade)), extra=skin*1.4, offset=offset, zoom=zoom)
        shape.fill(small, (*self.GLOW, int((45+45*pulse)*fade)), extra=skin*.6, offset=offset, zoom=zoom, folds=folds)
        screen.blit(pygame.transform.smoothscale(small, rect.size), offset)

    def _draw_eye_shine(self, screen, shape, skin, pulse, fade):
        """A soft glow round each eye, through a small layer over the head."""
        if fade <= 0:
            return
        size = math.ceil(skin*3.5)
        rect = pygame.Rect(0, 0, size, size)
        rect.center = (round(shape.curve[-1][0]), round(shape.curve[-1][1]))
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        eyes = shape.eyes(rect.topleft)
        for radius, alpha in ((.6, 30+30*pulse), (.4, 70+60*pulse)):
            for eye in eyes:
                pygame.draw.circle(layer, (*EYE, int(alpha*fade)), eye, skin*radius)
        screen.blit(layer, rect)
