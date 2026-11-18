"""The arrowhead, plume, immunity flash, hit bar and expanding death ring of a hunter-shaped Creature."""
import math
import pygame

import config
from rendering.thrust_plume import advance_thrust_particles, draw_thrust_plume
from rendering.ship_impact import ShipImpact, draw_ship_breakup

DEFAULT_COLOURS = {'color': (120, 255, 90), 'fill': (25, 110, 40), 'noseColor': (215, 255, 200),
            'cockpitColor': (15, 60, 30), 'engineColor': (170, 95, 20), 'engineLitColor': (255, 200, 80)}


HULL_POINTS = ((1.4, 0), (-0.9, 0.85), (-0.35, 0), (-0.9, -0.85))


class HunterShipBody:
    DEATH_DURATION = 48.0
    DEATH_FRICTION = 0.0
    BAR_WIDTH, BAR_HEIGHT = 40, 4

    def __init__(self, creature):
        self.impact = ShipImpact()
        creature.thrust_particles = []
        creature.thrusting = False
        self.colours = {key: tuple(creature.kind.tuning.get(key, value)) for key, value in DEFAULT_COLOURS.items()}
        self._shows_hit_points = creature.kind.tuning.get('showsHitPoints', False)

    def on_event(self, creature, name):
        if name == 'damaged':
            self.impact.hit()
        if name == 'hit_wall':
            dvx, dvy = getattr(creature, 'wall_impulse', (0.0, 0.0))
            speed = getattr(creature, 'max_speed', 0) or 1.0
            self.impact.hit(min(1.0, math.hypot(dvx, dvy) / speed))
        if name == 'integrated':
            advance_thrust_particles(creature, creature.integration_dt)

    def update(self, creature, dt, target=None):
        self.impact.update(dt)

    def update_death(self, creature, dt):
        pass

    def _place(self, c, x, y):
        heading = math.radians(c.angle)
        cs, sn = math.cos(heading), math.sin(heading)
        dx, dy = self.impact.offset(c.angle)
        return c.x + dx + (x * cs - y * sn) * c.radius, c.y + dy + (x * sn + y * cs) * c.radius

    def outline(self, c):
        return [self._place(c, x, y) for x, y in HULL_POINTS]

    def _nose_section(self, c, back_x):
        half_width = 0.85 * (1.4 - back_x) / 2.3
        return [self._place(c, 1.4, 0), self._place(c, back_x, half_width), self._place(c, back_x, -half_width)]

    def draw(self, screen, c):
        colours = self.colours
        colour = colours['color']
        if getattr(c, 'immunity_remaining', 0) and int(c.immunity_remaining * 20) % 2:
            colour = (235, 255, 230)
        outline = self.outline(c)
        pygame.draw.polygon(screen, colours['fill'], outline)
        middle = tuple((a + b) // 2 for a, b in zip(colours['fill'], colours['color']))
        pygame.draw.polygon(screen, middle, self._nose_section(c, -0.2))
        pygame.draw.polygon(screen, colours['noseColor'], self._nose_section(c, 0.45))
        pygame.draw.lines(screen, colour, True, outline, 1)
        cockpit = self._place(c, 0.5, 0)
        pygame.draw.circle(screen, colours['cockpitColor'], (round(cockpit[0]), round(cockpit[1])), 1)
        engine = self._place(c, -0.5, 0)
        burning = getattr(c, 'pilot_thrusting', False)
        pygame.draw.circle(screen, colours['engineLitColor'] if burning else colours['engineColor'],
                           (round(engine[0]), round(engine[1])), max(2, round(c.radius * 0.22)))
        draw_thrust_plume(c, screen, config.THRUST_PLUME_LENGTH * 0.6 if burning else 0)
        if self._shows_hit_points:
            self._draw_hit_points(screen, c)
        self.impact.draw(screen, c.get_pos(), c.radius, outline)

    def draw_death(self, screen, c):
        draw_ship_breakup(screen, c.get_pos(), c.radius, self.outline(c), self.colours['color'], c.death_progress)

    def _draw_hit_points(self, screen, c):
        bar = pygame.Rect(0, 0, round(self.BAR_WIDTH * c.radius / config.SHIP_SIZE), self.BAR_HEIGHT)
        bar.midbottom = (round(c.x), round(c.y - c.radius * 2.2))
        pygame.draw.rect(screen, (60, 20, 20), bar)
        fill = bar.copy()
        fill.width = round(bar.width * max(0, c.hit_points) / c.max_hit_points)
        if fill.width:
            pygame.draw.rect(screen, self.colours['color'], fill)
        pygame.draw.rect(screen, (150, 70, 70), bar, 1)
