"""Deterministic ship hit and breakup effects; visual state never touches physics or random draws."""
import math

import pygame


class ShipImpact:
    DURATION = 14.0

    def __init__(self):
        self.remaining = 0.0
        self.duration = self.DURATION
        self.severity = 1.0

    def hit(self, severity=1.0):
        """severity 0..1: how direct and hard the strike was; scales length, recoil, sparks and flash.

        Returns whether a fresh effect started, so a sound can play once per strike."""
        severity = min(1.0, max(0.0, severity))
        # Sustained contact cannot restart the recoil every frame, but a harder strike overrides a fading softer one.
        if self.remaining <= 0 or severity > self.severity * self.remaining / self.duration:
            self.severity = severity
            self.duration = self.DURATION * (0.45 + 0.4 * severity)
            self.remaining = self.duration
            return True
        return False

    def update(self, dt):
        self.remaining = max(0.0, self.remaining - dt)

    def offset(self, angle):
        fraction = self.remaining / self.duration
        recoil = -(1.0 + 2.5 * self.severity) * fraction * math.sin((1 - fraction) * math.pi * 3)
        heading = math.radians(angle)
        return recoil * math.cos(heading), recoil * math.sin(heading)

    def vertices(self, vertices, angle):
        dx, dy = self.offset(angle)
        return [(x + dx, y + dy) for x, y in vertices]

    def draw(self, screen, centre, radius, vertices):
        if self.remaining <= 0:
            return
        fraction = self.remaining / self.duration
        sev = self.severity
        # Weak glancing hits flash a dim orange; hard ones flash bright and white-hot.
        strength = fraction * (0.35 + 0.65 * sev)
        colour = (255, round(100 + 155 * strength), round(70 + 185 * strength))
        pygame.draw.lines(screen, colour, True, vertices, max(1, round((1 + sev) * fraction)))
        count = 2 + round(8 * sev)
        for i in range(count):
            angle = i * math.tau / count
            reach = radius * (1.05 + (0.5 + 1.0 * sev) * (1 - fraction))
            length = (3 + 3 * sev) * fraction
            start = (centre[0] + math.cos(angle) * reach, centre[1] + math.sin(angle) * reach)
            end = (start[0] + math.cos(angle) * length, start[1] + math.sin(angle) * length)
            pygame.draw.line(screen, colour, start, end, 1)


def draw_ship_breakup(screen, centre, radius, vertices, colour, progress):
    """Break the actual hull into spinning plates, with a fading flash, sparks and shockwave."""
    if not 0 <= progress < 1:
        return
    spread = 2.4
    reach = math.ceil(radius * 7 * spread + 16)
    layer = pygame.Surface((reach * 2, reach * 2), pygame.SRCALPHA)
    fade = 1 - progress
    local = [(x - centre[0], y - centre[1]) for x, y in vertices]
    for i, (a, b) in enumerate(zip(local, local[1:] + local[:1])):
        direction = math.atan2(a[1] + b[1], a[0] + b[0])
        drift = radius * 4 * progress * spread
        spin = progress * (i + 1) * (1 if i % 2 else -1)
        cs, sn = math.cos(spin), math.sin(spin)
        dx, dy = reach + math.cos(direction) * drift, reach + math.sin(direction) * drift
        points = [(dx + x * cs - y * sn, dy + x * sn + y * cs) for x, y in ((0, 0), a, b)]
        pygame.draw.polygon(layer, (*colour, round(150 * fade)), points)
        pygame.draw.lines(layer, (255, 220, 160, round(255 * fade)), True, points, 1)
    for i in range(16):
        angle = i * math.tau / 16 + 0.17
        distance = radius * (0.5 + progress * (3 + i % 4)) * spread
        x, y = reach + math.cos(angle) * distance, reach + math.sin(angle) * distance
        pygame.draw.line(layer, (255, 170 + i % 3 * 30, 90, round(255 * fade)),
                         (x, y), (x - math.cos(angle) * 8 * fade, y - math.sin(angle) * 8 * fade), 2)
    pygame.draw.circle(layer, (*colour, round(180 * fade)), (reach, reach),
                       max(1, round(radius * (1 + 5 * progress) * spread)), max(1, round(3 * fade)))
    if progress < 0.2:
        pygame.draw.circle(layer, (255, 245, 220, round(255 * (1 - progress / 0.2))),
                           (reach, reach), max(1, round(radius * (1 + progress * 5) * 1.7)))
    screen.blit(layer, (round(centre[0]) - reach, round(centre[1]) - reach))
