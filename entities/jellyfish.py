"""Jellyfish body for aggressive enemies.

An aggressive enemy chases the player and stings on contact. This module draws
it as a small jellyfish: a translucent bell that leads the way and pulses as it
swims, with stinging tentacles trailing behind.

The jellyfish is purely visual: it is the `jelly` Body of a Creature. Movement,
chasing and collision belong to the creature's traits.
"""

import math
import random
from typing import List, Optional, Tuple, TYPE_CHECKING

import pygame
import config
from entities.tentacle_chain import drag_chain
from rendering import visual_effects
from utils import distance

if TYPE_CHECKING:
    from creatures.creature import Creature

Point = Tuple[float, float]


def _ease(progress: float, span: Tuple[float, float]) -> float:
    """Eased progress (0.0 to 1.0) through a (start, end) span."""
    start, end = span
    t = max(0.0, min(1.0, (progress - start) / (end - start)))
    return t * t * (3.0 - 2.0 * t)


class Jellyfish:
    """Animation state and drawing for an aggressive enemy's jellyfish body.

    Lengths are multiples of the enemy radius, in the jellyfish's own frame:
    x runs along its heading and y across it.

    Attributes:
        heading: Direction the bell points, in degrees. Follows the enemy's
            angle smoothly.
        pulse_phase: Swimming pulse phase in radians.
        agitation: 0.0 drifting, 1.0 chasing the player.
        flutter: 0.0 steady, 1.0 backing away from an obstacle.
        tentacles: One chain of world-space points per tentacle.
    """

    COLOR = (150, 215, 255)  # Pale blue
    BELL_RIM_X = -0.25  # Open end of the bell
    BELL_APEX_X = 1.05  # Rounded front
    BELL_HALF_WIDTH = 1.1
    BELL_POINTS = 12  # Outline points per side
    BELL_SQUEEZE = 0.26  # Width lost at full contraction
    BELL_STRETCH = 0.14  # Length gained at full contraction
    BELL_FLARE = 0.12  # Extra width of the relaxed rim
    BELL_ALPHA = 135
    BELL_CORE_ALPHA = 185
    LOBE_OFFSET = 0.21  # Spacing of the four lobes inside the bell
    TURN_SPEED = 10.0  # Degrees per frame
    PULSE_SPEED = 0.06  # Radians per frame when drifting
    CHASE_PULSE_SPEED = 0.14  # Extra pulse speed when chasing
    MOOD_SMOOTHING = 0.1  # Fraction of the gap closed per frame
    TENTACLE_COUNT = 6
    TENTACLE_SEGMENTS = 6
    TENTACLE_LENGTH = 2.2
    TENTACLE_SPREAD_ANGLE = 16  # Degrees - fan half-angle of the relaxed tentacles
    TENTACLE_WIGGLE_ANGLE = 30  # Degrees - travelling wave amplitude at the tips
    TENTACLE_STIFFNESS_BASE = 0.35
    TENTACLE_STIFFNESS_TIP = 0.12
    ORAL_ARM_LENGTH = 0.9  # Short frilly arms in the middle
    ORAL_ARM_POINTS = 6
    # Death animation, as (start, end) spans of the enemy's death_progress
    DEATH_DURATION = 54.0  # Frames
    DEATH_FRICTION = 0.92  # Velocity kept per frame while the remains drift
    DEATH_COLLAPSE = (0.08, 0.7)
    DEATH_FADE = (0.62, 1.0)
    DEATH_GATHER = (0.16, 0.82)
    DEATH_CENTER_X = 0.25  # Bell and trailing tissue gather here

    def __init__(self, enemy: 'Creature'):
        """Initialize the jellyfish for an enemy."""
        self.pulse_phase = random.uniform(0, 2 * math.pi)  # Random start to avoid sync
        self.agitation = 0.0
        self.flutter = 0.0
        self.reset(enemy)

    def reset(self, enemy: 'Creature') -> None:
        """Point the bell along the enemy's current angle and re-lay the tentacles."""
        self.heading = enemy.angle % 360
        self.tentacles: List[List[Point]] = []

    @property
    def contraction(self) -> float:
        """Bell contraction (0.0 relaxed, 1.0 fully squeezed)."""
        return 0.5 + 0.5 * math.sin(self.pulse_phase)

    def bell_length_scale(self, death_progress: float) -> float:
        """Deflate toward a tiny knot, with one last pressure shudder."""
        shudder = 0.07 * math.sin(math.pi * min(1.0, death_progress / 0.16))
        return 1.0 + shudder - 0.93 * _ease(death_progress, self.DEATH_COLLAPSE)

    def on_event(self, enemy: 'Creature', name: str) -> None:
        """Its look follows the chase, so events change nothing."""

    def update(self, enemy: 'Creature', dt: float, player_pos: Optional[Point] = None) -> None:
        """Advance the swimming pulse, heading, mood and tentacles.

        Args:
            enemy: The enemy this jellyfish belongs to.
            dt: Delta time since last update.
            player_pos: Unused; the enemy's alert state says whether it is chasing.
        """
        turn = (enemy.angle - self.heading + 180) % 360 - 180
        max_turn = self.TURN_SPEED * dt
        self.heading = (self.heading + max(-max_turn, min(max_turn, turn))) % 360

        backing_off = enemy.is_alert and enemy.backing_off
        chasing = enemy.is_alert and not backing_off
        self.agitation += ((1.0 if chasing else 0.0) - self.agitation) * self.MOOD_SMOOTHING
        self.flutter += ((1.0 if backing_off else 0.0) - self.flutter) * self.MOOD_SMOOTHING

        self.pulse_phase += dt * (self.PULSE_SPEED + self.CHASE_PULSE_SPEED * self.agitation)
        self._update_tentacles(enemy, 0.0)

    def update_death(self, enemy: 'Creature', dt: float) -> None:
        """Reel the curling tentacles into the deflating bell."""
        self.agitation *= 0.85
        self.flutter *= 0.85
        self._update_tentacles(enemy, enemy.death_progress)

    def _bell_half_width(self, u: float, death_progress: float) -> float:
        """Half-width of the bell at u (0.0 = rim, 1.0 = apex)."""
        dome = max(0.0, 1.0 - u ** 2.3) ** 0.5
        squeeze = 1.0 - self.BELL_SQUEEZE * self.contraction
        # A nervous jellyfish shivers
        squeeze += 0.06 * self.flutter * math.sin(self.pulse_phase * 9.0)
        # The rim flares out as the bell relaxes
        flare = 1.0 + self.BELL_FLARE * (1.0 - self.contraction) * max(0.0, 1.0 - u / 0.2)
        collapse = _ease(death_progress, self.DEATH_COLLAPSE)
        # Tissue buckles into uneven folds as the volume drains away.
        folds = 1.0 - 0.32 * math.sin(math.pi * collapse) * (
            0.5 + 0.5 * math.cos(u * math.pi * 5.0 + death_progress * 9.0)
        )
        return self.BELL_HALF_WIDTH * dome * squeeze * flare * self.bell_length_scale(death_progress) * folds

    def _bell_x(self, u: float, death_progress: float) -> float:
        """Position along the heading of the bell at u (0.0 = rim, 1.0 = apex)."""
        apex = self.BELL_APEX_X * (1.0 + self.BELL_STRETCH * self.contraction)
        x = self.BELL_RIM_X + (apex - self.BELL_RIM_X) * u
        return self.DEATH_CENTER_X + (x - self.DEATH_CENTER_X) * self.bell_length_scale(death_progress)

    def _update_tentacles(self, enemy: 'Creature', death_progress: float) -> None:
        """Drag the tentacle chains behind the bell."""
        heading_rad = math.radians(self.heading)
        cos_heading, sin_heading = math.cos(heading_rad), math.sin(heading_rad)
        rear_angle = heading_rad + math.pi
        segment = enemy.radius * self.TENTACLE_LENGTH / self.TENTACLE_SEGMENTS
        gather = _ease(death_progress, self.DEATH_GATHER)
        segment *= 1.0 - 0.97 * gather
        # The tips lag behind the bell, curling as the tissue reels inward.
        spread = math.radians(self.TENTACLE_SPREAD_ANGLE) * (1.0 - 0.6 * self.contraction)
        spread *= 1.0 - gather
        wiggle = math.radians(self.TENTACLE_WIGGLE_ANGLE)
        rim_half_width = self._bell_half_width(0.0, death_progress) * 0.85

        if not self.tentacles:
            self.tentacles = [
                [(enemy.x, enemy.y)] * (self.TENTACLE_SEGMENTS + 1) for _ in range(self.TENTACLE_COUNT)
            ]

        for k, chain in enumerate(self.tentacles):
            side = k / (self.TENTACLE_COUNT - 1) * 2.0 - 1.0  # -1.0 to 1.0 across the rim
            local_x = self._bell_x(0.0, death_progress) * enemy.radius
            local_y = side * rim_half_width * enemy.radius
            anchor = (
                enemy.x + local_x * cos_heading - local_y * sin_heading,
                enemy.y + local_x * sin_heading + local_y * cos_heading,
            )
            # Snap to the rest pose on first use or after a jump (e.g. respawn)
            snap = distance(chain[0], anchor) > enemy.radius * 3

            def rest_angle(i: int, t: float, side: float = side, k: int = k) -> float:
                angle = rear_angle - side * spread * (1.0 - 0.4 * t)
                angle += side * gather * t * math.pi * 1.7
                return angle + math.sin(self.pulse_phase * 1.5 - i * 0.9 + k * 2.1) * wiggle * t * (1.0 - gather)

            drag_chain(
                chain, anchor, segment, rest_angle,
                self.TENTACLE_STIFFNESS_BASE + 0.5 * gather,
                self.TENTACLE_STIFFNESS_TIP + 0.7 * gather, snap or (death_progress > 0.0 and config.SETTINGS.game.lowGraphics)
            )

    def draw(self, screen: pygame.Surface, enemy: 'Creature') -> None:
        """Draw the living jellyfish.

        Args:
            screen: The pygame Surface to draw on.
            enemy: The enemy this jellyfish belongs to.
        """
        self._draw(screen, enemy, 0.0)

    def draw_death(self, screen: pygame.Surface, enemy: 'Creature') -> None:
        """Draw the dying jellyfish: the tissue folds inward into a fading knot.

        Args:
            screen: The pygame Surface to draw on.
            enemy: The enemy this jellyfish belongs to.
        """
        self._draw(screen, enemy, enemy.death_progress)

    def _draw(self, screen: pygame.Surface, enemy: 'Creature', death_progress: float) -> None:
        """Draw the jellyfish, part-way through dying if death_progress > 0."""
        fade = _ease(death_progress, self.DEATH_FADE)
        if fade >= 1.0:
            return
        if not self.tentacles:
            self._update_tentacles(enemy, death_progress)

        radius = enemy.radius
        center = (int(enemy.x), int(enemy.y))
        heading_rad = math.radians(self.heading)
        cos_heading, sin_heading = math.cos(heading_rad), math.sin(heading_rad)
        # Brighter when chasing, dimmer when backing off
        mood = 1.0 + 0.5 * self.agitation - 0.35 * self.flutter
        white = (255, 255, 255)
        bright = visual_effects.interpolate_color(self.COLOR, white, 0.5)

        collapse = _ease(death_progress, self.DEATH_COLLAPSE)
        compression_flash = math.sin(math.pi * collapse) ** 2

        # Stepped alpha keeps the glow cache small
        glow_alpha = int((45 + 45 * self.agitation + 65 * compression_flash) * (1.0 - fade)) // 10 * 10
        if glow_alpha > 0:
            glow = visual_effects.create_soft_glow_surface(radius * (2.4 - 1.6 * collapse), self.COLOR, glow_alpha)
            screen.blit(glow, glow.get_rect(center=center))

        # Everything is drawn on its own layer so the bell is see-through and
        # the whole creature can fade as one
        size = int(radius * 8)
        layer = pygame.Surface((size, size), pygame.SRCALPHA)
        half = size / 2

        def place(local_x: float, local_y: float) -> Point:
            return (
                half + (local_x * cos_heading - local_y * sin_heading) * radius,
                half + (local_x * sin_heading + local_y * cos_heading) * radius,
            )

        # Stinging tentacles, with sparks running down them during a chase
        tentacle_color = (*self.COLOR, min(255, int(190 * mood)))
        offset_x, offset_y = half - center[0], half - center[1]
        for k, chain in enumerate(self.tentacles):
            points = [(x + offset_x, y + offset_y) for x, y in chain]
            pygame.draw.lines(layer, tentacle_color, False, points, 1)
            if self.agitation > 0.3:
                spark = points[int(self.pulse_phase * 2.5 + k * 1.3) % len(points)]
                pygame.draw.circle(layer, (*white, int(255 * self.agitation)), (int(spark[0]), int(spark[1])), 1)

        # Short frilly arms under the middle of the bell
        arm_length = self.ORAL_ARM_LENGTH * (1.0 - _ease(death_progress, self.DEATH_COLLAPSE))
        if arm_length > 0.05:
            for sign in (-1.0, 1.0):
                arm = []
                for j in range(self.ORAL_ARM_POINTS):
                    t = j / (self.ORAL_ARM_POINTS - 1)
                    sway = 0.14 * t * math.sin(self.pulse_phase * 2.0 - j * 1.2 + sign)
                    arm.append(place(self._bell_x(0.0, death_progress) - arm_length * t,
                                     (sign * 0.16 + sway) * self.bell_length_scale(death_progress)))
                pygame.draw.lines(layer, (*bright, 220), False, arm, 2)

        # Bell: a see-through dome with a brighter core
        def bell_outline(scale: float) -> List[Point]:
            left, right = [], []
            for j in range(self.BELL_POINTS + 1):
                u = j / self.BELL_POINTS
                x = self._bell_x(u * (0.55 + 0.45 * scale), death_progress)
                half_width = self._bell_half_width(u, death_progress) * scale
                left.append(place(x, -half_width))
                right.append(place(x, half_width))
            return left + right[::-1]

        outline = bell_outline(1.0)
        pygame.draw.polygon(layer, (*self.COLOR, min(255, int(self.BELL_ALPHA * mood))), outline)
        pygame.draw.polygon(layer, (*bright, min(255, int(self.BELL_CORE_ALPHA * mood))), bell_outline(0.6))

        # Luminous creases travel inward across the crumpling membrane.
        if 0.0 < collapse < 1.0:
            crease_alpha = int(210 * math.sin(math.pi * collapse))
            for side in (-1.0, 1.0):
                for rib in (0.35, 0.7):
                    crease = []
                    for j in range(self.BELL_POINTS + 1):
                        u = j / self.BELL_POINTS
                        y = side * rib * self._bell_half_width(u, death_progress)
                        crease.append(place(self._bell_x(u, death_progress), y))
                    pygame.draw.lines(layer, (*bright, crease_alpha), False, crease, 1)

        # Four lobes in a clover, the jellyfish's glowing insides
        lobe_radius = max(1, int(round(radius * 0.14 * self.bell_length_scale(death_progress))))
        length_scale = self.bell_length_scale(death_progress)
        lobe_x = self._bell_x(0.42, death_progress)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            lobe = place(lobe_x + dx * self.LOBE_OFFSET * length_scale, dy * self.LOBE_OFFSET * length_scale)
            pygame.draw.circle(layer, (*white, 235), (int(lobe[0]), int(lobe[1])), lobe_radius)

        # Bright skin of the dome and a thicker rim
        pygame.draw.lines(layer, (*bright, 255), False, outline[:self.BELL_POINTS + 1], 1)
        pygame.draw.lines(layer, (*bright, 255), False, outline[self.BELL_POINTS + 1:], 1)
        pygame.draw.line(layer, (*white, 255), outline[0], outline[-1], 2)

        if fade > 0.0:
            layer.set_alpha(int(255 * (1.0 - fade)))
        screen.blit(layer, layer.get_rect(center=center))
