"""The left HUD strip in co-op: the team's score on top, Player 1's gauges above Player 2's."""
from typing import Optional, Sequence, Tuple

import pygame

import config
from rendering.fonts import get_font
from rendering.ui_elements import UIElementRenderer

HEADER_HEIGHT = 64


def _fit(image: pygame.Surface, width: int) -> pygame.Surface:
    """The image, shrunk to fit `width` when it is wider."""
    if image.get_width() <= width:
        return image
    return pygame.transform.smoothscale(image, (width, max(1, round(image.get_height() * width / image.get_width()))))


def draw_coop_hud(screen: pygame.Surface, players: Sequence, powers: Sequence[Tuple[float, float]],
                  team_score: float, level: int, bottom: Optional[int] = None) -> None:
    """Draw the strip from the top down to `bottom` (default: the screen's foot). `powers[i]` is (power, maximum) for
    players[i]. A half is dimmed while its player is out of play, and says why."""
    width = config.UI_ZONE_WIDTH
    bottom = screen.get_height() if bottom is None else min(bottom, screen.get_height())
    label_font = get_font(16)
    screen.blit(label_font.render(f"LEVEL {level}   TEAM SCORE", True, (150, 160, 195)),
                (12, 8))
    screen.blit(get_font(34).render(f"{int(team_score):,}", True, (235, 242, 255)), (12, 26))
    half = max(60, (bottom - HEADER_HEIGHT) // max(1, len(players)))
    for index, (player, (power, maximum)) in enumerate(zip(players, powers)):
        top = HEADER_HEIGHT + index * half
        rect = pygame.Rect(0, top, width, half)
        colour = player.color or config.COLOR_SHIP
        pygame.draw.line(screen, colour, (8, top), (width - 8, top), 2)
        status = "OUT OF POWER" if player.dropped_out else "THROUGH" if player.through else ""
        title = f"PLAYER {index + 1}" + (" · HUNTER" if player.driver is not None else "")
        screen.blit(_fit(get_font(18).render(title, True, colour), width - 24), (12, top + 6))
        if status:    # on a line of its own: beside the title it would run out into the playing area
            screen.blit(_fit(get_font(16).render(status, True, colour), width - 24), (12, top + 26))
        radius = max(14, min(34, (half - 90) // 2))
        centre_y = top + 46 + radius + 8
        percent = max(0.0, min(1.0, power / maximum)) if maximum else 0.0
        UIElementRenderer.draw_circular_gauge(
            screen, width // 4, centre_y, radius, percent, str(int(power)), colour,
            label_text="POWER", alert=percent <= 0.2 and player.flying)
        fraction, text, ammo_colour = player.ship.ammo_reading()
        UIElementRenderer.draw_circular_gauge(
            screen, 3 * width // 4, centre_y, radius, fraction, text, ammo_colour, label_text="AMMO")
        weapon = player.ship.weapon
        if weapon is not None:
            definition = weapon.definition
            screen.blit(get_font(18).render(definition.name.upper(), True, definition.color),
                        (12, centre_y + radius + 28))
        if not player.flying:
            veil = pygame.Surface(rect.size, pygame.SRCALPHA)
            veil.fill((0, 0, 0, 140))
            screen.blit(veil, rect.topleft)
