"""Level competition's two screens: whose turn is next, and how the level's pair of turns turned out."""
from typing import Optional

import pygame

import config
from rendering.fonts import get_font
from rendering.menu_components import (
    AnimatedBackground, TEXT_DIM, build_menu_backdrop, build_panel, menu_scale, render_hint_row, render_title
)

PLAYER_COLOURS = (config.COLOR_SHIP, (255, 170, 60))


def player_name(index: int, hunter: bool = False) -> str:
    return "CAMPAIGN HUNTER" if hunter else f"PLAYER {index + 1}"


class CompetitionScreens:
    """Draws the turn screen and the results screen over the menu backdrop."""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.scale = menu_scale()
        self.background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.title_font = get_font(self._scaled(84))
        self.big_font = get_font(self._scaled(56), bold=True)
        self.row_font = get_font(self._scaled(34))
        self.small_font = get_font(self._scaled(26))
        self.hints = render_hint_row(get_font(self._scaled(28)), [("ANY BUTTON", "Continue"), ("ESC", "Main menu")])

    def _scaled(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def update(self, dt: float) -> None:
        self.background.update(dt)

    def _begin(self, title: str, colour) -> int:
        """Backdrop and a title; returns the y where the content may start."""
        self.screen.blit(self.backdrop, (0, 0))
        self.background.draw(self.screen)
        image = self.title_font.render(title, True, colour)
        rect = image.get_rect(center=(config.SCREEN_WIDTH // 2, int(config.SCREEN_HEIGHT * 0.2)))
        self.screen.blit(image, rect)
        return rect.bottom + self._scaled(40)

    def _centred(self, font, text, colour, y) -> int:
        image = font.render(text, True, colour)
        rect = image.get_rect(midtop=(config.SCREEN_WIDTH // 2, y))
        self.screen.blit(image, rect)
        return rect.bottom + self._scaled(20)

    def _finish(self) -> None:
        self.screen.blit(self.hints, self.hints.get_rect(
            midbottom=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT - self._scaled(56))))

    def draw_turn(self, competition, hunter: bool = False) -> None:
        """Whose turn it is, on which level, and the points so far in a campaign."""
        player = competition.current
        y = self._begin(f"{player_name(player, hunter)}'S TURN", PLAYER_COLOURS[player])
        y = self._centred(self.big_font, f"LEVEL {competition.level}", config.COLOR_TEXT, y)
        if competition.turn == 1:
            other = competition.order[0]
            y = self._centred(self.row_font, f"{player_name(other)} SCORED {int(competition.scores[other]):,}",
                              TEXT_DIM, y)
        if not competition.from_levels:
            self._centred(self.row_font, points_line(competition.points), TEXT_DIM, y)
        self._finish()

    def draw_result(self, competition, hunter_player: Optional[int] = None) -> None:
        """Both scores, who won the level, and the points so far in a campaign."""
        winner = competition.winner
        if winner is not None:
            title, colour = f"{player_name(winner, winner == hunter_player)} WINS", PLAYER_COLOURS[winner]
        else:
            title, colour = ("BOTH OUT" if competition.both_out else "A TIE"), config.COLOR_TEXT
        y = self._begin(title, colour)
        panel = build_panel((self._scaled(900), self._scaled(200)), radius=self._scaled(22)).copy()
        for index in (0, 1):
            x = panel.get_width() * (1 + 2 * index) // 4
            name = self.small_font.render(player_name(index, index == hunter_player), True, PLAYER_COLOURS[index])
            panel.blit(name, name.get_rect(center=(x, self._scaled(52))))
            score = self.big_font.render(f"{int(competition.scores[index]):,}", True, config.COLOR_TEXT)
            panel.blit(score, score.get_rect(center=(x, self._scaled(122))))
        self.screen.blit(panel, panel.get_rect(midtop=(config.SCREEN_WIDTH // 2, y)))
        y += panel.get_height() + self._scaled(30)
        if competition.both_out and not competition.from_levels:
            y = self._centred(self.row_font, "NOBODY SCORES - THE LEVEL IS REPLAYED", TEXT_DIM, y)
        if not competition.from_levels:
            self._centred(self.row_font, points_line(competition.points), TEXT_DIM, y)
        self._finish()


def points_line(points) -> str:
    return f"POINTS   {points[0]} – {points[1]}"
