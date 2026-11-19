"""Level competition's two screens: whose turn is next, and how the level's pair of turns turned out."""
import pygame

import config
from rendering.fonts import get_font
from rendering.menu_components import (
    AnimatedBackground, TEXT_DIM, build_menu_backdrop, build_panel, menu_scale, render_hint_row, render_title
)

PLAYER_COLOURS = (config.COLOR_SHIP, (255, 170, 60))

# What a score is made of, in the order shown: label, the breakdown key, and the sign it has on the score.
SCORE_PARTS = (
    ("TIME", "time_penalty", -1),
    ("ENEMY HITS", "collision_penalty", -1),
    ("BULLET HITS", "enemy_bullet_penalty", -1),
    ("WALL BUMPS", "wall_collision_penalty", -1),
    ("SHOTS FIRED", "ammo_penalty", -1),
    ("FUEL", "fuel_penalty", -1),
    ("KILLS", "enemy_destruction_bonus", 1),
    ("CRYSTALS", "powerup_bonus", 1),
)


def score_rows(breakdowns):
    """(label, player 1's change, player 2's change) for each part of the score either player had."""
    rows = []
    for label, key, sign in SCORE_PARTS:
        values = [sign * int(round((b or {}).get(key, 0))) for b in breakdowns]
        if any(values):
            rows.append((label, *values))
    return rows


def player_name(index: int) -> str:
    return f"PLAYER {index + 1}"


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

    def draw_turn(self, competition) -> None:
        """Whose turn it is, on which level, and the points so far in a campaign."""
        player = competition.current
        y = self._begin(f"{player_name(player)}'S TURN", PLAYER_COLOURS[player])
        y = self._centred(self.big_font, f"LEVEL {competition.level}", config.COLOR_TEXT, y)
        if competition.turn == 1:
            other = competition.order[0]
            y = self._centred(self.row_font, f"{player_name(other)} SCORED {int(competition.scores[other]):,}",
                              TEXT_DIM, y)
        if not competition.from_levels:
            self._centred(self.row_font, points_line(competition.points), TEXT_DIM, y)
        self._finish()

    def draw_result(self, competition) -> None:
        """Both scores, who won the level, and the points so far in a campaign."""
        winner = competition.winner
        if winner is not None:
            title, colour = f"{player_name(winner)} WINS", PLAYER_COLOURS[winner]
        else:
            title, colour = ("BOTH OUT" if competition.both_out else "A TIE"), config.COLOR_TEXT
        y = self._begin(title, colour)
        rows = score_rows(competition.breakdowns)
        line = self._scaled(31)
        panel = build_panel((self._scaled(900), self._scaled(190) + line * len(rows)), radius=self._scaled(22)).copy()
        for index in (0, 1):
            x = panel.get_width() * (1 + 2 * index) // 4
            name = self.small_font.render(player_name(index), True, PLAYER_COLOURS[index])
            panel.blit(name, name.get_rect(center=(x, self._scaled(46))))
            score = self.big_font.render(f"{int(competition.scores[index]):,}", True, config.COLOR_TEXT)
            panel.blit(score, score.get_rect(center=(x, self._scaled(106))))
            if competition.scores[index] == 0 and competition.breakdowns[index] is not None:
                out = self.small_font.render("OUT OF POWER", True, TEXT_DIM)
                panel.blit(out, out.get_rect(center=(x, self._scaled(150))))
        for number, (label, *values) in enumerate(rows):
            row_y = self._scaled(178) + line * number
            text = self.small_font.render(label, True, TEXT_DIM)
            panel.blit(text, text.get_rect(center=(panel.get_width() // 2, row_y)))
            for index, value in enumerate(values):
                x = panel.get_width() * (1 + 2 * index) // 4
                figure = self.small_font.render((f"{value:+,}" if value else "–"), True, config.COLOR_TEXT if value else TEXT_DIM)
                panel.blit(figure, figure.get_rect(center=(x, row_y)))
        self.screen.blit(panel, panel.get_rect(midtop=(config.SCREEN_WIDTH // 2, y)))
        y += panel.get_height() + self._scaled(24)
        if competition.both_out and not competition.from_levels:
            y = self._centred(self.row_font, "NOBODY SCORES - THE LEVEL IS REPLAYED", TEXT_DIM, y)
        if not competition.from_levels:
            self._centred(self.row_font, points_line(competition.points), TEXT_DIM, y)
        self._finish()


def points_line(points) -> str:
    return f"POINTS   {points[0]} – {points[1]}"
