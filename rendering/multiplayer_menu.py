"""Multiplayer screen: how the game is played (alone or as a co-op pair) and who flies which ship."""

from typing import Callable, Dict, List, Optional, Tuple

import pygame

import config
from rendering.fonts import get_font
from rendering.menu_components import (
    AnimatedBackground, TEXT_DIM, TEXT_LABEL, build_menu_backdrop, build_panel, build_panel_glow, menu_scale,
    render_hint_row, render_pill, render_title
)

WARNING_TEXT = (255, 190, 120)
MODE_NAMES = {False: "SINGLE PLAYER", True: "CAMPAIGN CO-OP"}
PLAYER_2_NAMES = {"human": "HUMAN", "hunter": "CAMPAIGN HUNTER"}


class ChoiceRow:
    """One option with a few named values: its label, the value shown, and how to step to the next."""

    def __init__(self, label: str, read: Callable[[], str], step: Callable[[], None]):
        self.label = label
        self.read = read
        self.toggle = step


class MultiplayerMenu:
    """The choices for playing with two, over a panel saying who flies which ship with the devices at hand."""

    def __init__(self, screen: pygame.Surface, game):
        self.screen = screen
        self.game = game
        self.rows: List[ChoiceRow] = [
            ChoiceRow("MODE", lambda: MODE_NAMES[game.coop], lambda: game.set_coop(not game.coop)),
            ChoiceRow("PLAYER 2", lambda: PLAYER_2_NAMES[game.user_settings.player2],
                      lambda: game.set_player2('hunter' if game.user_settings.player2 == 'human' else 'human')),
        ]
        self.selected_index = 0
        self.scale = menu_scale()
        self.menu_background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.title = render_title(get_font(self._scaled(96)), "MULTIPLAYER")
        self.group_font = get_font(self._scaled(30), bold=True)
        self.row_font = get_font(self._scaled(32), bold=False)
        self.hints = render_hint_row(get_font(self._scaled(28)), [("ENTER / A", "Change"), ("ESC / B", "Back")])

    def _scaled(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def navigate(self, step: int) -> None:
        self.selected_index = (self.selected_index + step) % len(self.rows)

    def toggle_selected(self) -> None:
        self.rows[self.selected_index].toggle()

    def devices(self) -> Tuple[List[Tuple[str, str]], Optional[str]]:
        """Who flies which ship in the chosen mode, as (player, device) lines, and a warning when it cannot be played."""
        if not self.game.coop:
            return [("PLAYER 1", "KEYBOARD OR CONTROLLER")], None
        controllers = len(self.game.input_handler.get_controllers())
        if self.game.user_settings.player2 == 'hunter':
            return [("PLAYER 1", "CONTROLLER OR KEYBOARD" if controllers else "KEYBOARD"),
                    ("PLAYER 2", "CAMPAIGN HUNTER")], None
        if controllers >= 2:
            return [("PLAYER 1", "CONTROLLER 1 OR KEYBOARD"), ("PLAYER 2", "CONTROLLER 2")], None
        if controllers == 1:
            return [("PLAYER 1", "KEYBOARD"), ("PLAYER 2", "CONTROLLER")], None
        return [("PLAYER 1", "KEYBOARD"), ("PLAYER 2", "NO CONTROLLER")], "CONNECT A CONTROLLER FOR PLAYER 2"

    def layout(self) -> Dict[str, object]:
        center_x = config.SCREEN_WIDTH // 2
        title = self.title.get_rect(center=(center_x, int(config.SCREEN_HEIGHT * 0.12)))
        width = self._scaled(1040)
        left = center_x - width // 2
        row_height = self._scaled(84)
        top = title.bottom + self._scaled(30)
        rows = [pygame.Rect(left, top + i * (row_height + self._scaled(16)), width, row_height)
                for i in range(len(self.rows))]
        lines, warning = self.devices()
        panel_top = rows[-1].bottom + self._scaled(48)
        panel = pygame.Rect(left, panel_top, width,
                            self._scaled(96) + len(lines) * self._scaled(62) + (self._scaled(60) if warning else 0))
        hints = self.hints.get_rect(midbottom=(center_x, config.SCREEN_HEIGHT - self._scaled(56)))
        return {'title': title, 'rows': rows, 'panel': panel, 'hints': hints}

    def _render_row(self, size: Tuple[int, int], row: ChoiceRow) -> pygame.Surface:
        panel = build_panel(size, radius=self._scaled(22)).copy()
        pad = self._scaled(36)
        label = self.row_font.render(row.label, True, TEXT_DIM)
        panel.blit(label, label.get_rect(midleft=(pad, size[1] // 2)))
        value = render_pill(self.row_font, row.read(), padding=(self._scaled(18), self._scaled(6)),
                            border=max(1, self._scaled(1.5)))
        panel.blit(value, value.get_rect(midright=(size[0] - pad, size[1] // 2)))
        return panel

    def _render_panel(self, size: Tuple[int, int]) -> pygame.Surface:
        panel = build_panel(size, radius=self._scaled(22)).copy()
        pad = self._scaled(36)
        heading = self.group_font.render("WHO FLIES", True, TEXT_LABEL)
        panel.blit(heading, heading.get_rect(midleft=(pad, self._scaled(50))))
        rule_y = self._scaled(84)
        pygame.draw.line(panel, (60, 56, 100), (pad, rule_y), (size[0] - pad, rule_y), max(1, self._scaled(2)))
        lines, warning = self.devices()
        for i, (player, device) in enumerate(lines):
            y = self._scaled(96) + i * self._scaled(62) + self._scaled(31)
            colour = config.COLOR_SHIP if i == 0 else (255, 170, 60)
            panel.blit(self.row_font.render(player, True, colour), (pad, y - self._scaled(16)))
            text = self.row_font.render(device, True, TEXT_DIM)
            panel.blit(text, text.get_rect(midright=(size[0] - pad, y)))
        if warning:
            note = self.row_font.render(warning, True, WARNING_TEXT)
            panel.blit(note, note.get_rect(midleft=(pad, size[1] - self._scaled(38))))
        return panel

    def update(self, dt: float) -> None:
        self.menu_background.update(dt)

    def draw(self) -> None:
        layout = self.layout()
        self.screen.blit(self.backdrop, (0, 0))
        self.menu_background.draw(self.screen)
        self.screen.blit(self.title, layout['title'])
        for i, (row, rect) in enumerate(zip(self.rows, layout['rows'])):
            if i == self.selected_index:
                spread = self._scaled(22)
                glow = build_panel_glow(rect.size, radius=self._scaled(22), spread=spread)
                self.screen.blit(glow, (rect.x - spread, rect.y - spread))
            self.screen.blit(self._render_row(rect.size, row), rect)
        panel = layout['panel']
        self.screen.blit(self._render_panel(panel.size), panel)
        self.screen.blit(self.hints, layout['hints'])
