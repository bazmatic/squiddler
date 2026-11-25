"""Settings screen: the player's options, then the key and button mappings, kept off the main menu."""

from typing import Callable, Dict, List, Optional, Tuple

import pygame
import config
from rendering.fonts import get_font
from rendering.menu_components import (
    AnimatedBackground, TEXT_DIM, TEXT_LABEL, advance_pulse, build_menu_backdrop, ButtonSet, build_panel, menu_scale, selection_text_color,
    render_hint_row, render_pill, render_title
)
from user_settings import UserSettings

CONTROL_GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("KEYBOARD", [
        ("ARROWS / WASD", "Move"),
        ("SPACE", "Fire"),
        ("DOWN / S", "Next weapon"),
        ("ESC", "Quit level"),
        ("R", "Restart level"),
    ]),
    ("CONTROLLER", [
        ("STICKS", "Move"),
        ("R / ZR / B", "Fire"),
        ("A", "Next weapon"),
        ("L / ZL", "Thrust"),
        ("Y", "Restart level"),
    ]),
]

ON_TEXT = (150, 255, 200)
OFF_TEXT = (255, 130, 150)


class SettingRow:
    """One on/off option: its label, how to read it, and how to flip it."""

    def __init__(self, label: str, read: Callable[[], bool], write: Callable[[bool], None]):
        self.label = label
        self.read = read
        self.write = write

    def toggle(self) -> None:
        self.write(not self.read())


class SettingsMenu:
    """Shows the selectable settings above one panel of mappings per input device."""

    def __init__(self, screen: pygame.Surface, settings: UserSettings):
        """Initialize settings screen.

        Args:
            screen: The pygame Surface to draw on.
            settings: The player's settings, changed in place.
        """
        self.screen = screen
        self.settings = settings
        self.rows = [SettingRow("SOUND", lambda: settings.sound, settings.set_sound)]
        self.selected_index = 0
        self.phase = 0.0
        self.scale = menu_scale()
        self.menu_background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.title = render_title(get_font(self._scaled(96)), "SETTINGS")
        self.group_font = get_font(self._scaled(30), bold=True)
        self.row_font = get_font(self._scaled(32), bold=False)
        self.hints = render_hint_row(get_font(self._scaled(28)), [
            ("ENTER / A", "Change"), ("ESC / B", "Back")
        ])
        self.buttons = ButtonSet(self.row_font)
        self._panels: Optional[List[pygame.Surface]] = None

    def _scaled(self, value: float) -> int:
        """Scale a reference-layout length to the current screen size."""
        return max(1, int(round(value * self.scale)))

    def navigate(self, step: int) -> None:
        """Move the selection up (-1) or down (+1) through the settings."""
        self.selected_index = (self.selected_index + step) % len(self.rows)

    def toggle_selected(self) -> None:
        """Flip the selected setting."""
        self.rows[self.selected_index].toggle()

    def layout(self) -> Dict[str, object]:
        """Where each part of the screen sits: 'title', 'settings' (one rect per row), 'panels' (one
        rect per device) and 'hints'."""
        center_x = config.SCREEN_WIDTH // 2
        title = self.title.get_rect(center=(center_x, int(config.SCREEN_HEIGHT * 0.12)))

        rows = max(len(mappings) for _, mappings in CONTROL_GROUPS)
        panel_size = (self._scaled(500), self._scaled(96) + rows * self._scaled(62) + self._scaled(26))
        gap = self._scaled(40)
        total_width = panel_size[0] * len(CONTROL_GROUPS) + gap * (len(CONTROL_GROUPS) - 1)
        left = center_x - total_width // 2

        row_height = self._scaled(84)
        top = title.bottom + self._scaled(30)
        settings = [
            pygame.Rect(left, top + i * (row_height + self._scaled(16)), total_width, row_height)
            for i in range(len(self.rows))
        ]

        top = settings[-1].bottom + self._scaled(48)
        panels = [
            pygame.Rect(left + i * (panel_size[0] + gap), top, *panel_size)
            for i in range(len(CONTROL_GROUPS))
        ]
        hints = self.hints.get_rect(midbottom=(center_x, config.SCREEN_HEIGHT - self._scaled(56)))
        return {'title': title, 'settings': settings, 'panels': panels, 'hints': hints}

    def _render_setting(self, size: Tuple[int, int], row: SettingRow, blend: float) -> pygame.Surface:
        """Render one setting: its label on the left, its current value in a pill on the right."""
        panel = pygame.Surface(size, pygame.SRCALPHA)       # the frame is drawn beneath it by the ButtonSet
        pad = self._scaled(36)
        label = self.row_font.render(row.label, True, selection_text_color(blend))
        panel.blit(label, label.get_rect(midleft=(pad, size[1] // 2)))
        on = row.read()
        value = render_pill(self.row_font, "ON" if on else "OFF", text_color=ON_TEXT if on else OFF_TEXT,
                            padding=(self._scaled(18), self._scaled(6)), border=max(1, self._scaled(1.5)))
        panel.blit(value, value.get_rect(midright=(size[0] - pad, size[1] // 2)))
        return panel

    def _render_panel(self, size: Tuple[int, int], name: str, mappings: List[Tuple[str, str]]) -> pygame.Surface:
        """Render one device's panel: its name, then a key pill and action per row."""
        panel = build_panel(size, radius=self._scaled(22)).copy()
        pad = self._scaled(36)

        heading = self.group_font.render(name, True, TEXT_LABEL)
        panel.blit(heading, heading.get_rect(midleft=(pad, self._scaled(50))))
        rule_y = self._scaled(84)
        pygame.draw.line(panel, (60, 56, 100), (pad, rule_y), (size[0] - pad, rule_y), max(1, self._scaled(2)))

        for i, (key_text, action_text) in enumerate(mappings):
            center_y = self._scaled(96) + i * self._scaled(62) + self._scaled(31)
            key = render_pill(self.row_font, key_text, padding=(self._scaled(14), self._scaled(6)), border=max(1, self._scaled(1.5)))
            panel.blit(key, key.get_rect(midleft=(pad, center_y)))
            action = self.row_font.render(action_text, True, TEXT_DIM)
            panel.blit(action, action.get_rect(midright=(size[0] - pad, center_y)))
        return panel

    def update(self, dt: float) -> None:
        """Update background animation."""
        self.menu_background.update(dt)
        self.phase = advance_pulse(self.phase, dt)

    def draw(self) -> None:
        """Draw the settings screen."""
        layout = self.layout()
        if self._panels is None:
            self._panels = [
                self._render_panel(rect.size, name, mappings)
                for rect, (name, mappings) in zip(layout['panels'], CONTROL_GROUPS)
            ]

        self.screen.blit(self.backdrop, (0, 0))
        self.menu_background.draw(self.screen)
        self.screen.blit(self.title, layout['title'])
        for i, (row, rect) in enumerate(zip(self.rows, layout['settings'])):
            blend = self.buttons.draw(self.screen, i, rect, i == self.selected_index, self.phase)
            self.screen.blit(self._render_setting(rect.size, row, blend), rect)
        for panel, rect in zip(self._panels, layout['panels']):
            self.screen.blit(panel, rect)
        self.screen.blit(self.hints, layout['hints'])
