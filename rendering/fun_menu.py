"""The FUN screen: a list of experimental levels."""
from dataclasses import dataclass

import config
from rendering.fonts import get_font
from rendering.menu_components import AnimatedBackground, Button, advance_pulse, build_menu_backdrop, draw_buttons, menu_scale


@dataclass(frozen=True)
class FunEntry:
    title: str
    key: str                     # what Game.start_fun_entry starts


# Add an experimental level by adding a line here and a branch in Game.start_fun_entry.
FUN_ENTRIES = (
    FunEntry('Pilot fight: Rules vs Vibes', 'pilot_duel'),
    FunEntry('Flockers vs Wallcrawlers', 'team_battle'),
    FunEntry('Demo', 'demo'),
)


class FunMenu:
    def __init__(self, screen, entries=FUN_ENTRIES):
        self.screen = screen
        self.entries = list(entries)
        self.selected_index = 0
        self.notice = ''
        self.pulse_phase = 0.0
        self.scale = menu_scale()
        self.background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT).convert()
        self.title_font = get_font(self._scaled(72))
        self.notice_font = get_font(self._scaled(30))
        font = get_font(self._scaled(40))
        top = int(config.SCREEN_HEIGHT * 0.34)
        self.buttons = [Button(entry.title, (config.SCREEN_WIDTH // 2, top + i * self._scaled(80)), font,
                               width=self._scaled(760), height=self._scaled(64))
                        for i, entry in enumerate(self.entries)]

    def _scaled(self, value):
        return max(1, int(round(value * self.scale)))

    @property
    def selected_entry(self):
        return self.entries[self.selected_index] if self.entries else None

    def navigate(self, delta):
        if self.entries:
            self.selected_index = max(0, min(len(self.entries) - 1, self.selected_index + delta))

    def refresh(self):
        self.notice = ''

    def update(self, dt):
        self.background.update(dt)
        self.pulse_phase = advance_pulse(self.pulse_phase, dt)

    def draw(self):
        self.screen.blit(self.backdrop, (0, 0))
        self.background.draw(self.screen)
        title = self.title_font.render('FUN', True, (215, 225, 255))
        self.screen.blit(title, title.get_rect(center=(config.SCREEN_WIDTH // 2, int(config.SCREEN_HEIGHT * 0.18))))
        draw_buttons(self.screen, self.buttons, self.selected_index, self.pulse_phase)
        if self.notice:
            text = self.notice_font.render(self.notice, True, (255, 170, 120))
            self.screen.blit(text, text.get_rect(center=(config.SCREEN_WIDTH // 2, int(config.SCREEN_HEIGHT * 0.78))))
