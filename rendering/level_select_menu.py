"""A scrolling grid of unlocked levels and their personal bests."""
import math

import pygame
import config
from rendering.fonts import get_font
from rendering.menu_components import (
    AnimatedBackground, Button, TEXT_DIM, TEXT_LABEL, advance_pulse, build_menu_backdrop,
    menu_scale, render_hint_row, render_title,
)
from rendering.stars import draw_star
from utils.formatting import format_time


class LevelSelectMenu:
    """Navigation and layout for the active profile's level grid."""

    def __init__(self, screen, profile_manager):
        self.screen = screen
        self.profile_manager = profile_manager
        self.scale = menu_scale()
        self.background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.title = render_title(get_font(self._scaled(96)), 'LEVELS')
        self.number_font = get_font(self._scaled(42))
        self.detail_font = get_font(self._scaled(28))
        self.hints = render_hint_row(self.detail_font, [
            ('ARROWS', 'Move'), ('ENTER / A', 'Play'), ('ESC / B', 'Back'),
        ])
        self.phase = 0.0
        self.refresh()

    def _scaled(self, value):
        return max(1, round(value * self.scale))

    def refresh(self):
        """Open on the active profile's furthest level with current records."""
        self.profile = self.profile_manager.get_active_profile()
        self.unlocked = self.profile_manager.get_active_level()
        self.selected_level = self.unlocked
        self._buttons = {}

    @property
    def can_play_selected(self):
        return 1 <= self.selected_level <= self.unlocked

    @property
    def star_total(self):
        earned = sum(best.stars for level, best in self.profile.bests.items() if level <= self.unlocked)
        return earned, 5 * self.unlocked

    def navigate(self, dx, dy):
        """Move by columns or rows, clamping to playable tiles without wrapping."""
        columns = self.layout()['columns']
        index = self.selected_level - 1
        row, column = divmod(index, columns)
        if dx:
            column = min(columns - 1, max(0, column + dx))
        if dy:
            row = max(0, min((self.unlocked - 1) // columns, row + dy))
        self.selected_level = min(self.unlocked, row * columns + column + 1)

    def layout(self):
        """Visible tile rectangles and screen regions, scrolling by complete rows."""
        width, height = self.screen.get_size()
        title = self.title.get_rect(center=(width // 2, int(height * 0.14)))
        hints = self.hints.get_rect(midbottom=(width // 2, height - self._scaled(48)))
        details = pygame.Rect(0, 0, self._scaled(1000), self._scaled(42))
        details.midbottom = (width // 2, hints.top - self._scaled(34))
        tile_w, tile_h, gap = self._scaled(152), self._scaled(116), self._scaled(22)
        columns = max(1, min(6, (width - self._scaled(96) + gap) // (tile_w + gap)))
        top = title.bottom + self._scaled(40)
        visible_rows = max(1, (details.top - self._scaled(34) - top + gap) // (tile_h + gap))
        # Levels are procedurally generated without an upper limit, so the next
        # locked tile always exists. Only construct tiles in the visible window.
        total_rows = math.ceil((self.unlocked + 1) / columns)
        selected_row = (self.selected_level - 1) // columns
        first_row = min(max(0, selected_row - visible_rows + 1), max(0, total_rows - visible_rows))
        left = (width - columns * tile_w - (columns - 1) * gap) // 2
        tiles = []
        for index in range(first_row * columns, min(self.unlocked + 1, (first_row + visible_rows) * columns)):
            row, col = divmod(index, columns)
            level = index + 1
            best = self.profile.bests.get(level)
            tiles.append({
                'level': level, 'locked': level > self.unlocked,
                'stars': best.stars if best else 0,
                'rect': pygame.Rect(left + col * (tile_w + gap), top + (row - first_row) * (tile_h + gap), tile_w, tile_h),
            })
        return dict(title=title, hints=hints, details=details, tiles=tiles, columns=columns,
                    first_row=first_row, visible_rows=visible_rows, total_rows=total_rows)

    def update(self, dt):
        self.background.update(dt)
        self.phase = advance_pulse(self.phase, dt)

    def draw(self):
        layout = self.layout()
        self.screen.blit(self.backdrop, (0, 0))
        self.background.draw(self.screen)
        self.screen.blit(self.title, layout['title'])
        total = self.detail_font.render(f'{self.star_total[0]} / {self.star_total[1]}', True, (255, 224, 130))
        total_rect = total.get_rect(topright=(self.screen.get_width() - self._scaled(48), self._scaled(42)))
        self.screen.blit(total, total_rect)
        draw_star(self.screen, total_rect.left - self._scaled(22), total_rect.centery, self._scaled(26))
        visible_buttons = {}
        for tile in layout['tiles']:
            level, rect = tile['level'], tile['rect']
            button = self._buttons.get(level) or Button('', rect.center, self.number_font,
                                                        width=rect.width, height=rect.height)
            button.position = rect.center
            button.selected = level == self.selected_level and not tile['locked']
            button.draw(self.screen, self.phase)
            visible_buttons[level] = button
            if tile['locked']:
                # A drawn lock remains legible regardless of font glyph coverage.
                x, y = rect.centerx, rect.centery
                color = (91, 87, 118)
                pygame.draw.arc(self.screen, color, (x-self._scaled(12), y-self._scaled(24),
                                 self._scaled(24), self._scaled(28)), 0, math.pi, self._scaled(3))
                pygame.draw.rect(self.screen, color, (x-self._scaled(19), y-self._scaled(8),
                                 self._scaled(38), self._scaled(30)), border_radius=self._scaled(5))
            else:
                label = self.number_font.render(str(level), True, config.COLOR_TEXT)
                self.screen.blit(label, label.get_rect(center=(rect.centerx, rect.top + self._scaled(39))))
                for i in range(5):
                    draw_star(self.screen, rect.centerx + (i - 2) * self._scaled(24),
                              rect.bottom - self._scaled(27), self._scaled(19), i < tile['stars'])
        self._buttons = visible_buttons
        best = self.profile.bests.get(self.selected_level)
        text = f'BEST {format_time(best.time)}   SCORE {best.score}' if best else 'NOT CLEARED YET'
        details = self.detail_font.render(text, True, TEXT_LABEL)
        self.screen.blit(details, details.get_rect(center=layout['details'].center))
        self.screen.blit(self.hints, layout['hints'])
        # Scroll cues above and below the grid.
        for show, y, direction in [
            (layout['first_row'] > 0, layout['tiles'][0]['rect'].top - self._scaled(16), -1),
            (layout['first_row'] + layout['visible_rows'] < layout['total_rows'],
             layout['tiles'][-1]['rect'].bottom + self._scaled(16), 1),
        ]:
            if show:
                x, r = self.screen.get_width() // 2, self._scaled(7)
                pygame.draw.polygon(self.screen, TEXT_DIM, [(x-r,y-direction*r/2), (x+r,y-direction*r/2), (x,y+direction*r/2)])
