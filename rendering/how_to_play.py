"""The HOW TO PLAY screen: one page of the rules at a time."""
from typing import List, Tuple

import pygame

import config
from rendering.fonts import get_font
from rendering.how_to_play_pages import build_pages
from rendering.menu_components import (
    AnimatedBackground, TEXT_DIM, build_menu_backdrop, menu_scale, render_hint_row
)

TEXT_COLOUR = (225, 232, 255)
HEADING_COLOUR = (215, 225, 255)
TEXT_WIDTH_FRACTION = 0.7        # How much of the screen's width a line may fill
BODY_FONT_SIZES = (34, 30, 26, 22, 18)    # Reference sizes, tried largest first until the page fits


class HowToPlayMenu:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.scale = menu_scale()
        self.page_index = 0
        self.pulse_phase = 0.0
        self.controller = False
        self.pages = build_pages(False)
        self.background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT).convert()
        self.title_font = get_font(self._scaled(72))
        self.page_title_font = get_font(self._scaled(44))
        self.counter_font = get_font(self._scaled(26))
        self._hints = {False: render_hint_row(get_font(self._scaled(26)), [("LEFT / RIGHT", "Page"), ("ESC", "Back")]),
                       True: render_hint_row(get_font(self._scaled(26)), [("D-PAD", "Page"), ("B", "Back")])}
        self._laid_out = {}

    def _scaled(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def refresh(self, controller: bool) -> None:
        """Open on the first page, showing the controls of the device in use."""
        self.controller = controller
        self.pages = build_pages(controller)
        self.page_index = 0
        self._laid_out = {}

    def turn(self, delta: int) -> None:
        self.page_index = max(0, min(self.page_count - 1, self.page_index + delta))

    def update(self, dt: float) -> None:
        self.background.update(dt)
        self.pulse_phase = (self.pulse_phase + config.BUTTON_GLOW_PULSE_SPEED * dt / 60.0) % 6.2832

    def _wrap(self, font: pygame.font.Font, text: str, width: int) -> List[str]:
        lines, current = [], ''
        for word in text.split():
            trial = f"{current} {word}".strip()
            if current and font.size(trial)[0] > width:
                lines.append(current)
                current = word
            else:
                current = trial
        return lines + [current] if current else lines

    def _body_top(self) -> int:
        return int(config.SCREEN_HEIGHT * 0.34)

    def _body_bottom(self) -> int:
        return config.SCREEN_HEIGHT - self._scaled(145)

    def _layout(self) -> List[Tuple[pygame.Surface, pygame.Rect]]:
        """The current page's wrapped lines, largest font first that lets them all fit above the hints."""
        if self.page_index not in self._laid_out:
            width = int(config.SCREEN_WIDTH * TEXT_WIDTH_FRACTION)
            left = (config.SCREEN_WIDTH - width) // 2
            for size in BODY_FONT_SIZES:
                font = get_font(self._scaled(size))
                gap = font.get_height() // 2    # Between paragraphs
                placed, y = [], self._body_top()
                for paragraph in self.pages[self.page_index].lines:
                    for line in self._wrap(font, paragraph, width):
                        image = font.render(line, True, TEXT_COLOUR)
                        placed.append((image, image.get_rect(topleft=(left, y))))
                        y += font.get_height()
                    y += gap
                if y - gap <= self._body_bottom():
                    break
            self._laid_out[self.page_index] = placed
        return self._laid_out[self.page_index]

    def line_rects(self) -> List[pygame.Rect]:
        """Where each line of the current page is drawn."""
        return [rect for _, rect in self._layout()]

    def draw(self) -> None:
        centre = config.SCREEN_WIDTH // 2
        self.screen.blit(self.backdrop, (0, 0))
        self.background.draw(self.screen)
        title = self.title_font.render('HOW TO PLAY', True, HEADING_COLOUR)
        self.screen.blit(title, title.get_rect(center=(centre, int(config.SCREEN_HEIGHT * 0.12))))
        heading = self.page_title_font.render(self.pages[self.page_index].title, True, HEADING_COLOUR)
        self.screen.blit(heading, heading.get_rect(center=(centre, int(config.SCREEN_HEIGHT * 0.25))))
        for image, rect in self._layout():
            self.screen.blit(image, rect)
        counter = self.counter_font.render(f"{self.page_index + 1} / {self.page_count}", True, TEXT_DIM)
        self.screen.blit(counter, counter.get_rect(center=(centre, config.SCREEN_HEIGHT - self._scaled(100))))
        hint = self._hints[self.controller]
        self.screen.blit(hint, hint.get_rect(center=(centre, config.SCREEN_HEIGHT - self._scaled(50))))
