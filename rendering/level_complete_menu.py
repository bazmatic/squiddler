"""Level complete menu rendering.

This module handles the rendering of the level complete/failed screen.
"""

import math
import pygame
from typing import Optional, Dict, List, Sequence, Tuple
import config
from rendering.fonts import get_font
from rendering.enemy_death_portrait import EnemyDeathPortrait
from rendering.menu_components import (
    AnimatedBackground, Button, TEXT_LABEL, advance_pulse, as_light, build_menu_backdrop, draw_buttons, menu_scale,
    render_hint_row,
    render_title
)
from rendering.ui_elements import AnimatedStarRating
from rendering.stars import draw_star
from rendering.number_sprite import NumberSprite
from utils.resource_path import resource_path
from utils.formatting import format_time
from profiles import LevelResult
import creatures

STAR_SLOT_WIDTH = 170  # Reference width of each goal in the checklist under the stars
STAR_LINE_GAP = 14  # Between the stars' points and the checklist
STAR_LINE_MISSED = (96, 90, 138)  # A goal that was missed


def comparison_lines(result: Optional[LevelResult], score: int, elapsed: float) -> Tuple[str, str]:
    """Time and score captions describing this clear against the previous best."""
    if result is None:
        return '', ''
    if result.first_clear:
        return 'FIRST CLEAR', ''
    previous = result.previous_best
    time_line = ('NEW BEST' if result.new_best_time else
                 f'BEST {format_time(previous.time)}  {round(elapsed, 1) - previous.time:+.1f}s')
    score_line = ('NEW BEST' if result.new_best_score else
                  f'BEST {previous.score}  {int(score) - previous.score:+d}')
    return time_line, score_line


class LevelCompleteMenu:
    """Handles level complete/failed screen rendering."""

    # The game's own squids hover either side of the failed banner. Positions
    # are fractions of the banner, lengths are on the reference layout
    FAILED_SQUID_OFFSET_X = 0.365  # From the banner's centre line
    FAILED_SQUID_Y = 0.2
    FAILED_SQUID_RADIUS = 58
    FAILED_SQUID_TILT = 20  # Degrees the mantle tips lean in over the lettering
    FAILED_SQUID_SWAY = 7  # Degrees
    FAILED_SQUID_BOB = 7
    FAILED_SQUID_SPEED = 0.03  # Radians per frame

    def __init__(self, screen: pygame.Surface):
        """Initialize level complete menu.

        Args:
            screen: The pygame Surface to draw on.
        """
        self.screen = screen
        self.scale = menu_scale()
        self.level_complete_background: Optional[AnimatedBackground] = None
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        # Banner artwork (or a text heading if the file is missing), keyed by success
        self.banners: Dict[bool, pygame.Surface] = {}
        # Banners that are artwork on a dark background, and so are added to the screen as light
        self._banner_is_light: Dict[bool, bool] = {}
        self._level_numbers: Dict[Tuple[int, int], Optional[pygame.Surface]] = {}
        self.menu_pulse_phase = 0.0
        self.failed_squids = [creatures.build('replay', (0.0, 0.0)) for _ in range(2)]
        self.failed_squid_phase = 0.0
        self.complete_portraits: List[EnemyDeathPortrait] = []
        self.label_font = get_font(self._scaled(26), bold=True)
        self.comparison_font = get_font(self._scaled(21))
        self.value_font = get_font(self._scaled(52), bold=True)
        self.star_line_font = get_font(self._scaled(20), bold=True)
        self.star_line_rects: List[pygame.Rect] = []  # Where each star's line was last drawn
        self.hints = render_hint_row(
            get_font(self._scaled(28)), [("ARROWS", "Navigate"), ("SPACE / A", "Select"), ("R", "Retry level")]
        )
        self.hints_without_retry = render_hint_row(
            get_font(self._scaled(28)), [("ARROWS", "Navigate"), ("SPACE / A", "Select")])   # a won race is not retried
        self.number_sprite = NumberSprite()
        self.menu_options: list[str] = []
        self.menu_buttons: list[Button] = []
        self.menu_selected_index = 0
        self._initialize()
        self._update_failed_squids(0.0)

    def _scaled(self, value: float) -> int:
        """Scale a reference-layout length to the current screen size."""
        return max(1, int(round(value * self.scale)))

    def set_options(self, level_succeeded: bool, last_enemy=None) -> None:
        """Set menu options based on level success status.

        Args:
            level_succeeded: True if level was completed successfully.
        """
        self.complete_portraits = []
        if level_succeeded:
            self.menu_options = ["CONTINUE", "MAIN MENU"]
            if last_enemy is not None:
                self.complete_portraits = [EnemyDeathPortrait(last_enemy, self._scaled(self.FAILED_SQUID_RADIUS))
                                           for _ in range(2)]
        else:
            self.menu_options = ["RETRY LEVEL", "MAIN MENU"]
        self.menu_selected_index = 0
        self._create_buttons()

    def _create_buttons(self) -> None:
        """Create buttons for current menu options."""
        button_font = get_font(self._scaled(40))
        self.menu_buttons = []

        for i, option_text in enumerate(self.menu_options):
            button = Button(
                option_text,
                (0, 0),  # Placed from the layout when drawn
                button_font,
                width=self._scaled(440),
                height=self._scaled(64)
            )
            button.selected = (i == self.menu_selected_index)
            self.menu_buttons.append(button)

    def navigate_up(self) -> None:
        """Navigate to the previous menu option."""
        if self.menu_selected_index > 0:
            self.menu_selected_index -= 1

    def navigate_down(self) -> None:
        """Navigate to the next menu option."""
        if self.menu_selected_index < len(self.menu_options) - 1:
            self.menu_selected_index += 1

    def get_selected_option(self) -> str:
        """Get the currently selected menu option.

        Returns:
            The text of the currently selected option.
        """
        if not self.menu_options:
            return ""
        return self.menu_options[self.menu_selected_index]

    def _initialize(self) -> None:
        """Load the level complete and failed banners."""
        # (file, text fallback, height on the reference layout)
        sources = {
            True: ("assets/level_complete.png", "LEVEL COMPLETE", 330),
            False: ("assets/level_failed.png", "LEVEL FAILED", 400),
        }
        for succeeded, (path, fallback_text, reference_height) in sources.items():
            try:
                image = pygame.image.load(resource_path(path)).convert_alpha()
                # Trim transparent margins so the layout works from the artwork itself
                image = image.subsurface(image.get_bounding_rect())
                height = self._scaled(reference_height)
                width = int(image.get_width() * height / image.get_height())
                self.banners[succeeded] = as_light(pygame.transform.smoothscale(image, (width, height)))
                self._banner_is_light[succeeded] = True
            except (pygame.error, FileNotFoundError):
                self.banners[succeeded] = render_title(get_font(self._scaled(110)), fallback_text)
                self._banner_is_light[succeeded] = False

    def _level_number(self, level: int, height: int) -> Optional[pygame.Surface]:
        """Get the level number in the digit artwork, as light to add to the screen."""
        key = (level, height)
        if key not in self._level_numbers:
            number = self.number_sprite.render_number(level)
            if number is not None:
                width = int(number.get_width() * height / number.get_height())
                number = as_light(pygame.transform.smoothscale(number, (width, height)))
            self._level_numbers[key] = number
        return self._level_numbers[key]

    def layout(self, level_succeeded: bool) -> Dict[str, pygame.Rect]:
        """Where each part of the screen sits, top to bottom.

        Returns:
            Rectangles keyed 'title', 'level_number', 'stars' (success only),
            'stats', 'buttons' and 'hints'.
        """
        center_x = config.SCREEN_WIDTH // 2
        layout: Dict[str, pygame.Rect] = {}

        banner = self.banners[level_succeeded]
        layout['title'] = banner.get_rect(midtop=(center_x, self._scaled(24)))

        number_height = self._scaled(72)
        layout['level_number'] = pygame.Rect(0, 0, number_height * 4, number_height)
        layout['level_number'].midtop = (center_x, layout['title'].bottom + self._scaled(10))
        next_top = layout['level_number'].bottom

        if level_succeeded:
            star_size = config.LEVEL_COMPLETE_STAR_SIZE
            # The stars, and under them the checklist of goals, one slot wide each
            lines_height = self._scaled(STAR_LINE_GAP) + self.star_line_font.get_linesize()
            layout['stars'] = pygame.Rect(0, 0, self._scaled(STAR_SLOT_WIDTH) * 5, star_size + lines_height)
            layout['stars'].midtop = (center_x, next_top + self._scaled(26))
            next_top = layout['stars'].bottom

        layout['stats'] = pygame.Rect(0, 0, self._scaled(920), self._scaled(114 if level_succeeded else 86))
        layout['stats'].midtop = (center_x, next_top + self._scaled(30))

        button_height = self._scaled(64)
        button_spacing = self._scaled(84)
        layout['buttons'] = pygame.Rect(0, 0, self._scaled(440), button_height + button_spacing)
        layout['buttons'].midtop = (center_x, layout['stats'].bottom + self._scaled(34))

        layout['hints'] = self.hints.get_rect(midbottom=(center_x, config.SCREEN_HEIGHT - self._scaled(48)))
        return layout

    def _draw_star_lines(self, area: pygame.Rect, star_animation: AnimatedStarRating, lines: Sequence[str]) -> None:
        """The checklist under the stars: each goal with a small star, lit if it was met, and its figure."""
        self.star_line_rects = []
        top = star_animation.y + star_animation.star_size // 2 + self._scaled(STAR_LINE_GAP)
        slot = area.width // 5
        mark = self.star_line_font.get_height() * 0.8
        gap = self._scaled(6)
        room = slot - self._scaled(8) - round(mark) - gap
        for i, text in enumerate(lines):
            met = i < len(star_animation.earned) and star_animation.earned[i]
            line = self.star_line_font.render(text, True, TEXT_LABEL if met else STAR_LINE_MISSED)
            if line.get_width() > room:
                line = pygame.transform.smoothscale(
                    line, (room, max(1, round(line.get_height() * room / line.get_width()))))
            width = round(mark) + gap + line.get_width()
            rect = pygame.Rect(0, top, width, max(line.get_height(), round(mark)))
            rect.centerx = area.left + slot * i + slot // 2
            draw_star(self.screen, rect.left + mark / 2, rect.centery, mark, filled=met)
            self.screen.blit(line, line.get_rect(midleft=(rect.left + round(mark) + gap, rect.centery)))
            self.star_line_rects.append(rect)

    def _draw_stats(self, rect: pygame.Rect, stats: List[Tuple[str, str]], comparisons=()) -> None:
        """Draw a row of figures, each a small label over a large value, with dividers between."""
        column_width = rect.width // len(stats)
        for i, (label_text, value_text) in enumerate(stats):
            column = pygame.Rect(rect.left + i * column_width, rect.top, column_width, rect.height)
            label = self.label_font.render(label_text, True, TEXT_LABEL)
            self.screen.blit(label, label.get_rect(midtop=(column.centerx, column.top)))
            value = self.value_font.render(value_text, True, config.COLOR_TEXT)
            value_bottom = column.top + self._scaled(86)
            self.screen.blit(value, value.get_rect(midbottom=(column.centerx, value_bottom)))
            if i < len(comparisons) and comparisons[i]:
                color = (255, 220, 110) if comparisons[i] == 'NEW BEST' else TEXT_LABEL
                caption = self.comparison_font.render(comparisons[i], True, color)
                self.screen.blit(caption, caption.get_rect(midtop=(column.centerx, value_bottom + self._scaled(7))))
            if i > 0:
                pygame.draw.line(
                    self.screen, (70, 64, 120),
                    (column.left, column.top + self._scaled(8)), (column.left, column.bottom - self._scaled(8)),
                    max(1, self._scaled(2))
                )

    def _update_failed_squids(self, dt: float) -> None:
        """Hold the squids beside the failed banner, bobbing and swaying as they hover."""
        self.failed_squid_phase += self.FAILED_SQUID_SPEED * dt
        title = self.layout(False)['title']
        for side, squid in zip((-1, 1), self.failed_squids):
            phase = self.failed_squid_phase + (side + 1) * 0.9  # Out of step with each other
            squid.radius = self._scaled(self.FAILED_SQUID_RADIUS)
            squid.angle = -90 - side * self.FAILED_SQUID_TILT + self.FAILED_SQUID_SWAY * math.sin(phase * 0.6)
            squid.x = title.centerx + side * title.width * self.FAILED_SQUID_OFFSET_X
            squid.y = title.top + title.height * self.FAILED_SQUID_Y + self._scaled(self.FAILED_SQUID_BOB) * math.sin(phase)
            squid.body.update(squid, dt, None)

    def update(self, dt: float) -> None:
        """Update menu animations.

        Args:
            dt: Delta time since last update.
        """
        if self.level_complete_background:
            self.level_complete_background.update(dt)
        # Update pulse phase for button glow
        self.menu_pulse_phase = advance_pulse(self.menu_pulse_phase, dt)
        self._update_failed_squids(dt)
        for portrait in self.complete_portraits:
            portrait.update(dt)

    def draw(
        self,
        level: int,
        level_succeeded: bool,
        completion_time_seconds: float,
        level_score_breakdown: Dict,
        star_animation: Optional[AnimatedStarRating],
        show_quit_confirmation: bool,
        draw_quit_confirmation_callback,
        level_result: Optional[LevelResult] = None,
        star_lines: Sequence[str] = (),
        stats_override: Optional[Sequence[Tuple[str, str]]] = None
    ) -> None:
        """Draw level complete or failed screen.

        Args:
            level: Current level number.
            level_succeeded: True if level was completed successfully.
            completion_time_seconds: Time taken to complete the level.
            level_score_breakdown: Score breakdown dictionary.
            star_animation: Optional animated star rating.
            star_lines: What each star was for, one short line under each.
            show_quit_confirmation: Whether to show quit confirmation overlay.
            draw_quit_confirmation_callback: Callback to draw quit confirmation.
        """
        # Initialize background if needed
        if not self.level_complete_background:
            self.level_complete_background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        self.screen.blit(self.backdrop, (0, 0))
        self.level_complete_background.draw(self.screen)

        expected_options = ["CONTINUE", "MAIN MENU"] if level_succeeded else ["RETRY LEVEL", "MAIN MENU"]
        if self.menu_options != expected_options:
            self.set_options(level_succeeded)
        layout = self.layout(level_succeeded)

        # Banner, then the level number beneath it
        self.screen.blit(
            self.banners[level_succeeded], layout['title'],
            special_flags=pygame.BLEND_RGB_ADD if self._banner_is_light[level_succeeded] else 0
        )
        if level_succeeded:
            title = layout['title']
            for side, portrait in zip((-1, 1), self.complete_portraits):
                portrait.draw(self.screen, (
                    title.centerx + side * (title.width / 2 + self._scaled(100)),
                    title.centery,
                ))
        else:
            for squid in self.failed_squids:
                squid.draw(self.screen)
        number = self._level_number(level, layout['level_number'].height)
        if number is not None:
            self.screen.blit(
                number, number.get_rect(center=layout['level_number'].center), special_flags=pygame.BLEND_RGB_ADD
            )

        # Draw animated star rating in its slot
        if star_animation and 'stars' in layout:
            star_animation.x = layout['stars'].centerx - 2 * star_animation.star_spacing
            star_animation.y = layout['stars'].top + star_animation.star_size // 2
            star_animation.draw(self.screen)
            self._draw_star_lines(layout['stars'], star_animation, star_lines)

        # Figures for the level: time as MM:SS.{tenths} and scores
        total_score = int(level_score_breakdown.get('total_score', 0))
        if level_succeeded:
            stats = [
                ("TIME", format_time(completion_time_seconds)),
                ("LEVEL SCORE", f"{int(level_score_breakdown.get('final_score', 0)):,}"),
                ("PROGRESS SCORE", f"{total_score:,}"),
            ]
        else:
            stats = [("PROGRESS SCORE", f"{total_score:,}")]
        comparisons = comparison_lines(level_result, level_score_breakdown.get('final_score', 0), completion_time_seconds) if level_succeeded else ()
        if stats_override is not None:      # a race: who won and the points, not scores
            stats, comparisons = list(stats_override), ()
        self._draw_stats(layout['stats'], stats, comparisons)

        # Draw buttons
        for i, button in enumerate(self.menu_buttons):
            button.position = (
                layout['buttons'].centerx,
                layout['buttons'].top + button.height // 2 + i * self._scaled(84)
            )
        draw_buttons(self.screen, self.menu_buttons, self.menu_selected_index, self.menu_pulse_phase)

        hints = self.hints_without_retry if stats_override is not None and level_succeeded else self.hints
        self.screen.blit(hints, hints.get_rect(midbottom=layout['hints'].midbottom))

        # Draw quit confirmation overlay if active
        if show_quit_confirmation:
            draw_quit_confirmation_callback()
