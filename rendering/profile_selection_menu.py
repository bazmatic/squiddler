"""Render a profile selection UI for choosing or creating profiles."""

from __future__ import annotations

import pygame
from typing import List, Optional, Tuple

import config
from rendering.fonts import get_font
from profiles import Profile, ProfileManager
from rendering.menu_components import (
    AnimatedBackground, BUTTON_ACCENT_START, Button, TEXT_DIM, advance_pulse, build_menu_backdrop,
    menu_scale, render_hint_row, render_pill, render_title, selection_text_color
)


class ProfileSelectionMenu:
    """Menu that lists existing profiles and lets players add new ones."""

    CREATE_OPTION = "CREATE NEW PROFILE"
    MAX_NAME_LENGTH = 18
    # What a controller can spell with, one letter at a time
    PAD_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"

    def __init__(self, screen: pygame.Surface, profile_manager: ProfileManager):
        self.screen = screen
        self.profile_manager = profile_manager
        self.scale = menu_scale()
        self.menu_background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.title = render_title(get_font(self._scaled(96)), "SELECT PROFILE")
        self.entry_font = get_font(self._scaled(44))
        self.detail_font = get_font(self._scaled(28), bold=False)
        self.prompt_font = get_font(self._scaled(28))
        self.browse_hints = render_hint_row(
            self.prompt_font, [("ARROWS", "Move"), ("ENTER / A", "Select"), ("ESC / B", "Back")]
        )
        self.typing_hints = render_hint_row(self.prompt_font, [("ENTER", "Save"), ("ESC", "Cancel")])
        self.pad_typing_hints = render_hint_row(
            self.prompt_font,
            [("UP / DOWN", "Letter"), ("A / RIGHT", "Add"), ("X / LEFT", "Delete"), ("START", "Save"), ("B", "Cancel")]
        )
        self.menu_pulse_phase = 0.0
        self.profile_entries: List[Profile] = []
        self.options: List[str] = []
        self.buttons: List[Button] = []
        self.selected_index = 0
        self.creating_profile = False
        self.new_profile_name = ""
        # The letter being chosen on a controller, shown after the name but not yet part of it
        self.pending_character: Optional[str] = None
        self.feedback_message: Optional[str] = None
        self.refresh_profiles()

    def _scaled(self, value: float) -> int:
        """Scale a reference-layout length to the current screen size."""
        return max(1, int(round(value * self.scale)))

    def refresh_profiles(self) -> None:
        """Reload the profile list from the manager."""
        self.profile_entries = self.profile_manager.get_profiles()
        names = [profile.name for profile in self.profile_entries]
        self.options = names + [self.CREATE_OPTION]
        row_size = self._row_size()
        self.buttons = [
            Button("", (0, 0), self.entry_font, width=row_size[0], height=row_size[1]) for _ in self.options
        ]
        active_profile = self.profile_manager.get_active_profile()
        if active_profile and active_profile.name in names:
            self.selected_index = names.index(active_profile.name)
        elif self.selected_index >= len(self.options):
            self.selected_index = max(0, len(self.options) - 1)

    def navigate_up(self) -> None:
        """Move selection cursor up."""
        if self.selected_index > 0:
            self.selected_index -= 1

    def navigate_down(self) -> None:
        """Move selection cursor down."""
        if self.selected_index < len(self.options) - 1:
            self.selected_index += 1

    def get_selected_option(self) -> Optional[str]:
        """Return the text of the currently selected option."""
        if not self.options:
            return None
        return self.options[self.selected_index]

    def start_creating_profile(self, with_controller: bool = False) -> None:
        """Begin the create-profile flow; on a controller the name is picked letter by letter."""
        self.creating_profile = True
        self.new_profile_name = ""
        self.pending_character = self.PAD_CHARACTERS[0] if with_controller else None
        self.feedback_message = None

    def cancel_creating_profile(self) -> None:
        """Abort creating a profile."""
        self.creating_profile = False
        self.new_profile_name = ""
        self.pending_character = None
        self.feedback_message = "Profile creation cancelled."

    def step_pending_character(self, step: int) -> None:
        """Move the controller's letter along PAD_CHARACTERS, wrapping round."""
        if self.pending_character is None:
            self.pending_character = self.PAD_CHARACTERS[0]
            return
        index = self.PAD_CHARACTERS.index(self.pending_character)
        self.pending_character = self.PAD_CHARACTERS[(index + step) % len(self.PAD_CHARACTERS)]

    def commit_pending_character(self) -> None:
        """Add the controller's letter to the name; the next letter starts where this one was."""
        if self.pending_character is None:
            self.pending_character = self.PAD_CHARACTERS[0]
            return
        self.append_character(self.pending_character)

    def append_character(self, char: str) -> None:
        """Add a character to the active profile name."""
        if len(self.new_profile_name) >= self.MAX_NAME_LENGTH:
            return
        if not char.isprintable() or char in {"\r", "\n"}:
            return
        self.new_profile_name += char

    def backspace_character(self) -> None:
        """Remove the last character from the input."""
        self.new_profile_name = self.new_profile_name[:-1]

    def submit_new_profile(self) -> Optional[Profile]:
        """Attempt to create a new profile using the current input, including a letter still being chosen."""
        if self.pending_character is not None:
            self.append_character(self.pending_character)
            self.pending_character = None
        cleaned = self.new_profile_name.strip()
        if not cleaned:
            self.feedback_message = "Profile name cannot be empty."
            return None
        try:
            profile = self.profile_manager.create_profile(cleaned)
        except ValueError as exc:
            self.feedback_message = str(exc)
            return None

        self.creating_profile = False
        self.new_profile_name = ""
        self.pending_character = None
        self.feedback_message = f"Created profile '{profile.name}'."
        self.refresh_profiles()
        self.selected_index = self.options.index(profile.name)
        return profile

    def _row_size(self) -> Tuple[int, int]:
        return self._scaled(760), self._scaled(76)

    def title_rect(self) -> pygame.Rect:
        """Get the heading's rectangle."""
        return self.title.get_rect(center=(config.SCREEN_WIDTH // 2, int(config.SCREEN_HEIGHT * 0.14)))

    def _hints(self) -> pygame.Surface:
        if not self.creating_profile:
            return self.browse_hints
        return self.typing_hints if self.pending_character is None else self.pad_typing_hints

    def hints_rect(self) -> pygame.Rect:
        """Get the rectangle of the key hints at the bottom of the screen."""
        return self._hints().get_rect(midbottom=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT - self._scaled(56)))

    def _visible_range(self) -> Tuple[int, int, int]:
        """Work out which options fit on screen.

        Returns:
            (first visible index, number visible, y of the first row's top). The
            window follows the selection so it is always on screen.
        """
        spacing = self._scaled(92)
        top = self.title_rect().bottom + self._scaled(20)
        # Leave room above the hints for the feedback line
        bottom = self.hints_rect().top - self._scaled(64)
        visible = max(1, min(len(self.options), (bottom - top + spacing - self._row_size()[1]) // spacing))
        first = min(max(0, self.selected_index - visible // 2), max(0, len(self.options) - visible))
        return first, visible, top

    def row_rect(self, index: int) -> Optional[pygame.Rect]:
        """Get the rectangle of an option's row, or None if it is scrolled out of view."""
        first, visible, top = self._visible_range()
        if not first <= index < first + visible:
            return None
        width, height = self._row_size()
        rect = pygame.Rect(0, 0, width, height)
        rect.midtop = (config.SCREEN_WIDTH // 2, top + (index - first) * self._scaled(92))
        return rect

    def _draw_profile_row(self, rect: pygame.Rect, profile: Profile, blend_color, is_active: bool) -> None:
        """Draw a profile's name on the left and its progress on the right."""
        pad = self._scaled(32)
        name = self.entry_font.render(profile.name, True, blend_color)
        name_rect = name.get_rect(midleft=(rect.left + pad, rect.centery))
        self.screen.blit(name, name_rect)
        if is_active:
            badge = render_pill(
                self.detail_font, "ACTIVE", (215, 225, 255), BUTTON_ACCENT_START,
                (self._scaled(12), self._scaled(3)), max(1, self._scaled(1.5))
            )
            self.screen.blit(badge, badge.get_rect(midleft=(name_rect.right + self._scaled(16), rect.centery)))
        details = self.detail_font.render(
            f"LEVEL {profile.level}   \u00b7   SCORE {profile.total_score:,}", True, TEXT_DIM
        )
        self.screen.blit(details, details.get_rect(midright=(rect.right - pad, rect.centery)))

    def _draw_create_row(self, rect: pygame.Rect, blend_color) -> None:
        """Draw the create option, which turns into the name field while typing."""
        if not self.creating_profile:
            label = self.entry_font.render("+  NEW PROFILE", True, blend_color)
            self.screen.blit(label, label.get_rect(center=rect.center))
            return
        
        if self.new_profile_name:
            typed = self.entry_font.render(self.new_profile_name, True, config.COLOR_TEXT)
        elif self.pending_character is None:
            typed = self.entry_font.render("TYPE A NAME", True, (90, 90, 120))
        else:
            typed = self.entry_font.render("", True, config.COLOR_TEXT)
        typed_rect = typed.get_rect(midleft=(rect.left + self._scaled(32), rect.centery))
        self.screen.blit(typed, typed_rect)
        if self.pending_character is not None:
            self._draw_pending_character(typed_rect.right + self._scaled(4), rect.centery)
            return
        # Blinking caret
        if (pygame.time.get_ticks() // 450) % 2 == 0:
            caret_x = typed_rect.right + self._scaled(4) if self.new_profile_name else typed_rect.left - self._scaled(6)
            pygame.draw.rect(
                self.screen, BUTTON_ACCENT_START,
                (caret_x, rect.centery - self._scaled(18), max(2, self._scaled(3)), self._scaled(36))
            )

    def _draw_pending_character(self, left: int, centery: int) -> None:
        """The controller's letter in a lit box, with arrows above and below for choosing it."""
        letter = self.entry_font.render(self.pending_character, True, (255, 255, 255))
        box = pygame.Rect(0, 0, max(letter.get_width(), self.entry_font.size("W")[0]) + self._scaled(12), self._scaled(52))
        box.midleft = (left, centery)
        pygame.draw.rect(self.screen, BUTTON_ACCENT_START, box, max(2, self._scaled(3)), border_radius=self._scaled(6))
        self.screen.blit(letter, letter.get_rect(center=box.center))
        size = self._scaled(7)
        for direction, y in ((-1, box.top - self._scaled(8)), (1, box.bottom + self._scaled(8))):
            pygame.draw.polygon(
                self.screen, BUTTON_ACCENT_START,
                [(box.centerx - size, y - direction * size // 2), (box.centerx + size, y - direction * size // 2),
                 (box.centerx, y + direction * size // 2)]
            )

    def _draw_scroll_arrow(self, center: Tuple[int, int], direction: int) -> None:
        """Draw a small arrow showing there are more profiles above (-1) or below (1)."""
        size = self._scaled(9)
        x, y = center
        pygame.draw.polygon(
            self.screen, TEXT_DIM,
            [(x - size, y - direction * size // 2), (x + size, y - direction * size // 2), (x, y + direction * size // 2)]
        )

    def draw(self) -> None:
        """Draw the profile selection UI."""
        self.screen.blit(self.backdrop, (0, 0))
        self.menu_background.draw(self.screen)
        self.screen.blit(self.title, self.title_rect())

        active_profile = self.profile_manager.get_active_profile()
        active_name = active_profile.name if active_profile else None
        first, visible, _ = self._visible_range()

        for index in range(first, first + visible):
            rect = self.row_rect(index)
            button = self.buttons[index]
            button.position = rect.center
            button.selected = index == self.selected_index
            button.draw(self.screen, self.menu_pulse_phase)
            text_color = selection_text_color(button._selection_blend or 0.0)
            if self.options[index] == self.CREATE_OPTION:
                self._draw_create_row(rect, text_color)
            else:
                profile = self.profile_entries[index]
                self._draw_profile_row(rect, profile, text_color, profile.name == active_name)

        if first > 0:
            self._draw_scroll_arrow((config.SCREEN_WIDTH // 2, self.row_rect(first).top - self._scaled(10)), -1)
        if first + visible < len(self.options):
            self._draw_scroll_arrow(
                (config.SCREEN_WIDTH // 2, self.row_rect(first + visible - 1).bottom + self._scaled(12)), 1
            )

        hints_rect = self.hints_rect()
        self.screen.blit(self._hints(), hints_rect)

        if self.feedback_message:
            feedback_surface = self.prompt_font.render(self.feedback_message, True, (150, 255, 200))
            self.screen.blit(
                feedback_surface,
                feedback_surface.get_rect(center=(config.SCREEN_WIDTH // 2, hints_rect.top - self._scaled(36)))
            )

    def update(self, dt: float) -> None:
        """Update menu animations."""
        self.menu_background.update(dt)
        self.menu_pulse_phase = advance_pulse(self.menu_pulse_phase, dt)

