"""Main menu rendering.

This module handles the rendering of the main menu screen.
"""

import math
import pygame
from typing import Optional, Tuple
import config
from rendering.fonts import get_font
from rendering.menu_components import (
    AnimatedBackground, NeonText, Button, BUTTON_ACCENT_START, REFERENCE_HEIGHT, REFERENCE_WIDTH,
    build_menu_backdrop, menu_scale, render_pill
)
from rendering.visual_effects import create_radial_gradient_surface
from utils.resource_path import resource_path


TITLE_HEIGHT_FRACTION = 0.34
TITLE_CENTER_Y_FRACTION = 0.19
TITLE_FLOAT_SPEED = 0.9
BUTTON_HEIGHT = 64


class MainMenu:
    """Handles main menu rendering and state."""
    
    def __init__(self, screen: pygame.Surface):
        """Initialize main menu.
        
        Args:
            screen: The pygame Surface to draw on.
        """
        self.screen = screen
        self.menu_background: Optional[AnimatedBackground] = None
        self.menu_title: Optional[NeonText] = None
        self.menu_title_image: Optional[pygame.Surface] = None
        self.menu_title_rect: Optional[pygame.Rect] = None
        self.menu_options = ["START GAME", "LEVELS", "HOW TO PLAY", "FUN", "SELECT PROFILE", "MULTIPLAYER", "SETTINGS", "QUIT"]
        if config.editor_enabled():
            self.menu_options.insert(self.menu_options.index("QUIT"), "EDITOR")
        self.menu_buttons: list[Button] = []
        self.menu_selected_index = 0
        self.menu_pulse_phase = 0.0
        self.profile_name: Optional[str] = None
        self.profile_level: Optional[int] = None
        self.scale = menu_scale()
        self.profile_font = get_font(self._scaled(28))
        self.menu_time = 0.0
        self.backdrop: Optional[pygame.Surface] = None
        self.profile_badge: Optional[pygame.Surface] = None
        self._profile_badge_key: Optional[Tuple[Optional[str], Optional[int]]] = None
        self._initialize()
    
    def _scaled(self, value: float) -> int:
        """Scale a reference-layout length to the current screen size."""
        return max(1, int(round(value * self.scale)))
    
    def _initialize(self) -> None:
        """Initialize menu UI components."""
        # Create animated background
        self.menu_background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        
        # Load title graphic
        title_center = (config.SCREEN_WIDTH // 2, int(config.SCREEN_HEIGHT * TITLE_CENTER_Y_FRACTION))
        try:
            title_image = pygame.image.load(resource_path("assets/title.png")).convert_alpha()
            # Scale title image with the layout (maintain aspect ratio)
            title_height = self._scaled(REFERENCE_HEIGHT * TITLE_HEIGHT_FRACTION)
            title_width = int(title_image.get_width() * title_height / title_image.get_height())
            max_title_width = self._scaled(REFERENCE_WIDTH - 96)
            if title_width > max_title_width:
                title_height = round(title_height * max_title_width / title_width)
                title_width = max_title_width
            self.menu_title_image = pygame.transform.smoothscale(title_image, (title_width, title_height))
            self.menu_title_rect = self.menu_title_image.get_rect(center=title_center)
            # Soft halo behind the title, baked into the static backdrop
            halo = create_radial_gradient_surface(
                (int(title_width * 1.5), int(title_height * 1.5)),
                (110, 60, 220),
                90
            )
            self.backdrop.blit(halo, halo.get_rect(center=title_center))
        except (pygame.error, FileNotFoundError):
            # Fallback to text if image not found
            title_font = get_font(self._scaled(config.FONT_SIZE_TITLE * 2))
            self.menu_title = NeonText(
                "SQUIDDLER",
                title_font,
                title_center,
                config.COLOR_NEON_ASTER_START,
                config.COLOR_NEON_VOID_END,
                center=True
            )
            self.menu_title_image = None
        
        # Create buttons for all menu options
        button_font = get_font(self._scaled(40))
        self.menu_buttons = []
        
        # Position buttons in a group beneath the title
        title_bottom = (self.menu_title_rect.bottom if self.menu_title_image is not None
                        else title_center[1] + button_font.get_height())
        start_y, button_spacing = self._button_column(title_bottom)
        
        for i, option_text in enumerate(self.menu_options):
            button = Button(
                option_text,
                (config.SCREEN_WIDTH // 2, round(start_y + i * button_spacing)),
                button_font,
                width=self._scaled(440),
                height=self._scaled(BUTTON_HEIGHT)
            )
            button.selected = (i == self.menu_selected_index)
            self.menu_buttons.append(button)
        
        self.backdrop = self.backdrop.convert()
    
    def _button_column(self, title_bottom: int) -> Tuple[int, float]:
        """Where the first button sits and how far apart they are: the usual place and spacing, moved up and then
        packed closer only as far as it takes for the last one to clear the profile badge."""
        start_y, spacing = int(config.SCREEN_HEIGHT * 0.42), self._scaled(80)
        half = self._scaled(BUTTON_HEIGHT) // 2
        gaps = len(self.menu_options) - 1
        last_centre = self._badge_top() - self._scaled(16) - half
        highest_start = title_bottom + self._scaled(24) + half
        overflow = start_y + gaps * spacing - last_centre
        if overflow > 0:
            start_y = max(highest_start, start_y - overflow)
            spacing = min(spacing, (last_centre - start_y) / max(1, gaps))
        return start_y, spacing

    def _badge_top(self) -> int:
        """The top of the profile badge, which is as tall as its font and padding whatever it says."""
        return self._badge_bottom() - (self.profile_font.get_height() + 2 * self._scaled(8))

    def _badge_bottom(self) -> int:
        return config.SCREEN_HEIGHT - self._scaled(56)

    def profile_badge_rect(self) -> pygame.Rect:
        """Where the profile badge is drawn."""
        return self._profile_badge().get_rect(midbottom=(config.SCREEN_WIDTH // 2, self._badge_bottom()))

    def _profile_badge(self) -> pygame.Surface:
        badge_key = (self.profile_name, self.profile_level)
        if self.profile_badge is None or self._profile_badge_key != badge_key:
            self.profile_badge = render_pill(
                self.profile_font,
                f"{(self.profile_name or '').upper()}   \u00b7   LEVEL {self.profile_level or '-'}",
                (215, 225, 255),
                BUTTON_ACCENT_START,
                (self._scaled(20), self._scaled(8)),
                max(1, self._scaled(1.5))
            )
            self._profile_badge_key = badge_key
        return self.profile_badge

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
        return self.menu_options[self.menu_selected_index]
    
    def update(self, dt: float) -> None:
        """Update menu animations.
        
        Args:
            dt: Delta time since last update.
        """
        if self.menu_background:
            self.menu_background.update(dt)
        if self.menu_title:
            self.menu_title.update(dt)
        self.menu_time += dt / 60.0
        # Update pulse phase for button glow
        self.menu_pulse_phase += config.BUTTON_GLOW_PULSE_SPEED * dt / 60.0
        if self.menu_pulse_phase >= 2 * 3.14159:
            self.menu_pulse_phase -= 2 * 3.14159
    
    def draw(self) -> None:
        """Draw main menu with animated background and neon effects."""
        # Draw backdrop and animated background
        if self.backdrop:
            self.screen.blit(self.backdrop, (0, 0))
        if self.menu_background:
            self.menu_background.draw(self.screen)
        
        # Draw title graphic (gently floating) or neon text fallback
        if self.menu_title_image is not None:
            float_phase = self.menu_time * TITLE_FLOAT_SPEED
            title_rect = self.menu_title_rect.move(0, int(math.sin(float_phase) * self._scaled(6)))
            self.screen.blit(self.menu_title_image, title_rect)
        elif self.menu_title:
            self.menu_title.draw(self.screen)
        
        # Draw buttons
        for i, button in enumerate(self.menu_buttons):
            button.selected = (i == self.menu_selected_index)
            button.draw(self.screen, self.menu_pulse_phase)
        
        # Draw profile badge at the bottom
        if self.profile_name:
            self.screen.blit(self._profile_badge(), self.profile_badge_rect())

    def set_profile_info(self, name: Optional[str], level: Optional[int]) -> None:
        """Update the profile info shown on the menu."""
        self.profile_name = name
        self.profile_level = level

