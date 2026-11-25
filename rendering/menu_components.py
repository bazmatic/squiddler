"""Reusable menu UI components.

This module provides reusable UI components for menus including buttons,
controller icons, animated backgrounds, and neon text rendering.
"""

import pygame
import math
from typing import Dict, List, Tuple, Optional, Callable
import config
from rendering.fonts import get_font
from rendering.visual_effects import (
    draw_neon_text,
    draw_button_glow,
    interpolate_color,
    Starfield,
    MenuParticleSystem,
    create_radial_gradient_surface
)


# Accent gradient for selected buttons, matching the title logo (cyan to magenta)
BUTTON_ACCENT_START = config.COLOR_NEON_VOID_START
BUTTON_ACCENT_END = (255, 90, 220)
BUTTON_SELECTED_SCALE = 1.06
BUTTON_FADE_MS = 140
BUTTON_TEXT_IDLE = (160, 160, 190)

TEXT_DIM = (160, 160, 185)
TEXT_LABEL = (110, 120, 170)
KEY_TEXT = (210, 225, 255)
KEY_OUTLINE = (90, 110, 170)

# Menu layouts are authored for a 1080px-high screen and scaled to the real
# height. REFERENCE_WIDTH is the narrowest screen that layout fits; narrower
# (e.g. portrait) screens scale down by width instead so nothing runs off the sides
REFERENCE_HEIGHT = 1080
REFERENCE_WIDTH = 1200

# Button skins keyed by (width, height), shared by all buttons of that size
_button_skin_cache: dict = {}
_panel_cache: Dict[Tuple[int, int, int], pygame.Surface] = {}
_panel_glow_cache: Dict[Tuple[int, int, int], pygame.Surface] = {}


def menu_scale() -> float:
    """Factor that scales a reference-layout length to the current screen size."""
    return min(config.SCREEN_HEIGHT / REFERENCE_HEIGHT, config.SCREEN_WIDTH / REFERENCE_WIDTH)


def build_menu_backdrop(width: int, height: int) -> pygame.Surface:
    """Build the static menu backdrop: nebula tint and vignette over the background color."""
    # Composed without an alpha channel so every pixel is opaque. The display
    # surface can carry alpha, and pygame copies rather than blends
    # per-pixel-alpha surfaces onto pixels whose alpha is 0, which would turn
    # everything drawn on top into solid blocks
    backdrop = pygame.Surface((width, height), 0, 24)
    backdrop.fill(config.COLOR_BACKGROUND)
    
    # Nebula clouds as (color, alpha, center fraction, size fraction)
    clouds = [
        ((90, 40, 170), 70, (0.5, 0.42), (1.1, 1.2)),
        ((30, 90, 190), 45, (0.24, 0.2), (0.8, 0.9)),
        ((190, 50, 150), 40, (0.8, 0.78), (0.8, 0.9)),
    ]
    for color, alpha, center, size in clouds:
        cloud = create_radial_gradient_surface((int(width * size[0]), int(height * size[1])), color, alpha)
        backdrop.blit(cloud, cloud.get_rect(center=(int(width * center[0]), int(height * center[1]))))
    
    vignette = create_radial_gradient_surface((width, height), (0, 0, 0), 0, edge_alpha=150, falloff=0.6)
    backdrop.blit(vignette, (0, 0))
    return backdrop


def render_pill(
    font: pygame.font.Font,
    text: str,
    text_color: Tuple[int, int, int] = KEY_TEXT,
    outline_color: Tuple[int, int, int] = KEY_OUTLINE,
    padding: Tuple[int, int] = (10, 5),
    border: int = 1
) -> pygame.Surface:
    """Render text inside an outlined pill."""
    text_surface = font.render(text, True, text_color)
    pill = pygame.Surface(
        (text_surface.get_width() + padding[0] * 2, text_surface.get_height() + padding[1] * 2),
        pygame.SRCALPHA
    )
    radius = pill.get_height() // 2
    pygame.draw.rect(pill, (*outline_color, 255), pill.get_rect(), border_radius=radius)
    pygame.draw.rect(pill, (24, 20, 48, 200), pill.get_rect().inflate(-border * 2, -border * 2), border_radius=radius - border)
    pill.blit(text_surface, text_surface.get_rect(center=pill.get_rect().center))
    return pill


def render_hint_row(
    font: pygame.font.Font,
    hints: List[Tuple[str, str]],
    action_gap: int = 9,
    hint_gap: int = 28
) -> pygame.Surface:
    """Render a row of key hints: each key in a pill, followed by what it does."""
    pieces: List[pygame.Surface] = []
    gaps: List[int] = []
    for key_text, action_text in hints:
        pieces.append(render_pill(font, key_text))
        gaps.append(action_gap)
        pieces.append(font.render(action_text, True, TEXT_DIM))
        gaps.append(hint_gap)
    gaps[-1] = 0
    
    row = pygame.Surface(
        (sum(piece.get_width() for piece in pieces) + sum(gaps), max(piece.get_height() for piece in pieces)),
        pygame.SRCALPHA
    )
    x = 0
    for piece, gap in zip(pieces, gaps):
        row.blit(piece, piece.get_rect(midleft=(x, row.get_height() // 2)))
        x += piece.get_width() + gap
    return row


def build_panel(size: Tuple[int, int], radius: int = 18) -> pygame.Surface:
    """Build a dark rounded panel with the accent gradient as its border."""
    key = (size[0], size[1], radius)
    if key in _panel_cache:
        return _panel_cache[key]
    
    supersample = 2  # Drawn oversized and scaled down for smooth corners
    border = 2
    outer = pygame.Rect(0, 0, size[0] * supersample, size[1] * supersample)
    inner = outer.inflate(-border * 2 * supersample, -border * 2 * supersample)
    ring = pygame.Surface(outer.size, pygame.SRCALPHA)
    pygame.draw.rect(ring, (255, 255, 255, 255), outer, border_radius=radius * supersample)
    pygame.draw.rect(ring, (0, 0, 0, 0), inner, border_radius=(radius - border) * supersample)
    ring.blit(_accent_gradient(outer.size), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    panel = pygame.Surface(outer.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (18, 14, 38, 240), inner, border_radius=(radius - border) * supersample)
    panel.blit(ring, (0, 0))
    panel = pygame.transform.smoothscale(panel, size)
    _panel_cache[key] = panel
    return panel


def build_panel_glow(size: Tuple[int, int], radius: int = 18, spread: int = 22) -> pygame.Surface:
    """Build the soft accent glow that sits behind a panel; larger than it by spread on every side."""
    key = (size[0], size[1], radius)
    if key in _panel_glow_cache:
        return _panel_glow_cache[key]
    
    glow_rect = pygame.Rect(0, 0, size[0] + spread * 2, size[1] + spread * 2)
    glow = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
    for step in range(spread):
        alpha = int(70 * ((step + 1) / spread) ** 2)
        pygame.draw.rect(
            glow, (255, 255, 255, alpha), glow_rect.inflate(-step * 2, -step * 2),
            border_radius=radius + spread - step
        )
    glow.blit(_accent_gradient(glow_rect.size), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    _panel_glow_cache[key] = glow
    return glow


def as_light(image: pygame.Surface, floor: Tuple[int, int, int] = (16, 4, 24)) -> pygame.Surface:
    """Turn artwork painted on a dark background into light that can be added to the screen.

    The banner and digit images carry their own near-black background, which
    shows as a rectangle over anything but a matching flat color. Subtracting
    that floor leaves only the glowing artwork; blit the result with
    pygame.BLEND_RGB_ADD.
    """
    light = pygame.Surface(image.get_size(), 0, 24)
    light.blit(image, (0, 0))
    light.fill(floor, special_flags=pygame.BLEND_RGB_SUB)
    return light


def render_title(font: pygame.font.Font, text: str) -> pygame.Surface:
    """Render a screen heading in the accent gradient with a soft glow behind it."""
    pad = max(6, font.get_height() // 3)
    white = font.render(text, True, (255, 255, 255))
    lettering = pygame.Surface((white.get_width() + pad * 2, white.get_height() + pad * 2), pygame.SRCALPHA)
    lettering.blit(white, (pad, pad))
    lettering.blit(_accent_gradient(lettering.get_size()), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    # Blur by scaling down and back up
    small = pygame.transform.smoothscale(lettering, (max(1, lettering.get_width() // 8), max(1, lettering.get_height() // 8)))
    title = pygame.transform.smoothscale(small, lettering.get_size())
    title.blit(title, (0, 0))
    
    # Lift the letters towards white so they read as lit tubes over their own glow
    bright = lettering.copy()
    bright.fill((90, 90, 90, 0), special_flags=pygame.BLEND_RGBA_ADD)
    title.blit(bright, (0, 0))
    return title


def _accent_gradient(size: Tuple[int, int]) -> pygame.Surface:
    """Create a horizontal accent gradient surface."""
    width, height = size
    surface = pygame.Surface(size, pygame.SRCALPHA)
    for x in range(width):
        color = interpolate_color(BUTTON_ACCENT_START, BUTTON_ACCENT_END, x / max(1, width - 1))
        pygame.draw.line(surface, color, (x, 0), (x, height))
    return surface


def _build_button_skins(width: int, height: int) -> Tuple[pygame.Surface, pygame.Surface, pygame.Surface]:
    """Build the idle panel, selected panel and selected glow for a button size.
    
    Returns:
        (idle, selected, glow) surfaces. The selected panel is larger than the
        idle one by BUTTON_SELECTED_SCALE and the glow is larger again.
    """
    radius = max(2, int(height * 0.3))
    border = max(2, height // 22)
    supersample = 2  # Panels are drawn oversized and scaled down for smooth corners
    
    def panel_rects(panel_width: int, panel_height: int) -> Tuple[pygame.Rect, pygame.Rect]:
        outer = pygame.Rect(0, 0, panel_width * supersample, panel_height * supersample)
        inner = outer.inflate(-border * 2 * supersample, -border * 2 * supersample)
        return outer, inner
    
    # Idle: dark translucent panel with a faint outline
    outer, inner = panel_rects(width, height)
    idle = pygame.Surface(outer.size, pygame.SRCALPHA)
    pygame.draw.rect(idle, (110, 110, 165, 120), outer, border_radius=radius * supersample)
    pygame.draw.rect(idle, (20, 16, 40, 150), inner, border_radius=(radius - border) * supersample)
    idle = pygame.transform.smoothscale(idle, (width, height))
    
    # Selected: brighter panel with a gradient border
    selected_width = int(round(width * BUTTON_SELECTED_SCALE))
    selected_height = int(round(height * BUTTON_SELECTED_SCALE))
    outer, inner = panel_rects(selected_width, selected_height)
    ring = pygame.Surface(outer.size, pygame.SRCALPHA)
    pygame.draw.rect(ring, (255, 255, 255, 255), outer, border_radius=radius * supersample)
    pygame.draw.rect(ring, (0, 0, 0, 0), inner, border_radius=(radius - border) * supersample)
    ring.blit(_accent_gradient(outer.size), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    selected = pygame.Surface(outer.size, pygame.SRCALPHA)
    pygame.draw.rect(selected, (36, 24, 72, 220), inner, border_radius=(radius - border) * supersample)
    selected.blit(ring, (0, 0))
    selected = pygame.transform.smoothscale(selected, (selected_width, selected_height))
    
    # Glow: rounded rects stepping inwards from transparent to bright
    glow_pad = max(4, int(height * 0.3))
    glow_rect = pygame.Rect(0, 0, selected_width + glow_pad * 2, selected_height + glow_pad * 2)
    glow = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
    for step in range(glow_pad):
        alpha = int(120 * ((step + 1) / glow_pad) ** 2)
        layer_rect = glow_rect.inflate(-step * 2, -step * 2)
        pygame.draw.rect(glow, (255, 255, 255, alpha), layer_rect, border_radius=radius + glow_pad - step)
    glow.blit(_accent_gradient(glow_rect.size), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    return idle, selected, glow


class ControllerIcon:
    """Renders controller button icons."""
    
    @staticmethod
    def draw_a_button(
        screen: pygame.Surface,
        position: Tuple[int, int],
        size: int = 30,
        selected: bool = False
    ) -> None:
        """Draw A button icon.
        
        Args:
            screen: The pygame Surface to draw on.
            position: (x, y) center position.
            size: Icon size in pixels.
            selected: If True, adds pulsing glow effect.
        """
        x, y = position
        color = config.COLOR_BUTTON_A
        
        # Draw glow if selected
        if selected:
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 200.0)
            glow_intensity = 0.4 + 0.3 * pulse
            glow_radius = size * 0.6
            for layer in range(5):
                layer_radius = glow_radius - layer
                alpha = int(255 * glow_intensity * (1.0 - layer / 5))
                if alpha > 0 and layer_radius > 0:
                    glow_color = (*color, alpha)
                    glow_surf = pygame.Surface((int(layer_radius * 2) + 4, int(layer_radius * 2) + 4), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, glow_color, (int(layer_radius) + 2, int(layer_radius) + 2), int(layer_radius))
                    screen.blit(glow_surf, (x - int(layer_radius) - 2, y - int(layer_radius) - 2))
        
        # Draw button circle
        pygame.draw.circle(screen, color, (x, y), size)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), size, 2)
        
        # Draw "A" text
        font = get_font(size)
        text = font.render("A", True, (255, 255, 255))
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
    
    @staticmethod
    def draw_b_button(
        screen: pygame.Surface,
        position: Tuple[int, int],
        size: int = 30,
        selected: bool = False
    ) -> None:
        """Draw B button icon.
        
        Args:
            screen: The pygame Surface to draw on.
            position: (x, y) center position.
            size: Icon size in pixels.
            selected: If True, adds pulsing glow effect.
        """
        x, y = position
        color = config.COLOR_BUTTON_B
        
        # Draw glow if selected
        if selected:
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 200.0)
            glow_intensity = 0.4 + 0.3 * pulse
            glow_radius = size * 0.6
            for layer in range(5):
                layer_radius = glow_radius - layer
                alpha = int(255 * glow_intensity * (1.0 - layer / 5))
                if alpha > 0 and layer_radius > 0:
                    glow_color = (*color, alpha)
                    glow_surf = pygame.Surface((int(layer_radius * 2) + 4, int(layer_radius * 2) + 4), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, glow_color, (int(layer_radius) + 2, int(layer_radius) + 2), int(layer_radius))
                    screen.blit(glow_surf, (x - int(layer_radius) - 2, y - int(layer_radius) - 2))
        
        # Draw button circle
        pygame.draw.circle(screen, color, (x, y), size)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), size, 2)
        
        # Draw "B" text
        font = get_font(size)
        text = font.render("B", True, (255, 255, 255))
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)


class Button:
    """Reusable button component with hover/selected states and glow effects."""
    
    def __init__(
        self,
        text: str,
        position: Tuple[int, int],
        font: pygame.font.Font,
        callback: Optional[Callable[[], None]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        padding: int = 20
    ):
        """Initialize button.
        
        Args:
            text: Button text.
            position: (x, y) center position.
            font: Font to use for text.
            callback: Optional callback function when button is clicked.
            width: Optional fixed width (auto-calculated if None).
            height: Optional fixed height (auto-calculated if None).
            padding: Padding around text.
        """
        self.text = text
        self.position = position
        self.font = font
        self.callback = callback
        self.padding = padding
        
        # Calculate size
        text_surface = font.render(text, True, (255, 255, 255))
        text_width, text_height = text_surface.get_size()
        self.width = width if width is not None else text_width + padding * 2
        self.height = height if height is not None else text_height + padding * 2
        
        # State
        self.selected = False
        self.hover = False
        self._selection_blend: Optional[float] = None  # 0.0 idle to 1.0 selected
        self._last_draw_ticks = 0
    
    def get_rect(self) -> pygame.Rect:
        """Get button rectangle.
        
        Returns:
            Button rectangle centered on position.
        """
        return pygame.Rect(
            self.position[0] - self.width // 2,
            self.position[1] - self.height // 2,
            self.width,
            self.height
        )
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside button.
        
        Args:
            point: (x, y) point to check.
            
        Returns:
            True if point is inside button.
        """
        return self.get_rect().collidepoint(point)
    
    def draw(self, screen: pygame.Surface, pulse_phase: float = 0.0) -> None:
        """Draw button.
        
        Args:
            screen: The pygame Surface to draw on.
            pulse_phase: Phase for pulsing animation (0.0 to 2*pi).
        """
        rect = self.get_rect()
        
        skins = _button_skin_cache.get((self.width, self.height))
        if skins is None:
            skins = _build_button_skins(self.width, self.height)
            _button_skin_cache[(self.width, self.height)] = skins
        idle_skin, selected_skin, glow_skin = skins
        
        # Ease towards the selected/idle look (snaps on the first draw)
        now = pygame.time.get_ticks()
        target = 1.0 if self.selected else 0.0
        if self._selection_blend is None:
            self._selection_blend = target
        else:
            step = (now - self._last_draw_ticks) / BUTTON_FADE_MS
            if self._selection_blend < target:
                self._selection_blend = min(target, self._selection_blend + step)
            else:
                self._selection_blend = max(target, self._selection_blend - step)
        self._last_draw_ticks = now
        blend = self._selection_blend
        
        # Draw idle panel, cross-fading out as the button becomes selected
        if blend < 1.0:
            idle_skin.set_alpha(int(255 * (1.0 - blend)))
            screen.blit(idle_skin, rect)
        
        # Draw pulsing glow and selected panel
        if blend > 0.0:
            pulse = 0.5 + 0.5 * math.sin(pulse_phase)
            glow_skin.set_alpha(int(255 * blend * (0.55 + 0.45 * pulse)))
            screen.blit(glow_skin, glow_skin.get_rect(center=rect.center))
            selected_skin.set_alpha(int(255 * blend))
            screen.blit(selected_skin, selected_skin.get_rect(center=rect.center))
        
        # Draw text (dimmed if not selected)
        text_color = interpolate_color(BUTTON_TEXT_IDLE, config.COLOR_TEXT, blend)
        text_surface = self.font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.position)
        screen.blit(text_surface, text_rect)


def advance_pulse(phase: float, dt: float) -> float:
    """The button glow's phase after `dt` frames (the game's unit: 1.0 is one frame at 60 fps), kept within one turn."""
    return (phase + config.BUTTON_GLOW_PULSE_SPEED * dt / 60.0) % (2 * math.pi)


def draw_buttons(screen: pygame.Surface, buttons: List[Button], selected_index: int, phase: float) -> None:
    """Draw a menu's buttons, the one at `selected_index` lit."""
    for i, button in enumerate(buttons):
        button.selected = (i == selected_index)
        button.draw(screen, phase)


def selection_text_color(blend: float) -> Tuple[int, int, int]:
    """The text colour of a selectable item: dim when idle, white when selected, easing between."""
    return interpolate_color(BUTTON_TEXT_IDLE, config.COLOR_TEXT, blend)


class ButtonSet:
    """The look every selectable menu item shares: one Button per key, so each keeps its own fade."""

    def __init__(self, font: pygame.font.Font):
        self.font = font
        self._buttons: Dict[object, Button] = {}

    def draw(self, screen: pygame.Surface, key: object, rect: pygame.Rect, selected: bool, phase: float) -> float:
        """Draw the item's frame in `rect`; returns its selection blend (0 idle .. 1 selected) for the text on it."""
        button = self._buttons.get(key)
        if button is None:
            button = self._buttons[key] = Button('', rect.center, self.font, width=rect.width, height=rect.height)
        button.position, button.width, button.height = rect.center, rect.width, rect.height
        button.selected = selected
        button.draw(screen, phase)
        return button._selection_blend or 0.0


class AnimatedBackground:
    """Animated background for menus with starfield and particles."""
    
    def __init__(self, width: int, height: int):
        """Initialize animated background.
        
        Args:
            width: Screen width.
            height: Screen height.
        """
        self.width = width
        self.height = height
        self.starfield = Starfield(width, height)
        self.particles = MenuParticleSystem(width, height)
    
    def update(self, dt: float) -> None:
        """Update background animation.
        
        Args:
            dt: Delta time (normalized to 60fps).
        """
        self.starfield.update(dt)
        self.particles.update(dt)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw background.
        
        Args:
            screen: The pygame Surface to draw on.
        """
        # Draw starfield
        self.starfield.draw(screen)
        # Draw particles
        self.particles.draw(screen)


class NeonText:
    """Text rendering with neon glow effects."""
    
    def __init__(
        self,
        text: str,
        font: pygame.font.Font,
        position: Tuple[int, int],
        color_start: Tuple[int, int, int],
        color_end: Tuple[int, int, int],
        center: bool = True
    ):
        """Initialize neon text.
        
        Args:
            text: Text to render.
            font: Font to use.
            position: (x, y) position or center point if center=True.
            color_start: Start color for gradient.
            color_end: End color for gradient.
            center: If True, position is treated as center point.
        """
        self.text = text
        self.font = font
        self.position = position
        self.color_start = color_start
        self.color_end = color_end
        self.center = center
        self.pulse_phase = 0.0
    
    def update(self, dt: float) -> None:
        """Update animation.
        
        Args:
            dt: Delta time (normalized to 60fps).
        """
        dt_seconds = dt / 60.0
        self.pulse_phase += config.NEON_GLOW_PULSE_SPEED * dt_seconds
        if self.pulse_phase >= 2 * math.pi:
            self.pulse_phase -= 2 * math.pi
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw neon text.
        
        Args:
            screen: The pygame Surface to draw on.
        """
        draw_neon_text(
            screen,
            self.text,
            self.font,
            self.position,
            self.color_start,
            self.color_end,
            config.NEON_GLOW_INTENSITY,
            self.pulse_phase,
            self.center
        )


class ConfirmationDialog:
    """Reusable confirmation dialog component with two-button layout."""
    
    def __init__(
        self,
        screen: pygame.Surface,
        title: str,
        message: str,
        confirm_label: str = "OK",
        cancel_label: str = "Cancel",
        dialog_width: int = 550,
        dialog_height: int = 280,
        button_layout: str = "side_by_side"
    ):
        """Initialize confirmation dialog.
        
        Args:
            screen: The pygame Surface to draw on.
            title: Dialog title text.
            message: Dialog message text.
            confirm_label: Text for confirm button (default: "OK").
            cancel_label: Text for cancel button (default: "Cancel").
            dialog_width: Width of dialog box.
            dialog_height: Height of dialog box.
            button_layout: "side_by_side" or "stacked" button layout.
        """
        self.screen = screen
        self.title = title
        self.message = message
        self.confirm_label = confirm_label
        self.cancel_label = cancel_label
        self.dialog_width = dialog_width
        self.dialog_height = dialog_height
        self.button_layout = button_layout
        
        self.message_font = get_font(24)
        self.title_font = get_font(config.FONT_SIZE_SUBTITLE + 8)
        self.button_font = get_font(config.FONT_SIZE_BUTTON)
        self.hint_font = get_font(config.FONT_SIZE_HINT)
        # Buttons are kept between frames so their selection fade can play
        self._buttons: Optional[Tuple[Button, Button]] = None
        self._hints: Optional[pygame.Surface] = None
    
    def dialog_rect(self) -> pygame.Rect:
        """Get the dialog's rectangle, centered on the screen."""
        rect = pygame.Rect(0, 0, self.dialog_width, self.dialog_height)
        rect.center = (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2)
        return rect
    
    def button_rects(self) -> Tuple[pygame.Rect, pygame.Rect]:
        """Get the confirm and cancel button rectangles."""
        rect = self.dialog_rect()
        if self.button_layout == "side_by_side":
            width, height = 200, 54
            y = rect.top + 172
            centers = ((rect.centerx - 120, y), (rect.centerx + 120, y))
        else:
            width, height = 400, 54
            y = rect.top + 172
            centers = ((rect.centerx, y), (rect.centerx, y + 74))
        rects = []
        for center in centers:
            button_rect = pygame.Rect(0, 0, width, height)
            button_rect.center = center
            rects.append(button_rect)
        return rects[0], rects[1]
    
    def hints_rect(self) -> pygame.Rect:
        """Get the rectangle of the key hints along the bottom of the dialog."""
        if self._hints is None:
            self._hints = render_hint_row(self.hint_font, [("ENTER / A", "Select"), ("ESC / B", "Cancel")])
        return self._hints.get_rect(midbottom=(self.dialog_rect().centerx, self.dialog_rect().bottom - 26))
    
    def _get_buttons(self) -> Tuple[Button, Button]:
        """Get the confirm and cancel buttons, creating them on first use."""
        confirm_rect, cancel_rect = self.button_rects()
        if self._buttons is None:
            self._buttons = (
                Button(self.confirm_label, confirm_rect.center, self.button_font, width=confirm_rect.width, height=confirm_rect.height),
                Button(self.cancel_label, cancel_rect.center, self.button_font, width=cancel_rect.width, height=cancel_rect.height)
            )
        confirm_button, cancel_button = self._buttons
        confirm_button.position = confirm_rect.center
        cancel_button.position = cancel_rect.center
        return confirm_button, cancel_button
    
    def draw(
        self,
        menu_pulse_phase: float,
        selection_index: int = 0
    ) -> None:
        """Draw confirmation dialog.
        
        Args:
            menu_pulse_phase: Current pulse phase for button glow animation.
            selection_index: Which button is selected (0 = confirm, 1 = cancel).
        """
        # Draw semi-transparent overlay
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        dialog_rect = self.dialog_rect()
        
        # Rounded panel over a soft accent glow
        glow = build_panel_glow(dialog_rect.size)
        self.screen.blit(glow, glow.get_rect(center=dialog_rect.center))
        self.screen.blit(build_panel(dialog_rect.size), dialog_rect)
        
        # Draw title
        title_surface = self.title_font.render(self.title, True, config.COLOR_TEXT)
        self.screen.blit(title_surface, title_surface.get_rect(center=(dialog_rect.centerx, dialog_rect.top + 56)))
        
        # Draw message
        message_surface = self.message_font.render(self.message, True, TEXT_DIM)
        self.screen.blit(message_surface, message_surface.get_rect(center=(dialog_rect.centerx, dialog_rect.top + 104)))
        
        # Draw buttons
        confirm_button, cancel_button = self._get_buttons()
        draw_buttons(self.screen, [confirm_button, cancel_button], selection_index, menu_pulse_phase)
        
        # Key hints share one row at the bottom instead of crowding each button
        hints_rect = self.hints_rect()
        self.screen.blit(self._hints, hints_rect)
