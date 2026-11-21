"""The HOW TO PLAY screen: its copy, the paged screen and how the player gets in and out."""
import pygame
import pytest

import config
from rendering.how_to_play_pages import build_pages
from rendering.settings_menu import CONTROL_GROUPS


def page_text(page):
    return ' '.join(page.lines)


def test_there_are_four_pages_in_order():
    assert [page.title for page in build_pages(False)] == [
        "GOAL AND CONTROLS", "POWER", "CRYSTALS AND WEAPONS", "STARS AND ENEMIES"]


def test_power_page_quotes_the_settings(monkeypatch):
    monkeypatch.setattr(config, 'ENEMY_DESTRUCTION_BONUS', 37)
    monkeypatch.setattr(config, 'POWERUP_CRYSTAL_BONUS', 41)
    text = page_text(build_pages(False)[1])
    assert "+37" in text and "+41" in text
    assert str(config.SETTINGS.game.criticalWarningThreshold) in text


def test_crystal_page_quotes_the_settings(monkeypatch):
    monkeypatch.setattr(config, 'POWERUP_DURATION_SECONDS', 17)
    monkeypatch.setattr(config, 'POWERUP_AMMO_REFILL', 23)
    text = page_text(build_pages(False)[2])
    assert "17" in text and "23" in text
    assert f"{config.SETTINGS.powerups.shieldDurationSeconds:g}" in text


def test_controls_follow_the_device():
    groups = {name: [f"{label}: {action}" for label, action in rows] for name, rows in CONTROL_GROUPS}
    keyboard, controller = groups["KEYBOARD"], groups["CONTROLLER"]
    on_keys, on_pad = build_pages(False)[0].lines, build_pages(True)[0].lines
    assert all(line in on_keys for line in keyboard) and not any(line in on_keys for line in controller)
    assert all(line in on_pad for line in controller) and not any(line in on_pad for line in keyboard)


SIZE = (1352, 878)


@pytest.fixture
def screen(monkeypatch):
    pygame.init()
    pygame.display.set_mode((320, 240))
    monkeypatch.setattr(config, 'SCREEN_WIDTH', SIZE[0])
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', SIZE[1])
    surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    surface.fill((*config.COLOR_BACKGROUND, 255))
    return surface


def is_opaque(surface):
    return pygame.mask.from_surface(surface, 250).count() == surface.get_width() * surface.get_height()


@pytest.mark.parametrize('controller', [False, True])
def test_every_page_draws_and_leaves_the_screen_opaque(screen, controller):
    from rendering.how_to_play import HowToPlayMenu
    menu = HowToPlayMenu(screen)
    menu.refresh(controller)
    for _ in range(menu.page_count):
        menu.update(1.0)
        menu.draw()
        assert is_opaque(screen)
        menu.turn(1)


@pytest.mark.parametrize('size', [(1352, 878), (1280, 720), (1920, 1080)])
@pytest.mark.parametrize('controller', [False, True])
def test_page_lines_stay_on_screen_and_apart(monkeypatch, size, controller):
    from rendering.how_to_play import HowToPlayMenu
    pygame.init()
    pygame.display.set_mode((320, 240))
    monkeypatch.setattr(config, 'SCREEN_WIDTH', size[0])
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', size[1])
    menu = HowToPlayMenu(pygame.Surface(size))
    menu.refresh(controller)
    bounds = pygame.Rect(0, 0, *size)
    for _ in range(menu.page_count):
        rects = menu.line_rects()
        assert rects and all(bounds.contains(rect) for rect in rects)
        assert not any(a.colliderect(b) for i, a in enumerate(rects) for b in rects[i + 1:])
        menu.turn(1)


def test_turn_clamps_at_both_ends(screen):
    from rendering.how_to_play import HowToPlayMenu
    menu = HowToPlayMenu(screen)
    menu.turn(-1)
    assert menu.page_index == 0
    for _ in range(menu.page_count + 3):
        menu.turn(1)
    assert menu.page_index == menu.page_count - 1


def test_refresh_returns_to_the_first_page(screen):
    from rendering.how_to_play import HowToPlayMenu
    menu = HowToPlayMenu(screen)
    menu.turn(1)
    menu.refresh(False)
    assert menu.page_index == 0
