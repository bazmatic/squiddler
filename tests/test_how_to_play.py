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
        "GOAL AND CONTROLS", "POWER", "CRYSTALS AND WEAPONS", "STARS"]


def test_power_page_quotes_the_settings(monkeypatch):
    monkeypatch.setattr(config, 'ENEMY_DESTRUCTION_BONUS', 37)
    monkeypatch.setattr(config, 'POWERUP_CRYSTAL_BONUS', 41)
    text = page_text(build_pages(False)[1])
    assert "+37" in text and "+41" in text
    assert "pulse" not in text and "tick" not in text      # the warning signals are left to be discovered


def test_crystal_page_describes_the_weapon_crystals_players_meet():
    from entities.powerup_crystal import PowerupCrystal
    from weapons.catalog import CATALOG
    text = page_text(build_pages(False)[2])
    assert f"{PowerupCrystal.LIFETIME_FRAMES / config.FPS:g} seconds" in text
    assert f"{config.SETTINGS.powerups.shieldDurationSeconds:g} seconds" in text
    assert "basic gun never runs out" in text
    for weapon_id in ('twin_blaster', 'scattergun', 'railgun', 'seeker', 'inflator', 'grenade_launcher'):
        assert f"{CATALOG[weapon_id].name} {CATALOG[weapon_id].capacity}" in text
    assert "firepower" not in text      # the old gun-upgrade rules are not what a crystal does in play


def test_the_last_page_is_about_stars_and_does_not_describe_enemies():
    text = page_text(build_pages(False)[3]).lower()
    assert "star" in text and "par" in text
    for name in ("egg", "urchin", "crab", "jelly", "flocker", "wallcrawler", "boss"):
        assert name not in text


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


class TestGettingIn:
    def make_game(self):
        from unittest.mock import Mock
        from game_handlers.state_handlers import StateHandlerRegistry
        from rendering.how_to_play import HowToPlayMenu
        from rendering.main_menu import MainMenu
        game = Mock()
        game.state = config.STATE_MENU
        game.main_menu = MainMenu(pygame.Surface(SIZE))
        game.how_to_play_menu = HowToPlayMenu(pygame.Surface(SIZE))
        game.input_handler.get_controllers.return_value = []
        return game, StateHandlerRegistry()

    def key(self, key):
        return pygame.event.Event(pygame.KEYDOWN, key=key, unicode="")

    def press(self, game, registry, key):
        registry.get_handler(game.state).handle_keyboard(self.key(key), game)

    def open(self, screen):
        game, registry = self.make_game()
        options = game.main_menu.menu_options
        game.main_menu.menu_selected_index = options.index("HOW TO PLAY")
        self.press(game, registry, pygame.K_RETURN)
        return game, registry

    def test_entry_sits_after_levels_and_opens_the_screen(self, screen):
        game, registry = self.open(screen)
        options = game.main_menu.menu_options
        assert options[options.index("LEVELS") + 1] == "HOW TO PLAY"
        assert game.state == config.STATE_HOW_TO_PLAY
        assert game.how_to_play_menu.page_index == 0

    def test_a_controller_confirm_opens_it_too(self, screen):
        game, registry = self.make_game()
        game.main_menu.menu_selected_index = game.main_menu.menu_options.index("HOW TO PLAY")
        game.input_handler.is_controller_menu_confirm_pressed.return_value = True
        registry.get_handler(game.state).handle_controller(pygame.event.Event(pygame.JOYBUTTONDOWN, button=0), game)
        assert game.state == config.STATE_HOW_TO_PLAY

    def test_arrows_turn_pages_and_escape_goes_back(self, screen):
        game, registry = self.open(screen)
        self.press(game, registry, pygame.K_RIGHT)
        assert game.how_to_play_menu.page_index == 1
        self.press(game, registry, pygame.K_LEFT)
        self.press(game, registry, pygame.K_LEFT)
        assert game.how_to_play_menu.page_index == 0
        self.press(game, registry, pygame.K_ESCAPE)
        assert game.state == config.STATE_MENU

    def test_the_last_page_stops_turning(self, screen):
        game, registry = self.open(screen)
        for _ in range(game.how_to_play_menu.page_count + 2):
            self.press(game, registry, pygame.K_RIGHT)
        assert game.how_to_play_menu.page_index == game.how_to_play_menu.page_count - 1

    def test_the_controller_turns_pages_and_cancel_goes_back(self, screen):
        game, registry = self.open(screen)
        handler = registry.get_handler(game.state)
        handler.handle_controller(pygame.event.Event(pygame.JOYHATMOTION, hat=0, value=(1, 0)), game)
        assert game.how_to_play_menu.page_index == 1
        game.input_handler.is_controller_menu_cancel_pressed.return_value = True
        handler.handle_controller(pygame.event.Event(pygame.JOYBUTTONDOWN, button=1), game)
        assert game.state == config.STATE_MENU


    def test_a_keyboard_player_sees_keyboard_controls_even_with_a_pad_plugged_in(self, screen):
        game, registry = self.make_game()
        game.input_handler.get_controllers.return_value = [object()]
        game.main_menu.menu_selected_index = game.main_menu.menu_options.index("HOW TO PLAY")
        self.press(game, registry, pygame.K_RETURN)
        assert game.how_to_play_menu.controller is False

    def test_a_controller_player_sees_controller_controls(self, screen):
        game, registry = self.make_game()
        game.main_menu.menu_selected_index = game.main_menu.menu_options.index("HOW TO PLAY")
        game.input_handler.is_controller_menu_confirm_pressed.return_value = True
        registry.get_handler(game.state).handle_controller(pygame.event.Event(pygame.JOYBUTTONDOWN, button=0), game)
        assert game.how_to_play_menu.controller is True
