"""The profile, level complete and quit screens share the main menu's look and never overlap themselves."""
import pygame
import pytest

import config
from profiles import ProfileManager
from rendering import menu_components
from rendering.level_complete_menu import LevelCompleteMenu
from rendering.menu_components import ConfirmationDialog
from rendering.profile_selection_menu import ProfileSelectionMenu
from rendering.ui_elements import AnimatedStarRating

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
    # Antialiased lines round alpha down a step or two, which is not a hole
    return pygame.mask.from_surface(surface, 250).count() == surface.get_width() * surface.get_height()


def profile_menu(screen, tmp_path, names):
    manager = ProfileManager(tmp_path / "profiles.json")
    for name in names:
        if manager.get_profile(name) is None:
            manager.create_profile(name)
    return ProfileSelectionMenu(screen, manager)


class TestSharedPieces:
    def test_backdrop_is_opaque_and_tinted(self, screen):
        backdrop = menu_components.build_menu_backdrop(*SIZE)
        assert backdrop.get_size() == SIZE
        screen.blit(backdrop, (0, 0))
        assert is_opaque(screen)
        # The nebula lifts the middle of the screen above the flat background colour
        assert sum(backdrop.get_at((SIZE[0] // 2, int(SIZE[1] * 0.42)))[:3]) > sum(config.COLOR_BACKGROUND) + 20

    def test_hint_row_pairs_each_key_with_its_action(self, screen):
        font = pygame.font.Font(None, 24)
        one = menu_components.render_hint_row(font, [("ENTER", "Select")])
        two = menu_components.render_hint_row(font, [("ENTER", "Select"), ("ESC / B", "Back")])
        assert two.get_width() > one.get_width()
        assert two.get_height() == one.get_height()

    def test_panel_has_rounded_corners(self, screen):
        panel = menu_components.build_panel((400, 200))
        assert panel.get_size() == (400, 200)
        assert panel.get_at((0, 0))[3] == 0
        assert panel.get_at((399, 199))[3] == 0
        assert panel.get_at((200, 0))[3] > 200
        assert panel.get_at((200, 100))[3] > 200


class TestProfileSelection:
    def test_every_option_has_a_row_on_screen(self, screen, tmp_path):
        menu = profile_menu(screen, tmp_path, ["Ada", "Grace"])
        rows = [menu.row_rect(i) for i in range(len(menu.options))]
        assert all(row is not None for row in rows)
        assert all(screen.get_rect().contains(row) for row in rows)
        assert all(a.bottom <= b.top for a, b in zip(rows, rows[1:]))

    def test_long_lists_scroll_to_keep_the_selection_visible(self, screen, tmp_path):
        menu = profile_menu(screen, tmp_path, [f"Pilot {i}" for i in range(30)])
        hints_top = menu.hints_rect().top
        for index in range(len(menu.options)):
            menu.selected_index = index
            row = menu.row_rect(index)
            assert row is not None
            assert row.top >= menu.title_rect().bottom
            assert row.bottom <= hints_top
        menu.selected_index = 0
        assert menu.row_rect(len(menu.options) - 1) is None

    def test_selected_row_stands_out(self, screen, tmp_path):
        menu = profile_menu(screen, tmp_path, ["Ada", "Grace"])
        menu.selected_index = 1
        menu.draw()
        selected, idle = menu.row_rect(1), menu.row_rect(0)

        def border_brightness(rect):
            # The selected skin is drawn slightly larger, so look either side of the edge
            return max(sum(screen.get_at((rect.centerx, rect.top + dy))[:3]) for dy in range(-5, 5))

        assert border_brightness(selected) > border_brightness(idle) + 100

    def test_draws_in_every_state_and_leaves_the_screen_opaque(self, screen, tmp_path):
        menu = profile_menu(screen, tmp_path, ["Ada"])
        menu.draw()
        menu.start_creating_profile()
        for char in "Katherine":
            menu.append_character(char)
        menu.update(1.0)
        menu.draw()
        menu.cancel_creating_profile()
        menu.draw()
        assert is_opaque(screen)

    def test_name_being_typed_is_shown_in_the_create_row(self, screen, tmp_path):
        menu = profile_menu(screen, tmp_path, ["Ada"])
        menu.selected_index = len(menu.options) - 1
        menu.start_creating_profile()
        menu.draw()
        empty = screen.copy()
        for char in "WWWWWWWW":
            menu.append_character(char)
        menu.draw()
        row = menu.row_rect(menu.selected_index)
        changed = [
            (x, y) for x in range(row.left, row.right) for y in range(row.top, row.bottom)
            if screen.get_at((x, y)) != empty.get_at((x, y))
        ]
        assert len(changed) > 200


class TestLevelComplete:
    @pytest.mark.parametrize('succeeded', [True, False])
    def test_nothing_overlaps(self, screen, succeeded):
        menu = LevelCompleteMenu(screen)
        layout = menu.layout(succeeded)
        order = ['title', 'level_number', 'stars', 'stats', 'buttons', 'hints']
        rects = [layout[name] for name in order if name in layout]
        assert len(rects) >= 5
        for above, below in zip(rects, rects[1:]):
            assert above.bottom <= below.top
        assert all(screen.get_rect().contains(rect) for rect in rects)

    def test_time_is_no_longer_printed_over_the_banner(self, screen):
        menu = LevelCompleteMenu(screen)
        layout = menu.layout(True)
        assert not layout['stats'].colliderect(layout['title'])

    def test_stars_are_moved_into_their_slot(self, screen):
        menu = LevelCompleteMenu(screen)
        stars = AnimatedStarRating((True, True, False, False, False), 0, 0)
        menu.draw(3, True, 42.3, {'final_score': 55, 'total_score': 1234}, stars, False, lambda: None)
        slot = menu.layout(True)['stars']
        assert stars.y == slot.top + stars.star_size // 2
        assert stars.x + 2 * stars.star_spacing == slot.centerx

    def test_the_goals_are_listed_under_the_stars(self, screen):
        menu = LevelCompleteMenu(screen)
        stars = AnimatedStarRating((True, True, False, True, False), 0, 0)
        lines = ('NO HITS', '9/14 KILLED', '5 LEFT', '2/3 CRYSTALS', '1:42 / PAR 1:43')
        menu.draw(3, True, 42.3, {'final_score': 55, 'total_score': 1234}, stars, False, lambda: None,
                  star_lines=lines)
        assert len(menu.star_line_rects) == 5
        for rect in menu.star_line_rects:
            assert rect.top > stars.y + stars.star_size // 2
            text = pygame.mask.from_threshold(screen.subsurface(rect), (*config.COLOR_BACKGROUND, 255),
                                              (40, 40, 40, 255))
            text.invert()
            assert text.count() > 20

    @pytest.mark.parametrize('size', [(1352, 878), (1280, 720)])
    def test_the_star_lines_fit_their_slots(self, monkeypatch, size):
        pygame.init()
        pygame.display.set_mode((320, 240))
        monkeypatch.setattr(config, 'SCREEN_WIDTH', size[0])
        monkeypatch.setattr(config, 'SCREEN_HEIGHT', size[1])
        screen = pygame.Surface(size, pygame.SRCALPHA)
        menu = LevelCompleteMenu(screen)
        stars = AnimatedStarRating((True,) * 5, 0, 0)
        lines = ('NO HITS', '120/120 KILLED', 'NONE LEFT', '12/12 CRYSTALS', '12:05 / PAR 9:40')
        menu.draw(30, True, 725.0, {'final_score': 55, 'total_score': 123456}, stars, False, lambda: None,
                  star_lines=lines)
        layout = menu.layout(True)
        rects = menu.star_line_rects
        for left, right in zip(rects, rects[1:]):
            assert left.right <= right.left
        assert all(layout['stars'].contains(rect) for rect in rects)
        assert screen.get_rect().contains(layout['stars'])
        assert layout['stars'].bottom <= layout['stats'].top
        assert layout['stats'].bottom <= layout['buttons'].top

    @pytest.mark.parametrize('succeeded', [True, False])
    def test_draws_and_leaves_the_screen_opaque(self, screen, succeeded):
        menu = LevelCompleteMenu(screen)
        stars = AnimatedStarRating((True, True, False, False, False), 0, 0) if succeeded else None
        menu.update(1.0)
        menu.draw(3, succeeded, 42.3, {'final_score': 55, 'total_score': 1234}, stars, False, lambda: None)
        assert is_opaque(screen)

    def test_failed_banner_is_flanked_by_the_games_squids(self, screen):
        menu = LevelCompleteMenu(screen)
        title = menu.layout(False)['title']
        left, right = menu.failed_squids
        assert title.left < left.x < title.centerx < right.x < title.right
        assert title.top < left.y < title.centery

        def squid_pixels(succeeded):
            menu.draw(3, succeeded, 42.3, {'final_score': 55, 'total_score': 1234}, None, False, lambda: None)
            return pygame.mask.from_threshold(screen, (*config.REPLAY_ENEMY_COLOR, 255), (1, 1, 1, 255)).count()

        assert squid_pixels(False) > 200
        assert squid_pixels(True) == 0

    def test_last_enemy_portraits_play_once_and_restart_on_next_clear(self, screen):
        import creatures
        enemy = creatures.build('patrol', (4000, 3000))
        enemy.die()
        menu = LevelCompleteMenu(screen)
        menu.set_options(True, enemy)
        assert len(menu.complete_portraits) == 2
        assert all(p.enemy.type == "patrol" for p in menu.complete_portraits)
        menu.update(60.0)
        assert all(p.enemy.death_progress == pytest.approx(0.5) for p in menu.complete_portraits)
        menu.update(120.0)
        menu.draw(3, True, 42.3, {}, None, False, lambda: None)
        assert all(not p.enemy.is_dying for p in menu.complete_portraits)
        assert is_opaque(screen)
        menu.set_options(False)
        assert menu.complete_portraits == []
        menu.set_options(True, enemy)
        assert all(p.enemy.death_progress == 0.0 for p in menu.complete_portraits)
        menu.set_options(True)
        assert menu.complete_portraits == []

    def test_failed_squids_hover_and_trail_their_tentacles(self, screen):
        menu = LevelCompleteMenu(screen)
        squid = menu.failed_squids[0]
        before = (squid.y, squid.angle, list(squid.body.tentacles[0]))
        for _ in range(30):
            menu.update(1.0)
        assert (squid.y, squid.angle, list(squid.body.tentacles[0])) != before
        assert all(len(chain) > 1 for chain in squid.body.tentacles)


class TestStarRating:
    def test_unearned_stars_show_as_empty_outlines(self, screen):
        stars = AnimatedStarRating((True, False, False, False, False), 200, 200, star_size=40)
        assert stars.num_stars == 1
        stars.draw(screen)
        last = pygame.Rect(0, 0, 40, 40)
        last.center = (200 + 4 * stars.star_spacing, 200)
        outline = pygame.mask.from_threshold(
            screen.subsurface(last), (*config.COLOR_BACKGROUND, 255), (12, 12, 12, 255)
        )
        outline.invert()
        assert 20 < outline.count() < 40 * 40 * 0.5


class TestConfirmationDialog:
    SIZES = {'side_by_side': (560, 290), 'stacked': (620, 370)}

    def make(self, screen, layout):
        width, height = self.SIZES[layout]
        return ConfirmationDialog(
            screen, "QUIT LEVEL?", "Progress will be lost.",
            dialog_width=width, dialog_height=height, button_layout=layout
        )

    @pytest.mark.parametrize('layout', ['side_by_side', 'stacked'])
    def test_dialog_is_a_rounded_panel(self, screen, layout):
        dialog = self.make(screen, layout)
        dialog.draw(0.0, 0)
        rect = dialog.dialog_rect()
        border = sum(screen.get_at((rect.centerx, rect.top + 1))[:3])
        # The corner is cut away, so only the soft glow behind the panel shows there
        assert sum(screen.get_at(rect.topleft)[:3]) < border * 0.5
        assert sum(screen.get_at((rect.right - 1, rect.bottom - 1))[:3]) < border * 0.5
        assert is_opaque(screen)

    @pytest.mark.parametrize('layout', ['side_by_side', 'stacked'])
    def test_buttons_and_hints_fit_inside_without_overlapping(self, screen, layout):
        dialog = self.make(screen, layout)
        dialog.draw(0.0, 1)
        rect = dialog.dialog_rect()
        confirm, cancel = dialog.button_rects()
        hints = dialog.hints_rect()
        assert rect.contains(confirm) and rect.contains(cancel) and rect.contains(hints)
        assert not confirm.colliderect(cancel)
        assert max(confirm.bottom, cancel.bottom) <= hints.top


class TestSettingsScreen:
    def make_game(self, tmp_path):
        from unittest.mock import Mock
        from game_handlers.state_handlers import StateHandlerRegistry
        from rendering.main_menu import MainMenu
        from rendering.settings_menu import SettingsMenu
        game = Mock()
        game.state = config.STATE_MENU
        game.main_menu = MainMenu(pygame.Surface(SIZE))
        game.settings_menu = SettingsMenu(pygame.Surface(SIZE), self.settings(tmp_path))
        return game, StateHandlerRegistry()

    def settings(self, tmp_path):
        from user_settings import UserSettings
        return UserSettings(tmp_path / 'player_settings.json')  # Never the player's real settings file

    def key(self, key):
        return pygame.event.Event(pygame.KEYDOWN, key=key, unicode="")

    def test_main_menu_no_longer_carries_the_key_mappings(self, screen, tmp_path):
        game, _ = self.make_game(tmp_path)
        assert game.main_menu.menu_options == ["START GAME", "LEVELS", "FUN", "SELECT PROFILE", "MULTIPLAYER", "SETTINGS", "QUIT"]
        assert not hasattr(game.main_menu, 'controls_surface')

    def test_settings_option_opens_the_screen_and_escape_returns(self, screen, tmp_path):
        game, registry = self.make_game(tmp_path)
        game.main_menu.menu_selected_index = game.main_menu.menu_options.index("SETTINGS")
        registry.get_handler(game.state).handle_keyboard(self.key(pygame.K_RETURN), game)
        assert game.state == config.STATE_SETTINGS
        registry.get_handler(game.state).handle_keyboard(self.key(pygame.K_ESCAPE), game)
        assert game.state == config.STATE_MENU

    def test_sound_toggles_mutes_and_is_remembered(self, screen, tmp_path):
        from sounds import sound_manager
        game, registry = self.make_game(tmp_path)
        game.state = config.STATE_SETTINGS
        handler = registry.get_handler(game.state)
        try:
            assert game.settings_menu.settings.sound and not sound_manager._muted
            handler.handle_keyboard(self.key(pygame.K_RETURN), game)
            assert game.state == config.STATE_SETTINGS
            assert not game.settings_menu.settings.sound and sound_manager._muted
            assert not self.settings(tmp_path).sound
            handler.handle_keyboard(self.key(pygame.K_RIGHT), game)
            assert self.settings(tmp_path).sound and not sound_manager._muted
        finally:
            sound_manager.set_muted(False)

    def test_settings_sit_above_one_panel_per_device_side_by_side(self, screen, tmp_path):
        from rendering.settings_menu import CONTROL_GROUPS, SettingsMenu
        menu = SettingsMenu(screen, self.settings(tmp_path))
        layout = menu.layout()
        panels, settings = layout['panels'], layout['settings']
        assert len(panels) == len(CONTROL_GROUPS) == 2
        assert panels[0].right < panels[1].left
        assert layout['title'].bottom <= settings[0].top
        assert settings[-1].bottom < panels[0].top
        assert max(panel.bottom for panel in panels) <= layout['hints'].top
        assert all(screen.get_rect().contains(rect) for rect in panels + settings)

    def test_every_mapping_fits_inside_its_panel(self, screen, tmp_path):
        from rendering.settings_menu import CONTROL_GROUPS, SettingsMenu
        menu = SettingsMenu(screen, self.settings(tmp_path))
        pad = menu._scaled(36)
        for rect, (_, mappings) in zip(menu.layout()['panels'], CONTROL_GROUPS):
            for key_text, action_text in mappings:
                needed = menu.row_font.size(key_text)[0] + menu.row_font.size(action_text)[0] + menu._scaled(28) + 40
                assert needed <= rect.width - pad * 2

    def test_draws_and_leaves_the_screen_opaque(self, screen, tmp_path):
        from rendering.settings_menu import SettingsMenu
        menu = SettingsMenu(screen, self.settings(tmp_path))
        menu.update(1.0)
        menu.draw()
        assert is_opaque(screen)


@pytest.mark.parametrize('size', [(1352, 878), (1280, 720), (1920, 1080)])
@pytest.mark.parametrize('editor', [False, True])
def test_the_main_menu_buttons_fit_between_the_title_and_the_profile_badge(monkeypatch, size, editor):
    from rendering.main_menu import MainMenu
    pygame.init()
    pygame.display.set_mode((320, 240))
    monkeypatch.setattr(config, 'SCREEN_WIDTH', size[0])
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', size[1])
    if editor:
        monkeypatch.setenv('LEVEL_EDITOR', '1')
    else:
        monkeypatch.delenv('LEVEL_EDITOR', raising=False)
    menu = MainMenu(pygame.Surface(size))
    menu.set_profile_info('Player1', 30)
    buttons = [button.get_rect() for button in menu.menu_buttons]
    assert len(buttons) == (8 if editor else 7)
    assert buttons[-1].bottom < menu.profile_badge_rect().top
    assert buttons[0].top > menu.menu_title_rect.bottom


def test_seven_buttons_sit_a_little_higher_and_close_up_only_as_far_as_they_must(monkeypatch):
    from rendering.main_menu import MainMenu
    pygame.init()
    pygame.display.set_mode((320, 240))
    monkeypatch.setattr(config, 'SCREEN_WIDTH', SIZE[0])
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', SIZE[1])
    monkeypatch.delenv('LEVEL_EDITOR', raising=False)
    menu = MainMenu(pygame.Surface(SIZE))
    ys = [button.position[1] for button in menu.menu_buttons]
    assert ys[0] <= int(SIZE[1] * 0.42)                                    # moved up to make room, never down
    assert menu.menu_buttons[0].height < ys[1] - ys[0] <= menu._scaled(80)  # packed closer, not touching
