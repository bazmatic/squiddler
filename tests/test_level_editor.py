# tests/test_level_editor.py
import json
import os
from types import SimpleNamespace

import pygame
import pytest

import config
import level_config
from editor import files
from editor.model import PALETTE
from level_layout import parse_layout
from rendering.level_editor import LevelEditor
from tests.test_hunter_integration import game  # noqa: F401 - pytest fixture
from tests.test_layout_play import real_level_config


def key(k, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode='')


@pytest.fixture
def editor(game, monkeypatch, tmp_path):  # noqa: F811
    pygame.init()
    screen = pygame.display.set_mode((1352, 878))
    game.state = config.STATE_EDITOR
    real_level_config(monkeypatch)
    monkeypatch.setattr(files, 'level_path', lambda level: str(tmp_path / f'{level}.json'))
    monkeypatch.setattr(level_config, 'load_level_config', lambda level: (
        json.loads((tmp_path / f'{level}.json').read_text()) if (tmp_path / f'{level}.json').exists() else
        ({'seed': 5, 'maze': {'complexity': 'normal', 'grid_size': 12}, 'enemies': {'static': 2}}
         if level == 3 else None)))
    ed = LevelEditor(game, screen)
    ed.tmp = tmp_path
    return ed


def open_level(ed, level=3):
    ed.open_picker()
    ed.picker_index = level - 1
    ed.handle_key(key(pygame.K_RETURN))


def test_the_editor_env_var_gates_the_feature(monkeypatch):
    monkeypatch.delenv('LEVEL_EDITOR', raising=False)
    assert config.editor_enabled() is False
    monkeypatch.setenv('LEVEL_EDITOR', '')
    assert config.editor_enabled() is False
    monkeypatch.setenv('LEVEL_EDITOR', '1')
    assert config.editor_enabled() is True


def test_the_main_menu_has_an_editor_item_only_with_the_variable(monkeypatch):
    from rendering.main_menu import MainMenu
    pygame.init()
    screen = pygame.display.set_mode((1352, 878))
    monkeypatch.delenv('LEVEL_EDITOR', raising=False)
    assert 'EDITOR' not in MainMenu(screen).menu_options
    monkeypatch.setenv('LEVEL_EDITOR', '1')
    options = MainMenu(screen).menu_options
    assert options.index('EDITOR') == options.index('QUIT') - 1


def test_the_picker_chooses_a_level_with_the_arrows_and_opens_it(editor):
    editor.open_picker()
    assert editor.mode == 'picker' and editor.picker_index == 0
    editor.handle_key(key(pygame.K_RIGHT))
    editor.handle_key(key(pygame.K_DOWN))
    assert editor.picker_index == 1 + editor.PICKER_COLUMNS
    editor.picker_index = 2
    editor.handle_key(key(pygame.K_RETURN))
    assert editor.mode == 'edit' and editor.model.level == 3 and editor.model.grid_size == 12


def test_escape_leaves_the_picker_for_the_main_menu(editor):
    editor.open_picker()
    editor.handle_key(key(pygame.K_ESCAPE))
    assert editor.game.state == config.STATE_MENU


def test_arrows_tab_space_and_delete_edit_the_level(editor):
    open_level(editor)
    model = editor.model
    model.cursor = (1, model.cursor[1])                         # room to move right on any generated maze
    start = model.cursor
    editor.handle_key(key(pygame.K_RIGHT))
    assert model.cursor == (start[0] + 1, start[1])
    editor.handle_key(key(pygame.K_TAB))
    assert model.selected == PALETTE[1]
    editor.handle_key(key(pygame.K_TAB, pygame.KMOD_SHIFT))
    assert model.selected == 'wall'
    editor.handle_key(key(pygame.K_SPACE))
    assert model.contents(model.cursor) == 'wall' and model.dirty
    editor.handle_key(key(pygame.K_BACKSPACE))
    assert model.contents(model.cursor) is None


def test_saving_writes_the_layout_and_clears_the_dirty_mark(editor):
    open_level(editor)
    editor.handle_key(key(pygame.K_SPACE))                       # something to save (a wall, if it can)
    editor.model.dirty = True
    editor.handle_key(key(pygame.K_s))
    data = json.loads((editor.tmp / '3.json').read_text())
    assert parse_layout(data['layout'], 12) == editor.model.to_layout()
    assert editor.model.dirty is False and 'saved' in editor.flash.lower()


def test_a_save_that_cannot_be_written_says_so_and_keeps_the_edits(editor, monkeypatch):
    open_level(editor)
    editor.model.dirty = True
    monkeypatch.setattr(files, 'level_path', lambda level: str(editor.tmp / 'no' / 'dir' / '3.json'))
    editor.handle_key(key(pygame.K_s))
    assert 'could not' in editor.flash.lower() and editor.model.dirty is True


def test_leaving_with_unsaved_edits_asks_first(editor):
    open_level(editor)
    editor.model.dirty = True
    editor.handle_key(key(pygame.K_ESCAPE))
    assert editor.mode == 'edit' and editor.confirm is not None
    editor.handle_key(key(pygame.K_n))
    assert editor.confirm is None and editor.mode == 'edit'
    editor.handle_key(key(pygame.K_ESCAPE))
    editor.handle_key(key(pygame.K_y))
    assert editor.mode == 'picker'


def test_leaving_a_clean_level_does_not_ask(editor):
    open_level(editor)
    editor.handle_key(key(pygame.K_ESCAPE))
    assert editor.mode == 'picker' and editor.confirm is None


def test_x_asks_then_reverts_to_the_generated_level(editor):
    open_level(editor)
    editor.handle_key(key(pygame.K_s))                           # a layout now exists on disk
    assert (editor.tmp / '3.json').exists()
    editor.handle_key(key(pygame.K_x))
    assert editor.confirm is not None
    editor.handle_key(key(pygame.K_y))
    assert not (editor.tmp / '3.json').exists()                  # the editor made that file, so it is deleted
    assert 'revert' in editor.flash.lower() and editor.model.dirty is False


def test_x_with_no_layout_says_there_is_nothing_to_revert(editor):
    open_level(editor)
    editor.handle_key(key(pygame.K_x))
    assert editor.confirm is None and 'nothing' in editor.flash.lower()


def test_every_screen_and_palette_item_draws(editor):
    editor.open_picker()
    editor.update(1.0)
    editor.draw()
    open_level(editor)
    model = editor.model
    model.walls.clear()                                           # open floor, so a gate's plank fits anywhere
    free = [(col, row) for row in range(1, model.grid_size - 1) for col in range(1, model.grid_size - 1)
            if model.contents((col, row)) is None]
    for index, item in enumerate(PALETTE):
        model.cursor, model.palette_index = free[index * (len(free)//len(PALETTE))], index
        assert model.place(), item                                # every item lands, so every sprite is drawn
    editor.update(1.0)
    editor.draw()                                                 # once with them all, not once per item (~17 s)
    editor.model.dirty = True
    editor.handle_key(key(pygame.K_ESCAPE))
    editor.draw()                                                # with the confirmation showing


def test_holding_an_arrow_repeats_and_releasing_stops(editor, monkeypatch):
    open_level(editor)
    editor.model.cursor = (1, editor.model.cursor[1])
    start = editor.model.cursor
    editor.handle_key(key(pygame.K_RIGHT))
    pressed = {pygame.K_RIGHT: True}
    monkeypatch.setattr(pygame.key, 'get_pressed', lambda: type('K', (), {'__getitem__': lambda s, k: pressed.get(k, False)})())
    for _ in range(40):
        editor.update(0.05 * config.FPS)
    assert editor.model.cursor[0] > start[0] + 2
    pressed.clear()
    stopped = editor.model.cursor
    for _ in range(10):
        editor.update(0.05 * config.FPS)
    assert editor.model.cursor == stopped


def test_drawing_does_not_read_the_level_file_every_frame(editor, monkeypatch):
    open_level(editor)
    reads = []
    monkeypatch.setattr(editor, '_saved_layout_exists', lambda: reads.append(1) or False)
    for _ in range(5):
        editor.update(1.0)
        editor.draw()
    assert reads == []


def test_e_and_a_change_the_policies(editor):
    open_level(editor)
    exit_before, ammo_before = editor.model.exit_policy, editor.model.ammo_policy
    editor.handle_key(key(pygame.K_e))
    editor.handle_key(key(pygame.K_a))
    assert editor.model.exit_policy != exit_before and editor.model.ammo_policy != ammo_before


def test_saving_writes_both_policies(editor):
    open_level(editor)
    editor.handle_key(key(pygame.K_e))
    editor.handle_key(key(pygame.K_a))
    editor.handle_key(key(pygame.K_s))
    data = json.loads((editor.tmp / '3.json').read_text())
    assert data['exit_policy'] == editor.model.exit_policy.to_json()
    assert data['ammo_policy'] == editor.model.ammo_policy.to_json()


def test_a_generated_boss_level_opens_with_its_policies(editor):
    from policies.ammo import UnlimitedAmmo
    from policies.exit import AllCleared
    open_level(editor, 6)
    assert (editor.model.exit_policy, editor.model.ammo_policy) == (AllCleared(), UnlimitedAmmo())


def test_the_policy_keys_and_their_settings(editor):
    from policies.exit import OpenAfterTime, TypeCleared
    open_level(editor)
    editor.handle_key(key(pygame.K_e))                                    # always open -> timer 60s
    editor.handle_key(key(pygame.K_EQUALS))
    assert editor.model.exit_policy == OpenAfterTime(70)
    editor.handle_key(key(pygame.K_MINUS))
    assert editor.model.exit_policy == OpenAfterTime(60)
    editor.handle_key(key(pygame.K_t))
    assert editor.flash                                                   # not a type_cleared exit: said why
    editor.handle_key(key(pygame.K_e))                                    # -> kill eggs
    editor.model.palette_index = PALETTE.index('static')
    editor.handle_key(key(pygame.K_t))
    assert editor.model.exit_policy == TypeCleared(('egg', 'static'))
    editor.draw()


def test_the_preview_shows_the_jev_beacon(editor):  # noqa: F811
    open_level(editor)
    model = editor.model
    model.walls.clear()
    model.enemies.clear()
    model.beacon = None
    editor.draw()
    before = editor.screen.copy()
    model.beacon = (3, 3)
    editor.draw()
    calc = editor._preview[0].position_calculator
    x, y = calc.grid_center_to_screen(3, 3)
    area = pygame.Rect(0, 0, calc.cell_size_x, calc.cell_size_y)
    area.center = (x, y)
    area = area.clip(editor.screen.get_rect())
    assert area.width > 10 and area.height > 10
    differ = [(i, j) for i in range(area.left, area.right, 2) for j in range(area.top, area.bottom, 2)
              if before.get_at((i, j)) != editor.screen.get_at((i, j))]
    assert len(differ) > 20


def test_a_controller_opens_edits_and_saves_a_level(editor, monkeypatch):
    names = {0: 'a', 1: 'b', 2: 'x', 5: 'rightshoulder', 7: 'start'}
    held = [(0, 0)]
    monkeypatch.setattr(editor.game, 'input_handler', SimpleNamespace(
        controller_button_name=names.get, held_direction=lambda: held[0]), raising=False)

    def button(name):
        editor.handle_controller(pygame.event.Event(
            pygame.JOYBUTTONDOWN, button={v: k for k, v in names.items()}[name]))

    def dpad(dx, dy):
        held[0] = (dx, dy)
        editor.handle_controller(pygame.event.Event(pygame.JOYHATMOTION, value=(dx, -dy)))
        held[0] = (0, 0)
        editor.update(1.0)                                          # let go

    editor.open_picker()
    dpad(1, 0)
    dpad(1, 0)
    button('a')
    assert editor.mode == 'edit' and editor.model.level == 3
    model = editor.model
    model.cursor = (1, model.cursor[1])                             # room to move right on any generated maze
    start = model.cursor
    dpad(1, 0)
    assert model.cursor == (start[0] + 1, start[1])
    button('rightshoulder')
    button('rightshoulder')
    assert model.selected == PALETTE[2]
    button('x')
    button('a')
    assert model.contents(model.cursor) == PALETTE[2]

    button('start')                                                 # the menu: its first row saves
    button('a')
    assert editor.menu_index is None
    data = json.loads((editor.tmp / '3.json').read_text())
    assert parse_layout(data['layout'], 12) == model.to_layout()
