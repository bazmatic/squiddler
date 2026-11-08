# tests/test_editor_gates.py
import json

import pygame
import pytest

import config

from editor import files
from editor.model import PALETTE, EditorModel
from level_layout import GatePlacement, Layout, parse_layout, plank_cells
from tests.test_hunter_integration import game  # noqa: F401 - pytest fixture
from tests.test_level_editor import editor, key, open_level  # noqa: F401 - pytest fixtures and helpers


def model_with(**kwargs):
    layout = Layout(12, frozenset(kwargs.pop('walls', ())), (1, 1), (10, 10), tuple(kwargs.pop('enemies', ())),
                    kwargs.pop('beacon', None), tuple(kwargs.pop('gates', ())))
    return EditorModel(1, layout)


def select(model, item):
    model.palette_index = PALETTE.index(item)


def test_placing_turning_and_flipping_a_gate(tmp_path):
    model = model_with()
    model.cursor = (5, 5)
    select(model, 'motor_gate')
    assert model.place() and model.contents((5, 5)) == 'motor_gate'
    assert model.gates[(5, 5)] == GatePlacement((5, 5), 0, 'cw')
    assert model.turn_gate() and model.gates[(5, 5)].angle == 45
    assert model.flip_gate() and model.gates[(5, 5)].motor == 'ccw'
    layout = model.to_layout()
    assert layout.gates == (GatePlacement((5, 5), 45, 'ccw'),)
    path = str(tmp_path / '1.json')
    files.save_layout(path, layout, {'seed': 1})
    reloaded = parse_layout(json.loads(open(path).read())['layout'], 12)
    assert reloaded == layout
    assert EditorModel(1, reloaded).gates == model.gates


def test_turning_wraps_at_180_and_a_free_gate_cannot_flip():
    model = model_with(gates=[GatePlacement((5, 5), 135)])
    model.cursor = (5, 5)
    assert model.turn_gate() and model.gates[(5, 5)].angle == 0
    assert model.flip_gate() is False and model.gates[(5, 5)].motor is None
    model.cursor = (6, 6)
    assert model.turn_gate() is False and model.flip_gate() is False


def test_a_gate_replaces_the_other_kind():
    model = model_with(gates=[GatePlacement((5, 5), 90)])
    model.cursor = (5, 5)
    select(model, 'motor_gate')
    assert model.place() and model.gates[(5, 5)] == GatePlacement((5, 5), 0, 'cw')
    select(model, 'gate')
    assert model.place() and model.gates[(5, 5)] == GatePlacement((5, 5))


def test_a_gate_cannot_go_across_a_wall():
    model = model_with(walls=[(6, 5), (4, 4), (6, 4), (5, 4)])
    model.cursor = (5, 5)
    select(model, 'gate')
    assert model.place() is False and model.gates == {} and not model.dirty
    model = model_with(walls=[(6, 5)])
    model.cursor = (5, 5)
    select(model, 'gate')
    assert model.place() and model.gates[(5, 5)].angle == 45             # 0 would cross the wall
    model = model_with(walls=[(6, 6)], gates=[GatePlacement((5, 5))])
    model.cursor = (5, 5)
    assert model.turn_gate() and model.gates[(5, 5)].angle == 90         # 45 would cross the wall


def test_a_gate_cannot_reach_the_border():
    model = model_with()
    model.cursor = (10, 1)
    select(model, 'gate')
    assert model.place() is False
    model.cursor = (1, 5)
    assert model.place() and model.gates[(1, 5)].angle == 90             # along the border, not into it


def test_a_wall_cannot_go_across_a_plank():
    model = model_with(gates=[GatePlacement((5, 5))])
    model.cursor = (6, 5)
    select(model, 'wall')
    assert model.place() is False and (6, 5) not in model.walls
    model.cursor = (5, 5)
    assert model.place() is False
    model.cursor = (6, 7)
    assert model.place() is True


def test_a_gate_cannot_share_a_cell_and_can_be_removed():
    model = model_with(enemies=[('static', (4, 4))], beacon=(7, 7), gates=[GatePlacement((5, 5), 0, 'cw')])
    select(model, 'gate')
    for cell in ((1, 1), (10, 10), (7, 7), (4, 4)):
        model.cursor = cell
        assert model.place() is False
    for item in ('static', 'start', 'exit', 'beacon'):
        model.cursor = (5, 5)
        select(model, item)
        assert model.place() is False
    model.cursor = (5, 5)
    assert model.remove() and model.gates == {} and model.contents((5, 5)) is None


def test_r_and_f_turn_and_flip_in_the_editor(editor):  # noqa: F811
    open_level(editor)
    model = editor.model
    model.cursor = (5, 5)
    model.walls.clear()
    select(model, 'motor_gate')
    editor.handle_key(key(pygame.K_SPACE))
    editor.handle_key(key(pygame.K_r))
    editor.handle_key(key(pygame.K_f))
    assert model.gates[(5, 5)] == GatePlacement((5, 5), 45, 'ccw')


def test_the_preview_draws_gates(editor):  # noqa: F811
    open_level(editor)
    model = editor.model
    model.walls.clear()
    model.enemies.clear()
    model.cursor = (3, 3)
    editor.draw()
    before = editor.screen.copy()
    select(model, 'motor_gate')
    editor.handle_key(key(pygame.K_SPACE))
    editor.draw()
    calc = editor._preview[0].position_calculator
    x, y = calc.grid_center_to_screen(3, 3)
    area = pygame.Rect(0, 0, 3 * calc.cell_size_x, 3 * calc.cell_size_y)
    area.center = (x, y)
    area = area.clip(editor.screen.get_rect())
    differ = [(i, j) for i in range(area.left, area.right, 3) for j in range(area.top, area.bottom, 3)
              if before.get_at((i, j)) != editor.screen.get_at((i, j))]
    assert len(differ) > 10
    key_before_turn = editor._preview_key
    editor.handle_key(key(pygame.K_r))              # turning rebuilds the preview
    editor.draw()
    assert editor._preview_key != key_before_turn and editor.model.gates[(3, 3)].angle == 45


def test_a_gate_cannot_be_placed_or_turned_into_another_plank():
    model = model_with(gates=[GatePlacement((5, 5), 0), GatePlacement((7, 5), 90)])
    select(model, 'gate')
    model.cursor = (6, 5)
    assert model.place() is False and (6, 5) not in model.gates
    model.cursor = (7, 5)
    assert model.turn_gate() is True and model.gates[(7, 5)].angle == 135    # still clear of (4..6, 5)
    assert model.turn_gate() is True and model.gates[(7, 5)].angle == 45     # at 0 its tip would reach (6, 5)


def test_planks_that_share_no_cell_are_fine_and_a_gate_can_replace_its_own_kind():
    model = model_with(gates=[GatePlacement((5, 5), 0), GatePlacement((7, 5), 90)])
    model.cursor = (5, 5)
    select(model, 'motor_gate')
    assert model.place() is True and model.gates[(5, 5)] == GatePlacement((5, 5), 0, 'cw')


@pytest.fixture
def real_screen(monkeypatch):
    """The real fullscreen, where cells are wider than tall."""
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)


def grid_model(grid, walls=(), gates=(), enemies=()):
    return EditorModel(1, Layout(grid, frozenset(walls), (1, 1), (grid - 2, grid - 2), tuple(enemies), None,
                                 tuple(gates)))


@pytest.mark.parametrize('grid', [12, 20])
@pytest.mark.parametrize('wall', [(5, 4), (5, 6)])
def test_a_diagonal_plank_that_would_cut_a_wall_is_refused(real_screen, grid, wall):
    # Cells are wider than tall: at 45 and 135 degrees the plank cuts the cell above or below its tips
    model = grid_model(grid, walls=[wall], gates=[GatePlacement((5, 5))])
    model.cursor = (5, 5)
    assert model.turn_gate() is False and model.gates[(5, 5)].angle == 0      # 45, 90 and 135 all cut the wall
    for angle in (45, 135):
        model = grid_model(grid, gates=[GatePlacement((5, 5), angle)])
        model.cursor = wall
        select(model, 'wall')
        assert model.place() is False and wall not in model.walls


@pytest.mark.parametrize('grid', [12, 20])
def test_a_gate_that_fits_beside_walls_is_accepted(real_screen, grid):
    model = grid_model(grid, walls=[(6, 4), (4, 6)])
    model.cursor = (5, 5)
    select(model, 'gate')
    assert model.place() and model.gates[(5, 5)].angle == 0
    assert model.turn_gate() and model.gates[(5, 5)].angle == 45


def test_a_gate_is_placed_at_the_first_angle_that_fits():
    model = model_with(walls=[(4, 5), (6, 5), (4, 4), (6, 6)])
    model.cursor = (5, 5)
    select(model, 'motor_gate')
    assert model.place() and model.gates[(5, 5)] == GatePlacement((5, 5), 90, 'cw')
    model = model_with(walls=[(4, 5), (6, 5), (5, 4), (4, 4), (6, 4)])
    model.cursor = (5, 5)
    select(model, 'motor_gate')
    assert model.place() is False and model.gates == {}


def test_placing_the_same_kind_again_keeps_the_angle_and_direction():
    model = model_with(gates=[GatePlacement((5, 5), 45, 'ccw')])
    model.cursor = (5, 5)
    select(model, 'motor_gate')
    assert model.place() and model.gates[(5, 5)] == GatePlacement((5, 5), 45, 'ccw')
    model = model_with(gates=[GatePlacement((5, 5), 135)])
    model.cursor = (5, 5)
    select(model, 'gate')
    assert model.place() and model.gates[(5, 5)] == GatePlacement((5, 5), 135)


def test_turning_steps_past_an_angle_that_does_not_fit():
    model = model_with(walls=[(4, 4), (6, 6)], gates=[GatePlacement((5, 5))])
    model.cursor = (5, 5)
    angles = []
    for _ in range(3):
        assert model.turn_gate()
        angles.append(model.gates[(5, 5)].angle)
    assert angles == [90, 135, 0]


@pytest.mark.parametrize('item', ['start', 'exit', 'anemone', 'flighthouse'])
def test_a_plank_cannot_cover_the_start_the_exit_or_an_anchored_enemy(item):
    start, exit_ = {'start': ((4, 5), (10, 10)), 'exit': ((1, 1), (4, 5))}.get(item, ((1, 1), (10, 10)))
    enemies = [(item, (4, 5))] if item in ('anemone', 'flighthouse') else []
    model = EditorModel(1, Layout(12, frozenset(), start, exit_, tuple(enemies)))
    model.cursor = (5, 5)
    select(model, 'gate')
    assert model.place() and model.gates[(5, 5)].angle != 0        # at 0 it would cover (4, 5)
    assert (4, 5) not in plank_cells(model.gates[(5, 5)])
    # ... and none of them can be put under a plank
    model = model_with(gates=[GatePlacement((5, 5))])
    model.cursor = (6, 5)
    select(model, item)
    assert model.place() is False
    assert model.contents((6, 5)) is None
    select(model, 'static')
    assert model.place() is True


def test_a_gate_that_does_not_fit_this_screen_does_not_lock_the_editor(real_screen):
    # Saved on a screen of another shape: at 45 degrees its plank cuts the wall below here
    model = grid_model(12, walls=[(5, 6)], gates=[GatePlacement((5, 5), 45)])
    assert model.misfit_gates() == [(5, 5)]
    for item, cell in (('wall', (9, 2)), ('start', (8, 2)), ('exit', (2, 8)), ('anemone', (9, 9))):
        model.cursor = cell
        select(model, item)
        assert model.place(), item
    assert model.misfit_gates() == [(5, 5)]
    model.cursor = (5, 5)
    assert model.turn_gate() and model.misfit_gates() == []


def test_the_cursor_says_when_a_gate_does_not_fit(editor, real_screen):  # noqa: F811
    open_level(editor)
    model = editor.model
    model.walls.clear()
    model.enemies.clear()
    model.walls.add((5, 6))
    model.gates[(5, 5)] = GatePlacement((5, 5), 45)
    model.cursor = (5, 5)
    assert "doesn't fit" in editor.cursor_label()
    model.cursor = (8, 8)
    assert "doesn't fit" not in editor.cursor_label()
