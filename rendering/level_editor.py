"""The level editor screen: a level picker, then a cell-by-cell editor drawn with the game's own maze and sprites.

Switched on by LEVEL_EDITOR (see config.editor_enabled). The rules live in editor.model, the files in editor.files;
this screen only reads keys and controller buttons, and draws.

On a controller the common edits have buttons (D-pad or left stick move, A place, X remove, LB/RB item, Y turn gate,
B back) and Start opens a menu of the rest.
"""
import json
import math
import os

import pygame

import config
import enemy_kinds
from input.input_handler import menu_direction
import level_config
from editor import files
from editor.generated import generated_layout
from editor.model import PALETTE, TIMER_STEP, EditorModel
from entities.command_recorder import CommandRecorder
from entities.jev_beacon import JevBeacon
from entities.ship import Ship
from game_handlers.spawn_manager import make_enemy
from maze.config import MazeComplexity
from maze.generator import Maze
from policies.exit import OpenAfterTime, TypeCleared
from physics.obstacle_world import plank_size
from rendering.gates import draw_gate
from rendering.fonts import get_font
from rendering.menu_components import (TEXT_DIM, TEXT_LABEL, AnimatedBackground, build_menu_backdrop, build_panel,
                                       build_panel_glow, menu_scale, render_hint_row, render_pill, render_title)

LEVELS_SHOWN = 31
REPEAT_DELAY = 0.25          # seconds a cursor key is held before it repeats
REPEAT_INTERVAL = 0.05
FLASH_SECONDS = 2.0
ARROWS = {pygame.K_LEFT: (-1, 0), pygame.K_RIGHT: (1, 0), pygame.K_UP: (0, -1), pygame.K_DOWN: (0, 1)}

LABELS = {'wall': 'Wall', 'strong_wall': 'Strong wall', **{kind: enemy_kinds.name(kind) for kind in enemy_kinds.kinds()},
          'start': 'Start', 'exit': 'Exit', 'beacon': 'Ally beacon', 'gate': 'Gate', 'motor_gate': 'Motor gate'}

# The chip beside each palette entry, in the colours the things have in play
CHIPS = {'wall': (120, 130, 150), 'strong_wall': (176, 192, 214), 'static': (255, 90, 90), 'patrol': (255, 150, 70), 'aggressive': (255, 70, 140), 'neural_jelly': (255, 110, 170),
         'replay': (190, 110, 255), 'flocker': (255, 200, 80), 'flighthouse': (255, 230, 140), 'vibehouse': (255, 90, 230), 'egg': (240, 220, 200),
         'wallcrawler': (225, 190, 110), 'inchworm': (150, 220, 85), 'sea_snake': (120, 70, 160), 'mini_sea_snake': (150, 100, 200), 'teeny_sea_snake': (185, 140, 225), 'anemone': (255, 120, 200), 'split_boss': (170, 90, 255), 'mother_boss': (220, 80, 255),
         'rules_hunter': (255, 90, 90), 'neural_hunter': (255, 90, 230),
         'boss_viber': (255, 90, 230), 'vibelet': (255, 90, 230), 'baby': (215, 170, 255), 'belemnite': (240, 120, 60), 'start': (90, 170, 255), 'exit': (80, 230, 255), 'beacon': (120, 255, 90),
         'gate': (120, 235, 245), 'motor_gate': (255, 190, 80)}

CURSOR_COLOR = config.COLOR_NEON_VOID_START
CONFIRM_KEYS = (pygame.K_y, pygame.K_RETURN)
CANCEL_KEYS = (pygame.K_n, pygame.K_ESCAPE)


class LevelEditor:
    PICKER_COLUMNS = 6

    def __init__(self, game, screen):
        self.game = game
        self.screen = screen
        self.scale = menu_scale()
        self.background = AnimatedBackground(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.backdrop = build_menu_backdrop(config.SCREEN_WIDTH, config.SCREEN_HEIGHT).convert()
        self.title_font = get_font(self._scaled(64))
        self.tile_font = get_font(self._scaled(40))
        self.hint_font = get_font(self._scaled(22))
        self.small_font = get_font(self._scaled(20))
        self.message_font = get_font(self._scaled(30))
        self.mode = 'picker'
        self.picker_index = 0
        self.with_layout = set()
        self.model = None
        self.defaults = None
        self.flash = ''
        self.flash_left = 0.0
        self.confirm = None
        self.phase = 0.0
        self._preview = None
        self._preview_key = None
        self._palette_font = None
        self._saved = False          # whether the level file holds a layout; read when it can change
        self.menu_index = None       # the row chosen in the controller's action menu, None while it is shut
        self.using_controller = False  # which hints to show: the last input's
        self._pad_direction = None   # the D-pad or stick direction already acted on, until it is let go
        self._clear_held()

    def _scaled(self, value):
        return max(1, int(round(value * self.scale)))

    def _clear_held(self):
        self._held = None            # (direction, whether it is still held), repeating the cursor's move
        self._repeat_left = 0.0

    def _hold(self, direction, still_held):
        self.model.move(*direction)
        self._held, self._repeat_left = (direction, still_held), REPEAT_DELAY

    def _say(self, text):
        self.flash, self.flash_left = text, FLASH_SECONDS

    # --- the picker ---
    def open_picker(self):
        self.mode = 'picker'
        self.confirm = None
        self.menu_index = None
        self.flash, self.flash_left = '', 0.0
        self._clear_held()
        self.with_layout = {level for level in range(1, LEVELS_SHOWN + 1)
                            if level_config.get_level_layout(level) is not None}

    def _picker_move(self, dx, dy):
        index = self.picker_index + dx + dy * self.PICKER_COLUMNS
        if 0 <= index < LEVELS_SHOWN:
            self.picker_index = index

    def _picker_key(self, event):
        if event.key in ARROWS:
            self._picker_move(*ARROWS[event.key])
            return True
        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._open_level(self.picker_index + 1)
            return True
        if event.key == pygame.K_ESCAPE:
            self.game.state = config.STATE_MENU
            return True
        return False

    # --- editing ---
    def _open_level(self, level):
        # Read the seed, maze and policies first: generating the level starts it in the game
        layout = level_config.get_level_layout(level)
        policies = level_config.get_level_exit_policy(level), level_config.get_level_ammo_policy(level)
        self.defaults = {'seed': level_config.get_level_seed(level),
                         'maze': {'complexity': level_config.get_maze_complexity(level).name.lower(),
                                  'grid_size': level_config.get_maze_grid_size(level)}}
        if layout is None:
            layout = generated_layout(self.game, level)
            self.defaults['maze']['grid_size'] = layout.grid_size
        self.model = EditorModel(level, layout, *policies)
        self._saved = self._saved_layout_exists()
        self.mode = 'edit'
        self.confirm = None
        self.menu_index = None
        self._preview = self._preview_key = None
        self._clear_held()

    def _place(self):
        if not self.model.place():
            self._say(f"{LABELS[self.model.selected]} can't go there")

    def _turn_gate(self):
        model = self.model
        if not model.turn_gate():
            self._say('No gate here' if model.contents(model.cursor) not in ('gate', 'motor_gate')
                      else "Can't turn: a wall is in the way")

    def _flip_gate(self):
        if not self.model.flip_gate():
            self._say('Motor gate only')

    def _adjust_timer(self, step):
        if not self.model.adjust_timer(step):
            self._say('Timer only' if not isinstance(self.model.exit_policy, OpenAfterTime) else 'Shortest timer')

    def _toggle_kill_type(self):
        model = self.model
        if not isinstance(model.exit_policy, TypeCleared):
            self._say('Kill-type exit only')
        elif not model.toggle_exit_type(model.selected):
            self._say('Keep at least one type' if model.selected in model.exit_policy.enemy_types
                      else f"{LABELS[model.selected]} isn't an enemy")

    def _edit_key(self, event):
        model = self.model
        if event.key in ARROWS:
            self._hold(ARROWS[event.key], lambda key=event.key: pygame.key.get_pressed()[key])
            return True
        if event.key == pygame.K_TAB:
            model.step_palette(-1 if event.mod & pygame.KMOD_SHIFT else 1)
            return True
        if event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            model.step_palette(-1 if event.key == pygame.K_LEFTBRACKET else 1)
            return True
        if event.key == pygame.K_SPACE:
            self._place()
            return True
        if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            model.remove()
            return True
        if event.key == pygame.K_e:
            model.cycle_exit_policy()
            return True
        if event.key == pygame.K_a:
            model.cycle_ammo_policy()
            return True
        if event.key == pygame.K_r:
            self._turn_gate()
            return True
        if event.key == pygame.K_f:
            self._flip_gate()
            return True
        if event.key in (pygame.K_MINUS, pygame.K_EQUALS):
            self._adjust_timer(TIMER_STEP if event.key == pygame.K_EQUALS else -TIMER_STEP)
            return True
        if event.key == pygame.K_t:
            self._toggle_kill_type()
            return True
        if event.key == pygame.K_s:
            self._save()
            return True
        if event.key == pygame.K_x:
            self._ask_revert()
            return True
        if event.key == pygame.K_ESCAPE:
            self._ask_leave()
            return True
        return False

    # --- the controller's action menu, for the edits without a button ---
    def menu_rows(self):
        """(label, what A does, what left and right do or None), top to bottom."""
        model = self.model
        shut = lambda action: lambda: (self._shut_menu(), action())
        return [
            ('Save', shut(self._save), None),
            ('Revert to generated', shut(self._ask_revert), None),
            (f'Exit: {model.exit_policy.label()}', model.cycle_exit_policy, None),
            (f'Legacy ammo: {model.ammo_policy.label()}', model.cycle_ammo_policy, None),
            ('Exit timer  (left / right)', lambda: self._adjust_timer(TIMER_STEP),
             lambda step: self._adjust_timer(step * TIMER_STEP)),
            (f'Kill type: {LABELS[model.selected]}', self._toggle_kill_type, None),
            ('Flip motor gate', shut(self._flip_gate), None),
            ('Back to editing', self._shut_menu, None),
        ]

    def _shut_menu(self):
        self.menu_index = None

    def _menu_direction(self, dx, dy):
        rows = self.menu_rows()
        if dy:
            self.menu_index = (self.menu_index + dy) % len(rows)
        elif rows[self.menu_index][2] is not None:
            rows[self.menu_index][2](dx)

    def _menu_choose(self):
        self.menu_rows()[self.menu_index][1]()

    # --- the controller ---
    def handle_controller(self, event):
        """Act on a controller button, D-pad or stick. True when the editor used it."""
        if event.type == pygame.JOYBUTTONDOWN:
            self.using_controller = True
            return self._pad_button(self.game.input_handler.controller_button_name(event.button))
        if event.type not in (pygame.JOYHATMOTION, pygame.JOYAXISMOTION) or not any(menu_direction(event)):
            return False
        # Read where the pad is held rather than the event, so a stick pushed diagonally settles on one direction
        direction = self.game.input_handler.held_direction()
        if not any(direction) or direction == self._pad_direction:
            return False
        self.using_controller = True
        self._pad_direction = direction
        if self.confirm is not None:
            return False
        if self.mode == 'picker':
            self._picker_move(*direction)
        elif self.menu_index is not None:
            self._menu_direction(*direction)
        else:
            self._hold(direction, lambda: self.game.input_handler.held_direction() == direction)
        return True

    def _pad_button(self, name):
        if self.confirm is not None:
            if name == 'a':
                self._answer_confirm()
                return True
            if name in ('b', 'back'):
                self.confirm = None
                return True
            return False
        if self.mode == 'picker':
            if name in ('a', 'start'):
                self._open_level(self.picker_index + 1)
                return True
            if name in ('b', 'back'):
                self.game.state = config.STATE_MENU
                return True
            return False
        if self.menu_index is not None:
            if name == 'a':
                self._menu_choose()
                return True
            if name in ('b', 'back', 'start'):
                self._shut_menu()
                return True
            return False
        model = self.model
        actions = {
            'a': self._place,
            'x': model.remove,
            'y': self._turn_gate,
            'leftshoulder': lambda: model.step_palette(-1),
            'rightshoulder': lambda: model.step_palette(1),
            'b': self._ask_leave,
            'back': self._ask_leave,
        }
        if name == 'start':
            self.menu_index = 0
            self._clear_held()
            return True
        if name in actions:
            actions[name]()
            return True
        return False

    def _save(self):
        try:
            files.save_layout(files.level_path(self.model.level), self.model.to_layout(), self.defaults,
                              extra={'exit_policy': self.model.exit_policy.to_json(),
                                     'ammo_policy': self.model.ammo_policy.to_json()})
        except (OSError, ValueError) as exc:
            self._say(f'Could not save: {exc}')
            return
        self.model.dirty = False
        self._saved = True
        self.with_layout.add(self.model.level)
        self._say(f'Saved level {self.model.level}')

    def _saved_layout_exists(self):
        path = files.level_path(self.model.level)
        if not os.path.exists(path):
            return False
        try:
            with open(path) as handle:
                return 'layout' in json.load(handle)
        except (OSError, ValueError):
            return False

    def _ask_revert(self):
        if not self._saved_layout_exists():
            self._say('Nothing to revert: no layout saved')
            return
        self.confirm = ('revert', 'Delete this layout and return to the generated level?')

    def _revert(self):
        level = self.model.level
        try:
            files.revert_layout(files.level_path(level))
        except (OSError, ValueError) as exc:
            self._say(f'Could not revert: {exc}')
            return
        self.with_layout.discard(level)
        self._open_level(level)
        self._say('Reverted to the generated level')

    def _ask_leave(self):
        if self.model.dirty:
            self.confirm = ('leave', 'Discard unsaved changes?')
        else:
            self.open_picker()

    def _answer_confirm(self):
        action, self.confirm = self.confirm[0], None
        if action == 'revert':
            self._revert()
        else:
            self.open_picker()

    def handle_key(self, event):
        """Act on a key press. True when the editor used it."""
        self.using_controller = False
        if self.confirm is not None:
            if event.key in CONFIRM_KEYS:
                self._answer_confirm()
                return True
            if event.key in CANCEL_KEYS:
                self.confirm = None
                return True
            return False
        if self.mode == 'picker':
            return self._picker_key(event)
        if self.menu_index is not None:
            if event.key in ARROWS:
                self._menu_direction(*ARROWS[event.key])
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._menu_choose()
            elif event.key == pygame.K_ESCAPE:
                self._shut_menu()
            else:
                return False
            return True
        return self._edit_key(event)

    def update(self, dt):
        """Advance by `dt` frames (the game's unit; 1.0 is one frame at config.FPS)."""
        self.background.update(dt)
        seconds = dt / config.FPS
        self.phase = (self.phase + seconds * 4.0) % (2 * math.pi)
        self.flash_left = max(0.0, self.flash_left - seconds)
        if self._pad_direction is not None and self.game.input_handler.held_direction() != self._pad_direction:
            self._pad_direction = None
        if self._held is None or self.mode != 'edit' or self.confirm is not None or self.menu_index is not None:
            return
        direction, still_held = self._held
        if not still_held():
            self._clear_held()
            return
        self._repeat_left -= seconds
        while self._repeat_left <= 0.0:
            self.model.move(*direction)
            self._repeat_left += REPEAT_INTERVAL

    # --- drawing ---
    def draw(self):
        if self.mode == 'picker':
            self._draw_picker()
        else:
            self._draw_edit()
        if self.flash and self.flash_left > 0.0:
            pill = render_pill(self.message_font, self.flash)
            self.screen.blit(pill, pill.get_rect(midtop=(config.UI_ZONE_WIDTH + (config.SCREEN_WIDTH - config.UI_ZONE_WIDTH) // 2,
                                                         self._scaled(18))))
        if self.menu_index is not None and self.mode == 'edit':
            self._draw_menu()
        if self.confirm is not None:
            self._draw_confirm()

    def _draw_hints(self, hints, rows_up=0):
        row = render_hint_row(self.hint_font, hints)
        bottom = config.SCREEN_HEIGHT - self._scaled(16) - rows_up * (row.get_height() + self._scaled(24))
        place = row.get_rect(midbottom=(config.SCREEN_WIDTH // 2, bottom))
        backing = pygame.Surface(place.inflate(self._scaled(36), self._scaled(14)).size, pygame.SRCALPHA)
        pygame.draw.rect(backing, (8, 6, 20, 215), backing.get_rect(), border_radius=backing.get_height() // 2)
        self.screen.blit(backing, backing.get_rect(center=place.center))
        self.screen.blit(row, place)

    def _draw_picker(self):
        self.screen.blit(self.backdrop, (0, 0))
        self.background.draw(self.screen)
        title = render_title(self.title_font, 'EDIT LEVELS')
        self.screen.blit(title, title.get_rect(center=(config.SCREEN_WIDTH // 2, int(config.SCREEN_HEIGHT * 0.13))))
        size = (self._scaled(130), self._scaled(96))
        gap = self._scaled(22)
        rows = math.ceil(LEVELS_SHOWN / self.PICKER_COLUMNS)
        width = self.PICKER_COLUMNS * size[0] + (self.PICKER_COLUMNS - 1) * gap
        left = (config.SCREEN_WIDTH - width) // 2
        top = int(config.SCREEN_HEIGHT * 0.25)
        for index in range(LEVELS_SHOWN):
            col, row = index % self.PICKER_COLUMNS, index // self.PICKER_COLUMNS
            rect = pygame.Rect(left + col * (size[0] + gap), top + row * (size[1] + gap), *size)
            level = index + 1
            if index == self.picker_index:
                glow = build_panel_glow(size, radius=16)
                self.screen.blit(glow, glow.get_rect(center=rect.center))
                self.screen.blit(build_panel(size, radius=16), rect)
                color = (255, 255, 255)
            else:
                pygame.draw.rect(self.screen, (22, 20, 44), rect, border_radius=16)
                pygame.draw.rect(self.screen, (60, 60, 110), rect, 1, border_radius=16)
                color = TEXT_DIM
            text = self.tile_font.render(str(level), True, color)
            self.screen.blit(text, text.get_rect(center=rect.center))
            if level in self.with_layout:
                pygame.draw.circle(self.screen, CHIPS['exit'],
                                   (rect.right - self._scaled(16), rect.top + self._scaled(16)), self._scaled(6))
        legend = self.small_font.render('A dot marks a level with a hand-made layout', True, TEXT_LABEL)
        self.screen.blit(legend, legend.get_rect(center=(config.SCREEN_WIDTH // 2,
                                                         top + rows * (size[1] + gap) + self._scaled(10))))
        if self.using_controller:
            self._draw_hints([('D-pad', 'choose'), ('A', 'edit'), ('B', 'back')])
        else:
            self._draw_hints([('Arrows', 'choose'), ('Enter', 'edit'), ('Esc', 'back')])

    def _preview_scene(self):
        """The maze and sprites of the model as it is, rebuilt only when it changes."""
        model = self.model
        key = (frozenset(model.walls), frozenset(model.strong_walls), tuple(sorted(model.enemies.items())), model.start, model.exit, model.beacon,
               tuple(sorted(model.gates.values(), key=lambda g: g.cell)))
        if self._preview is not None and key == self._preview_key:
            return self._preview
        layout = model.to_layout()
        # The level's seed, so the cave rock bulges as it will in play
        maze = Maze(model.level, complexity=MazeComplexity.EMPTY, grid_size=model.grid_size, layout=layout,
                    seed=level_config.get_level_seed(model.level))
        centre = maze.position_calculator.grid_center_to_screen
        recorder = CommandRecorder()
        sprites = []
        for kind, cell in layout.enemies:
            sprites.append((kind, make_enemy(kind, centre(*cell), model.level, recorder)))
        ship = Ship(centre(*model.start))
        exit_pos = centre(*model.exit)
        ship.angle = math.degrees(math.atan2(exit_pos[1] - ship.y, exit_pos[0] - ship.x)) % 360
        sprites.append(('start', ship))
        if model.beacon is not None:
            beacon = JevBeacon(centre(*model.beacon))
            beacon.age = beacon.SPAWN_FRAMES      # fully popped into view: the preview never updates it
            sprites.append(('beacon', beacon))
        self._preview, self._preview_key = (maze, sprites), key
        return self._preview

    def _draw_sprite(self, kind, sprite):
        try:
            sprite.draw(self.screen)
        except Exception:      # a sprite that cannot draw out of play still shows where it is
            pygame.draw.circle(self.screen, CHIPS.get(kind, (255, 255, 255)),
                               (round(sprite.x), round(sprite.y)), max(6, round(getattr(sprite, 'radius', 10))))

    def _draw_edit(self):
        self.screen.fill(config.COLOR_BACKGROUND)
        maze, sprites = self._preview_scene()
        maze.draw(self.screen)
        half_length, thickness = plank_size(maze)
        for gate in self.model.gates.values():
            draw_gate(self.screen, maze.position_calculator.grid_center_to_screen(*gate.cell),
                      math.radians(gate.angle), half_length, thickness, gate.motor)
        for kind, sprite in sprites:
            self._draw_sprite(kind, sprite)
        self._draw_cursor(maze)
        self._draw_palette()
        if self.using_controller:
            self._draw_hints([('D-pad', 'move'), ('LB RB', 'item'), ('A', 'place'), ('X', 'remove'),
                              ('Y', 'turn gate'), ('Start', 'more'), ('B', 'back')])
            return
        self._draw_hints([('Arrows', 'move'), ('Tab', 'item'), ('Space', 'place'), ('Del', 'remove'),
                          ('S', 'save'), ('X', 'revert'), ('Esc', 'back')])
        self._draw_hints([('E', f'Exit: {self.model.exit_policy.label()}'),
                          ('A', f'Legacy ammo: {self.model.ammo_policy.label()}'), ('- =', 'timer'), ('T', 'kill type'),
                          ('R', 'turn gate'), ('F', 'flip')],
                         rows_up=1)

    def _draw_cursor(self, maze):
        calc = maze.position_calculator
        col, row = self.model.cursor
        x, y = calc.grid_to_screen(col, row)
        rect = pygame.Rect(round(x), round(y), math.ceil(calc.cell_size_x), math.ceil(calc.cell_size_y))
        pulse = 0.5 + 0.5 * math.sin(self.phase)
        glow = pygame.Surface(rect.inflate(16, 16).size, pygame.SRCALPHA)
        pygame.draw.rect(glow, (*CURSOR_COLOR, int(40 + 50 * pulse)), glow.get_rect(), border_radius=12)
        self.screen.blit(glow, rect.inflate(16, 16))
        pygame.draw.rect(self.screen, CURSOR_COLOR, rect, 2 + round(pulse), border_radius=8)
        label = render_pill(self.small_font, self.cursor_label())
        above = rect.top - label.get_height() - 6 >= maze.offset_y
        anchor = rect.midtop if above else rect.midbottom
        place = label.get_rect(midbottom=(anchor[0], anchor[1] - 6)) if above else label.get_rect(midtop=(anchor[0], anchor[1] + 6))
        place.clamp_ip(pygame.Rect(config.UI_ZONE_WIDTH, 0, config.SCREEN_WIDTH - config.UI_ZONE_WIDTH, config.SCREEN_HEIGHT))
        self.screen.blit(label, place)

    def cursor_label(self):
        """What the pill above the cursor says: what is in the cell, and whether a gate there fits."""
        model = self.model
        here = model.contents(model.cursor)
        if here is None:
            return f'Empty  {model.cursor[0]},{model.cursor[1]}'
        if here in ('gate', 'motor_gate') and model.cursor in model.misfit_gates():
            return f"{LABELS[here]} doesn't fit: {'Y' if self.using_controller else 'R'} to turn"
        return LABELS[here]

    def _fitting_font(self, width):
        """The largest font, up to the screen's scale, in which every palette label fits `width`."""
        if self._palette_font is None:
            size = self._scaled(26)
            while size > 11 and max(get_font(size).size(text)[0] for text in LABELS.values()) > width:
                size -= 1
            self._palette_font = get_font(size)
            self._header_font = get_font(min(self._scaled(34), round(size * 1.5)))
        return self._palette_font

    def _draw_palette(self):
        zone = config.UI_ZONE_WIDTH
        pygame.draw.rect(self.screen, (12, 10, 26), (0, 0, zone, config.SCREEN_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 110), (zone - 1, 0), (zone - 1, config.SCREEN_HEIGHT))
        chip_room = self._scaled(30)
        font = self._fitting_font(zone - 24 - chip_room)
        header = self._header_font.render(f'LEVEL {self.model.level}', True, (215, 225, 255))
        self.screen.blit(header, (16, 16))
        status, colour = (('Unsaved changes', (255, 170, 120)) if self.model.dirty
                          else ('Saved' if self._saved else 'Generated level', TEXT_LABEL))
        status_text = font.render(status, True, colour)
        self.screen.blit(status_text, (16, 16 + header.get_height()))
        top = 16 + header.get_height() + status_text.get_height() + self._scaled(20)
        bottom = config.SCREEN_HEIGHT - self._scaled(60)
        row_height = min(self._scaled(56), (bottom - top) // len(PALETTE))
        chip = max(10, min(chip_room - 10, row_height - 16))
        for index, item in enumerate(PALETTE):
            y = top + index * row_height
            row = pygame.Rect(8, y, zone - 16, row_height - 4)
            if index == self.model.palette_index:
                self.screen.blit(build_panel(row.size, radius=10), row)
                colour = (255, 255, 255)
            else:
                colour = TEXT_DIM
            chip_rect = pygame.Rect(0, 0, chip, chip)
            chip_rect.midleft = (row.left + 10, row.centery)
            pygame.draw.rect(self.screen, CHIPS[item], chip_rect, border_radius=3 if item in ('wall', 'strong_wall') else chip // 2)
            text = font.render(LABELS[item], True, colour)
            self.screen.blit(text, text.get_rect(midleft=(chip_rect.right + 10, row.centery)))

    def _draw_confirm(self):
        shade = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        self.screen.blit(shade, (0, 0))
        text = self.message_font.render(self.confirm[1], True, (235, 240, 255))
        hints = render_hint_row(self.hint_font, [('A', 'yes'), ('B', 'no')] if self.using_controller
                                else [('Y', 'yes'), ('N', 'no')])
        size = (max(text.get_width(), hints.get_width()) + self._scaled(80), text.get_height() + hints.get_height()
                + self._scaled(70))
        panel = build_panel(size)
        rect = panel.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2))
        glow = build_panel_glow(size)
        self.screen.blit(glow, glow.get_rect(center=rect.center))
        self.screen.blit(panel, rect)
        self.screen.blit(text, text.get_rect(midtop=(rect.centerx, rect.top + self._scaled(26))))
        self.screen.blit(hints, hints.get_rect(midbottom=(rect.centerx, rect.bottom - self._scaled(22))))

    def _draw_menu(self):
        shade = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        self.screen.blit(shade, (0, 0))
        labels = [self.message_font.render(label, True, (255, 255, 255) if index == self.menu_index else TEXT_DIM)
                  for index, (label, _, _) in enumerate(self.menu_rows())]
        hints = render_hint_row(self.hint_font, [('A', 'choose'), ('B', 'close')])
        row_height = max(label.get_height() for label in labels) + self._scaled(18)
        width = max(max(label.get_width() for label in labels), hints.get_width()) + self._scaled(100)
        size = (width, row_height * len(labels) + hints.get_height() + self._scaled(80))
        panel = build_panel(size)
        rect = panel.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2))
        glow = build_panel_glow(size)
        self.screen.blit(glow, glow.get_rect(center=rect.center))
        self.screen.blit(panel, rect)
        top = rect.top + self._scaled(30)
        for index, label in enumerate(labels):
            row = pygame.Rect(rect.left + self._scaled(24), top + index * row_height, width - self._scaled(48), row_height)
            if index == self.menu_index:
                pygame.draw.rect(self.screen, CURSOR_COLOR, row, 2, border_radius=row_height // 2)
            self.screen.blit(label, label.get_rect(center=row.center))
        self.screen.blit(hints, hints.get_rect(midbottom=(rect.centerx, rect.bottom - self._scaled(20))))
