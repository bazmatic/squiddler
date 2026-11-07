"""State-specific event handlers to eliminate duplication.

This module provides state handlers that unify keyboard and controller
input handling, eliminating code duplication in the main game loop.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import pygame
import time
import config
from input.input_handler import menu_direction
if TYPE_CHECKING:
    from game import Game
    from rendering.profile_selection_menu import ProfileSelectionMenu


class StateHandler(ABC):
    """Base class for state-specific event handlers."""
    
    @abstractmethod
    def handle_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle keyboard event for this state.
        
        Args:
            event: The pygame keyboard event.
            game: The game instance.
            
        Returns:
            True if event was handled, False otherwise.
        """
        pass
    
    @abstractmethod
    def handle_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle controller button event for this state.
        
        Args:
            event: The pygame controller button event.
            game: The game instance.
            
        Returns:
            True if event was handled, False otherwise.
        """
        pass


class MenuStateHandler(StateHandler):
    """Handler for menu state events."""
    
    def __init__(self):
        """Initialize menu state handler with navigation debounce."""
        self.last_navigation_time = 0.0
        self.navigation_debounce_interval = 0.15  # 150ms between navigations
    
    def handle_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle keyboard events in menu state."""
        if event.key == pygame.K_UP:
            game.main_menu.navigate_up()
            return True
        elif event.key == pygame.K_DOWN:
            game.main_menu.navigate_down()
            return True
        elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
            selected_option = game.main_menu.get_selected_option()
            if selected_option == "START GAME":
                game.state = config.STATE_PLAYING
                game.level = game.initial_start_level if game.initial_start_level else game.profile_manager.get_active_level()
                game.start_level()
            elif selected_option == "LEVELS":
                game.level_select_menu.refresh()
                game.state = config.STATE_LEVEL_SELECT
            elif selected_option == "FUN":
                game.fun_menu.refresh()
                game.state = config.STATE_FUN
            elif selected_option == "EDITOR":
                game.level_editor.open_picker()
                game.state = config.STATE_EDITOR
            elif selected_option == "SELECT PROFILE":
                game.profile_selection_menu.refresh_profiles()
                game.state = config.STATE_PROFILE_SELECTION
            elif selected_option == "SETTINGS":
                game.state = config.STATE_SETTINGS
            elif selected_option == "QUIT":
                game.running = False
            return True
        elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
            game.running = False
            return True
        return False
    
    def handle_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle controller events in menu state."""
        if event.type == pygame.JOYBUTTONDOWN:
            if game.input_handler.is_controller_menu_confirm_pressed(event.button):
                selected_option = game.main_menu.get_selected_option()
                if selected_option == "START GAME":
                    game.state = config.STATE_PLAYING
                    game.level = game.initial_start_level if game.initial_start_level else game.profile_manager.get_active_level()
                    game.start_level()
                elif selected_option == "LEVELS":
                    game.level_select_menu.refresh()
                    game.state = config.STATE_LEVEL_SELECT
                elif selected_option == "FUN":
                    game.fun_menu.refresh()
                    game.state = config.STATE_FUN
                elif selected_option == "EDITOR":
                    game.level_editor.open_picker()
                    game.state = config.STATE_EDITOR
                elif selected_option == "SELECT PROFILE":
                    game.profile_selection_menu.refresh_profiles()
                    game.state = config.STATE_PROFILE_SELECTION
                elif selected_option == "SETTINGS":
                    game.state = config.STATE_SETTINGS
                elif selected_option == "QUIT":
                    game.running = False
                return True
        elif event.type == pygame.JOYHATMOTION:
            # Handle d-pad navigation
            hat_value = event.value
            # hat_value is a tuple (x, y) where y is 1 for up and -1 for down
            current_time = time.time()
            if current_time - self.last_navigation_time >= self.navigation_debounce_interval:
                if hat_value[1] == 1:  # D-pad up
                    game.main_menu.navigate_up()
                    self.last_navigation_time = current_time
                    return True
                elif hat_value[1] == -1:  # D-pad down
                    game.main_menu.navigate_down()
                    self.last_navigation_time = current_time
                    return True
        elif event.type == pygame.JOYAXISMOTION:
            # Handle analog stick navigation (with debounce to prevent rapid navigation)
            current_time = time.time()
            if current_time - self.last_navigation_time >= self.navigation_debounce_interval:
                if event.axis == 1:  # Left stick Y-axis
                    if event.value < -config.CONTROLLER_DEADZONE:
                        game.main_menu.navigate_up()
                        self.last_navigation_time = current_time
                        return True
                    elif event.value > config.CONTROLLER_DEADZONE:
                        game.main_menu.navigate_down()
                        self.last_navigation_time = current_time
                        return True
                elif event.axis == 3:  # Right stick Y-axis
                    if event.value < -config.CONTROLLER_DEADZONE:
                        game.main_menu.navigate_up()
                        self.last_navigation_time = current_time
                        return True
                    elif event.value > config.CONTROLLER_DEADZONE:
                        game.main_menu.navigate_down()
                        self.last_navigation_time = current_time
                        return True
        return False


class ProfileSelectionStateHandler(StateHandler):
    """Handler for profile selection events."""

    def __init__(self):
        self.last_navigation_time = 0.0
        self.navigation_debounce_interval = 0.15

    def handle_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        menu = game.profile_selection_menu
        if menu.creating_profile:
            return self._handle_creating_keyboard(event, game, menu)

        if event.key == pygame.K_UP:
            menu.navigate_up()
            return True
        elif event.key == pygame.K_DOWN:
            menu.navigate_down()
            return True
        elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
            return self._select_profile(menu, game)
        elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
            menu.feedback_message = None
            game.state = config.STATE_MENU
            return True
        return False

    def _handle_creating_keyboard(
        self,
        event: 'pygame.event.Event',
        game: 'Game',
        menu: 'ProfileSelectionMenu'
    ) -> bool:
        menu.pending_character = None  # the keyboard takes over from a controller's letter picker
        if event.key == pygame.K_RETURN:
            profile = menu.submit_new_profile()
            if profile:
                game.reset_scoring_to_profile_state()
                game.state = config.STATE_MENU
            return True
        if event.key == pygame.K_ESCAPE:
            menu.cancel_creating_profile()
            return True
        if event.key == pygame.K_BACKSPACE:
            menu.backspace_character()
            return True

        if event.unicode:
            menu.append_character(event.unicode)
            return True

        return False

    def _select_profile(self, menu: 'ProfileSelectionMenu', game: 'Game', with_controller: bool = False) -> bool:
        """Handle profile selection or creation entry."""
        selected = menu.get_selected_option()
        if selected == menu.CREATE_OPTION:
            menu.start_creating_profile(with_controller)
            return True

        if selected:
            game.profile_manager.set_active_profile(selected)
            menu.feedback_message = f"Selected profile '{selected}'."
            menu.refresh_profiles()
            game.reset_scoring_to_profile_state()
            game.state = config.STATE_MENU
        return True

    def handle_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        menu = game.profile_selection_menu
        current_time = time.time()

        if menu.creating_profile:
            return self._handle_creating_controller(event, game, menu)

        if event.type == pygame.JOYBUTTONDOWN:
            if game.input_handler.is_controller_menu_confirm_pressed(event.button):
                return self._select_profile(menu, game, with_controller=True)
            if game.input_handler.is_controller_menu_cancel_pressed(event.button):
                game.state = config.STATE_MENU
                return True
            return False

        _, dy = menu_direction(event)
        if not dy or current_time - self.last_navigation_time < self.navigation_debounce_interval:
            return False
        if dy < 0:
            menu.navigate_up()
        else:
            menu.navigate_down()
        self.last_navigation_time = current_time
        return True

    def _handle_creating_controller(
        self,
        event: 'pygame.event.Event',
        game: 'Game',
        menu: 'ProfileSelectionMenu'
    ) -> bool:
        """Spell the name a letter at a time: up and down choose it, A or right adds it, X or left deletes,
        Start saves, B or Back cancels."""
        if event.type == pygame.JOYBUTTONDOWN:
            name = game.input_handler.controller_button_name(event.button)
            if name == 'start':
                if menu.submit_new_profile():
                    game.reset_scoring_to_profile_state()
                    game.state = config.STATE_MENU
                return True
            if name == 'a':
                menu.commit_pending_character()
                return True
            if name == 'x':
                menu.backspace_character()
                return True
            if name in ('b', 'back'):
                menu.cancel_creating_profile()
                return True
            return False
        dx, dy = menu_direction(event)
        if not (dx or dy):
            return False
        now = time.time()
        if now - self.last_navigation_time < self.navigation_debounce_interval:
            return False
        self.last_navigation_time = now
        if dy:
            menu.step_pending_character(-dy)
        elif dx > 0:
            menu.commit_pending_character()
        else:
            menu.backspace_character()
        return True


class LevelSelectStateHandler(StateHandler):
    """Keyboard, d-pad and stick navigation for the level grid."""

    def __init__(self):
        self.last_navigation_time = 0.0

    def _play(self, game):
        menu = game.level_select_menu
        if menu.can_play_selected:
            game.level = menu.selected_level
            game.state = config.STATE_PLAYING
            game.start_level()
        return True

    def handle_keyboard(self, event, game):
        directions = {pygame.K_LEFT: (-1, 0), pygame.K_RIGHT: (1, 0),
                      pygame.K_UP: (0, -1), pygame.K_DOWN: (0, 1)}
        if event.key in directions:
            game.level_select_menu.navigate(*directions[event.key])
            return True
        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            return self._play(game)
        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            game.state = config.STATE_MENU
            return True
        return False

    def handle_controller(self, event, game):
        if event.type == pygame.JOYBUTTONDOWN:
            if game.input_handler.is_controller_menu_confirm_pressed(event.button):
                return self._play(game)
            if game.input_handler.is_controller_menu_cancel_pressed(event.button):
                game.state = config.STATE_MENU
                return True
        now = time.monotonic()
        if now - self.last_navigation_time < 0.15:
            return False
        dx, dy = 0, 0
        if event.type == pygame.JOYHATMOTION:
            dx, dy = event.value[0], -event.value[1]
        elif event.type == pygame.JOYAXISMOTION and abs(event.value) > config.CONTROLLER_DEADZONE:
            direction = 1 if event.value > 0 else -1
            if event.axis in (0, 2):
                dx = direction
            elif event.axis in (1, 3):
                dy = direction
        if dx or dy:
            game.level_select_menu.navigate(dx, dy)
            self.last_navigation_time = now
            return True
        return False


class SettingsStateHandler(StateHandler):
    """Handler for the settings screen: up and down choose a setting, confirm or left and right
    change it, cancel goes back to the menu."""

    def __init__(self):
        self.last_navigation_time = 0.0
        self.navigation_debounce_interval = 0.15

    def handle_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        menu = game.settings_menu
        if event.key in (pygame.K_ESCAPE, pygame.K_q, pygame.K_BACKSPACE):
            game.state = config.STATE_MENU
        elif event.key in (pygame.K_UP, pygame.K_w):
            menu.navigate(-1)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            menu.navigate(1)
        elif event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_a, pygame.K_d):
            menu.toggle_selected()
        else:
            return False
        return True

    def handle_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        menu = game.settings_menu
        if event.type == pygame.JOYBUTTONDOWN:
            if game.input_handler.is_controller_menu_cancel_pressed(event.button):
                game.state = config.STATE_MENU
                return True
            if game.input_handler.is_controller_menu_confirm_pressed(event.button):
                menu.toggle_selected()
                return True
        elif event.type == pygame.JOYHATMOTION:
            dx, dy = event.value
            if dy:
                menu.navigate(-dy)
                return True
            if dx:
                menu.toggle_selected()
                return True
        elif event.type == pygame.JOYAXISMOTION and abs(event.value) > config.CONTROLLER_DEADZONE:
            now = time.time()
            if now - self.last_navigation_time < self.navigation_debounce_interval:
                return False
            self.last_navigation_time = now
            if event.axis in (1, 3):
                menu.navigate(1 if event.value > 0 else -1)
            elif event.axis in (0, 2):
                menu.toggle_selected()
            else:
                return False
            return True
        return False


class PlayingStateHandler(StateHandler):
    """Handler for playing state events."""
    
    def handle_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle keyboard events in playing state."""
        if event.key == pygame.K_r:
            game.restart_level()
            return True
        if game.game_over_active:
            game.complete_level(success=False)
            return True
        if event.key == pygame.K_ESCAPE:
            game.state = config.STATE_QUIT_CONFIRM
            return True
        return False
    
    def handle_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle controller events in playing state."""
        if event.type != pygame.JOYBUTTONDOWN:
            return False
        if game.input_handler.is_controller_restart_pressed(event.button):
            game.restart_level()
            return True
        if game.game_over_active:
            game.complete_level(success=False)
            return True
        if game.input_handler.is_controller_quit_pressed(event.button):
            game.state = config.STATE_QUIT_CONFIRM
            return True
        return False


class QuitConfirmStateHandler(StateHandler):
    """Handler for quit confirmation state events."""

    def __init__(self):
        self.last_navigation_time = 0.0
        self.navigation_debounce_interval = 0.15  # 150ms between navigations

    def handle_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle keyboard events in quit confirm state."""
        if event.key in (pygame.K_LEFT, pygame.K_UP):
            if game.quit_confirmation_selection > 0:
                game.quit_confirmation_selection -= 1
            return True
        if event.key in (pygame.K_RIGHT, pygame.K_DOWN):
            if game.quit_confirmation_selection < 1:
                game.quit_confirmation_selection += 1
            return True
        if event.key in (pygame.K_y, pygame.K_RETURN, pygame.K_SPACE):
            return self._apply_quit_selection(game)
        elif event.key == pygame.K_q:
            game.running = False
            return True
        elif event.key in (pygame.K_n, pygame.K_ESCAPE):
            game.state = config.STATE_PLAYING
            game.reset_quit_confirmation_selection()
            return True
        return False
    
    def handle_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle controller events in quit confirm state."""
        if event.type == pygame.JOYBUTTONDOWN:
            if game.input_handler.is_controller_menu_confirm_pressed(event.button):
                return self._apply_quit_selection(game)
            if game.input_handler.is_controller_menu_cancel_pressed(event.button):
                game.state = config.STATE_PLAYING
                game.reset_quit_confirmation_selection()
                return True
            return False
        # D-pad or stick, any direction: up/left picks the first choice, down/right the second
        dx, dy = menu_direction(event)
        step = dx or dy
        if not step:
            return False
        now = time.time()
        if now - self.last_navigation_time < self.navigation_debounce_interval:
            return False
        self.last_navigation_time = now
        game.quit_confirmation_selection = min(1, max(0, game.quit_confirmation_selection + step))
        return True

    def _apply_quit_selection(self, game: 'Game') -> bool:
        if game.quit_confirmation_selection == 0:
            game.stop_autoplay()
            game.state = config.STATE_MENU
            game.reset_scoring_to_profile_state()
        else:
            game.state = config.STATE_PLAYING
        game.reset_quit_confirmation_selection()
        return True


class LevelCompleteStateHandler(StateHandler):
    """Handler for level complete state events."""
    
    def __init__(self):
        """Initialize level complete state handler with navigation debounce."""
        self.last_navigation_time = 0.0
        self.navigation_debounce_interval = 0.15  # 150ms between navigations
    
    def handle_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle keyboard events in level complete state."""
        if game.level_complete_quit_confirm:
            return self._handle_quit_confirm_keyboard(event, game)
        else:
            return self._handle_normal_keyboard(event, game)
    
    def _handle_quit_confirm_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle keyboard events when quit confirmation is active."""
        if event.key in (pygame.K_UP, pygame.K_LEFT):
            if game.quit_confirmation_selection > 0:
                game.quit_confirmation_selection -= 1
            return True
        if event.key in (pygame.K_DOWN, pygame.K_RIGHT):
            if game.quit_confirmation_selection < 1:
                game.quit_confirmation_selection += 1
            return True
        if event.key in (pygame.K_y, pygame.K_RETURN, pygame.K_SPACE):
            return self._apply_quit_selection(game)
        if event.key in (pygame.K_n, pygame.K_ESCAPE):
            game.level_complete_quit_confirm = False
            game.reset_quit_confirmation_selection()
            return True
        return False
    
    def _handle_normal_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle keyboard events in normal level complete state."""
        if event.key == pygame.K_UP:
            game.level_complete_menu.navigate_up()
            return True
        elif event.key == pygame.K_DOWN:
            game.level_complete_menu.navigate_down()
            return True
        elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
            selected_option = game.level_complete_menu.get_selected_option()
            if selected_option == "CONTINUE":
                game.level += 1
                game.state = config.STATE_PLAYING
                game.start_level()
            elif selected_option == "RETRY LEVEL":
                game.restart_level()
            elif selected_option == "MAIN MENU":
                game.state = config.STATE_MENU
                game.reset_scoring_to_profile_state()
            return True
        elif event.key == pygame.K_r:
            # R key still works for quick retry
            game.restart_level()
            return True
        elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
            # ESC/Q still shows quit confirmation (for backwards compatibility)
            game.level_complete_quit_confirm = True
            game.reset_quit_confirmation_selection()
            return True
        return False
    
    def handle_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle controller events in level complete state."""
        if game.level_complete_quit_confirm:
            return self._handle_quit_confirm_controller(event, game)
        else:
            return self._handle_normal_controller(event, game)
    
    def _handle_quit_confirm_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle controller events when quit confirmation is active."""
        current_time = time.time()

        if event.type == pygame.JOYBUTTONDOWN:
            if game.input_handler.is_controller_menu_confirm_pressed(event.button):
                return self._apply_quit_selection(game)
            if game.input_handler.is_controller_menu_cancel_pressed(event.button):
                game.level_complete_quit_confirm = False
                game.reset_quit_confirmation_selection()
                return True
            return False

        if event.type in (pygame.JOYHATMOTION, pygame.JOYAXISMOTION):
            if current_time - self.last_navigation_time < self.navigation_debounce_interval:
                return False
            direction = game.input_handler.get_controller_menu_navigation()
            if direction == "up":
                if game.quit_confirmation_selection > 0:
                    game.quit_confirmation_selection -= 1
                    self.last_navigation_time = current_time
                    return True
                return False
            if direction == "down":
                if game.quit_confirmation_selection < 1:
                    game.quit_confirmation_selection += 1
                    self.last_navigation_time = current_time
                    return True
                return False
        return False

    def _apply_quit_selection(self, game: 'Game') -> bool:
        """Execute the active quit confirmation choice."""
        if game.quit_confirmation_selection == 0:
            game.state = config.STATE_MENU
            game.reset_scoring_to_profile_state()
        else:
            game.level_complete_quit_confirm = False
        game.reset_quit_confirmation_selection()
        return True
    
    def _handle_normal_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        """Handle controller events in normal level complete state."""
        if event.type == pygame.JOYBUTTONDOWN:
            if game.input_handler.is_controller_menu_confirm_pressed(event.button):
                selected_option = game.level_complete_menu.get_selected_option()
                if selected_option == "CONTINUE":
                    game.level += 1
                    game.state = config.STATE_PLAYING
                    game.start_level()
                elif selected_option == "RETRY LEVEL":
                    game.restart_level()
                elif selected_option == "MAIN MENU":
                    game.state = config.STATE_MENU
                    game.reset_scoring_to_profile_state()
                return True
            elif game.input_handler.is_controller_menu_cancel_pressed(event.button):
                # B button: Show quit confirm (for backwards compatibility)
                game.level_complete_quit_confirm = True
                game.reset_quit_confirmation_selection()
                return True
        elif event.type in (pygame.JOYHATMOTION, pygame.JOYAXISMOTION):
            current_time = time.time()
            if current_time - self.last_navigation_time < self.navigation_debounce_interval:
                return False
            direction = game.input_handler.get_controller_menu_navigation()
            if direction == "up":
                game.level_complete_menu.navigate_up()
                self.last_navigation_time = current_time
                return True
            if direction == "down":
                game.level_complete_menu.navigate_down()
                self.last_navigation_time = current_time
                return True
        return False


class FunStateHandler(StateHandler):
    """The FUN list: up and down choose, confirm starts, cancel goes back to the main menu."""

    def __init__(self):
        self.last_navigation_time = 0.0

    def _start(self, game):
        entry = game.fun_menu.selected_entry
        if entry is not None:
            game.start_fun_entry(entry.key)
        return True

    def handle_keyboard(self, event, game):
        if event.key == pygame.K_UP:
            game.fun_menu.navigate(-1)
            return True
        if event.key == pygame.K_DOWN:
            game.fun_menu.navigate(1)
            return True
        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            return self._start(game)
        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            game.state = config.STATE_MENU
            return True
        return False

    def handle_controller(self, event, game):
        if event.type == pygame.JOYBUTTONDOWN:
            if game.input_handler.is_controller_menu_confirm_pressed(event.button):
                return self._start(game)
            if game.input_handler.is_controller_menu_cancel_pressed(event.button):
                game.state = config.STATE_MENU
                return True
        now = time.monotonic()
        if now - self.last_navigation_time < 0.15:
            return False
        step = 0
        if event.type == pygame.JOYHATMOTION:
            step = event.value[1] * -1
        elif event.type == pygame.JOYAXISMOTION and event.axis in (1, 3) and abs(event.value) > config.CONTROLLER_DEADZONE:
            step = 1 if event.value > 0 else -1
        if step:
            game.fun_menu.navigate(step)
            self.last_navigation_time = now
            return True
        return False


class DuelStateHandler(StateHandler):
    """During a duel the only input is cancel, which ends it."""

    def handle_keyboard(self, event, game):
        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            game.end_duel()
            return True
        return False

    def handle_controller(self, event, game):
        if event.type == pygame.JOYBUTTONDOWN and game.input_handler.is_controller_menu_cancel_pressed(event.button):
            game.end_duel()
            return True
        return False


class EditorStateHandler(StateHandler):
    """The level editor: the editor itself reads the keys."""

    def handle_keyboard(self, event, game):
        return game.level_editor.handle_key(event)

    def handle_controller(self, event, game):
        return game.level_editor.handle_controller(event)


class StateHandlerRegistry:
    """Registry for state handlers."""
    
    def __init__(self):
        """Initialize registry with all state handlers."""
        self.handlers = {
            config.STATE_MENU: MenuStateHandler(),
            config.STATE_PROFILE_SELECTION: ProfileSelectionStateHandler(),
            config.STATE_SETTINGS: SettingsStateHandler(),
            config.STATE_LEVEL_SELECT: LevelSelectStateHandler(),
            config.STATE_FUN: FunStateHandler(),
            config.STATE_EDITOR: EditorStateHandler(),
            config.STATE_DUEL: DuelStateHandler(),
            config.STATE_PLAYING: PlayingStateHandler(),
            config.STATE_QUIT_CONFIRM: QuitConfirmStateHandler(),
            config.STATE_LEVEL_COMPLETE: LevelCompleteStateHandler(),
        }
    
    def get_handler(self, state: str) -> StateHandler:
        """Get handler for a state.
        
        Args:
            state: The game state.
            
        Returns:
            The state handler, or a no-op handler if state not found.
        """
        return self.handlers.get(state, _NoOpStateHandler())


class _NoOpStateHandler(StateHandler):
    """No-op handler for unknown states."""
    
    def handle_keyboard(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        return False
    
    def handle_controller(self, event: 'pygame.event.Event', game: 'Game') -> bool:
        return False

