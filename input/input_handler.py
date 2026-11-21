"""Input handler for processing keyboard and controller input.

This module provides the InputHandler class that processes keyboard and controller
input and maps them to game commands without side effects.
"""

import pygame
from pygame._sdl2 import controller as sdl_controller
from typing import Dict, List, Optional, Tuple
from entities.command_recorder import CommandType
import config

# SDL's D-pad names, as the hat values pygame reports (y is +1 for up)
_DPAD_HATS = {'dpup': (0, 1), 'dpdown': (0, -1), 'dpleft': (-1, 0), 'dpright': (1, 0)}
# The common raw layout, for pads SDL can't name
_FALLBACK_BUTTON_NAMES = {0: 'a', 1: 'b', 2: 'x', 3: 'y', 4: 'leftshoulder', 5: 'rightshoulder', 6: 'back', 7: 'start'}


def menu_direction(event: 'pygame.event.Event') -> Tuple[int, int]:
    """Where a D-pad or stick event points, as (dx, dy) with dy positive downward; (0, 0) for a stick at rest
    or any other event."""
    if event.type == pygame.JOYHATMOTION:
        return event.value[0], -event.value[1]
    if event.type == pygame.JOYAXISMOTION and event.axis in (0, 1, 2, 3) and abs(event.value) > config.CONTROLLER_DEADZONE:
        step = 1 if event.value > 0 else -1
        return (step, 0) if event.axis in (0, 2) else (0, step)
    return 0, 0

class InputHandler:
    """Handles keyboard and controller input and maps them to commands.
    
    This class processes keyboard and controller input and returns a list of commands
    without executing any game logic, following the Single Responsibility Principle.
    
    Attributes:
        key_mappings: Dictionary mapping key combinations to command types.
        controllers: List of active pygame joystick objects.
    """
    
    def __init__(self):
        """Initialize input handler with key mappings and controller support."""
        # Map key combinations to command types
        # Each entry maps a tuple of key codes to a CommandType
        self.key_mappings = {
            (pygame.K_LEFT, pygame.K_a, pygame.K_j): CommandType.ROTATE_LEFT,
            (pygame.K_RIGHT, pygame.K_d, pygame.K_l): CommandType.ROTATE_RIGHT,
            (pygame.K_UP, pygame.K_w, pygame.K_i): CommandType.APPLY_THRUST,
            (pygame.K_DOWN, pygame.K_s): CommandType.ACTIVATE_SHIELD,
        }
        
        # Track active controllers
        self.controllers: List[pygame.joystick.Joystick] = []
        self._button_names_cache: Dict[int, Dict[int, str]] = {}
        self._initialize_controllers()
    
    def _initialize_controllers(self) -> None:
        """Initialize all connected controllers."""
        self.controllers = []
        joystick_count = pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.controllers.append(joystick)
    
    def _button_names(self) -> Dict[int, str]:
        """Raw button number -> SDL's standard name ('a', 'start', 'dpup', ...) for the first controller.

        Empty when SDL doesn't recognise the pad; callers then fall back to raw button numbers.
        """
        if not self.controllers:
            return {}
        joystick = self.controllers[0]
        instance_id = joystick.get_instance_id()
        if instance_id not in self._button_names_cache:
            names = {}
            try:
                if not sdl_controller.get_init():
                    sdl_controller.init()
                mapping = sdl_controller.Controller.from_joystick(joystick).get_mapping()
                names = {int(raw[1:]): name for name, raw in mapping.items()
                         if raw.startswith('b') and raw[1:].isdigit()}
            except (pygame.error, TypeError):
                pass  # not a pad SDL can describe
            self._button_names_cache[instance_id] = names
        return self._button_names_cache[instance_id]

    def dpad_button_as_hat(self, event: 'pygame.event.Event') -> 'pygame.event.Event':
        """Turn a D-pad press from a pad that reports its D-pad as buttons into the hat event menus listen for."""
        if event.type != pygame.JOYBUTTONDOWN:
            return event
        hat = _DPAD_HATS.get(self._button_names().get(event.button))
        if hat is None:
            return event
        return pygame.event.Event(pygame.JOYHATMOTION, joy=event.joy, instance_id=event.instance_id, hat=0, value=hat)

    def controller_button_name(self, button: int) -> Optional[str]:
        """SDL's standard name for a raw button ('a', 'start', 'leftshoulder', ...), or None for an unknown button."""
        if not self.controllers:
            return None
        names = self._button_names()
        return names.get(button) if names else _FALLBACK_BUTTON_NAMES.get(button)

    def held_direction(self) -> Tuple[int, int]:
        """Where the first controller's D-pad or left stick is held right now, as (dx, dy) with dy positive downward."""
        if not self.controllers:
            return 0, 0
        controller = self.controllers[0]
        if controller.get_numhats():
            x, y = controller.get_hat(0)
            if x or y:
                return x, -y
        for raw, name in self._button_names().items():
            if name in _DPAD_HATS and raw < controller.get_numbuttons() and controller.get_button(raw):
                x, y = _DPAD_HATS[name]
                return x, -y
        if controller.get_numaxes() > 1:
            x, y = controller.get_axis(0), controller.get_axis(1)
            if max(abs(x), abs(y)) > config.CONTROLLER_DEADZONE:
                if abs(x) >= abs(y):
                    return (1 if x > 0 else -1), 0
                return 0, (1 if y > 0 else -1)
        return 0, 0

    def get_controllers(self) -> List[pygame.joystick.Joystick]:
        """Get list of active controllers.
        
        Returns:
            List of active pygame joystick objects.
        """
        return self.controllers
    
    def add_controller(self, joystick_id: int) -> None:
        """Add a newly connected controller.
        
        Args:
            joystick_id: The ID of the joystick to add.
        """
        if joystick_id < pygame.joystick.get_count():
            joystick = pygame.joystick.Joystick(joystick_id)
            joystick.init()
            if joystick not in self.controllers:
                self.controllers.append(joystick)
    
    def remove_controller(self, instance_id: int) -> None:
        """Remove a disconnected controller.

        Args:
            instance_id: The instance ID from the JOYDEVICEREMOVED event.
        """
        self.controllers = [
            ctrl for ctrl in self.controllers
            if ctrl.get_instance_id() != instance_id
        ]
    
    def _controller_at(self, index: int):
        """The controller driving a player's ship, or None when that player has none."""
        return self.controllers[index] if 0 <= index < len(self.controllers) else None

    def process_controller_input(self, index: int = 0) -> List[CommandType]:
        """Process controller input and return detected commands.
        
        Returns:
            List of CommandType values detected from controller input.
            Returns empty list when no controller input is detected.
        """
        commands = []
        
        controller = self._controller_at(index)
        if controller is None:
            return commands
        
        # Get number of axes and buttons
        num_axes = controller.get_numaxes()
        num_buttons = controller.get_numbuttons()
        
        # Process rotation from analog sticks
        # Left stick X-axis (axis 0) or right stick X-axis (axis 2)
        left_stick_x = controller.get_axis(0) if num_axes > 0 else 0.0
        right_stick_x = controller.get_axis(2) if num_axes > 2 else 0.0
        
        # Use whichever stick has input above deadzone (prioritize left stick if both have input)
        stick_x = 0.0
        if abs(left_stick_x) > config.CONTROLLER_DEADZONE:
            stick_x = left_stick_x
        elif abs(right_stick_x) > config.CONTROLLER_DEADZONE:
            stick_x = right_stick_x
        
        if abs(stick_x) > config.CONTROLLER_DEADZONE:
            if stick_x < 0:
                commands.append(CommandType.ROTATE_LEFT)
            elif stick_x > 0:
                commands.append(CommandType.ROTATE_RIGHT)
        
        # Process thrust: L button (button 4) OR ZL (left trigger/axis 5)
        thrust_active = False
        if num_buttons >= 5:
            if controller.get_button(4):  # L button
                thrust_active = True
        
        if not thrust_active and num_axes >= 6:
            left_trigger = controller.get_axis(5)
            # Triggers may range from -1.0 to 1.0 (unpressed to pressed)
            # or 0.0 to 1.0 (unpressed to pressed)
            # Check for positive values above threshold
            if left_trigger > config.CONTROLLER_TRIGGER_THRESHOLD:
                thrust_active = True
        
        if thrust_active:
            commands.append(CommandType.APPLY_THRUST)
        
        return commands
    
    def process_input(self, keys: List[bool]) -> List[CommandType]:
        """Process keyboard and controller input and return detected commands.
        
        Args:
            keys: List of key states from pygame.key.get_pressed().
            
        Returns:
            List of CommandType values detected from current input.
            Returns empty list when no input is detected.
        """
        commands = []
        
        # Process keyboard input
        for key_codes, command_type in self.key_mappings.items():
            # Check if any of the mapped keys are pressed
            if any(keys[key_code] for key_code in key_codes):
                commands.append(command_type)
        
        # Process controller input and combine with keyboard
        controller_commands = self.process_controller_input()
        for cmd in controller_commands:
            if cmd not in commands:
                commands.append(cmd)
        
        return commands
    
    def is_controller_fire_pressed(self, index: int = 0) -> bool:
        """Check if controller fire button is pressed.
        
        Returns:
            True if fire button is pressed, False otherwise.
        """
        controller = self._controller_at(index)
        if controller is None:
            return False
        num_axes = controller.get_numaxes()
        num_buttons = controller.get_numbuttons()
        
        # Fire: B button (button 1) OR R button (button 5) OR ZR (right trigger/axis 4)
        if num_buttons >= 2:
            if controller.get_button(1):  # B button
                return True
        
        if num_buttons >= 6:
            if controller.get_button(5):  # R button
                return True
        
        if num_axes >= 5:
            right_trigger = controller.get_axis(4)
            # Triggers may range from -1.0 to 1.0 (unpressed to pressed)
            # or 0.0 to 1.0 (unpressed to pressed)
            # Check for positive values above threshold
            if right_trigger > config.CONTROLLER_TRIGGER_THRESHOLD:
                return True
        
        return False
    
    def is_controller_shield_pressed(self, index: int = 0) -> bool:
        """Check if controller shield button is pressed.
        
        Returns:
            True if shield button is pressed, False otherwise.
        """
        controller = self._controller_at(index)
        if controller is None:
            return False
        num_buttons = controller.get_numbuttons()
        
        # Shield: A button (button 0)
        if num_buttons > 0:
            if controller.get_button(0):  # A button
                return True
        
        return False
    
    def is_controller_menu_confirm_pressed(self, button: int) -> bool:
        """Check if controller confirm button was pressed.
        
        Args:
            button: Button ID that was pressed.
            
        Returns:
            True if confirm button (A or Start) was pressed, False otherwise.
        """
        if not self.controllers:
            return False
        
        names = self._button_names()
        if names:
            return names.get(button) in ('a', 'start')

        controller = self.controllers[0]
        num_buttons = controller.get_numbuttons()
        
        # Confirm: A button (button 0) or Start button (button 7, typical for Xbox/PlayStation)
        if button == 0:  # A button
            return True
        
        if num_buttons > 7:
            if button == 7:  # Start button
                return True
        
        return False
    
    def is_controller_menu_cancel_pressed(self, button: int) -> bool:
        """Check if controller cancel button was pressed.
        
        Args:
            button: Button ID that was pressed.
            
        Returns:
            True if cancel button (B or Back) was pressed, False otherwise.
        """
        if not self.controllers:
            return False
        
        names = self._button_names()
        if names:
            return names.get(button) in ('b', 'back')

        controller = self.controllers[0]
        num_buttons = controller.get_numbuttons()
        
        # Cancel: B button (button 1) or Back button (button 6, typical for Xbox/PlayStation)
        if button == 1:  # B button
            return True
        
        if num_buttons > 6:
            if button == 6:  # Back button
                return True
        
        return False
    
    def is_controller_restart_pressed(self, button: int) -> bool:
        """Y (button 3) retries the current level."""
        names = self._button_names()
        if names:
            return names.get(button) == 'y'
        return bool(self.controllers) and button == 3

    def is_controller_quit_pressed(self, button: int) -> bool:
        """Check if controller quit button was pressed.
        
        Args:
            button: Button ID that was pressed.
            
        Returns:
            True if quit button (X) was pressed, False otherwise.
        """
        if not self.controllers:
            return False
        
        names = self._button_names()
        if names:
            return names.get(button) == 'x'

        # Exclude squeeze/grip buttons (typically buttons 6+)
        # These should not trigger any actions
        if button >= 6:
            return False
        
        # Quit: X button (button 2) only
        if button == 2:  # X button
            return True
        
        return False
    
    def get_controller_menu_navigation(self) -> Optional[str]:
        """Get controller menu navigation direction.
        
        Checks d-pad buttons and analog stick Y-axis for up/down navigation.
        
        Returns:
            "up" if d-pad up or stick up is pressed, "down" if d-pad down or stick down is pressed,
            None otherwise.
        """
        if not self.controllers:
            return None
        
        controller = self.controllers[0]
        num_axes = controller.get_numaxes()
        num_buttons = controller.get_numbuttons()
        
        # D-pad as a hat
        for hat in range(controller.get_numhats()):
            hat_y = controller.get_hat(hat)[1]
            if hat_y:
                return "up" if hat_y > 0 else "down"

        # D-pad as buttons: SDL's layout when it knows the pad, else the common 11=up, 12=down
        buttons = {name: raw for raw, name in self._button_names().items()}
        up_button = buttons.get('dpup', 11)
        down_button = buttons.get('dpdown', 12)
        if up_button < num_buttons and controller.get_button(up_button):
            return "up"
        if down_button < num_buttons and controller.get_button(down_button):
            return "down"
        
        # Check analog stick Y-axis (axis 1 for left stick, axis 3 for right stick)
        # Negative Y = up, positive Y = down
        left_stick_y = controller.get_axis(1) if num_axes > 1 else 0.0
        right_stick_y = controller.get_axis(3) if num_axes > 3 else 0.0
        
        # Use whichever stick has input above deadzone (prioritize left stick)
        stick_y = 0.0
        if abs(left_stick_y) > config.CONTROLLER_DEADZONE:
            stick_y = left_stick_y
        elif abs(right_stick_y) > config.CONTROLLER_DEADZONE:
            stick_y = right_stick_y
        
        if abs(stick_y) > config.CONTROLLER_DEADZONE:
            if stick_y < 0:
                return "up"
            elif stick_y > 0:
                return "down"
        
        return None


