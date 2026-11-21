"""Left-handed keys: I thrusts, J and L pivot, alongside the arrow keys and WASD."""
from collections import defaultdict

import pygame
import pytest

from input.input_handler import InputHandler
from entities.command_recorder import CommandType


@pytest.mark.parametrize('key, command', [
    (pygame.K_i, CommandType.APPLY_THRUST),
    (pygame.K_j, CommandType.ROTATE_LEFT),
    (pygame.K_l, CommandType.ROTATE_RIGHT),
])
def test_the_left_hand_keys_fly_the_ship(key, command):
    pygame.init()
    keys = defaultdict(bool)
    keys[key] = True
    assert command in InputHandler().process_input(keys)
