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
