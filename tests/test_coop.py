"""Campaign co-op: two ships fly one level together as a team."""
import pygame
import pytest

import config
from tests.test_campaign_demo import make_game  # noqa: F401


@pytest.fixture
def coop(make_game):
    game = make_game(autoplay=False)
    game.coop = True
    game.state = config.STATE_PLAYING
    game.start_level()
    game.level_start_time -= 10      # past the opening moment when input is ignored
    return game


def hold(keys_down):
    class Keys:
        def __getitem__(self, key):
            return key in keys_down
    return lambda: Keys()


def step(game, monkeypatch, *keys, frames=1):
    monkeypatch.setattr(pygame.key, 'get_pressed', hold(set(keys)))
    for _ in range(frames):
        game.update(1.0)


def open_exit(game):
    game.exit_opened = True
    game.maze.exit.set_activated(True, game.sound_manager)


def put_in_exit(player, game):
    player.ship.x, player.ship.y = game.maze.exit.get_pos()
    player.ship.vx = player.ship.vy = 0.0


def test_both_ships_start_side_by_side_without_overlapping(coop):
    a, b = (p.ship for p in coop.players)
    assert len(coop.players) == 2
    assert ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 >= a.radius + b.radius
    assert b.palette is not None and a.palette is None      # each in its own colour


def test_enemies_and_hunter_wake_when_either_ship_moves_but_the_drain_starts_per_ship(coop, monkeypatch):
    first, second = coop.players
    assert not coop.player_has_moved
    monkeypatch.setattr(first, 'controller', None)
    step(coop, monkeypatch, pygame.K_UP)       # the keyboard flies Player 1 only
    assert first.has_moved and not second.has_moved and coop.player_has_moved
    now = coop.clock()
    # Player 2 hasn't moved, so its power hasn't drained; Player 1's clock runs from its first move
    assert second.scoring.level_start_time >= now - 0.5
    assert first.scoring.level_start_time <= second.scoring.level_start_time


def test_the_keyboard_does_not_fly_the_second_ship(coop, monkeypatch):
    second_ship = coop.players[1].ship
    before = (second_ship.vx, second_ship.vy, second_ship.angle)
    step(coop, monkeypatch, pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE, frames=5)
    assert (second_ship.vx, second_ship.vy, second_ship.angle) == before and not coop.players[1].has_moved


def test_devices_follow_the_controllers_connected(coop, monkeypatch):
    pads = [object(), object()]
    monkeypatch.setattr(coop.input_handler, 'get_controllers', lambda: pads[:1])
    coop._assign_devices()
    assert (coop.players[0].keyboard, coop.players[0].controller) == (True, None)
    assert (coop.players[1].keyboard, coop.players[1].controller) == (False, 0)
    monkeypatch.setattr(coop.input_handler, 'get_controllers', lambda: pads)
    coop._assign_devices()
    assert (coop.players[0].keyboard, coop.players[0].controller) == (True, 0)
    assert (coop.players[1].keyboard, coop.players[1].controller) == (False, 1)


def test_a_ship_out_of_power_drops_out_and_the_level_carries_on(coop, monkeypatch):
    first, second = coop.players
    first.has_moved = second.has_moved = True
    first.scoring.level_start_time -= 10_000     # all its power drained away
    step(coop, monkeypatch)
    assert first.dropped_out and not first.ship.active and first.ship.is_dying
    assert not second.dropped_out and second.ship.active
    assert not coop.game_over_active


def test_the_level_fails_when_both_ships_run_out_of_power(coop, monkeypatch):
    for player in coop.players:
        player.has_moved = True
        player.scoring.level_start_time -= 10_000
    step(coop, monkeypatch)
    assert all(p.dropped_out for p in coop.players) and coop.game_over_active


def test_a_ship_through_the_portal_leaves_play_with_its_power_frozen(coop, monkeypatch):
    first, second = coop.players
    first.has_moved = second.has_moved = True
    open_exit(coop)
    put_in_exit(first, coop)
    step(coop, monkeypatch)
    assert first.through and not first.ship.active and first.final_power > 0
    assert not coop.exit_explosion_active          # the other is still flying
    frozen = first.final_power
    first.scoring.level_start_time -= 50
    step(coop, monkeypatch)
    assert first.final_power == frozen


def test_the_level_ends_when_every_ship_still_flying_is_through_and_the_team_adds_their_power(coop, monkeypatch):
    first, second = coop.players
    first.has_moved = second.has_moved = True
    open_exit(coop)
    put_in_exit(first, coop)
    step(coop, monkeypatch)
    put_in_exit(second, coop)
    step(coop, monkeypatch)
    assert coop.exit_explosion_active
    assert coop.team_power(coop.clock()) == pytest.approx(first.final_power + second.final_power)
    coop.complete_level(success=True)
    assert coop.level_score_breakdown['final_score'] == pytest.approx(first.final_power + second.final_power)


def test_the_level_ends_when_the_other_ship_dropped_out_after_one_got_through(coop, monkeypatch):
    first, second = coop.players
    first.has_moved = second.has_moved = True
    open_exit(coop)
    put_in_exit(first, coop)
    step(coop, monkeypatch)
    second.scoring.level_start_time -= 10_000
    step(coop, monkeypatch)
    step(coop, monkeypatch)
    assert second.dropped_out and coop.exit_explosion_active and not coop.game_over_active
    assert coop.team_power(coop.clock()) == pytest.approx(first.final_power)


def test_ships_bounce_off_each_other_and_each_pays_the_wall_penalty(coop, monkeypatch):
    first, second = coop.players
    second.ship.x, second.ship.y = first.ship.x + first.ship.radius, first.ship.y
    first.ship.shield_active = second.ship.shield_active = False
    first.ship.shield_initial_timer = second.ship.shield_initial_timer = 0
    first.ship.powerup_shield_timer = second.ship.powerup_shield_timer = 0
    coop._bounce_ships()
    assert first.scoring.wall_collisions == 1 and second.scoring.wall_collisions == 1
    gap = ((first.ship.x - second.ship.x) ** 2 + (first.ship.y - second.ship.y) ** 2) ** 0.5
    assert gap > first.ship.radius


def test_enemies_go_for_the_nearest_living_ship(coop):
    from game_handlers.combat_targets import nearest_friendly
    first, second = coop.players
    foe = type('Foe', (), {'x': second.ship.x + 1, 'y': second.ship.y})()
    assert nearest_friendly(foe, coop.ships) is second.ship
    second.ship.die()
    assert nearest_friendly(foe, coop.ships) is first.ship


def test_a_kill_counts_for_the_player_who_fired_the_shot(coop):
    first, second = coop.players
    shot = type('Shot', (), {'source': 'player', 'owner': second})()
    coop.collision_handler.record_kill(shot, type('Enemy', (), {})())
    assert second.scoring.enemies_destroyed == 1 and first.scoring.enemies_destroyed == 0


def test_the_hunter_is_shown_the_nearest_living_ship_as_its_friend(coop):
    first, second = coop.players
    near_second = type('Spot', (), {'x': second.ship.x + 5, 'y': second.ship.y})()
    assert coop.nearest_ship(near_second) is second.ship
    first.dropped_out = True       # only the second ship is left to follow
    assert coop.nearest_ship(type('Spot', (), {'x': first.ship.x, 'y': first.ship.y})()) is second.ship
    assert coop.hunter_surroundings.friend() is second.ship


def test_the_coop_hud_marks_each_player_in_their_colour_and_dims_a_dropped_out_half(coop):
    coop.draw_game()
    from rendering.coop_hud import HEADER_HEIGHT
    live = coop.screen.get_at((100, HEADER_HEIGHT))[:3]
    coop.players[1].dropped_out = True
    coop.players[1].final_power = 0.0
    coop.draw_game()
    assert live != (0, 0, 0)                                      # Player 1's coloured rule is drawn


def test_single_player_is_unchanged_by_the_coop_code(make_game, monkeypatch):
    game = make_game(autoplay=False)
    game.state = config.STATE_PLAYING
    game.start_level()
    assert len(game.players) == 1 and game.players[0].keyboard and game.players[0].controller == 0
    assert game.ship.palette is None


def test_the_multiplayer_menu_switches_the_mode_and_remembers_it(make_game, tmp_path):
    from user_settings import UserSettings
    game = make_game(autoplay=False)
    game.user_settings = UserSettings(tmp_path / 'player_settings.json')   # never the player's real file
    game.set_mode('compete')
    assert game.multiplayer_menu.rows[0].read() == 'LEVEL COMPETITION'
    game.multiplayer_menu.toggle_selected()
    assert game.coop and game.multiplayer_menu.rows[0].read() == 'CAMPAIGN CO-OP'
    assert UserSettings(tmp_path / 'player_settings.json').multiplayer_mode == 'coop'


def test_start_game_begins_at_level_one_in_co_op_whatever_the_profile(make_game):
    game = make_game(autoplay=False)
    game.profile_manager.get_active_level = lambda: 7
    game.coop = False
    assert game.first_level() == 7
    game.coop = True
    assert game.first_level() == 1


def test_the_multiplayer_menu_says_when_player_two_has_no_controller(make_game, monkeypatch):
    game = make_game(autoplay=False)
    game.coop = True
    monkeypatch.setattr(game.input_handler, 'get_controllers', lambda: [])
    lines, warning = game.multiplayer_menu.devices()
    assert lines[1] == ('PLAYER 2', 'NO CONTROLLER') and warning
    monkeypatch.setattr(game.input_handler, 'get_controllers', lambda: [object()])
    assert game.multiplayer_menu.devices() == ([('PLAYER 1', 'KEYBOARD'), ('PLAYER 2', 'CONTROLLER')], None)


class _Pilot:
    """A stand-in Campaign Hunter: flies straight ahead and fires; remembers whose ship it was asked about."""

    def __init__(self):
        self.asked = []

    def reset(self):
        pass

    def decide(self, game, player=None):
        from hunter.campaign_pilot import ShipCommand
        self.asked.append(player)
        return ShipCommand(0, True, False)


def test_the_campaign_hunter_can_fly_player_two_with_no_controller(coop, monkeypatch):
    pilot = _Pilot()
    monkeypatch.setattr(coop, '_hunter_driver', lambda: pilot)
    coop.user_settings.player2 = 'hunter'
    monkeypatch.setattr(coop.input_handler, 'get_controllers', lambda: [])
    coop.start_level()
    coop.level_start_time -= 10
    first, second = coop.players
    assert second.driver is pilot and second.controller is None and not second.keyboard
    before = (second.ship.x, second.ship.y)
    step(coop, monkeypatch, frames=10)
    assert second.has_moved and pilot.asked and all(p is second for p in pilot.asked)
    assert (second.ship.x, second.ship.y) != before and not first.has_moved     # the hunter moved, the keyboard did not


def test_the_hunter_as_player_two_is_flown_with_its_own_power_and_friend(make_game, monkeypatch):
    from hunter.campaign_pilot import CampaignPilot
    from hunter.campaign_senses import CampaignSurroundings, ShipBody
    game = make_game(autoplay=False)
    game.coop = True
    game.state = config.STATE_PLAYING
    game.start_level()
    first, second = game.players
    second.scoring.record_wall_collision()
    body = ShipBody(second.ship, game, CampaignPilot.__init__.__globals__['hunter_settings_for']('neural'), player=second)
    assert body.player is second
    assert CampaignSurroundings(game, second).friend() is first.ship      # the other ship is a friend
    assert CampaignSurroundings(game, first).friend() is second.ship
    second.dropped_out = True
    assert CampaignSurroundings(game, first).friend() is game.hunter      # a ship out of play is no friend to follow


def test_the_menu_chooses_a_hunter_for_player_two_and_remembers_it(make_game, tmp_path):
    from user_settings import UserSettings
    game = make_game(autoplay=False)
    game.user_settings = UserSettings(tmp_path / 'player_settings.json')
    game.coop = True
    row = game.multiplayer_menu.rows[1]
    assert row.read() == 'HUMAN'
    row.toggle()
    assert row.read() == 'CAMPAIGN HUNTER' and UserSettings(tmp_path / 'player_settings.json').player2 == 'hunter'
    lines, warning = game.multiplayer_menu.devices()
    assert lines[1] == ('PLAYER 2', 'CAMPAIGN HUNTER') and warning is None


def test_the_real_campaign_pilot_flies_player_two_through_frames_of_a_co_op_level(make_game, monkeypatch):
    from hunter.campaign_pilot import CampaignPilot
    from tests.test_campaign_demo import StubPolicy
    game = make_game(autoplay=False, policy=StubPolicy((1, True, False)))    # turn right and thrust
    game.coop = True
    game.user_settings.player2 = 'hunter'
    monkeypatch.setattr(game.input_handler, 'get_controllers', lambda: [])
    game.state = config.STATE_PLAYING
    game.start_level()
    game.level_start_time -= 10
    first, second = game.players
    assert isinstance(second.driver, CampaignPilot) and second.controller is None
    monkeypatch.setattr(pygame.key, 'get_pressed', hold(set()))
    for _ in range(300):
        game.update(1.0)
    assert second.has_moved                       # the hunter set off by itself, with no controller
    assert not first.has_moved or first.keyboard  # nobody touched the keyboard
    assert not game.game_over_active or all(p.dropped_out for p in game.players)


# --- Campaign race ---------------------------------------------------------------------------------------------

@pytest.fixture
def race(coop):
    coop.race = True
    coop.new_race()
    return coop


def test_the_first_ship_through_wins_the_level_and_the_point_and_the_level_ends_there(race, monkeypatch):
    first, second = race.players
    first.has_moved = second.has_moved = True
    open_exit(race)
    put_in_exit(second, race)           # Player 2 gets there first, while Player 1 is still flying
    step(race, monkeypatch)
    assert race.race_points == [0, 1] and race.exit_explosion_active and first.flying
    put_in_exit(first, race)            # arriving during the finish adds nothing
    step(race, monkeypatch)
    assert race.race_points == [0, 1] and race.race_winner == 1


def test_if_one_player_runs_out_of_power_the_other_can_still_win(race, monkeypatch):
    first, second = race.players
    first.has_moved = second.has_moved = True
    first.scoring.level_start_time -= 10_000
    step(race, monkeypatch)
    assert first.dropped_out and not race.game_over_active and race.race_points == [0, 0]
    open_exit(race)
    put_in_exit(second, race)
    step(race, monkeypatch)
    assert race.race_points == [0, 1] and race.exit_explosion_active


def test_nobody_scores_when_both_run_out_and_the_level_is_replayed(race, monkeypatch):
    for player in race.players:
        player.has_moved = True
        player.scoring.level_start_time -= 10_000
    step(race, monkeypatch)
    assert race.game_over_active and race.race_points == [0, 0]
    race.complete_level(success=False)
    assert race.race_summary == [("WINNER", "NOBODY SCORES"), ("POINTS", "0 – 0")]
    assert race.level_complete_menu.menu_options[0] == "RETRY LEVEL"


def test_a_won_race_reports_the_winner_and_points_and_no_score_or_stars(race, monkeypatch):
    first, second = race.players
    first.has_moved = second.has_moved = True
    open_exit(race)
    put_in_exit(first, race)
    step(race, monkeypatch)
    race.complete_level(success=True)
    assert race.race_summary == [("WINNER", "PLAYER 1"), ("POINTS", "1 – 0")] and race.star_animation is None
    race.draw()      # the level-complete screen draws with the race figures


def test_a_race_cannot_be_restarted_and_a_new_run_starts_the_points_again(race, monkeypatch):
    from game_handlers.state_handlers import StateHandlerRegistry
    race.race_points = [3, 2]
    restarted = []
    monkeypatch.setattr(race, 'restart_level', lambda: restarted.append(True))
    handler = StateHandlerRegistry().get_handler(config.STATE_PLAYING)
    handler.handle_keyboard(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r), race)
    assert not restarted
    race.race = False
    handler.handle_keyboard(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r), race)
    assert restarted
    race.new_race()
    assert race.race_points == [0, 0]


def test_the_race_hud_shows_each_players_points_and_no_restart_hint(race):
    race.race_points = [2, 1]
    race.draw_game()      # draws without a team score or the restart hint


def test_the_race_works_with_the_campaign_hunter_as_the_rival(make_game, monkeypatch):
    from tests.test_campaign_demo import StubPolicy
    game = make_game(autoplay=False, policy=StubPolicy((1, True, False)))
    game.set_mode('race')
    game.user_settings.player2 = 'hunter'
    monkeypatch.setattr(game.input_handler, 'get_controllers', lambda: [])
    game.state = config.STATE_PLAYING
    game.start_level()
    assert game.race and game.players[1].driver is not None and game.players[1].controller is None


def test_the_mode_cycles_through_every_mode_and_is_remembered(make_game):
    from user_settings import UserSettings
    game = make_game(autoplay=False)
    game.set_mode('compete')
    names = []
    for _ in range(3):
        game.multiplayer_menu.toggle_selected()
        names.append(game.multiplayer_menu.rows[0].read())
    assert names == ['CAMPAIGN CO-OP', 'CAMPAIGN RACE', 'LEVEL COMPETITION']
    for _ in range(2):
        game.multiplayer_menu.toggle_selected()
    assert game.mode == 'race' and UserSettings().multiplayer_mode == 'race'


# --- per-player recordings, and the Replay enemies that copy them -----------------------------------------------------

def test_each_player_records_their_own_commands(coop, monkeypatch):
    first, second = coop.players
    assert first.recorder is not second.recorder
    monkeypatch.setattr(first, 'controller', None)
    step(coop, monkeypatch, pygame.K_UP, frames=3)          # the keyboard flies Player 1 only
    assert first.recorder.get_command_count() == 3
    assert second.recorder.get_command_count() == 3         # an idle ship records NO_ACTION each frame
    from entities.command_recorder import CommandType
    assert CommandType.APPLY_THRUST in first.recorder.get_replay_commands()
    assert CommandType.APPLY_THRUST not in second.recorder.get_replay_commands()


def test_replay_enemies_alternate_between_the_players_in_spawn_order(coop):
    from game_handlers.spawn_manager import make_enemy
    first, second = coop.players
    replays = [make_enemy('replay', (600.0 + 30 * i, 400.0), coop.level, coop.command_recorder) for i in range(4)]
    assert [r.recorder for r in replays] == [first.recorder, second.recorder, first.recorder, second.recorder]


def test_single_players_replay_enemy_copies_the_one_recording(make_game):
    from game_handlers.spawn_manager import make_enemy
    game = make_game(autoplay=False)
    game.state = config.STATE_PLAYING
    game.start_level()
    assert game.command_recorder is game.players[0].recorder
    enemy = make_enemy('replay', (600.0, 400.0), game.level, game.command_recorder)
    assert enemy.recorder is game.players[0].recorder


def test_a_new_level_starts_every_recording_afresh(coop, monkeypatch):
    step(coop, monkeypatch, pygame.K_UP, frames=3)
    coop.start_level()
    assert all(p.recorder.get_command_count() == 0 for p in coop.players)


def _press(game, code):
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=code, unicode=''))
    game.handle_events()


def _open_from_main_menu(game, option):
    game.state = config.STATE_MENU
    game.main_menu.menu_selected_index = game.main_menu.menu_options.index(option)
    _press(game, pygame.K_RETURN)


def test_single_player_offers_start_game_and_levels_in_one_ship_mode(make_game):
    game = make_game()
    assert "START GAME" not in game.main_menu.menu_options and "LEVELS" not in game.main_menu.menu_options
    game.set_mode('coop')                                       # whatever multiplayer was left as
    _open_from_main_menu(game, "SINGLE PLAYER")
    assert game.state == config.STATE_SINGLE_PLAYER and game.mode == 'single'
    assert [row.read_label() for row in game.single_player_menu.rows] == ["START GAME", "LEVELS"]
    _press(game, pygame.K_RETURN)                               # START GAME is ready
    assert game.state == config.STATE_PLAYING and len(game.players) == 1
    _open_from_main_menu(game, "SINGLE PLAYER")
    _press(game, pygame.K_DOWN)
    _press(game, pygame.K_RETURN)
    assert game.state == config.STATE_LEVEL_SELECT
    _press(game, pygame.K_ESCAPE)
    assert game.state == config.STATE_SINGLE_PLAYER             # back to where LEVELS lives
    _press(game, pygame.K_ESCAPE)
    assert game.state == config.STATE_MENU


def test_multiplayer_opens_in_the_remembered_mode_and_stepping_never_starts_play(make_game):
    game = make_game()
    _open_from_main_menu(game, "MULTIPLAYER")
    assert game.state == config.STATE_MULTIPLAYER and game.mode == game.user_settings.multiplayer_mode != 'single'
    assert game.mode == 'compete' and len(game.multiplayer_menu.rows) == 4     # the length and Player 2 are asked
    _press(game, pygame.K_RIGHT)                                # the start row is selected: stepping does nothing
    assert game.state == config.STATE_MULTIPLAYER
    _press(game, pygame.K_RETURN)
    assert game.state == config.STATE_TURN
