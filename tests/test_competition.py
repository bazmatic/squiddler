"""Level competition: two players take turns on one level, each a single attempt; the higher score wins it."""
from rendering.competition_screens import score_rows
import pygame
import pytest

import config
from game_handlers.competition import Competition
from tests.test_campaign_demo import make_game  # noqa: F401
from tests.test_coop import hold, open_exit, step


# --- the pair of turns, on its own ----------------------------------------------------------------------------

def test_the_higher_score_wins_the_level_and_the_point():
    c = Competition(3, False)
    c.record(40)
    assert c.current == 1 and not c.finished and c.winner is None
    c.record(65)
    assert c.finished and c.winner == 1 and c.points == [0, 1]


def test_a_tie_gives_nobody_the_point():
    c = Competition(3, False)
    c.record(50)
    c.record(50)
    assert c.winner is None and c.points == [0, 0] and not c.both_out


def test_a_player_who_ran_out_of_power_scores_zero_and_loses_to_one_who_got_through():
    c = Competition(3, False)
    c.record(0)
    c.record(12)
    assert c.winner == 1 and not c.both_out


def test_both_out_scores_nobody_and_a_campaign_replays_the_level_with_the_same_first_player():
    c = Competition(3, False, first=1)
    c.record(0)
    c.record(0)
    assert c.both_out and c.winner is None and c.points == [0, 0]
    again = c.next_round()
    assert (again.level, again.first) == (3, 1) and again.scores == [None, None]


def test_the_first_turn_alternates_by_level_and_the_points_carry_on():
    c = Competition(1, False)
    assert c.order == (0, 1)
    c.record(70)
    c.record(30)
    nxt = c.next_round()
    assert (nxt.level, nxt.order, nxt.points) == (2, (1, 0), [1, 0])
    assert nxt.current == 1


def test_a_level_picked_from_levels_is_played_once_then_back_to_the_list():
    c = Competition(7, True)
    c.record(10)
    c.record(20)
    assert c.next_round() is None


# --- in the game ------------------------------------------------------------------------------------------------

@pytest.fixture
def compete(make_game):
    game = make_game(autoplay=False)
    game.set_mode('compete')
    return game


def fly_to_the_exit(game, monkeypatch):
    game.level_start_time -= 10
    game.player_has_moved = True
    open_exit(game)
    game.ship.x, game.ship.y = game.maze.exit.get_pos()
    game.ship.vx = game.ship.vy = 0.0
    step(game, monkeypatch, frames=3)
    for _ in range(120):
        game.update(1.0)
        if game.state != config.STATE_PLAYING:
            break


def run_out_of_power(game, monkeypatch):
    game.level_start_time -= 10
    game.player_has_moved = True
    game.scoring.level_start_time -= 10_000
    step(game, monkeypatch, frames=2)
    game.complete_level(success=False)


def test_start_game_begins_with_player_ones_turn_on_level_one(compete):
    compete.begin_play(compete.first_level())
    assert compete.state == config.STATE_TURN and compete.competition.current == 0 and compete.competition.level == 1


def test_both_turns_face_the_same_enemy_layout(compete):
    compete.begin_play(4)
    layouts = []
    for _ in range(2):
        compete.start_turn()
        layouts.append(sorted((e.type, round(e.x), round(e.y)) for e in compete.roster.all()))
    assert layouts[0] == layouts[1] and layouts[0]


def test_a_turn_ends_on_reaching_the_exit_and_the_other_player_goes_next(compete, monkeypatch):
    compete.begin_play(3)
    compete.start_turn()
    fly_to_the_exit(compete, monkeypatch)
    assert compete.state == config.STATE_TURN and compete.competition.current == 1
    assert compete.competition.scores[0] > 0


def test_after_both_turns_the_results_name_the_winner_and_levels_returns_to_the_list(compete, monkeypatch):
    compete.begin_play(3, from_levels=True)
    compete.start_turn()
    run_out_of_power(compete, monkeypatch)             # Player 1 scores nothing
    assert compete.state == config.STATE_TURN and compete.competition.scores[0] == 0
    compete.start_turn()
    fly_to_the_exit(compete, monkeypatch)              # Player 2 gets through
    assert compete.state == config.STATE_COMPETITION_RESULT and compete.competition.winner == 1
    compete.draw()
    compete.leave_competition_result()
    assert compete.state == config.STATE_LEVEL_SELECT and compete.competition is None


def test_the_results_break_each_score_down_into_what_made_it(compete, monkeypatch):
    compete.begin_play(3, from_levels=True)
    compete.start_turn()
    fly_to_the_exit(compete, monkeypatch)
    compete.start_turn()
    fly_to_the_exit(compete, monkeypatch)
    competition = compete.competition
    assert [b['final_score'] for b in competition.breakdowns] == pytest.approx(competition.scores)
    compete.draw()


def test_only_the_parts_of_the_score_either_player_had_are_listed():
    ours = {"time_penalty": 30, "wall_collision_penalty": 4, "enemy_destruction_bonus": 15}
    theirs = {"time_penalty": 70, "collision_penalty": 40}
    assert score_rows([ours, theirs]) == [
        ("TIME", -30, -70), ("ENEMY HITS", 0, -40), ("WALL BUMPS", -4, 0), ("KILLS", 15, 0)]


def test_a_campaign_goes_on_to_the_next_level_with_the_other_player_first(compete, monkeypatch):
    compete.begin_play(1)
    compete.start_turn(); fly_to_the_exit(compete, monkeypatch)
    compete.start_turn(); run_out_of_power(compete, monkeypatch)
    assert compete.competition.winner == 0 and compete.competition.points == [1, 0]
    compete.leave_competition_result()
    assert compete.state == config.STATE_TURN
    assert (compete.competition.level, compete.competition.current) == (2, 1)


def test_both_out_replays_the_same_level(compete, monkeypatch):
    compete.begin_play(2)
    compete.start_turn(); run_out_of_power(compete, monkeypatch)
    compete.start_turn(); run_out_of_power(compete, monkeypatch)
    assert compete.competition.both_out
    compete.leave_competition_result()
    assert compete.competition.level == 2 and compete.state == config.STATE_TURN


def test_a_turn_cannot_be_restarted_and_never_touches_the_profile(compete, monkeypatch):
    from game_handlers.state_handlers import StateHandlerRegistry
    compete.begin_play(3)
    compete.start_turn()
    before = compete.profile_manager.get_active_level()
    restarted = []
    monkeypatch.setattr(compete, 'restart_level', lambda: restarted.append(True))
    handler = StateHandlerRegistry().get_handler(config.STATE_PLAYING)
    handler.handle_keyboard(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r), compete)
    assert not restarted
    fly_to_the_exit(compete, monkeypatch)
    assert compete.profile_manager.get_active_level() == before


def test_every_device_flies_whichever_ship_is_flying(compete, monkeypatch):
    pads = [object(), object()]
    monkeypatch.setattr(compete.input_handler, 'get_controllers', lambda: pads)
    compete.begin_play(3)
    compete.start_turn()
    me = compete.players[0]
    assert me.keyboard and compete._controllers_of(me) == [0, 1]


def test_the_hunter_takes_player_twos_turn_when_chosen(compete, monkeypatch):
    from tests.test_coop import _Pilot
    pilot = _Pilot()
    monkeypatch.setattr(compete, '_hunter_driver', lambda: pilot)
    compete.user_settings.player2 = 'hunter'
    compete.begin_play(3)
    compete.start_turn()
    assert compete.players[0].driver is None            # Player 1 is a person
    compete.competition.turn = 1
    compete.start_turn()
    assert compete.players[0].driver is pilot and compete.competition.current == 1


def test_the_turn_and_result_screens_draw(compete):
    compete.begin_play(3)
    compete.draw()
    compete.competition.record(40)
    compete.draw()
    compete.competition.record(25)
    compete.state = config.STATE_COMPETITION_RESULT
    compete.draw()


def test_the_mode_menu_offers_level_competition(compete):
    compete.set_mode('single')
    compete.multiplayer_menu.toggle_selected()
    assert compete.multiplayer_menu.rows[0].read() == 'LEVEL COMPETITION' and compete.mode == 'compete'


# --- the multiplayer menu's promises (#46) -----------------------------------------------------------------------

def test_a_simultaneous_mode_will_not_start_without_a_controller_for_player_two(make_game, monkeypatch):
    game = make_game(autoplay=False)
    monkeypatch.setattr(game.input_handler, 'get_controllers', lambda: [])
    for mode in ('coop', 'race'):
        game.set_mode(mode)
        game.state = config.STATE_MENU
        game.begin_play(1)
        assert game.state == config.STATE_MULTIPLAYER           # the screen that says a controller is needed
    game.user_settings.player2 = 'hunter'                        # a hunter needs none
    game.begin_play(1)
    assert game.state == config.STATE_PLAYING and len(game.players) == 2
    game.set_mode('compete')                                     # one ship at a time: no controller needed
    game.user_settings.player2 = 'human'
    game.begin_play(1)
    assert game.state == config.STATE_TURN


def test_the_profile_file_is_never_written_in_a_multiplayer_mode(make_game, monkeypatch):
    from tests.test_campaign_demo import profile_bytes
    from tests.test_coop import put_in_exit
    game = make_game(autoplay=False)
    monkeypatch.setattr(game.input_handler, 'get_controllers', lambda: [object()])
    before = profile_bytes(make_game)
    for mode in ('coop', 'race', 'compete'):
        game.set_mode(mode)
        game.begin_play(1)
        if mode == 'compete':
            for _ in range(2):
                game.start_turn()
                fly_to_the_exit(game, monkeypatch)
            assert game.state == config.STATE_COMPETITION_RESULT
            continue
        for player in game.players:
            player.has_moved = True
        open_exit(game)
        for player in game.players:
            put_in_exit(player, game)
        step(game, monkeypatch, frames=3)
        game.complete_level(success=True)
        assert game.state == config.STATE_LEVEL_COMPLETE
    assert profile_bytes(make_game) == before
