"""The sea snake in an Encounter: chasing, being shot link by link, stinging, and surviving a failing solver."""
import copy
import math
from types import SimpleNamespace

import pygame

import creatures
from entities.projectile import Projectile
from physics.sea_snake import SeaSnakeCommand
from tests.creature_parts import encounter, open_maze, world


def ship_at(pos, shielded=False):
    return SimpleNamespace(x=pos[0], y=pos[1], active=True, radius=8., get_pos=lambda: pos,
                           is_shield_active=lambda: shielded)


def setting(ship_pos=None, shielded=False, walls=()):
    # Born level (heading +x): an unset angle is random, which once made these tests flaky.
    c = creatures.build('sea_snake', (400., 286.), angle=0.)
    ship = ship_at(ship_pos, shielded) if ship_pos else None
    enc = encounter(c, world_=world(ship=ship, maze=open_maze(list(walls))))
    enc.ready()
    return c, enc, ship


def test_snake_swims_toward_the_ship():
    c, enc, ship = setting((700., 286.))
    start = math.dist(c.get_pos(), ship.get_pos())
    for _ in range(300):
        enc.step(1.)
    assert math.dist(c.get_pos(), ship.get_pos()) < start-100.


def test_shot_kills_after_its_hit_points_and_hits_any_link():
    c, enc, _ = setting()
    tail = c.snake_pose[0][0]
    assert math.dist(tail, c.get_pos()) > c.radius
    shot = Projectile(tail, 0.)
    hit_points = c.hit_points
    assert enc.shoot(shot)
    assert c.hit_points == hit_points-1 and not shot.active
    # Beside the body, just clear of its skin and the shot's own radius, at whatever heading it was born.
    clear = c.snake_skin+shot.radius+4.
    heading = math.radians(c.angle)
    miss = Projectile((c.x-math.sin(heading)*clear, c.y+math.cos(heading)*clear), 0.)
    assert not enc.shoot(miss) and miss.active
    for _ in range(hit_points-1):
        assert enc.shoot(Projectile(c.snake_pose[3][0], 0.))
    assert not c.active


def test_a_shot_knocks_the_snake_along_and_it_glides():
    def drift(shoot):
        c, enc, _ = setting()
        c.snake_command = SeaSnakeCommand(muscles=(0.,)*7)      # limp, so only the knock moves it
        enc.step(1.)
        start = c.x
        shot = Projectile(c.snake_pose[0][0], 0.)
        if shoot:
            assert enc.shoot(shot)
        for _ in range(60):
            enc.step(1.)
        return (c.x-start)*math.copysign(1., shot.vx)
    assert drift(shoot=False) < 5.
    assert drift(shoot=True) > 40.


def test_touching_the_ship_stings_once_per_cooldown():
    c, enc, _ = setting((400., 286.))
    c.snake_command = SeaSnakeCommand()
    for _ in range(100):
        enc.step(1.)
    assert enc.world.scoring.record_enemy_collision.call_count == 3
    c2, enc2, _ = setting((400., 286.), shielded=True)
    for _ in range(100):
        enc2.step(1.)
    assert enc2.world.scoring.record_enemy_collision.call_count == 0


def test_a_sting_makes_the_ship_flinch_and_throws_it_clear():
    c, enc, ship = setting((400., 270.))
    ship.vx = ship.vy = 0.
    ship.flinch = lambda: setattr(ship, 'flinched', True)
    c.snake_command = SeaSnakeCommand()
    enc.step(1.)
    assert ship.flinched
    assert math.hypot(ship.vx, ship.vy) > 2.


def test_a_sting_lands_through_any_link_not_just_the_centre():
    c, enc, _ = setting((700., 286.))
    c.snake_command = SeaSnakeCommand()
    tail = c.snake_pose[0][0]
    enc.world.ship = ship_at(tail)
    enc.step(1.)
    assert enc.world.scoring.record_enemy_collision.call_count == 1


def test_a_snake_whose_physics_blows_up_does_not_take_the_game_down(monkeypatch):
    from physics.sea_snake import SeaSnakePhysics
    c, enc, _ = setting((700., 286.))

    def explode(self, *args, **kwargs):
        raise RuntimeError('sea snake physics exceeded its stable substep budget')
    monkeypatch.setattr(SeaSnakePhysics, 'step', explode)
    for _ in range(10):
        enc.step(1.)
    assert c.snake_resets >= 1
    assert not c.active
    for _ in range(int(c.body.DEATH_DURATION)):
        c.update_death(1.)


def test_death_portrait_deepcopy_excludes_solver():
    c, enc, _ = setting((700., 286.))
    enc.step(1.)
    portrait = copy.deepcopy(c)
    assert portrait.snake_physics is None and portrait.snake_driver is None
    screen = pygame.Surface((800, 600))
    c.body.draw(screen, c)
    portrait.body.draw(screen, portrait)
    enc.kill(c)
    c.update_death(1.)
    c.body.draw_death(screen, c)
    enc.close()
    enc.close()
    assert c.snake_physics is None


def test_snake_born_against_a_wall_stays_inside_the_maze():
    import enemy_kinds
    reach = 400.+enemy_kinds.kind('sea_snake').tuning['size']*2.8   # where a tip lies when it is born level
    wall = SimpleNamespace(start=(reach, 0.), end=(reach, 600.), active=True)
    c, enc, _ = setting((reach+300., 286.), walls=[wall])
    for _ in range(200):
        enc.step(1.)
    points = [p for segment in c.snake_pose for p in segment]
    assert all(math.isfinite(v) for p in points for v in p)
    assert all(p[0] <= reach+.5 for p in points)


def test_a_snake_killed_on_the_move_still_stops_at_rock():
    from creatures.traits.sea_snake import snapshot
    wall = SimpleNamespace(start=(560., 0.), end=(560., 600.), active=True)
    c, enc, _ = setting(walls=[wall])
    c.snake_physics.impulse((6.*c.snake_physics.mass, 0.))      # swimming hard at the wall when it dies
    snapshot(c)
    enc.kill(c)
    for _ in range(90):
        c.body.update_death(c, 1.)
    assert all(p[0] <= 560.5 for segment in c.snake_pose for p in segment)


def test_no_ship_means_no_errors():
    c, enc, _ = setting()
    for _ in range(100):
        enc.step(1.)
    assert all(math.isfinite(v) for v in (c.x, c.y, c.vx, c.vy))
    assert c.active


def test_the_body_draws_from_any_pose_without_failing(monkeypatch):
    import config
    c, enc, _ = setting()
    screen = pygame.Surface((800, 600))
    point = (c.x, c.y)
    for pose in (c.snake_pose, None, tuple((point, point) for _ in c.snake_pose)):
        c.snake_pose = pose
        for low in (False, True):
            monkeypatch.setattr(config.SETTINGS.game, 'lowGraphics', low)
            screen.fill(config.COLOR_BACKGROUND)
            c.body.draw(screen, c)
    c.snake_pose = None
    monkeypatch.setattr(config.SETTINGS.game, 'lowGraphics', False)
    screen.fill(config.COLOR_BACKGROUND)
    c.body.draw(screen, c)
    assert screen.get_at((int(c.x), int(c.y)))[:3] != tuple(config.COLOR_BACKGROUND)[:3]


def test_a_dying_snake_bursts_link_by_link_and_its_blast_goes_off_at_the_finale():
    import config
    from entities.sea_snake import FINALE
    from physics.sea_snake import LINKS
    c, enc, _ = setting()
    enc.kill(c)
    (blast,), _ = enc.world.collision_handler.explosions.trigger.call_args
    assert blast.fuse == FINALE and blast.damage > 0 and blast.radius > c.radius
    screen = pygame.Surface((800, 600))
    shown = []
    for frame in range(int(c.DEATH_DURATION)):
        c.update_death(1.)
        screen.fill(config.COLOR_BACKGROUND)
        c.draw_death(screen)
        shown.append(c.body.gone)
    assert shown == sorted(shown) and shown[0] == 0 and shown[-1] == LINKS      # one after another, head to tail
    assert c.body.finale and not c.body.bits and not c.body.rings      # the embers and rings have all burnt out
    # one bang for each link (the kill's own bang is made by the Encounter)
    assert enc.world.sound_manager.play_enemy_destroy.call_count >= LINKS


def test_a_snake_is_born_curved_not_as_a_straight_rod():
    c, enc, _ = setting()
    heads = [math.atan2(b[1]-a[1], b[0]-a[0]) for a, b in c.snake_pose]
    spread = max(heads)-min(heads)
    assert spread > .5              # the links point different ways, so it lies in a curve
    for a, b in zip(c.snake_pose, c.snake_pose[1:]):
        assert math.dist(a[1], b[0]) < 1.       # and is still joined end to end


def test_a_snake_that_jams_now_and_then_lives_but_one_that_keeps_jamming_dies(monkeypatch):
    from physics.sea_snake import SeaSnakePhysics

    def jam(self, *a, **k):
        raise RuntimeError('jammed')
    c, enc, _ = setting()
    real = SeaSnakePhysics.step
    for _ in range(3):                       # three jams, each long after the last: all forgiven
        monkeypatch.setattr(SeaSnakePhysics, 'step', jam)
        enc.step(1.)
        monkeypatch.setattr(SeaSnakePhysics, 'step', real)
        for _ in range(700):
            c.snake_calm += 1.
        enc.step(1.)
    assert c.active
    monkeypatch.setattr(SeaSnakePhysics, 'step', jam)
    for _ in range(5):                       # but jam after jam kills it
        enc.step(1.)
    assert not c.active or c.is_dying


def test_a_small_snake_is_hurried_so_it_covers_more_of_its_own_lengths(monkeypatch):
    monkeypatch.setenv('SEA_SNAKE_POLICY', '')      # the scripted wave: the same swimming for every size

    def lengths_covered(kind):
        c = creatures.build(kind, (400., 286.), angle=0.)
        enc = encounter(c, world_=world(ship=ship_at((4000., 286.)), maze=open_maze([])))
        enc.ready()
        for _ in range(300):
            enc.step(1.)
        return math.dist((400., 286.), c.get_pos())/c.radius
    assert lengths_covered('teeny_sea_snake') > lengths_covered('sea_snake')*1.3


def test_snakes_are_solid_to_one_another_whatever_their_size():
    from creatures.traits.sea_snake import _closest

    def deepest_overlap(first, second):
        a = creatures.build(first, (300., 286.), angle=0.)
        b = creatures.build(second, (500., 286.), angle=180.)
        # Both are lured to the middle, where they would pass straight through one another.
        enc = encounter(a, b, world_=world(ship=ship_at((400., 286.)), maze=open_maze([])))
        enc.ready()
        deepest = 0.
        for _ in range(300):
            enc.step(1.)
            for p0, p1 in a.snake_pose:
                for q0, q1 in b.snake_pose:
                    near, far = _closest(p0, p1, q0, q1)
                    deepest = max(deepest, a.snake_skin+b.snake_skin-math.dist(near, far))
        return deepest/(a.snake_skin+b.snake_skin)      # 1 would be the two centre lines on top of each other
    for first, second in (('sea_snake', 'sea_snake'), ('sea_snake', 'mini_sea_snake'), ('mini_sea_snake', 'teeny_sea_snake')):
        assert deepest_overlap(first, second) < .8     # a brush at speed may sink in for a frame, but the bodies never pass through one another
