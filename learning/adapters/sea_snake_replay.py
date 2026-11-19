"""Exact recorded-action replay of sea snake episodes, with physics/schema checks."""
import math
from types import SimpleNamespace

from learning.replay import check_rules
from learning.sea_snake_rewards import recovered_settings


def metadata_args(path, manifest):
    return SimpleNamespace(horizon=manifest['horizon'], **recovered_settings({'manifest': manifest}))


def episode(job):
    from learning.adapters.sea_snake_scenario import STAGES
    return dict(level=job['level'], seed=job['seed'], lesson=STAGES[job['stage']-1].capitalize())


LESSONS = ('Open', 'Moving', 'Maze')       # demo-video order (the curriculum's)


def showcase(row):
    return 1. if row['success'] else None


def focus(env):
    """Demo videos crop to the snake and its target: (x, y, width)."""
    start, goal = env.snake.get_pos(), env.target.get_pos()
    return (start[0]+goal[0])/2, (start[1]+goal[1])/2, max(12*env.unit, math.dist(start, goal)*1.35+8*env.unit)


def open_environment(path, manifest):
    from learning.adapters import sea_snake_scenario as scenario
    check_rules(path, manifest, scenario.metadata, metadata_args=metadata_args)
    return scenario.environment(horizon=manifest['horizon'], arenas=manifest['arenas'],
                                **scenario.environment_options(manifest))


def draw(env, screen):
    import pygame
    for wall in env.world.maze.walls:
        if wall.active:
            pygame.draw.line(screen, (110, 130, 145), wall.start, wall.end, 3)
    pygame.draw.circle(screen, (100, 200, 240), env.target.get_pos(), env.target.radius, 2)
    env.world.roster.draw(screen)


def status(env, job):
    from learning.adapters.sea_snake_scenario import STAGES
    return [f'SEA SNAKE — {STAGES[env.stage-1]} — {env.frames/60:.1f}s',
            f'Stung {env.success}   Head distance {env.distance():.1f} radii   Solver resets {env.resets}']
