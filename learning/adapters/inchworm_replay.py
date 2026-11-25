"""Exact recorded-action replay with physics/schema checks."""
import math
from types import SimpleNamespace
from learning.replay import check_rules
from learning.inchworm_rewards import recovered_settings


def metadata_args(path,manifest):
    return SimpleNamespace(horizon=manifest['horizon'],**recovered_settings({'manifest':manifest}))


def episode(job):
    from learning.adapters.inchworm_scenario import STAGES
    return dict(level=job['level'],seed=job['seed'],lesson=STAGES[job['stage']-1].capitalize())


LESSONS = ('Crawl','Rear','Curl','Turn','About-face','Gap','Reach','Feed','Recover','Leap','Pursue')           # demo-video order (the curriculum's)
CLIP_SECONDS = {'Crawl':(8.,25.),'Recover':(6.,40.),'Rear':(1.5,20.)}      # readable clip lengths; others use the default


def showcase(row):
    return 1. if row['success'] else None


def focus(env):
    """Demo videos crop to the worm and its target: (x, y, width)."""
    start, goal = env.worm.get_pos(), env.world.worm_goal
    if goal is None and env.prey is not None:
        goal = env.prey.get_pos()
    if goal is None and env.pursuit_target is not None:
        goal = env.pursuit_target.get_pos()
    if goal is None:
        return start[0],start[1],420.
    return (start[0]+goal[0])/2,(start[1]+goal[1])/2,max(300.,math.dist(start,goal)*1.35+120)


def open_environment(path,manifest):
    from learning.adapters import inchworm_scenario as scenario
    check_rules(path,manifest,scenario.metadata,metadata_args=metadata_args)
    return scenario.environment(horizon=manifest['horizon'],arenas=manifest['arenas'],
                                **scenario.environment_options(manifest))


def draw(env,screen):
    import pygame
    for wall in env.world.maze.walls:
        if wall.active:
            pygame.draw.line(screen,(110,130,145),wall.start,wall.end,3)
    goal = env.world.worm_goal
    if goal is not None:
        pygame.draw.circle(screen,(100,200,240),goal,12 if env.stage in (6,7) else 14 if env.stage == 8 else 10,2)
    if env.stage == 7 and env.curl_anchor is not None:
        pygame.draw.circle(screen,(230,180,80),env.curl_anchor.foot,4,2)
    if env.pursuit_target is not None:
        env.pursuit_target.draw(screen)
    env.world.roster.draw(screen)


def status(env,job):
    from learning.adapters.inchworm_scenario import STAGES
    lines = [f'INCHWORM — {STAGES[env.stage-1]} — {env.frames/60:.1f}s',
             f'Bites {env.bites}   Goal distance {env.distance():.1f}   Success {env.success}']
    if env.stage == 7:
        lines.append('Starting rear anchor: '+('lost — attempt ended' if env.curl_anchor_lost else 'held'))
    if env.stage == 8:
        grounded = env.worm.worm_front_grip or env.worm.worm_rear_grip
        lines.append(f'Head distance < 14 px; planted grip: {"yes" if grounded else "no"}')
    return lines
