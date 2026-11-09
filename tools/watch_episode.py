"""Replay a training episode: list an iteration's episodes by how much the hunter panicked, or watch one in a window.

    python -m tools.watch_episode RUN ITERATION            # fly the iteration's episodes, the most panic first
    python -m tools.watch_episode RUN ITERATION EPISODE    # watch that episode in the game window (Esc to stop)

Episodes are not recorded but they are reproducible: each is fixed by its level, its seed, and the checkpoint and dice
seed it flew with, which the run's config.json and checkpoints give back. RUN is a run directory written by
tools.ppo_neural (it needs the 'shaping' its config.json has kept since panic was added). A Campaign Hunter run's
episodes replay the same way; listing them shows how each ended rather than panic.
"""
import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CAMPAIGN = 'campaign_hunter'


def episode_jobs(run, iteration):
    """The iteration's jobs, (level, seed, spec) each, exactly as training made them, and the run's shaping. A
    Campaign Hunter job also names its task, (level, seed, spec, 'campaign_hunter'), as training's do.

    Iteration 0 flew the run's start, which a campaign run first migrated to its senses (into iter_start.npz, which
    training deletes when the run ends): its jobs name the start, and flying one migrates it again."""
    from tools.ppo_neural import make_jobs
    config = json.loads((Path(run) / 'config.json').read_text())
    if 'shaping' not in config:
        raise SystemExit(f'{run}/config.json has no shaping: the run predates replays')
    episode = config['episode']
    weights = Path(run) / f'iter{iteration - 1:04d}.npz' if iteration else Path(config['start'])
    levels = tuple(episode['levels']) if episode.get('levels') else None
    task = config.get('task', 'hunter')
    jobs = make_jobs(iteration, episode['episodes_per_iteration'], weights, config.get('decision_frames', 6),
                     config.get('turn_rate', 1.0), levels, task)
    if task == CAMPAIGN:
        # A resume that changed the episode count or levels rewrites config.episode: the iterations before it then
        # list (and fly) as the new settings would have made them.
        return jobs, dict(config['shaping'])
    # Historical runs recorded a separate Rules-level share. Reproduce those jobs exactly;
    # new training uses only the selected level cycle and never writes this field.
    share = episode.get('rules_share', 0.0)
    if share > 0.0 and iteration < episode.get('level_cycle_from_iteration', float('inf')):
        from experiments.run import TUNE_LEVELS
        others = tuple(level for level in (levels or TUNE_LEVELS) if level != 25)
        if others:
            count = episode['episodes_per_iteration']
            jobs = [(25 if int((iteration * count + i + 1) * share + 1e-9)
                     > int((iteration * count + i) * share + 1e-9)
                     else others[(iteration * count + i) % len(others)], seed, spec)
                    for i, (_, seed, spec) in enumerate(jobs)]
    shaping = dict(config['shaping'])
    if iteration < episode.get('random_maze_from_iteration', 0):
        shaping['random_maze'] = False
    return jobs, shaping


def _fly_campaign(job, shaping, **watching):
    """A campaign job flown as training flew it (see learning.adapters.campaign_hunter), its policy loaded as
    training's start was: a hunter policy is migrated to the campaign senses in memory."""
    from experiments.campaign_rollout import fly_campaign
    from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
    from learning.adapters.campaign_hunter import DEFAULT_HORIZON
    from learning.campaign_rewards import CampaignShaping
    from tools.ppo_neural import start_policy
    level, seed, spec, _ = job
    rewards = dict(shaping)
    horizon = rewards.pop('horizon', DEFAULT_HORIZON)
    return fly_campaign(level, seed, spec, horizon=horizon, shaping=CampaignShaping(**rewards),
                        policy=start_policy(spec.weights, CAMPAIGN_FEATURE_NAMES), **watching)


def _fly(item):
    """A job flown again: (level, panic share, return, kills, end, seconds), or for a campaign job
    (level, end, return, own kills, seconds)."""
    job, shaping = item
    from experiments import run
    run._use_dummy_drivers()
    if len(job) == 4:
        flown = _fly_campaign(job, shaping)
        result = flown['result']
        return job[0], result.end, float(flown['rewards'].sum()), result.own_kills, result.elapsed
    from experiments.rollout import fly_rollout
    level, seed, spec = job
    flown = fly_rollout(level, seed, spec, **shaping)
    temperatures = flown['temperatures']
    return (level, float((temperatures > 1.0).mean()) if len(temperatures) else 0.0, float(flown['rewards'].sum()),
            flown['result'].kills, flown['result'].end, len(temperatures) * spec.decision_frames / 60.0)


def fly_all(run, iteration, workers):
    """Each of the iteration's episodes flown again, as _fly reports it."""
    jobs, shaping = episode_jobs(run, iteration)
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_fly, [(job, shaping) for job in jobs]))


def find(run, iteration, workers):
    rows = fly_all(run, iteration, workers)
    if rows and isinstance(rows[0][1], str):          # a campaign run's: how each episode ended
        print(f'{"episode":>7} {"level":>5} {"return":>7} {"kills":>5} {"seconds":>7}  end')
        for index, (level, end, ret, kills, seconds) in enumerate(rows):
            print(f'{index:7d} {level:5d} {ret:7.1f} {kills:5d} {seconds:7.1f}  {end}')
        return
    print(f'{"episode":>7} {"level":>5} {"panic":>6} {"return":>7} {"kills":>5} {"seconds":>7}  end')
    for index, (level, panic, ret, kills, end, seconds) in sorted(enumerate(rows), key=lambda r: -r[1][1]):
        if panic > 0:
            print(f'{index:7d} {level:5d} {100 * panic:5.0f}% {ret:7.1f} {kills:5d} {seconds:7.1f}  {end}')
    print(f'{sum(1 for r in rows if r[1] > 0)} of {len(rows)} episodes panicked')


def _open_window():
    """The window the replay is shown in: a function that shows the game's screen, or None to show it as it is.

    The game world is as big as its screen (1352x878), and a replay must keep that size to fly as training did.
    On a smaller desktop the game draws on its full-size screen, hidden, and each frame is shown shrunk in a
    window that fits.
    """
    import pygame
    from experiments import harness
    from utils.window_size import fit_window_size
    pygame.init()
    sizes = pygame.display.get_desktop_sizes()
    size = fit_window_size(harness.SCREEN, sizes[0] if sizes else None)
    if size == harness.SCREEN:
        return None
    from unittest import mock
    from pygame._sdl2.video import Renderer, Texture, Window
    set_mode = pygame.display.set_mode
    with mock.patch.object(pygame.display, 'set_mode', lambda size, flags=0, *a, **k: set_mode(size, flags | pygame.HIDDEN)):
        harness._game = harness._make_game()
    window = Window('Episode replay', size=size)
    renderer = Renderer(window)

    def show():
        renderer.clear()
        Texture.from_surface(renderer, pygame.display.get_surface()).draw(dstrect=(0, 0, *size))
        renderer.present()
    return show


def _drawer(speed):
    """Opens the window and gives back draw(game, text, alarm=False): the game's frame shown with `text` on top (on
    red for an alarm), at `speed`. Esc or closing the window stops the replay."""
    import pygame
    show = _open_window()
    clock, font, flip = pygame.time.Clock(), None, pygame.display.flip

    def draw(game, text, alarm=False):
        nonlocal font
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.WINDOWCLOSE) or (event.type == pygame.KEYDOWN
                                                                     and event.key == pygame.K_ESCAPE):
                raise SystemExit
        pygame.display.flip = lambda: None          # the overlay goes on before the frame is shown
        # Drawing draws random numbers (sparks, debris): put them back, or the episode would play out differently.
        states = random.getstate(), np.random.get_state()
        try:
            game.draw()
        finally:
            pygame.display.flip = flip
            random.setstate(states[0])
            np.random.set_state(states[1])
        font = font or pygame.font.SysFont(None, 34)
        label = font.render(text, True, (255, 255, 255))
        screen = pygame.display.get_surface()
        pygame.draw.rect(screen, (190, 30, 30) if alarm else (20, 20, 30), (0, 0, label.get_width() + 24, 40))
        screen.blit(label, (12, 8))
        (show or flip)()
        clock.tick(60 * speed)
    return draw


def watch(run, iteration, episode, speed):
    jobs, shaping = episode_jobs(run, iteration)
    heading = f'run {Path(run).name}  iteration {iteration}  episode {episode}  level {jobs[episode][0]}'
    if len(jobs[episode]) == 4:
        draw = _drawer(speed)
        # The window keeps the tentacles' drag physics; the episode does not depend on it.
        flown = _fly_campaign(jobs[episode], shaping, low_graphics=False,
                              trace=lambda seconds, game: draw(game, f'{heading}  {seconds:5.1f} s'))
        result = flown['result']
        print(f'episode {episode}: level {result.level}, {result.end}, own kills {result.own_kills}, '
              f'return {flown["rewards"].sum():.1f}')
        return
    from experiments.rollout import fly_rollout
    level, seed, spec = jobs[episode]
    draw = _drawer(speed)

    def frame(seconds, game, hunter):
        stub = game.scoring.calculate_current_potential_score
        if not getattr(stub, 'for_drawing', False):
            # The harness stands in a bare score; the HUD reads a couple more fields from it.
            def for_drawing(*a, **k):
                return {'max_score': 1000, 'final_score': 0, **stub(*a, **k)}
            for_drawing.for_drawing = True
            game.scoring.calculate_current_potential_score = for_drawing
        temperature = getattr(game.hunter_flight.controller, 'temperature', 1.0)
        panic = temperature > 1.0
        text = f'{heading}  {seconds:5.1f} s'
        if panic:
            text += f'   PANIC (temperature {temperature:g})'
        draw(game, text, panic)

    # Training skips the tentacles' drag physics; a watched episode needs it, or they hang where they started.
    flown = fly_rollout(level, seed, spec, trace=frame, low_graphics=False, **shaping)
    temperatures = flown['temperatures']
    print(f'episode {episode}: level {level}, {flown["result"].end}, kills {flown["result"].kills}, '
          f'return {flown["rewards"].sum():.1f}, panic {100 * float((temperatures > 1.0).mean()):.0f}%')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('iteration', type=int)
    parser.add_argument('episode', type=int, nargs='?')
    parser.add_argument('--check',action='store_true',help='validate creature replay rules without opening a window')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed when watching')
    parser.add_argument('--workers', type=int, default=8, help='processes when finding')
    args = parser.parse_args(argv)
    import json
    config_path = Path(args.run)/'config.json'
    from learning.tasks import task
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    replay = task(config.get('task','hunter')).replay()
    if replay is not None:
        if args.check:
            try:
                env=replay.open_environment(args.run);env.close()
            except (ValueError,OSError,KeyError) as exc:
                raise SystemExit(str(exc)) from None
            return
        if args.episode is None:
            try:
                rows = replay.fly(args.run, args.iteration)
            except (ValueError, OSError, KeyError) as exc:     # changed rules, a diverged episode, a missing file
                raise SystemExit(str(exc)) from None
            print(json.dumps(rows, indent=2))
        else:
            replay.watch(args.run, args.iteration, args.episode, args.speed)
        return
    if args.episode is None:
        find(args.run, args.iteration, args.workers)
    else:
        watch(args.run, args.iteration, args.episode, args.speed)


if __name__ == '__main__':
    main(sys.argv[1:])
