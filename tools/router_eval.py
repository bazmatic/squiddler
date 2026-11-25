"""Held-out trials of the walker, the leaper and the router built from them.

    venv/bin/python -m tools.router_eval WALKER.npz LEAPER.npz [--seeds train|heldout] [--count N] [--leap-count N]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

LESSONS = {'crawl':1,'about_face':8,'gap':9,'leap':11}
SEED_BASES = {'train':6000,'heldout':90000}      # tune on 'train'; look at 'heldout' once, at the end


def trial_level(kind, k):
    """The arena of trial k: the heldout trials use arenas neither specialist trained on."""
    from learning.adapters import inchworm_scenario as scenario
    levels = scenario.HOLDOUT_LEVELS if kind == 'heldout' else scenario.TRAIN_LEVELS
    return levels[k%len(levels)]


def trial_seeds(lesson, count, kind):
    return [SEED_BASES[kind]+LESSONS[lesson]*1000+k for k in range(count)]


def _run(policy, lesson, count, kind):
    from learning.adapters import inchworm_scenario as scenario
    from learning.training import collect
    env = scenario.environment(horizon=40.)
    try:
        return [collect(env,policy,seed,dict(level=trial_level(kind,k),stage=LESSONS[lesson]))[1]['success']
                for k,seed in enumerate(trial_seeds(lesson,count,kind))]
    finally:
        env.close()


def evaluate(policy, lesson, count, kind):
    """Success rate over `count` trials of one lesson, on the 'train' or 'heldout' seeds."""
    results = _run(policy,lesson,count,kind)
    return sum(results)/max(1,len(results))


class _Watched:
    """A router that remembers the state of every episode, to count its hand-overs afterwards."""
    def __new__(cls, router):
        router.states = []
        original = router.initial_state
        router.initial_state = lambda: router.states.append(original()) or router.states[-1]
        return router


def handover_rate(router, count, kind, lesson='leap'):
    """The fraction of trials of a lesson (Leap by default) in which the router handed over to the leaper at least once."""
    _Watched(router)
    _run(router,lesson,count,kind)
    return sum(s.handovers > 0 for s in router.states)/max(1,len(router.states))


def _job(args):
    walker_path, leaper_path, who, lesson, count, kind = args
    from learning.adapters.inchworm import load_policy
    from learning.router import RoutedPolicy
    walker, leaper = load_policy(walker_path), load_policy(leaper_path)
    policy = {'walker':walker,'leaper':leaper,'router':RoutedPolicy(walker,leaper)}[who]
    os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    if who == 'router' and lesson == 'handover':
        return handover_rate(policy,count,kind)
    return evaluate(policy,lesson,count,kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('walker'); parser.add_argument('leaper')
    parser.add_argument('--seeds',choices=sorted(SEED_BASES),default='train')
    parser.add_argument('--count',type=int,default=150); parser.add_argument('--leap-count',type=int,default=240)
    parser.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)-2))
    args = parser.parse_args()
    jobs = [(args.walker,args.leaper,who,lesson,args.leap_count if lesson in ('leap','handover') else args.count,args.seeds)
            for who in ('walker','leaper','router') for lesson in ('crawl','gap','about_face','leap')]
    jobs.append((args.walker,args.leaper,'router','handover',args.leap_count,args.seeds))
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(zip(((j[2],j[3]) for j in jobs),pool.map(_job,jobs)))
    print(f'{args.seeds} seeds: {args.count} trials per lesson, {args.leap_count} for Leap')
    print(f'{"":8}'+''.join(f'{lesson:>12}' for lesson in ('crawl','gap','about_face','leap')))
    for who in ('walker','leaper','router'):
        print(f'{who:8}'+''.join(f'{results[(who,lesson)]:12.2f}' for lesson in ('crawl','gap','about_face','leap')))
    print(f'router handed over in {results[("router","handover")]:.2f} of Leap trials')


if __name__ == '__main__':
    main()
