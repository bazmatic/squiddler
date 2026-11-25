"""Optional deployment weights, loaded once per encounter; no learner imports."""
import logging
import os
from pathlib import Path


def _routed_policy(paths):
    """INCHWORM_ROUTER=<walker.npz>,<leaper.npz>: a walker that hands leaps to a leaper (learning/router.py)."""
    if not paths:
        return None
    try:
        from learning.adapters.inchworm import load_policy
        from learning.router import RoutedPolicy
        walker, leaper = (load_policy(Path(path)) for path in paths.split(','))
        return RoutedPolicy(walker, leaper)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.getLogger(__name__).warning('Inchworm router %s unavailable; using the single policy: %s', paths, exc)
        return None


ASSETS = Path(__file__).resolve().parents[1]/'assets/neural'


def _router_paths():
    """The router to use: INCHWORM_ROUTER if set, else the shipped walker and leaper unless INCHWORM_POLICY chose a policy."""
    selected = os.getenv('INCHWORM_ROUTER')
    shipped = (ASSETS/'inchworm_walker.npz', ASSETS/'inchworm_leaper.npz')
    if selected is None and os.getenv('INCHWORM_POLICY') is None and all(path.is_file() for path in shipped):
        return ','.join(str(path) for path in shipped)
    return selected


def driver_for(enc):
    from learning.adapters.inchworm import InchwormDriver, load_policy
    if not hasattr(enc, '_inchworm_policy'):
        if hasattr(enc.world, 'inchworm_policy'):
            enc._inchworm_policy = enc.world.inchworm_policy
        elif (routed := _routed_policy(_router_paths())) is not None:
            enc._inchworm_policy = routed
        else:
            selected = os.getenv('INCHWORM_POLICY')
            path = Path(selected) if selected else ASSETS/'inchworm.npz'
            enc._inchworm_policy = None
            if selected != '' and (selected is not None or path.is_file()):
                try:
                    enc._inchworm_policy = load_policy(path)
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    logging.getLogger(__name__).warning('Inchworm policy %s unavailable; using reflex controller: %s',path,exc)
    return InchwormDriver(enc._inchworm_policy) if enc._inchworm_policy is not None else None
