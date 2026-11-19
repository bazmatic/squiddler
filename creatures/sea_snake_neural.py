"""Optional deployment weights, loaded once per encounter; no learner imports."""
import logging
import os
from pathlib import Path


def driver_for(enc):
    from learning.adapters.sea_snake import SeaSnakeDriver, load_policy
    if not hasattr(enc, '_sea_snake_policy'):
        if hasattr(enc.world, 'sea_snake_policy'):
            enc._sea_snake_policy = enc.world.sea_snake_policy
        else:
            selected = os.getenv('SEA_SNAKE_POLICY')
            path = Path(selected) if selected else Path(__file__).resolve().parents[1]/'assets/neural/sea_snake.npz'
            enc._sea_snake_policy = None
            if selected != '' and (selected is not None or path.is_file()):
                try:
                    enc._sea_snake_policy = load_policy(path)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logging.getLogger(__name__).warning('Sea snake policy %s unavailable; using scripted controller: %s', path, exc)
    return SeaSnakeDriver(enc._sea_snake_policy) if enc._sea_snake_policy is not None else None
