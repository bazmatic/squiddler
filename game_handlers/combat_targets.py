"""Shared target selection for enemies; ties preserve the player target. An observer is nobody's target."""


def nearest_friendly(enemy, player, hunter=None):
    ships = player if isinstance(player, (list, tuple)) else [player]
    candidates = [obj for obj in (*ships, hunter)
                  if obj is not None and obj.active and not getattr(obj, 'observer', False)]
    return min(candidates, key=lambda obj:(obj.x-enemy.x)**2+(obj.y-enemy.y)**2, default=None)


def nearest_friendly_pos(enemy, player, hunter=None):
    target = nearest_friendly(enemy, player, hunter)
    return target.get_pos() if target is not None else None
