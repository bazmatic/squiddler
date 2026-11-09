"""Level-based enemy rules and scaling.

This module centralizes all rules for enemy counts and strength scaling based on level.
It provides a single source of truth for level-based difficulty adjustments.

Architecture:
    This module follows the Configuration Object pattern, providing functions that
    calculate enemy properties based on level. This makes it easy to:
    - Tune game balance
    - Adjust difficulty curves
    - Modify enemy scaling formulas
    - Test different configurations
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import config
from maze.config import MazeComplexity, MazeComplexityPresets


@dataclass
class EnemyCounts:
    """Enemy count configuration for a level.
    
    Attributes:
        total: Total number of regular enemies
        static: Number of static enemies
        patrol: Number of patrol enemies
        aggressive: Number of aggressive enemies
        replay: Number of replay enemy ships
        flocker: Number of flocker enemy ships
        flighthouse: Number of flighthouse enemies
        egg: Number of egg enemies
        anemone: Number of anemone enemies
    """
    total: int
    static: int
    patrol: int
    aggressive: int
    replay: int
    flocker: int
    flighthouse: int
    egg: int
    anemone: int = 0
    vibehouse: int = 0


@dataclass
class EnemyStrength:
    """Enemy strength configuration for a level.
    
    Attributes:
        patrol_speed: Movement speed for patrol enemies
        aggressive_speed: Movement speed for aggressive enemies
        damage: Damage dealt by enemies
        fire_interval_min: Minimum frames between enemy shots
        fire_interval_max: Maximum frames between enemy shots
        fire_range: Maximum distance to player for firing (pixels)
    """
    patrol_speed: float
    aggressive_speed: float
    damage: int
    fire_interval_min: int
    fire_interval_max: int
    fire_range: float


# The designed arc: levels 1 to ARC_LEVELS each have a level file. The formulas
# below are what the endless game uses, and a safety net for an arc level whose
# file is missing.
ARC_LEVELS = 24
BOSS_ARENA_SIZE = 16
MAX_BOSSES = 3

# Level each enemy type first appears on (anemones: config.ANEMONE_FIRST_LEVEL)
FIRST_LEVELS = {'static': 1, 'patrol': 2, 'aggressive': 3, 'replay': 8,
                'flocker': 5, 'flighthouse': 10, 'egg': 4, 'wallcrawler': 11}

# The types a level can be built around, and the most of each that one level carries (a Flocker team is at most eight)
THEMES = ('aggressive', 'flocker', 'patrol', 'wallcrawler')
DOMINANT_CAP = {'aggressive': 12, 'flocker': 8, 'patrol': 12}
# What fills out the active roster beside the dominant type; Jellies and Flockers are never a token one or two
SUPPORT_ORDER = ('patrol', 'replay')

# What level 23, the last ordinary level of the arc, carries
ARC_END_MIX = {'static': 4, 'patrol': 3, 'aggressive': 4, 'replay': 3, 'flocker': 3,
               'flighthouse': 2, 'egg': 1, 'anemone': 4}

# The last level whose mission is to destroy its Eggs; after it, a level with Eggs must be cleared of every enemy
LAST_EGG_MISSION_LEVEL = 8


def first_level(enemy_type: str) -> int:
    """Level an enemy type first appears on."""
    if enemy_type == 'anemone':
        return config.ANEMONE_FIRST_LEVEL
    return FIRST_LEVELS[enemy_type]


def is_boss_level(level: int) -> bool:
    """Whether a level without a level file is a boss level."""
    return level % config.BOSS_LEVEL_INTERVAL == 0


def generated_exit_policy(level: int) -> Optional[Dict]:
    """The exit policy of a level without a level file, as JSON, or None for the default."""
    if _must_clear_all(level):
        return {'type': 'all_cleared'}
    if get_enemy_counts(level).egg > 0:
        return {'type': 'type_cleared', 'enemy_types': ['egg']}
    return None


def generated_ammo_policy(level: int) -> Optional[Dict]:
    """The ammo policy of a level without a level file, as JSON, or None for the default."""
    return {'type': 'unlimited'} if _must_clear_all(level) else None


def _must_clear_all(level: int) -> bool:
    """Whether a level without a level file opens only when every enemy is dead (and so has unlimited ammo)."""
    return is_boss_level(level) or (level > LAST_EGG_MISSION_LEVEL and get_enemy_counts(level).egg > 0)


def get_boss_counts(level: int) -> Tuple[int, int]:
    """Get the bosses on a level without a level file.

    Every boss level carries a mother boss; the cycle varies how many mothers and split bosses. Each full
    turn of the cycle adds a boss, up to MAX_BOSSES on a level.

    Args:
        level: Current level number (1-based).

    Returns:
        (split boss count, mother boss count); (0, 0) on an ordinary level.
    """
    if not is_boss_level(level):
        return (0, 0)
    # Level 30 opens the first full cycle of the endless game
    cycle = level // config.BOSS_LEVEL_INTERVAL - 5
    kind, turn = cycle % 3, max(0, cycle // 3)
    if kind == 0:
        return (min(MAX_BOSSES - 1, 2 + turn), 1)
    if kind == 1:
        return (0, min(MAX_BOSSES, 1 + turn))
    return (min(MAX_BOSSES - 1, 1 + turn), 1)


def get_split_boss_count(level: int) -> int:
    """Get number of SplitBoss enemies for a level without a level file."""
    return get_boss_counts(level)[0]


def get_mother_boss_count(level: int) -> int:
    """Get number of Mother Boss enemies for a level without a level file."""
    return get_boss_counts(level)[1]


def _boss_escort(level: int) -> Dict[str, int]:
    """Enemies that accompany the bosses on a boss level."""
    split, mother = get_boss_counts(level)
    if split and mother:
        return {'replay': 2, 'flocker': 2, 'egg': 2}
    if mother:
        return {'egg': 4}
    return {'flocker': 6}


def get_enemy_count(level: int) -> int:
    """Get the number of enemies on a level without a level file.

    Bosses and the hunter are not counted. Anemones are.

    Args:
        level: Current level number (1-based).

    Returns:
        The escort on a boss level. Otherwise a number that rises by one a
        level through the arc, then by one every two levels, up to
        config.MAX_ENEMY_COUNT.
    """
    if is_boss_level(level):
        return sum(_boss_escort(level).values())
    arc_end_count = sum(ARC_END_MIX.values())
    if level <= ARC_LEVELS:
        return min(arc_end_count, 3 + level)
    return min(config.MAX_ENEMY_COUNT, arc_end_count + (level - (ARC_LEVELS - 1)) // 2)


def get_enemy_mix(level: int) -> Dict[str, int]:
    """Get how many of each enemy type a level without a level file has.

    Args:
        level: Current level number (1-based).

    Returns:
        Count for each of static, patrol, aggressive, replay, flocker,
        flighthouse, egg and anemone. They add up to get_enemy_count(level).
    """
    mix = dict.fromkeys(ARC_END_MIX, 0)
    if is_boss_level(level):
        mix.update(_boss_escort(level))
        return mix

    count = get_enemy_count(level)
    if level > ARC_LEVELS:
        mix.update({n: ARC_END_MIX[n] for n in ('static', 'flighthouse', 'egg', 'anemone')})
        mix['static'] = 3
    else:
        # Only what the player has already met; one flighthouse and one egg once they have
        for enemy_type in ('flighthouse', 'egg'):
            if level >= first_level(enemy_type):
                mix[enemy_type] = 1
    # Every level has one dominant active type, in numbers; whatever the roster cannot fill with it is urchins
    theme = dominant_type(level)
    room = count - sum(mix.values())
    if theme in ('aggressive', 'flocker', 'patrol'):
        dominant = min(DOMINANT_CAP[theme], max(1, -(-room * 2 // 3)))
        mix[theme] = min(room, dominant)
        support = min(room - mix[theme], mix[theme] // 2)
    elif theme == 'wallcrawler':
        support = min(room, -(-generated_wallcrawler_count(level) // 2))
    else:
        support = 0
    minors = [t for t in SUPPORT_ORDER if t != theme and level >= first_level(t)]
    for i in range(support if minors else 0):
        mix[minors[i % len(minors)]] += 1
    mix['static'] += count - sum(mix.values())
    return mix


def dominant_type(level: int) -> str:
    """The active enemy type a level is built around: the Jellies, Flockers, Crabs and Wallcrawlers it has met, in turn."""
    met = [t for t in THEMES if level >= first_level(t)]
    return met[level % len(met)] if met else 'static'


def generated_wallcrawler_count(level: int) -> int:
    """Wallcrawlers on a level without a level file: none, unless it is built around them, and then a crowd."""
    return min(8, 5 + (level - 11) // 12) if dominant_type(level) == 'wallcrawler' else 0


def get_anemone_count(level: int) -> int:
    """Get number of anemones for a level without a level file."""
    return get_enemy_mix(level)['anemone']


def anemones_that_fit(grid_size: int) -> int:
    """Get the most anemones a maze has room for.
    
    Their fields of pull together may cover only a share of the maze, so a
    small maze is not blanketed. There is always room for one.
    
    Args:
        grid_size: Width and height of the maze in cells.
    """
    field_cells = math.pi * config.ANEMONE_REACH_CELLS ** 2
    return max(1, int(config.ANEMONE_MAX_COVERAGE * grid_size * grid_size / field_cells))


def get_flighthouse_spawn_interval(level: int) -> float:
    """Get spawn interval for Flighthouse enemies at a given level.
    
    Spawn interval starts at 3.0 seconds and decreases to 0.5 seconds as level increases.
    Uses linear interpolation between start and end values.
    
    Args:
        level: Current level number (1-based).
        
    Returns:
        Spawn interval in seconds (between 0.5 and 3.0).
    """
    start_interval = 3.0  # seconds at level 1
    end_interval = 0.5    # seconds at high levels
    
    # Tutorial levels use start interval
    if level <= config.TUTORIAL_LEVELS:
        return start_interval
    
    # Calculate effective level (after tutorial)
    effective_level = level - config.TUTORIAL_LEVELS
    
    # Use a scaling curve that reaches end_interval by around level 20-30
    # Linear interpolation: start at effective_level 1, reach end by effective_level 20
    max_effective_level = 20.0
    progress = min(1.0, (effective_level - 1) / (max_effective_level - 1))
    
    # Interpolate between start and end
    interval = start_interval - (start_interval - end_interval) * progress
    
    # Ensure we don't go below end_interval
    return max(end_interval, interval)


def get_enemy_speed(level: int, enemy_type: str) -> float:
    """Get movement speed for an enemy type at a given level.
    
    Args:
        level: Current level number (1-based).
        enemy_type: Type of enemy ('static', 'patrol', or 'aggressive').
        
    Returns:
        Movement speed for the enemy type at this level.
    """
    if enemy_type == "static":
        return 0.0
    
    # Base speed from config
    if enemy_type == "patrol":
        base_speed = config.ENEMY_PATROL_SPEED
    elif enemy_type == "aggressive":
        base_speed = config.ENEMY_AGGRESSIVE_SPEED
    else:
        raise ValueError(f"Unknown enemy type: {enemy_type}")
    
    # Tutorial levels use base speed, scaling starts after
    if level <= config.TUTORIAL_LEVELS:
        return base_speed
    
    # Scale speed by effective level, up to a ceiling
    effective_level = level - config.TUTORIAL_LEVELS
    speed_multiplier = min(config.ENEMY_SPEED_CEILING, 1.0 + (effective_level - 1) * config.ENEMY_SPEED_GROWTH)
    return base_speed * speed_multiplier


def get_enemy_damage(level: int) -> int:
    """Get damage value for enemies at a given level.
    
    Args:
        level: Current level number (1-based).
        
    Returns:
        Damage value for enemies at this level.
    """
    # Tutorial levels use base damage, scaling starts after
    if level <= config.TUTORIAL_LEVELS:
        return config.ENEMY_DAMAGE
    
    # Base damage from config, scaled by effective level up to a ceiling
    effective_level = level - config.TUTORIAL_LEVELS
    damage_multiplier = min(config.ENEMY_DAMAGE_CEILING, 1.0 + (effective_level - 1) * config.ENEMY_DAMAGE_GROWTH)
    return int(config.ENEMY_DAMAGE * damage_multiplier)


def get_enemy_fire_interval(level: int) -> Tuple[int, int]:
    """Get fire interval range for enemies at a given level.
    
    Args:
        level: Current level number (1-based).
        
    Returns:
        Tuple of (min_interval, max_interval) in frames.
        Intervals decrease (faster firing) as level increases.
    """
    # Base intervals from config
    base_min = config.ENEMY_FIRE_INTERVAL_MIN
    base_max = config.ENEMY_FIRE_INTERVAL_MAX
    
    # Tutorial levels use base intervals, scaling starts after
    if level <= config.TUTORIAL_LEVELS:
        return (base_min, base_max)
    
    # Reduce interval by 5% per effective level (faster firing at higher levels)
    # Minimum interval is 60% of base (40% reduction max)
    effective_level = level - config.TUTORIAL_LEVELS
    reduction_factor = min(0.4, (effective_level - 1) * 0.05)
    interval_multiplier = 1.0 - reduction_factor
    
    min_interval = int(base_min * interval_multiplier)
    max_interval = int(base_max * interval_multiplier)
    
    # Ensure minimum values
    min_interval = max(min_interval, 30)  # At least 0.5 seconds at 60 FPS
    max_interval = max(max_interval, min_interval + 60)  # At least 1 second range
    
    return (min_interval, max_interval)


def get_enemy_fire_range(level: int) -> float:
    """Get fire range for enemies at a given level.
    
    Args:
        level: Current level number (1-based).
        
    Returns:
        Maximum distance to player for firing (pixels).
        Range increases as level increases, up to ENEMY_MAX_FIRE_RANGE.
    """
    # Base range from config
    base_range = config.ENEMY_FIRE_RANGE
    
    # Tutorial levels use base range, scaling starts after
    if level <= config.TUTORIAL_LEVELS:
        return base_range
    
    # Increase range by 5% per effective level
    effective_level = level - config.TUTORIAL_LEVELS
    range_multiplier = 1.0 + (effective_level - 1) * 0.05
    return min(base_range * range_multiplier, config.ENEMY_MAX_FIRE_RANGE)


def get_enemy_strength(level: int) -> EnemyStrength:
    """Get complete enemy strength configuration for a level.
    
    Args:
        level: Current level number (1-based).
        
    Returns:
        EnemyStrength dataclass with all strength properties.
    """
    fire_interval_min, fire_interval_max = get_enemy_fire_interval(level)
    
    return EnemyStrength(
        patrol_speed=get_enemy_speed(level, "patrol"),
        aggressive_speed=get_enemy_speed(level, "aggressive"),
        damage=get_enemy_damage(level),
        fire_interval_min=fire_interval_min,
        fire_interval_max=fire_interval_max,
        fire_range=get_enemy_fire_range(level)
    )


def get_enemy_counts(level: int) -> EnemyCounts:
    """Get complete enemy count configuration for a level.
    
    Args:
        level: Current level number (1-based).
        
    Returns:
        EnemyCounts dataclass with all enemy counts.
    """
    mix = get_enemy_mix(level)
    return EnemyCounts(
        total=mix['static'] + mix['patrol'] + mix['aggressive'],
        static=mix['static'],
        patrol=mix['patrol'],
        aggressive=mix['aggressive'],
        replay=mix['replay'],
        flocker=mix['flocker'],
        flighthouse=mix['flighthouse'],
        egg=mix['egg'],
        anemone=mix['anemone']
    )


def get_maze_complexity(level: int) -> MazeComplexity:
    """Get default maze complexity for a level.
    
    Args:
        level: Current level number (1-based).
        
    Returns:
        EMPTY for level 1 and for boss levels (an open arena), then SIMPLE to
        level 5, NORMAL to 11, COMPLEX to 17 and EXTREME after.
    """
    if level == 1 or is_boss_level(level):
        return MazeComplexity.EMPTY
    if level <= 5:
        return MazeComplexity.SIMPLE
    if level <= 11:
        return MazeComplexity.NORMAL
    if level <= 17:
        return MazeComplexity.COMPLEX
    return MazeComplexity.EXTREME


def get_maze_grid_size(level: int) -> int:
    """Get default maze grid size for a level.
    
    Args:
        level: Current level number (1-based).
        
    Returns:
        Grid size (width/height in cells). Maze is always square. Boss levels
        get a small arena; other levels grow by MAZE_SIZE_INCREMENT a level
        up to MAX_MAZE_SIZE.
    """
    if is_boss_level(level):
        return BOSS_ARENA_SIZE
    return min(config.BASE_MAZE_SIZE + (level - 1) * config.MAZE_SIZE_INCREMENT, config.MAX_MAZE_SIZE)
