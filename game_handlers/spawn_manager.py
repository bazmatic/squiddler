"""Enemy spawning management.

This module handles all enemy spawning logic, eliminating duplication
in the start_level() method.
"""

from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, TYPE_CHECKING
import logging
import os
import random
import math
import enemy_kinds
import level_rules
import config as game_config
if TYPE_CHECKING:
    from entities.command_recorder import CommandRecorder
    from game_handlers.enemy_roster import EnemyRoster
    from level_rules import EnemyCounts


@dataclass
class SpawnConfig:
    """Configuration for spawning a type of entity.
    
    Attributes:
        count: Number of entities to spawn.
        kind: The kind of enemy (e.g., "replay").
        factory_func: Factory function that takes (pos, command_recorder) and returns entity.
        requires_command_recorder: Whether command_recorder is needed for factory.
        post_create_hook: Optional function to call after entity creation (entity) -> None.
        radius: How far the entity reaches at its largest; positions with that much room
            from the walls are used first.
    """
    count: int
    kind: str
    factory_func: Callable[[Tuple[float, float], Optional['CommandRecorder']], object]
    requires_command_recorder: bool = False
    post_create_hook: Optional[Callable[[object], None]] = None
    radius: float = 0.0


def make_enemy(kind, pos, level, command_recorder):
    """One enemy of a kind at a position.

    Every spawn path makes its Creatures here. The Encounter gives a pilot its view of the game at spawn time.
    """
    if kind not in enemy_kinds.kinds() or enemy_kinds.kind(kind).body is None:
        raise ValueError(f'cannot make an enemy of type {kind!r}')
    import creatures
    if kind == 'aggressive' and not os.getenv('SCRIPTED_JELLIES'):
        kind = 'neural_jelly'   # neural is the default; it chases like the scripted Jelly while no trained brain loads
    return creatures.build(kind, pos, level, command_recorder)


class SpawnManager:
    """Manages enemy spawning for levels."""
    
    def __init__(self, roster: 'EnemyRoster'):
        """Initialize spawn manager.
        
        Args:
            roster: The roster to add spawned enemies to.
        """
        self.roster = roster
        self._anemone_keep_clear: Optional[Tuple[Tuple[float, float], float]] = None
        self._maze = None
    
    def _update_available_positions(
        self,
        used_positions: List[Tuple[float, float]],
        all_positions: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Calculate available positions after excluding used ones.
        
        Args:
            used_positions: Positions that have been used.
            all_positions: All available spawn positions.
            
        Returns:
            List of positions that are still available.
        """
        return [pos for pos in all_positions if pos not in used_positions]
    
    def _spawn_entities(
        self,
        config: SpawnConfig,
        available_positions: List[Tuple[float, float]],
        command_recorder: Optional['CommandRecorder'] = None
    ) -> List[Tuple[float, float]]:
        """Spawn entities based on configuration.
        
        Args:
            config: Spawn configuration.
            available_positions: Available spawn positions.
            command_recorder: Command recorder (required if config.requires_command_recorder).
            
        Returns:
            List of positions used by spawned entities.
        """
        if config.count <= 0 or len(available_positions) == 0:
            return []

        positions_to_use = self._select_positions(config, available_positions)
        used_positions: List[Tuple[float, float]] = []
        
        for pos in positions_to_use:
            entity = self._create_entity(config, self._seat(config.kind, config.radius, pos), command_recorder)
            if config.post_create_hook:
                config.post_create_hook(entity)
            self.roster.add(entity)
            used_positions.append(pos)
        
        return used_positions

    def _seat(self, kind: str, radius: float, pos: Tuple[float, float]) -> Tuple[float, float]:
        """Where to put an enemy chosen for pos: there if its whole body fits in open space, else the nearest spot where
        it does. When no spot nearby has room for it, it goes at pos anyway, and the level says so."""
        maze = self._maze
        if maze is None or radius <= 0 or maze.is_open(pos, radius):
            return pos
        seat = maze.nearest_open(pos, radius, reach=2 * max(maze.cell_size_x, maze.cell_size_y))
        if seat is None:
            logging.getLogger(__name__).warning('No room for a %s near %s: it starts touching a wall', kind, pos)
            return pos
        return seat

    def _create_entity(
        self,
        config: SpawnConfig,
        pos: Tuple[float, float],
        command_recorder: Optional['CommandRecorder']
    ):
        """Create an entity using the provided factory and recorder requirement."""
        if config.requires_command_recorder:
            if command_recorder is None:
                raise ValueError(f"command_recorder required for {config.kind}")
            return config.factory_func(pos, command_recorder)
        return config.factory_func(pos, None)

    def _select_positions(
        self,
        config: SpawnConfig,
        available_positions: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Select spawn positions with special handling for flockers and anemones."""
        if config.kind == "anemone" and self._anemone_keep_clear is not None:
            (clear_x, clear_y), distance = self._anemone_keep_clear
            available_positions = [
                pos for pos in available_positions
                if math.hypot(pos[0] - clear_x, pos[1] - clear_y) > distance
            ]
        
        spawn_count = min(config.count, len(available_positions))
        if spawn_count <= 0:
            return []

        if self._maze is not None and config.radius > 0:
            # Positions with room for the whole body first; the rest only if those run out
            available_positions = sorted(available_positions,
                                         key=lambda pos: not self._maze.is_open(pos, config.radius))

        if config.kind == "flocker":
            return self._select_flocker_positions(available_positions, spawn_count)

        return available_positions[:spawn_count]

    def _select_flocker_positions(
        self,
        available_positions: List[Tuple[float, float]],
        spawn_count: int
    ) -> List[Tuple[float, float]]:
        """Select clustered positions for flockers using configurable radius."""
        anchor = available_positions[0]
        radius_sq = game_config.FLOCKER_ENEMY_CLUSTER_RADIUS * game_config.FLOCKER_ENEMY_CLUSTER_RADIUS

        within_radius = [
            p for p in available_positions
            if (p[0] - anchor[0]) ** 2 + (p[1] - anchor[1]) ** 2 <= radius_sq
        ]

        positions_sorted = sorted(
            within_radius if within_radius else available_positions,
            key=lambda p: (p[0] - anchor[0]) ** 2 + (p[1] - anchor[1]) ** 2
        )

        if len(positions_sorted) >= spawn_count:
            return positions_sorted[:spawn_count]

        # Allow overlap if not enough slots; repeat anchor to fill
        return positions_sorted + [anchor] * (spawn_count - len(positions_sorted))
    
    def spawn_all_enemies(
        self,
        level: int,
        spawn_positions: List[Tuple[float, float]],
        command_recorder: 'CommandRecorder',
        enemy_counts: 'EnemyCounts',
        split_boss_count: int,
        mother_boss_count: int,
        anemone_keep_clear: Optional[Tuple[Tuple[float, float], float]] = None,
        maze=None,
        belemnite_count: int = 0,
        wallcrawler_count: int = 0
    ) -> None:
        """Spawn all enemies for a level.
        
        Args:
            level: Current level number.
            spawn_positions: Available spawn positions.
            command_recorder: Command recorder for replay enemies.
            enemy_counts: Enemy count configuration.
            split_boss_count: Number of SplitBoss enemies to spawn.
            mother_boss_count: Number of Mother Boss enemies to spawn.
            anemone_keep_clear: Optional (position, distance); no anemone is placed
                within that distance of the position (the player's start and its reach).
            maze: When given, big enemies are placed where they do not touch a wall.
            belemnite_count: Number of Belemnites to spawn.
            wallcrawler_count: Number of wall-clinging guns to spawn.
        """
        self._anemone_keep_clear = anemone_keep_clear
        self._maze = maze
        
        # Clear existing enemies
        self.roster.clear()
        
        # Create regular enemies based on enemy_counts
        new_enemies = self._create_enemies_from_counts(
            level, spawn_positions, enemy_counts
        )
        self.roster.extend(new_enemies)
        
        # Track used positions
        used_positions = [e.get_pos() for e in new_enemies]
        available_positions = self._update_available_positions(used_positions, spawn_positions)
        
        # Define spawn configurations for all enemy types
        spawn_configs = self._create_spawn_configs(
            enemy_counts, split_boss_count, mother_boss_count, level, belemnite_count, wallcrawler_count
        )
        
        # Spawn all entities using configuration
        for config in spawn_configs:
            newly_used = self._spawn_entities(config, available_positions, command_recorder)
            used_positions.extend(newly_used)
            available_positions = self._update_available_positions(used_positions, spawn_positions)
    
    def spawn_from_layout(self, layout, maze, level, command_recorder):
        """Create only the authored enemies; wall crawlers start on a nearby clear surface.

        Returns (kind, pixel position) for each of its enemy hunters, which the game makes and flies itself.
        """
        self._anemone_keep_clear = None
        self.roster.clear()
        hunters = []
        for kind, cell in layout.enemies:
            pos = maze.position_calculator.grid_center_to_screen(*cell)
            if kind in enemy_kinds.hunters():
                hunters.append((kind, pos))
                continue
            if kind not in ('wallcrawler', 'inchworm'):
                self._maze = maze
                pos = self._seat(kind, enemy_kinds.kind(kind).spawn_radius, pos)
            enemy = make_enemy(kind, pos, level, command_recorder)
            if kind in ('wallcrawler', 'inchworm'):
                self._place_layout_crawler(enemy, maze)
            self.roster.add(enemy)
        return hunters

    def _place_layout_crawler(self, crawler, maze):
        """Resolve an authored cell to a planted spawn before the Encounter's spawn hooks."""
        from creatures.traits.wallcrawl import contact, clear, nearby
        from creatures.traits.rooted import wall_between
        origin = crawler.get_pos()
        reach = max(maze.cell_size_x, maze.cell_size_y) * 2
        surfaces = [contact(maze, origin, [wall]) for wall in nearby(maze, origin, reach)]
        surfaces = sorted((surface for surface in surfaces if surface is not None),
                          key=lambda surface: surface[3])
        for _, foot, normal, distance in surfaces:
            if distance > reach:
                continue
            gap = crawler.radius + 2.
            position = (foot[0]+normal[0]*gap, foot[1]+normal[1]*gap)
            if not clear(maze, position, crawler.radius) or wall_between(origin, position, maze):
                continue
            start = getattr(maze, 'start_pos', None)
            if (start is not None and math.dist(origin, start) >= game_config.ENEMY_START_CLEARANCE
                    and math.dist(position, start) < game_config.ENEMY_START_CLEARANCE):
                continue
            if any(math.dist(position, other.get_pos()) < crawler.radius+other.radius+2
                   for other in self.roster.all()):
                continue
            crawler.x, crawler.y = position
            return

    def _create_spawn_configs(
        self,
        enemy_counts: 'EnemyCounts',
        split_boss_count: int,
        mother_boss_count: int,
        level: int = 1,
        belemnite_count: int = 0,
        wallcrawler_count: int = 0
    ) -> List[SpawnConfig]:
        """Create spawn configurations for all enemy types.
        
        Args:
            enemy_counts: Enemy count configuration.
            split_boss_count: Number of SplitBoss enemies.
            mother_boss_count: Number of Mother Boss enemies.
            
        Returns:
            List of SpawnConfig objects.
        """
        counts = (('replay', enemy_counts.replay), ('flocker', enemy_counts.flocker),
                  ('flighthouse', enemy_counts.flighthouse), ('vibehouse', enemy_counts.vibehouse),
                  ('split_boss', split_boss_count),
                  ('mother_boss', mother_boss_count), ('egg', enemy_counts.egg), ('anemone', enemy_counts.anemone),
                  ('belemnite', belemnite_count), ('wallcrawler', wallcrawler_count))
        # Big enemies need their whole body clear of walls, not just their centre
        return [SpawnConfig(count=count, kind=kind,
                            factory_func=lambda pos, cr, kind=kind: make_enemy(kind, pos, level, cr),
                            requires_command_recorder=enemy_kinds.kind(kind).replays_player, post_create_hook=None,
                            radius=enemy_kinds.kind(kind).spawn_radius)
                for kind, count in counts if count > 0]
    
    def _create_enemies_from_counts(
        self,
        level: int,
        spawn_positions: List[Tuple[float, float]],
        enemy_counts: 'EnemyCounts'
    ) -> List:
        """Create enemies based on enemy_counts configuration.
        
        Args:
            level: Current level number.
            spawn_positions: List of valid spawn positions.
            enemy_counts: Enemy count configuration.
            
        Returns:
            The enemies made.
        """
            
        enemies = []
        used_positions = []
        
        # Shuffle positions
        available_positions = spawn_positions.copy()
        random.shuffle(available_positions)
        
        # Define enemy type configurations
        enemy_types = [
            ("static", enemy_counts.static),
            ("patrol", enemy_counts.patrol),
            ("aggressive", enemy_counts.aggressive)
        ]
        
        # Create enemies for each type
        for enemy_type, count in enemy_types:
            if count <= 0:
                continue
            
            remaining_positions = self._update_available_positions(used_positions, available_positions)
            spawn_count = min(count, len(remaining_positions))
            
            for i in range(spawn_count):
                pos = remaining_positions[i]
                enemies.append(make_enemy(enemy_type, pos, level, None))
                used_positions.append(pos)
        
        return enemies
    

