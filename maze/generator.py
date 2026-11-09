"""Procedural maze generation."""

import random
import pygame
import math
from contextlib import contextmanager
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple
import config
from utils import (
    distance,
    distance_squared,
    get_angle_to_point,
    point_in_rect,
    circle_circle_collision,
    circle_line_collision
)
from maze.wall_segment import WallSegment
from maze.config import MazeComplexity, MazeGenerationConfig, MazeComplexityPresets
from maze.positioning import MazePositionCalculator
from maze.converter import GridToWallsConverter, SIDES
from hunter.visibility import simplify_outline
from maze.cave import CaveShape
from utils.spatial_grid import SpatialGrid
from entities.exit import ExitPortal
from flora import Flora
from rendering.flora_renderer import draw_flora
from rendering.wall_renderer import WallRenderer
from rendering.wall_debris import WallDebris


class RecursiveBacktrackingGenerator:
    """Carve a connected room graph, then expand it into broad, aligned corridors.

    Visiting logical rooms separately from the rendered grid prevents wide passages
    from accidentally marking neighbouring rooms as visited. Extra routes join rooms
    through whole doorways instead of eroding random patches of masonry.
    """

    def __init__(self, config: MazeGenerationConfig, grid_width: int, grid_height: int, level: int):
        self.config = config
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.level = level

    def generate(self) -> List[List[int]]:
        grid = [[1 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        empty = (self.config.extra_paths_multiplier == 0 and
                 self.config.passage_width == 0 and self.config.clear_radius == 0)
        if empty:
            self._clear(grid, 1, 1, self.grid_width-1, self.grid_height-1)
            return grid

        # Later complexity presets create more junctions, never narrower than a
        # comfortable flight corridor. Distribute spare cells evenly at the edges.
        position = MazePositionCalculator(self.grid_width, self.grid_height)
        room_width = max(3, 7-self.config.step_size//2)
        xs = self._bands(self.grid_width, max(room_width, math.ceil(70/position.cell_size_x)))
        ys = self._bands(self.grid_height, max(room_width, math.ceil(70/position.cell_size_y)))
        rooms = [(x,y) for y in range(len(ys)) for x in range(len(xs))]
        for x,y in rooms:
            self._clear(grid, xs[x][0], ys[y][0], xs[x][1], ys[y][1])

        def neighbours(room):
            x,y = room
            return [(nx,ny) for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1))
                    if 0 <= nx < len(xs) and 0 <= ny < len(ys)]

        def connect(a,b):
            x,y = a
            nx,ny = b
            if x != nx:
                wall = xs[min(x,nx)][1]
                self._clear(grid, wall, ys[y][0], wall+1, ys[y][1])
            else:
                wall = ys[min(y,ny)][1]
                self._clear(grid, xs[x][0], wall, xs[x][1], wall+1)

        start = random.choice(rooms)
        visited = {start}
        stack = [start]
        passages = set()
        while stack:
            current = stack[-1]
            choices = [room for room in neighbours(current) if room not in visited]
            if not choices:
                stack.pop()
                continue
            following = random.choice(choices)
            connect(current,following)
            passages.add(frozenset((current,following)))
            visited.add(following)
            stack.append(following)

        # A few deliberate shortcuts support combat and flanking. Never remove
        # all four sides around a junction: that would leave a lonely square block.
        closed = [(a,b) for a in rooms for b in neighbours(a)
                  if a < b and frozenset((a,b)) not in passages]
        random.shuffle(closed)
        budget = round(len(rooms)/(self.config.clear_radius+2))
        for a,b in closed:
            if budget <= 0:
                break
            edge = frozenset((a,b))
            passages.add(edge)
            if self._isolates_pillar(passages,a,b):
                passages.remove(edge)
                continue
            connect(a,b)
            budget -= 1
        return grid

    @staticmethod
    def _bands(size, minimum):
        """Half-open room intervals separated by exactly one wall cell."""
        interior = size-2
        count = max(1,(interior+1)//(minimum+1))
        available = interior-(count-1)
        width,remainder = divmod(available,count)
        bands=[]
        start=1
        for index in range(count):
            end=start+width+(index < remainder)
            bands.append((start,end))
            start=end+1
        return bands

    @staticmethod
    def _isolates_pillar(passages,a,b):
        for x in range(min(a[0],b[0])-1,max(a[0],b[0])+1):
            for y in range(min(a[1],b[1])-1,max(a[1],b[1])+1):
                cycle=((x,y),(x+1,y),(x+1,y+1),(x,y+1))
                if all(frozenset((cycle[i],cycle[(i+1)%4])) in passages for i in range(4)):
                    return True
        return False

    @staticmethod
    def _clear(grid,x0,y0,x1,y1):
        for y in range(y0,y1):
            for x in range(x0,x1):
                grid[y][x]=0

    def clear_corner_area(self, grid: List[List[int]], corner: Tuple[int, int]) -> None:
        # Corner rooms are already open and connected. Do not erase their walls:
        # doing so would undo the corridor structure near the start and exit.
        grid[corner[1]][corner[0]]=0


# Spatial grid bucket size (px) for curved mazes: their many short segments would crowd 150 px buckets
CURVED_BUCKET = 50.0


class Maze:
    """Procedurally generated maze."""
    
    def __init__(self, level: int, complexity: Optional[MazeComplexity] = None, grid_size: int = 0, layout=None,
                 seed: Optional[int] = None, curved: Optional[bool] = None):
        """Generate a maze for the given level.
        
        Args:
            level: Current level number (1-based).
            complexity: Optional maze complexity level. If None, calculated from level.
            grid_size: Grid size (width/height in cells). Always provided by level_config.get_maze_grid_size().
            layout: A level_layout.Layout to build the maze from instead of generating one (optional).
            seed: Seed for the cave rock's bulges (the level number if None). The global random is not used.
            curved: Curved cave walls (config.CAVE.enabled if None); False keeps the square block faces.
        """
        self.level = level
        self.curved = config.CAVE.enabled if curved is None else curved
        self.cave_seed = level if seed is None else seed
        self.layout = layout
        self.cave: Optional[CaveShape] = None
        
        # Determine complexity
        if complexity is None:
            complexity = MazeComplexityPresets.get_complexity_from_level(level)
        
        # Get generation config
        gen_config = MazeComplexityPresets.get_config(complexity)
        
        # Set grid dimensions (grid_size is already calculated in level_config if None was passed)
        self.grid_width = grid_size
        self.grid_height = grid_size
        
        # Create position calculator
        self.position_calculator = MazePositionCalculator(self.grid_width, self.grid_height)
        self.cell_size_x = self.position_calculator.cell_size_x
        self.cell_size_y = self.position_calculator.cell_size_y
        # Keep cell_size property for backward compatibility (use average)
        self.cell_size = (self.cell_size_x + self.cell_size_y) / 2
        self.offset_x = self.position_calculator.offset_x
        self.offset_y = self.position_calculator.offset_y
        
        # Wall blocks no shot can destroy; only a hand-made layout has them
        self.strong_cells: FrozenSet[Tuple[int, int]] = frozenset()
        if layout is None:
            # Generate maze grid
            generator = RecursiveBacktrackingGenerator(gen_config, self.grid_width, self.grid_height, level)
            self.grid = generator.generate()

            # Select random opposite corners for start and exit
            corner_combinations = [
                ((1, 1), (self.grid_width - 2, self.grid_height - 2)),  # TL -> BR
                ((self.grid_width - 2, 1), (1, self.grid_height - 2)),  # TR -> BL
                ((1, self.grid_height - 2), (self.grid_width - 2, 1)),  # BL -> TR
                ((self.grid_width - 2, self.grid_height - 2), (1, 1)),  # BR -> TL
            ]
            start_grid, exit_grid = random.choice(corner_combinations)

            # Clear areas around selected corners to ensure they're accessible
            generator.clear_corner_area(self.grid, start_grid)
            generator.clear_corner_area(self.grid, exit_grid)
        else:
            # A hand-made layout: its walls, its start and its exit
            self.grid = layout.grid()
            self.strong_cells = layout.strong_walls
            start_grid, exit_grid = layout.start, layout.exit
        
        # Set start position (before the walls: the cave rock keeps it clear)
        if layout is None:
            self.start_pos = self.position_calculator.get_start_position(start_grid)
        else:
            self.start_pos = self.position_calculator.grid_center_to_screen(*start_grid)
        
        # Create exit object
        exit_pos = self.position_calculator.grid_center_to_screen(exit_grid[0], exit_grid[1])
        # Use average cell size for exit radius
        exit_radius = (self.cell_size_x + self.cell_size_y) / 4
        self.exit = ExitPortal(exit_pos, exit_radius)

        # Convert grid to blocks and their wall segments
        self.converter = GridToWallsConverter(self.position_calculator, self.strong_cells)
        self.wall_renderer = WallRenderer()
        self.wall_debris = WallDebris()
        self.flora: Optional[Flora] = None  # living weed, when the game grows it here
        # Spatial grid for efficient collision detection
        self.spatial_grid = SpatialGrid(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            cell_size=CURVED_BUCKET if self.curved else 150.0  # square walls keep the bucket they always had
        )
        self.wall_generation = 0  # Goes up each time a block is destroyed
        self._sight_cache = None  # (walls list, its length, wall_generation, sight segments)
        # Curved blocks destroyed inside batched_destroys(), whose rock is rebuilt when it ends
        self._batch_depth = 0
        self._pending_opens: Dict[Tuple[int, int], None] = {}
        # Centre lines of the physics gates' planks, for everything that sees walls (set by the game each frame)
        self.gate_segments: List[WallSegment] = []
        self.rebuild_walls()

    
    def check_exit_reached(self, pos: Tuple[float, float], radius: float) -> bool:
        """Check if player reached the exit."""
        return self.exit.check_circle_collision(pos, radius)
    
    def _gate_keep_clear(self) -> List[Tuple[Tuple[float, float], float]]:
        """Circles the cave rock must stay out of: each gate's swing, with a little room to spare."""
        if self.layout is None or not self.layout.gates:
            return []
        from physics.obstacle_world import plank_size  # here, not at the top: obstacle_world imports this module
        half, thickness = plank_size(self)
        radius = half + thickness / 2 + 2
        return [(self.position_calculator.grid_center_to_screen(*gate.cell), radius) for gate in self.layout.gates]

    def _spawn_keep_clear(self) -> List[Tuple[Tuple[float, float], float]]:
        """Circles the cave rock must stay out of so nothing starts inside a bulge: the ship's start,
        the exit portal, and a layout's beacon and creatures (each its own size, plus a pixel)."""
        out = [(self.start_pos, config.SHIP_SIZE + 1), (self.exit.get_pos(), self.exit.radius)]
        if self.layout is None:
            return out  # generated mazes place creatures later, only where is_open finds room
        import enemy_kinds  # here, not at the top: it loads the enemy catalogue
        from entities.jev_beacon import JevBeacon
        calc = self.position_calculator
        if self.layout.beacon is not None:
            # The beacon's turning ring, the widest part of it
            out.append((calc.grid_center_to_screen(*self.layout.beacon),
                        JevBeacon.RING_RADIUS * config.POWERUP_CRYSTAL_SIZE + 1))
        out.extend((calc.grid_center_to_screen(*cell), enemy_kinds.kind(kind).tuning['size'] + 1)
                   for kind, cell in self.layout.enemies)
        return out

    def rebuild_walls(self) -> None:
        """Build the blocks, their exposed faces and the collision index from the grid."""
        # Hit points left in each wall block, keyed by grid cell (x, y). A
        # block is solid rock: shots at any of its faces wear down the block
        self.blocks: Dict[Tuple[int, int], int] = {
            (x, y): config.WALL_HIT_POINTS
            for y, row in enumerate(self.grid) for x, value in enumerate(row) if value == 1
        }
        if self.curved:
            # Curved rock: many short segments per block, each knowing the block it damages
            self.cave = CaveShape(
                self.grid, self.position_calculator, self.converter, strong=self.strong_cells,
                keep_clear=self._gate_keep_clear() + self._spawn_keep_clear(), seed=self.cave_seed, settings=config.CAVE,
                hit_points=lambda cell: self.blocks.get(cell, config.WALL_HIT_POINTS))
            self.walls = self.cave.segments()
        else:
            self.cave = None
            self.walls = self.converter.convert(self.grid)
        # Screen areas of the outer ring's blocks, which wear down but are never destroyed
        self.unbreakable_blocks = frozenset(
            self.converter.cell_rect(x, y) for x, y in self.blocks if self.converter.on_perimeter(self.grid, x, y)
        )
        # Screen areas of the strong walls, which shots never mark
        self.strong_blocks = frozenset(self.converter.cell_rect(x, y) for x, y in self.strong_cells)
        # Where each block has been hit, as points on its edge, oldest first
        self.wounds: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._pending_opens = {}
        self.spatial_grid.add_walls(self.walls)
        self._refresh_block_fills()
        self._refresh_territories()
    
    # Blocks this many cells from a destroyed one may have their rock re-formed
    TERRITORY_REACH = 2

    @property
    def block_territories(self) -> Optional[Dict[Tuple[int, int, int, int], Tuple[pygame.Rect, pygame.mask.Mask]]]:
        """Each block's curved rock, keyed like block_fills, for the renderer: (screen area, mask of its
        pixels). None for square walls. Worked out when first asked for after the rock changes, so a
        destroy does not pay for it; then only blocks near those destroyed are worked out again."""
        if self.cave is None:
            return None
        if self._territories is None or self._stale_territories:
            if self._territories is None:
                by_cell, cells = {}, list(self.blocks)
            else:
                by_cell = {cell: value for cell, value in self._territories_by_cell.items() if cell in self.blocks}
                reach, destroyed = self.TERRITORY_REACH, self._stale_territories
                cells = [cell for cell in self.blocks
                         if any(abs(cell[0] - x) <= reach and abs(cell[1] - y) <= reach for x, y in destroyed)]
            for cell in cells:
                by_cell[cell] = self.cave.territory(cell)
            self._territories_by_cell = by_cell
            self._stale_territories = []
            self._territories = {
                self.converter.cell_rect(*cell): value for cell, value in by_cell.items() if value is not None}
        return self._territories

    def _refresh_territories(self, destroyed: Optional[List[Tuple[int, int]]] = None) -> None:
        """Mark the curved rock's territories out of date: all of them, or those near destroyed blocks."""
        if destroyed is None:
            self._territories = None
            self._territories_by_cell = {}
            self._stale_territories = []
        else:
            self._stale_territories = self._stale_territories + list(destroyed)

    def _refresh_block_fills(self) -> None:
        """Rebuild the screen areas of the blocks, which the renderer fills with rock."""
        self.block_fills = {self.converter.cell_rect(x, y): hit_points for (x, y), hit_points in self.blocks.items()}
        self.block_wounds = {self.converter.cell_rect(x, y): tuple(wounds) for (x, y), wounds in self.wounds.items()}
    
    @staticmethod
    def _wound(wall: WallSegment, impact: Optional[Tuple[float, float]]) -> Tuple[int, int]:
        """The point on a face nearest to where a shot landed; its middle if that is unknown."""
        (ax, ay), (bx, by) = wall.start, wall.end
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        along = 0.5
        if impact is not None and length_sq > 0:
            along = ((impact[0] - ax) * dx + (impact[1] - ay) * dy) / length_sq
        # Kept off the very corner, which belongs to the neighbouring side as much as this one
        inset = min(0.5, 3.0 / math.sqrt(length_sq)) if length_sq > 0 else 0.5
        along = max(inset, min(1.0 - inset, along))
        return (int(round(ax + dx * along)), int(round(ay + dy * along)))
    
    def damage_wall(self, wall: WallSegment, impact: Optional[Tuple[float, float]] = None,
                    *, damage: int = 1) -> bool:
        """Damage the block a wall segment is a face of. Returns True if the block was destroyed.
        
        The whole block goes at once, and the faces of its neighbours that
        looked onto it become walls. Either way the shot knocks rock loose.
        A block of the outer ring cracks and sheds rock but is never destroyed.

        Args:
            wall: The wall segment that was hit.
            impact: Where the shot landed, for the chips it knocks off.
            damage: Hit points removed in one impact (rail shots remove the whole block).
            
        Returns:
            True if the block was destroyed (hit points reached 0), False otherwise.
        """
        if not wall.active or wall not in self.walls:
            # Gone already, perhaps replaced by a segment with the same ends when the rock was rebuilt
            return False

        cell = wall.cell
        if cell in self._pending_opens:
            # Destroyed earlier in this batch: its old segments wait for the rebuild at the end
            return False
        if cell in self.strong_cells:
            # Metal: the shot strikes sparks and leaves no mark
            self.wall_debris.spark(impact if impact is not None else wall.start)
            return False
        if cell not in self.blocks:
            # A loose segment that belongs to no block
            destroyed = wall.damage(damage)
            if destroyed:
                self.spatial_grid.update_wall(wall)
                self.walls = [w for w in self.walls if w.active]
            return destroyed

        faces = [w for w in self.walls if w.cell == cell]
        hit_points = self.blocks[cell] - damage
        if not wall.destructible:
            # The outer ring wears down like any block, but its last hit point never goes
            hit_points = max(1, hit_points)
        for face in faces:
            face.hit_points = hit_points
        if hit_points > 0:
            if hit_points < self.blocks[cell]:
                self.blocks[cell] = hit_points
                self.wounds.setdefault(cell, []).append(self._wound(wall, impact))
                self._refresh_block_fills()
            self.wall_debris.hit(self.converter.cell_rect(*cell), impact)
            return False
        
        del self.blocks[cell]
        self.wounds.pop(cell, None)
        self.wall_debris.destroy(self.converter.cell_rect(*cell))
        self.grid[cell[1]][cell[0]] = 0
        if self.cave is not None:
            if self._batch_depth:
                self._pending_opens[cell] = None
            else:
                self._reshape_cave([cell])
            return True
        for face in faces:
            face.active = False
            self.spatial_grid.update_wall(face)
        self.walls = [w for w in self.walls if w.active]
        # Neighbouring blocks now have a face onto the gap
        for dx, dy in SIDES:
            neighbour = (cell[0] + dx, cell[1] + dy)
            if neighbour in self.blocks:
                face = self.converter.face(self.grid, neighbour, (-dx, -dy), self.blocks[neighbour])
                self.walls.append(face)
                self.spatial_grid.add_wall(face)
        self.wall_generation += 1
        self._refresh_block_fills()
        return True
    
    @contextmanager
    def batched_destroys(self) -> Iterator[None]:
        """Defer the curved rock rebuild for blocks destroyed inside the block to its end, so a
        blast that breaks several blocks rebuilds the rock once. Until then a destroyed block's
        old segments stay in place, and damage_wall ignores them. Square walls are unaffected."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if not self._batch_depth and self._pending_opens:
                cells, self._pending_opens = list(self._pending_opens), {}
                self._reshape_cave(cells)

    def _reshape_cave(self, cells: List[Tuple[int, int]]) -> None:
        """Rebuild the curved rock around destroyed blocks and patch the walls and their index."""
        if len(cells) == 1:
            removed, added = self.cave.open_cell(cells[0])
        else:
            removed, added = self.cave.open_cells(cells)
        # Removed first: a segment whose owner alone changed comes back with the same ends. WallSegment
        # equality and hashing use only the ends, so the spatial grid's wall_to_index maps both to one
        # entry; update_wall looks the index up there and drops it from the segment's cells. Added
        # first, the new segment's index would be the one dropped, and the old one would stay indexed
        for segment in removed:
            segment.active = False
            self.spatial_grid.update_wall(segment)
        self.walls = [w for w in self.walls if w.active]
        for segment in added:
            segment.hit_points = self.blocks[segment.cell]
            self.walls.append(segment)
            self.spatial_grid.add_wall(segment)
        self.wall_generation += len(cells)  # one per destroyed block, as for square walls
        self._refresh_block_fills()
        self._refresh_territories(cells)

    def sight_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """The active walls as (start, end) for line of sight. Curved rock's short segments are
        merged within hunter.visibility.SIGHT_TOLERANCE, which makes sight several times cheaper;
        square walls are passed through. Cached until the walls change."""
        walls = self.walls
        cached = self._sight_cache
        if cached is not None and cached[0] is walls and cached[1] == len(walls) and cached[2] == self.wall_generation:
            return cached[3]
        segments = [(w.start, w.end) for w in walls if w.active]
        if self.curved:
            segments = simplify_outline(segments)
        self._sight_cache = (walls, len(walls), self.wall_generation, segments)
        return segments

    def get_valid_spawn_positions(self, count: int, min_distance: float = 100,
                                  start_clearance: Optional[float] = None) -> List[Tuple[float, float]]:
        """Get valid spawn positions for enemies, avoiding walls.
        
        Args:
            count: Number of positions wanted.
            min_distance: Minimum distance between positions, and from the exit.
            start_clearance: Minimum distance from the player's start (defaults to min_distance).
        
        When the maze is too full to fit them all, positions are packed closer together;
        the distances from the start and the exit are never reduced.
        """
        if start_clearance is None:
            start_clearance = min_distance
        start_clearance_sq = start_clearance * start_clearance
        min_distance_sq = min_distance * min_distance
        spacing_sq = min_distance_sq
        positions = []
        attempts = 0
        max_attempts = count * 50
        
        while len(positions) < count:
            if attempts >= max_attempts:
                # Out of room at this spacing: pack the rest closer, down to half the spacing
                spacing_sq *= 0.64
                if spacing_sq < min_distance_sq * 0.25:
                    break
                attempts = 0
            attempts += 1
            
            # Random position in maze (with offset)
            x = random.uniform(
                self.offset_x + self.cell_size_x * 2,
                self.offset_x + (self.grid_width - 2) * self.cell_size_x
            )
            y = random.uniform(
                self.offset_y + self.cell_size_y * 2,
                self.offset_y + (self.grid_height - 2) * self.cell_size_y
            )
            pos = (x, y)
            
            # Check if too close to start or exit (use squared distance for comparison)
            exit_pos = self.exit.get_pos()
            if (distance_squared(pos, self.start_pos) < start_clearance_sq or
                distance_squared(pos, exit_pos) < min_distance_sq):
                continue
            
            # Check if too close to other spawns
            too_close = False
            for existing_pos in positions:
                if distance_squared(pos, existing_pos) < spacing_sq:
                    too_close = True
                    break
            if too_close:
                continue
            
            if not self.is_open(pos, 15):
                continue
            
            positions.append(pos)
        
        return positions
    
    def is_open(self, pos: Tuple[float, float], radius: float) -> bool:
        """Whether a circle of this radius at pos sits in open space: centre inside the maze and outside every block, touching no wall."""
        cell = (int((pos[0] - self.offset_x) // self.cell_size_x), int((pos[1] - self.offset_y) // self.cell_size_y))
        if not (0 <= cell[0] < self.grid_width and 0 <= cell[1] < self.grid_height) or cell in self.blocks:
            return False
        if self.cave is not None and self.cave.is_rock(pos):
            return False
        for wall in self.spatial_grid.get_nearby_walls(pos, radius):
            if wall.active:
                segment = wall.get_segment()
                if circle_line_collision(pos, radius, segment[0], segment[1]):
                    return False
        return True
    
    def open_spot_near(self, origin: Tuple[float, float], radius: float,
                       max_offset: float, min_offset: float = 0.0, attempts: int = 20) -> Tuple[float, float]:
        """A random open position between min_offset and max_offset from origin, for something appearing there.
        
        Falls back to origin itself when no open spot turns up, so nothing is placed inside a block.
        """
        for _ in range(attempts):
            angle = random.uniform(0, 2 * math.pi)
            offset = random.uniform(min_offset, max_offset)
            pos = (origin[0] + math.cos(angle) * offset, origin[1] + math.sin(angle) * offset)
            if self.is_open(pos, radius):
                return pos
        return origin
    
    def nearest_open(self, origin: Tuple[float, float], radius: float, reach: float,
                     step: float = 4.0) -> Optional[Tuple[float, float]]:
        """The closest position to origin (within reach) where a circle of this radius sits in open space: origin itself
        if it does, else the first open point on rings growing out from it; None when there is none."""
        if self.is_open(origin, radius):
            return origin
        distance = step
        while distance <= reach:
            points = max(8, int(2 * math.pi * distance / step))
            for n in range(points):
                angle = 2 * math.pi * n / points
                pos = (origin[0] + math.cos(angle) * distance, origin[1] + math.sin(angle) * distance)
                if self.is_open(pos, radius):
                    return pos
            distance += step
        return None

    def update(self, dt: float) -> None:
        """Advance the rock knocked loose by shots."""
        self.wall_debris.update(dt)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the maze."""
        # Living weed takes the place of the rock's painted weed
        self.wall_renderer.draw(
            screen, self.walls, animate=self.flora is None, blocks=self.block_fills, wounds=self.block_wounds,
            unbreakable=self.unbreakable_blocks, strong=self.strong_blocks, territories=self.block_territories,
            strong_cells=self.strong_cells
        )
        if self.flora is not None:
            draw_flora(screen, self.flora)
        self.wall_debris.draw(screen)
        
        # Draw exit marker
        if self.exit.active:
            self.exit.draw(screen)

