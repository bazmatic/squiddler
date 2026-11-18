"""Spatial partitioning grid for efficient collision detection.

This module provides a grid-based spatial partitioning system to reduce
the number of collision checks from O(n*m) to O(k) where k is the number
of entities in nearby grid cells.
"""

import math
from typing import List, Tuple, Set
import config


class SpatialGrid:
    """Grid-based spatial partition for collision detection optimization.
    
    Divides the game space into a grid of cells. Each wall is assigned to
    the cells it overlaps with, allowing entities to only check walls
    in nearby cells instead of all walls.
    """
    
    def __init__(self, width: float, height: float, cell_size: float = 100.0):
        """Initialize spatial grid.
        
        Args:
            width: Total width of the game space.
            height: Total height of the game space.
            cell_size: Size of each grid cell (default: 100 pixels).
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Calculate grid dimensions
        self.grid_cols = int(width / cell_size) + 1
        self.grid_rows = int(height / cell_size) + 1
        
        # Grid: list of sets, each set contains wall indices
        # Using indices instead of wall objects for efficiency
        self.grid: List[List[Set[int]]] = [
            [set() for _ in range(self.grid_cols)]
            for _ in range(self.grid_rows)
        ]
        
        # Store all walls with their indices
        self.walls: List = []
        self.wall_to_index: dict = {}
    
    def clear(self) -> None:
        """Clear all walls from the grid."""
        for row in self.grid:
            for cell in row:
                cell.clear()
        self.walls.clear()
        self.wall_to_index.clear()
    
    def add_walls(self, walls: List) -> None:
        """Add walls to the spatial grid.
        
        Args:
            walls: List of wall segments (WallSegment instances or tuples).
        """
        self.clear()
        
        for wall in walls:
            # Handle both WallSegment and tuple formats
            if hasattr(wall, 'get_segment'):
                if not wall.active:
                    continue
                segment = wall.get_segment()
            else:
                segment = wall
            
            # Add wall to storage
            wall_index = len(self.walls)
            self.walls.append(wall)
            self.wall_to_index[wall] = wall_index
            
            # Find all grid cells this wall overlaps with
            cells = self._get_cells_for_line(segment[0], segment[1])
            for row, col in cells:
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    self.grid[row][col].add(wall_index)
    
    def get_nearby_walls(
        self,
        pos: Tuple[float, float],
        radius: float
    ) -> List:
        """Get walls that are potentially colliding with an entity.
        
        Args:
            pos: Entity position (x, y).
            radius: Entity collision radius.
            
        Returns:
            List of walls that might be colliding (need further collision check).
        """
        # Calculate bounding box around entity
        min_x = pos[0] - radius
        max_x = pos[0] + radius
        min_y = pos[1] - radius
        max_y = pos[1] + radius
        
        # Get grid cells that overlap with bounding box
        min_col = max(0, int(min_x / self.cell_size))
        max_col = min(self.grid_cols - 1, int(max_x / self.cell_size))
        min_row = max(0, int(min_y / self.cell_size))
        max_row = min(self.grid_rows - 1, int(max_y / self.cell_size))
        
        # Collect unique wall indices from overlapping cells
        wall_indices: Set[int] = set()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                wall_indices.update(self.grid[row][col])
        
        # Return actual wall objects
        return [self.walls[i] for i in wall_indices]
    
    def get_walls_along_path(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        radius: float
    ) -> List:
        """Get walls that might intersect with a swept path.
        
        This method queries walls along the entire movement path, not just
        at the end position. This prevents tunneling through walls during
        large movements or high-speed scenarios.
        
        Args:
            start_pos: Starting position (x, y).
            end_pos: Ending position (x, y).
            radius: Entity collision radius.
            
        Returns:
            List of walls that might intersect the swept path.
        """
        # Calculate bounding box that covers entire movement path
        # Include radius on all sides to account for entity size
        min_x = min(start_pos[0], end_pos[0]) - radius
        max_x = max(start_pos[0], end_pos[0]) + radius
        min_y = min(start_pos[1], end_pos[1]) - radius
        max_y = max(start_pos[1], end_pos[1]) + radius
        
        cell = self.cell_size
        min_col = max(0, int(min_x / cell))
        max_col = min(self.grid_cols - 1, int(max_x / cell))
        min_row = max(0, int(min_y / cell))
        max_row = min(self.grid_rows - 1, int(max_y / cell))
        grid = self.grid
        wall_indices: Set[int] = set()

        if (max_col - min_col + 1) * (max_row - min_row + 1) <= 6:
            # A short path: its whole bounding box is only a few cells.
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    wall_indices.update(grid[row][col])
        else:
            # A long slanted path touches only a strip of its bounding box. Walk it along its longer axis one cell
            # band at a time and take only the cells the swept circle can reach within each band.
            x0, y0 = start_pos
            x1, y1 = end_pos
            along_x = abs(x1 - x0) >= abs(y1 - y0)
            if not along_x:
                x0, y0, x1, y1 = y0, x0, y1, x1
            lo_band, hi_band = (min_col, max_col) if along_x else (min_row, max_row)
            cross_count = self.grid_rows if along_x else self.grid_cols
            span = x1 - x0
            slope = (y1 - y0) / span if span else 0.0
            low_y, high_y = (y0, y1) if y0 <= y1 else (y1, y0)
            lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
            for band in range(lo_band, hi_band + 1):
                # The stretch of the path whose along-axis position lies within radius of this band
                band_lo = max(lo_x, band * cell - radius)
                band_hi = min(hi_x, (band + 1) * cell + radius)
                if band_lo > band_hi:
                    continue
                ya, yb = y0 + (band_lo - x0) * slope, y0 + (band_hi - x0) * slope
                if ya > yb:
                    ya, yb = yb, ya
                lo_cross = max(0, int((ya - radius) / cell))
                hi_cross = min(cross_count - 1, int((yb + radius) / cell))
                if along_x:
                    for row in range(lo_cross, hi_cross + 1):
                        wall_indices.update(grid[row][band])
                else:
                    row_cells = grid[band]
                    for col in range(lo_cross, hi_cross + 1):
                        wall_indices.update(row_cells[col])

        # Return actual wall objects
        walls = self.walls
        return [walls[i] for i in wall_indices]
    
    def _get_cells_for_line(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> List[Tuple[int, int]]:
        """Get all grid cells that a line segment passes through.
        
        Every cell the line's bounding box touches.
        
        Args:
            start: Line start point (x, y).
            end: Line end point (x, y).
            
        Returns:
            List of (row, col) tuples for cells the line passes through.
        """
        # Every cell the segment's bounding box touches: exact for horizontal and vertical lines, and
        # never misses a cell a slanted line crosses (it may add a few it does not)
        x1, y1 = start
        x2, y2 = end
        col1, col2 = sorted((int(x1 / self.cell_size), int(x2 / self.cell_size)))
        row1, row2 = sorted((int(y1 / self.cell_size), int(y2 / self.cell_size)))
        cells = [(row, col) for row in range(row1, row2 + 1) for col in range(col1, col2 + 1)]
        return cells
    
    def add_wall(self, wall) -> None:
        """Add one more wall to the grid, keeping those already there.
        
        Args:
            wall: The wall segment to add (WallSegment instance or tuple).
        """
        segment = wall.get_segment() if hasattr(wall, 'get_segment') else wall
        wall_index = len(self.walls)
        self.walls.append(wall)
        self.wall_to_index[wall] = wall_index
        for row, col in self._get_cells_for_line(segment[0], segment[1]):
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                self.grid[row][col].add(wall_index)
    
    def update_wall(self, wall) -> None:
        """Update a wall's position in the grid (e.g., when it's destroyed).
        
        Args:
            wall: The wall segment to update.
        """
        if wall not in self.wall_to_index:
            return
        
        wall_index = self.wall_to_index[wall]
        
        # Remove from the cells it was added to (a wall's ends never move)
        segment = wall.get_segment() if hasattr(wall, 'get_segment') else wall
        for row, col in self._get_cells_for_line(segment[0], segment[1]):
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                self.grid[row][col].discard(wall_index)
        
        # Re-add if still active
        if hasattr(wall, 'active') and wall.active:
            if hasattr(wall, 'get_segment'):
                segment = wall.get_segment()
            else:
                segment = wall
            
            cells = self._get_cells_for_line(segment[0], segment[1])
            for row, col in cells:
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    self.grid[row][col].add(wall_index)



