"""Rotating thruster ship base class.

This module provides the RotatingThrusterShip base class for ships that use
rotating thruster movement mechanics. This includes rotation, thrust application,
momentum-based physics, and collision detection.

Subclasses can implement their own control mechanisms (keyboard, AI, etc.)
and add additional features like fuel management, sound effects, etc.
"""

import pygame
import math
import random
from typing import Tuple, List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
import config
from physics import thrust

if TYPE_CHECKING:
    from entities.base import GameEntity
from utils import (
    angle_to_radians,
    normalize_angle,
    rotate_point,
    circle_line_collision,
    circle_line_collision_swept,
    circle_circle_collision,
    distance,
    distance_squared,
    get_wall_normal,
    get_closest_point_on_line,
    reflect_velocity,
)
from utils.math_utils import apply_circle_collision_physics, apply_wall_collision_physics
from entities.base import GameEntity
from entities.collidable import Collidable
from entities.drawable import Drawable


class RotatingThrusterShip(GameEntity, Collidable, Drawable):
    """Base class for ships with rotating thruster movement mechanics.
    
    This class provides the core movement system including:
    - Rotation (left/right)
    - Thrust application in facing direction
    - Momentum-based physics with friction
    - Edge bouncing
    - Collision detection (walls and circles)
    - Thrust particle visualization
    
    Subclasses can:
    - Override apply_thrust() to add constraints (e.g., fuel checks)
    - Override draw() to customize appearance
    - Add their own control mechanisms (keyboard, AI, etc.)
    - Add additional features (fuel, ammo, sound, etc.)
    
    Attributes:
        angle: Current facing angle in degrees (0 = right, 90 = down).
        thrust_particles: List of thrust particle dictionaries for visualization.
        thrusting: Whether thrust is currently being applied this frame.
        prev_x: Previous X position for swept collision detection.
        prev_y: Previous Y position for swept collision detection.
    """
    
    def __init__(self, start_pos: Tuple[float, float], radius: float):
        """Initialize rotating thruster ship at starting position.
        
        Args:
            start_pos: Starting position as (x, y) tuple.
            radius: Collision radius of the ship.
        """
        super().__init__(start_pos, radius, 0.0, 0.0)
        self.angle = 0.0  # degrees, 0 = right, 90 = down
        self.thrust_particles: List[dict] = []  # For enhanced thrust visualization
        self.thrusting = False  # Track when thrust is actively being applied
        self.prev_x = self.x  # Previous position for swept collision detection
        self.prev_y = self.y
        self.base_rotation_speed = config.SHIP_ROTATION_SPEED
        self.rotation_speed_multiplier = 1.0
        self.wall_impulse = (0.0, 0.0)  # velocity change from the most recent wall check only
    
    @property
    def max_speed(self) -> float:
        """Get the maximum speed for this ship.
        
        Subclasses can override this to provide different max speeds.
        
        Returns:
            Maximum speed value.
        """
        return config.SHIP_MAX_SPEED

    @property
    def thrust_force(self) -> float:
        """Get the acceleration applied per thrust frame.

        Subclasses can override this to provide a different thrust.

        Returns:
            Thrust force value.
        """
        return config.SHIP_THRUST_FORCE
    
    @property
    def friction(self) -> float:
        return config.SHIP_FRICTION

    def rotate_left(self) -> None:
        """Rotate ship counter-clockwise."""
        self.angle -= self.current_rotation_speed
        self.angle = normalize_angle(self.angle)
    
    def rotate_right(self) -> None:
        """Rotate ship clockwise."""
        self.angle += self.current_rotation_speed
        self.angle = normalize_angle(self.angle)

    @property
    def current_rotation_speed(self) -> float:
        return self.base_rotation_speed * self.rotation_speed_multiplier
    
    def apply_thrust(self) -> bool:
        """Apply thrust in current direction.
        
        This base implementation always applies thrust. Subclasses can override
        to add constraints (e.g., fuel checks).
        
        Returns:
            True if thrust was applied, False otherwise.
        """
        thrust.apply_thrust(self, self.thrust_force, self.max_speed)
        self.thrusting = True
        return True

    def update(self, dt: float) -> None:
        """Update ship position and apply friction.
        
        Args:
            dt: Delta time since last update.
        """
        # Save thrusting state from previous frame (set by apply_thrust)
        # This allows it to persist through the draw() call
        was_thrusting = self.thrusting
        
        # Clear thrusting flag at start of update (will be set again if apply_thrust is called this frame)
        self.thrusting = False
        
        # Move with friction (remembering where it was, for swept wall collision), bouncing off screen edges
        for _ in range(thrust.coast(self, dt, self.friction)):
            self.on_edge_collision()
        
        # Update thrust particles (only when thrusting from previous frame)
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if was_thrusting and speed > 0.0:
            # Add new particles based on speed
            angle_rad = angle_to_radians(self.angle)
            for _ in range(int(speed * 0.5)):
                if len(self.thrust_particles) < config.THRUST_PLUME_PARTICLES * 3:
                    particle_x = -math.cos(angle_rad) * self.radius * 0.8
                    particle_y = -math.sin(angle_rad) * self.radius * 0.8
                    particle_vx = -math.cos(angle_rad) * speed * 0.3
                    particle_vy = -math.sin(angle_rad) * speed * 0.3
                    self.thrust_particles.append({
                        'x': particle_x,
                        'y': particle_y,
                        'vx': particle_vx,
                        'vy': particle_vy,
                        'life': config.THRUST_PLUME_LENGTH,
                        'size': random.uniform(2, 4)
                    })
        
        # Update existing particles (use list comprehension instead of remove)
        for particle in self.thrust_particles:
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['life'] -= 1
        # Filter out dead particles
        self.thrust_particles = [p for p in self.thrust_particles if p['life'] > 0]
        
        # Enforce maximum speed limit (after all physics updates)
        # This ensures speed never exceeds max, regardless of collisions, bounces, etc.
        thrust.cap_speed(self, self.max_speed)
    
    def check_wall_collision(
        self,
        walls: List,
        spatial_grid=None
    ) -> bool:
        """Check collision with walls using continuous collision detection.
        
        Uses swept collision detection to prevent tunneling through walls
        at high speeds. Finds the earliest collision along the movement path
        and stops movement at that point.
        
        Args:
            walls: List of wall segments (WallSegment instances or tuples).
            spatial_grid: Optional spatial grid for optimized collision detection.
            
        Returns:
            True if collision occurred, False otherwise.
        """
        before = self.vx, self.vy
        self.wall_impulse = (0.0, 0.0)
        if thrust.sweep_walls(self, walls, spatial_grid):
            self.wall_impulse = (self.vx - before[0], self.vy - before[1])
            self.on_wall_collision()   # hook for damage, sound, etc.
            return True
        return False

    def check_circle_collision(
        self,
        other_pos: Tuple[float, float],
        other_radius: float,
        other_entity: Optional[GameEntity] = None
    ) -> bool:
        """Check collision with another circular entity.
        
        Uses proper elastic collision physics when other_entity is provided,
        otherwise falls back to simple push-back for backward compatibility.
        
        Args:
            other_pos: Position of the other entity (x, y).
            other_radius: Radius of the other entity.
            other_entity: Optional GameEntity object. If provided, both objects'
                         velocities will be updated using conservation of momentum.
            
        Returns:
            True if collision occurred, False otherwise.
        """
        if circle_circle_collision(
            (self.x, self.y), self.radius,
            other_pos, other_radius
        ):
            if other_entity is not None:
                # Use proper physics with conservation of momentum
                apply_circle_collision_physics(self, other_entity, config.COLLISION_RESTITUTION)
            else:
                # Backward compatibility: simple push-back
                dx = self.x - other_pos[0]
                dy = self.y - other_pos[1]
                dist_sq = distance_squared((self.x, self.y), other_pos)
                if dist_sq > 0:
                    dist = math.sqrt(dist_sq)
                    push_force = 2.0
                    self.vx += (dx / dist) * push_force
                    self.vy += (dy / dist) * push_force
            
            # Notify subclass of collision (hook for damage, sound, etc.)
            self.on_circle_collision()
            return True
        return False
    
    def on_edge_collision(self) -> None:
        """Hook method called when screen edge collision occurs.
        
        Subclasses can override this to handle collision effects
        (e.g., damage, sound, visual feedback).
        """
        pass
    
    def on_wall_collision(self) -> None:
        """Hook method called when wall collision occurs.
        
        Subclasses can override this to handle collision effects
        (e.g., damage, sound, visual feedback).
        """
        pass
    
    def on_circle_collision(self) -> None:
        """Hook method called when circle collision occurs.
        
        Subclasses can override this to handle collision effects
        (e.g., damage, sound, visual feedback).
        """
        pass
    
    def draw_thrust_plume(self, screen: pygame.Surface, min_length: float = 0.0) -> None:
        """Draw the shared thrust plume."""
        from rendering.thrust_plume import draw_thrust_plume
        draw_thrust_plume(self, screen, min_length)

    def get_vertices(self) -> List[Tuple[float, float]]:
        """Get ship vertices for rendering (triangle shape).
        
        Returns:
            List of (x, y) vertex coordinates.
        """
        # Triangle pointing right (0 degrees) with near-equilateral proportions
        half_height = self.radius * math.sqrt(3) * 0.5
        rear_x = -self.radius * 0.5
        base_vertices = [
            (self.radius, 0.0),           # Nose
            (rear_x, -half_height),       # Left rear
            (rear_x, half_height)         # Right rear
        ]
        
        # Rotate vertices around center
        center = (0, 0)
        rotated_vertices = []
        for vertex in base_vertices:
            rotated = rotate_point(vertex, center, self.angle)
            rotated_vertices.append((rotated[0] + self.x, rotated[1] + self.y))
        
        return rotated_vertices
    
    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the ship on screen.
        
        Subclasses must implement this to provide their own visual representation.
        
        Args:
            screen: The pygame Surface to draw on.
        """
        pass

