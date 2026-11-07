"""Collision detection and handling for game entities.

This module centralizes all collision detection logic, eliminating duplication
and making collision handling easier to maintain and extend.
"""

from copy import deepcopy
import math
import random
from typing import List, Optional, Tuple, TYPE_CHECKING
import config
from entities.powerup_crystal import PowerupCrystal
from entities.jev_beacon import JevBeacon
from entities.explosion import Explosions, push, reaches
from entities.projectile import Projectile
from collections import Counter
from creatures.traits.energy import damage, register_hit


def _damage_hunter(hunter):
    return damage(hunter, hunter.encounter)

if TYPE_CHECKING:
    from entities.ship import Ship
    from entities.explosion import Blast
    from maze.generator import Maze
    from scoring.system import ScoringSystem
    from sounds import SoundManager


class CollisionHandler:
    """Handles all collision detection and response in the game."""
    
    def __init__(
        self,
        sound_manager: 'SoundManager',
        scoring: 'ScoringSystem',
        command_recorder
    ):
        """Initialize collision handler.
        
        Args:
            sound_manager: Sound manager for playing collision sounds.
            scoring: Scoring system for recording collisions.
            command_recorder: Command recorder for spawning replay enemies.
        """
        self.sound_manager = sound_manager
        self.scoring = scoring
        self.command_recorder = command_recorder
        self.last_player_kill = None
        # This level's kills and crystals by the player and the Jev hunter, for the stars
        self.level_kills = 0
        self.level_crystals_collected = 0
        self.weapon_kills = Counter()
        self.weapon_pickups = Counter()
        # Blasts lit and fireballs burning; the game applies each blast with handle_blast as it goes off
        self.explosions = Explosions()
    
    def record_kill(self, projectile, enemy) -> None:
        """Count a kill for whoever fired the shot (or lit the blast), and light the blast of anything that explodes."""
        if projectile.source in ('player', 'hunter'):
            self.level_kills += 1
            weapon_id = getattr(projectile, 'weapon_id', None)
            if weapon_id:
                self.weapon_kills[weapon_id] += 1
        if projectile.source == 'player':
            self.scoring.record_enemy_destroyed()
            self.last_player_kill = deepcopy(enemy)
        blast = getattr(enemy, 'blast', None)
        if blast is not None:
            self.explosions.trigger(blast(projectile.source))

    def drop_pickup(
        self,
        pos: Tuple[float, float],
        powerup_crystals: List['PowerupCrystal'],
        jev_beacon_allowed: bool
    ) -> None:
        """Roll for what a destroyed enemy leaves behind.
        
        A Jev beacon takes the place of the crystal, and only one lies on the
        field at a time.
        
        Args:
            pos: Where the enemy died.
            powerup_crystals: List to add the pickup to.
            jev_beacon_allowed: Whether this kill may drop a Jev beacon.
        """
        if (jev_beacon_allowed
                and not any(isinstance(pickup, JevBeacon) and pickup.active for pickup in powerup_crystals)
                and random.random() < config.JEV_BEACON_SPAWN_CHANCE):
            powerup_crystals.append(JevBeacon(pos))
            self.sound_manager.play_beacon_arrival()
        elif random.random() < config.POWERUP_CRYSTAL_SPAWN_CHANCE:
            from weapons.catalog import CATALOG, PICKUP_RULESETS
            catalog = getattr(self, 'weapon_catalog', CATALOG)
            weapon_id = catalog.choose_drop(random, getattr(self, 'weapon_allowlist', None)) if getattr(self, 'combat_ruleset', 'legacy') in PICKUP_RULESETS else None
            powerup_crystals.append(PowerupCrystal(pos, weapon_id, catalog))

    def handle_projectile_ship_collision(
        self,
        projectile: 'Projectile',
        ship: 'Ship',
        scoring: 'ScoringSystem'
    ) -> bool:
        """Handle collision between enemy projectile and ship.
        
        Args:
            projectile: The enemy projectile.
            ship: The player ship.
            scoring: Scoring system for recording collisions.
            
        Returns:
            True if collision occurred, False otherwise.
        """
        if not projectile.is_enemy or not projectile.active:
            return False
        
        if ship.is_shield_active():
            return False
        
        if projectile.check_circle_collision((ship.x, ship.y), ship.radius):
            ship.flinch()
            callback = getattr(self, 'on_player_projectile_hit', None)
            if callback is not None:
                callback(projectile)
            scoring.record_enemy_bullet_hit(projectile.strength)  # a bullet costs the bullet penalty, nothing more
            self.sound_manager.play_bad_hit()
            # Apply physics-based collision (projectile has very small mass)
            # Use proper conservation of momentum for realistic impact
            from utils.math_utils import apply_circle_collision_physics, calculate_entity_mass
            # Projectiles have very small mass compared to ship
            # Use a reduced restitution for projectile impacts (more inelastic)
            projectile_restitution = 0.3  # Projectiles stick/transfer more energy
            before = ship.vx, ship.vy
            apply_circle_collision_physics(ship, projectile, projectile_restitution)
            # A weaker bullet pushes the ship less
            ship.vx = before[0] + (ship.vx - before[0]) * projectile.strength
            ship.vy = before[1] + (ship.vy - before[1]) * projectile.strength
            return True
        
        return False
    
    def handle_blast(self, blast: 'Blast', ship, hunter, enemies, hit_enemy, maze: Optional['Maze'] = None) -> None:
        """A blast goes off: it shoves and hurts the player, the hunter, each enemy and each wall block it reaches.
        The player and the hunter take one hit, as from touching an enemy; each enemy takes the blast's damage.

        `enemies` are the enemies on the level. `hit_enemy(shot)` deals a shot to the enemies as a projectile would
        (the Encounter's shoot), so a blast kills each kind of enemy the way a shot does: bosses split, pickups drop,
        and the kill counts for the blast's source. Anything it kills that explodes lights its own blast, which is how
        a chain of them goes off.
        """
        self.sound_manager.play_enemy_destroy()
        if ship is not None and ship.active and reaches(blast, ship.get_pos(), ship.radius):
            dvx, dvy = push(blast, ship.get_pos())
            ship.vx += dvx
            ship.vy += dvy
            if not ship.is_shield_active():
                ship.flinch()
                self.scoring.record_enemy_collision()   # it costs what touching an enemy does
                self.sound_manager.play_bad_hit()
        if hunter is not None and hunter.active and reaches(blast, hunter.get_pos(), hunter.radius):
            dvx, dvy = push(blast, hunter.get_pos())
            hunter.vx += dvx
            hunter.vy += dvy
            register_hit(hunter, dvx, dvy)
            _damage_hunter(hunter)
        caught = [e for e in enemies if e.active and reaches(blast, e.get_pos(), getattr(e,'hit_radius',e.radius))]
        for enemy in caught:
            dvx, dvy = push(blast, enemy.get_pos())
            if hasattr(enemy, 'apply_momentum'):
                enemy.apply_momentum(dvx, dvy)    # the anchored kinds ignore it
            else:
                enemy.vx += dvx
                enemy.vy += dvy
            for _ in range(blast.damage):
                if not enemy.active:
                    break
                hit_enemy(self._blast_shot(blast, enemy))
        if maze is not None and blast.wall_hits:
            self._blast_walls(blast, maze)

    @staticmethod
    def _blast_shot(blast: 'Blast', enemy) -> Projectile:
        """One hit of a blast on an enemy, as a still, friendly shot at its centre whose kill counts for the blast's
        source. (The blast has already shoved it, so the shot carries no momentum.)"""
        shot = Projectile(enemy.get_pos(), 0.0, source='player')
        shot.vx = shot.vy = 0.0
        shot.source = blast.source            # may be 'enemy': still a hit on enemies, but a kill for no one
        shot.is_enemy = False
        shot.weapon_id = blast.weapon_id
        shot._blast_target = enemy
        return shot

    @staticmethod
    def _blast_walls(blast: 'Blast', maze: 'Maze') -> None:
        """Each wall block the blast reaches takes its wall hits once, struck at the nearest point of the face."""
        from utils.math_utils import circle_line_collision, get_closest_point_on_line
        faces = {}
        for wall in maze.spatial_grid.get_nearby_walls(blast.pos, blast.radius):
            if not wall.active:
                continue
            start, end = wall.get_segment()
            if circle_line_collision(blast.pos, blast.radius, start, end):
                faces.setdefault(wall.cell if wall.cell is not None else id(wall), (wall, get_closest_point_on_line(blast.pos, start, end)))
        # One curved rock rebuild for every block the blast breaks, once all the hits are in
        with maze.batched_destroys():
            for wall, impact in faces.values():
                if not wall.active and wall.cell is not None:
                    # Rebuilt since it was found: hit the block through one of its current segments
                    wall = next((w for w in maze.walls if w.cell == wall.cell), wall)
                for _ in range(blast.wall_hits):
                    if maze.damage_wall(wall, impact):
                        break

    def handle_projectile_hunter_collision(self, projectile, hunter) -> bool:
        """Enemy bullets damage the hunter without changing player statistics."""
        if hunter is None or not hunter.active or not projectile.active or not projectile.is_enemy:
            return False
        if not projectile.check_circle_collision(hunter.get_pos(), hunter.radius):
            return False
        projectile.active = False
        from utils.math_utils import apply_circle_collision_physics
        apply_circle_collision_physics(hunter, projectile, 0.3)
        if _damage_hunter(hunter):
            self.sound_manager.play_enemy_destroy()
        else:
            self.sound_manager.play_bad_hit()
        return True

    def handle_hunter_contacts(self, hunter, player, enemies) -> None:
        if hunter is None or not hunter.active:
            return
        if player is not None and player.active:
            hunter.check_circle_collision(player.get_pos(), player.radius, player)
        for enemy in enemies:
            if not hunter.active:
                break
            if enemy.active and hunter.check_circle_collision(enemy.get_pos(), enemy.radius, enemy):
                if _damage_hunter(hunter):
                    self.sound_manager.play_enemy_destroy()

    def handle_ship_crystal_collision(
        self,
        ship: 'Ship',
        crystal: 'PowerupCrystal',
        scoring: 'ScoringSystem'
    ) -> bool:
        """Handle collision between ship and powerup crystal.
        
        Args:
            ship: The player ship.
            crystal: The powerup crystal.
            
        Returns:
            True if collision occurred, False otherwise.
        """
        if not crystal.active:
            return False
        
        if crystal.check_circle_collision((ship.x, ship.y), ship.radius):
            if getattr(ship, 'weapon', None) is not None:
                weapon_id = crystal.weapon_id or 'twin_blaster'
                ship.equip_weapon(weapon_id)
                self.weapon_pickups[weapon_id] += 1
            else:
                ship.activate_gun_upgrade()
            scoring.record_powerup_collected()
            self.level_crystals_collected += 1
            return True
        
        return False
    
    def handle_hunter_crystal_collision(self, hunter, crystal: 'PowerupCrystal') -> bool:
        """The hunter takes a crystal's weapon and fires it until the ammo runs out, or a shield's immunity;
        under the legacy rules a crystal has no weapon and its Gun takes the pickup as an upgrade."""
        if hunter is None or not hunter.active or not crystal.active:
            return False
        if not crystal.check_circle_collision(hunter.get_pos(), hunter.radius):
            return False
        weapon_id = crystal.weapon_id
        if weapon_id is not None and crystal.catalog[weapon_id].behaviour == 'shield':
            hunter.immunity_remaining = max(hunter.immunity_remaining, config.SETTINGS.powerups.shieldDurationSeconds)
        elif weapon_id is not None:
            from weapons import EquippedWeapon
            if getattr(hunter, 'weapon', None) is None:
                hunter.weapon = EquippedWeapon(crystal.catalog)
            hunter.weapon.equip(weapon_id)
        hunter.emit('pickup', hunter.encounter)
        self.level_crystals_collected += 1
        self.sound_manager.play_powerup_activation(weapon_id or 'upgrade')
        return True

