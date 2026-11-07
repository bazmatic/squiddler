"""Combat: how a creature hurts, is hurt, is pushed about, and what its death leaves."""
import math
import random
from typing import Optional

import config
import level_rules
from creatures.trait import Trait
from game_handlers.fire_rate_calculator import cooldown_for_upgrade, shots_per_volley
from entities.explosion import Blast
from entities.projectile import Projectile
from utils import circle_line_collision, get_angle_to_point
from utils.math_utils import apply_wall_collision_physics
from weapons.state import held_weapon


class TakesHits(Trait):
    """Each friendly shot costs it a hit point; the last one kills it. With `growth`, it has that share of `hitPoints`
    more for each level past `fromLevel`, rounded, up to `maxHitPoints`."""
    NAME = 'takesHits'

    def __init__(self, hitPoints: int, growth: float = 0.0, fromLevel: int = 1, maxHitPoints: Optional[int] = None):
        self.hit_points, self.growth, self.from_level, self.max_hit_points = hitPoints, growth, fromLevel, maxHitPoints

    def on_spawn(self, c, enc):
        c.hit_points = c.max_hit_points = self.hit_points_at(c.level)

    def hit_points_at(self, level: int) -> int:
        grown = round(self.hit_points * (1 + self.growth * max(0, level - self.from_level)))
        return grown if self.max_hit_points is None else min(self.max_hit_points, grown)

    def on_shot(self, c, shot, enc):
        c.hit_points -= getattr(shot, 'creature_damage', 1.)
        c.emit('hit', enc)
        if c.hit_points <= 0:
            enc.kill(c, shot)


class ShovedByShots(Trait):
    """A shot that strikes it passes on some of its speed. Put it before takesHits, so the killing shot shoves too."""
    NAME = 'shovedByShots'

    def __init__(self, transfer: float = config.MOMENTUM_TRANSFER_FACTOR):
        self.transfer = transfer

    def on_shot(self, c, shot, enc):
        c.vx += shot.vx * self.transfer
        c.vy += shot.vy * self.transfer


class RamsShip(Trait):
    """Touching the player's ship costs the player, unless the shield is up; the two bounce apart."""
    NAME = 'ramsShip'

    def on_frame(self, c, enc, dt):
        ship = enc.ship
        if (ship is not None and ship.active and not ship.is_shield_active()
                and ship.check_circle_collision(c.get_pos(), c.radius, c)):
            enc.scoring.record_enemy_collision()
            c.emit('rammed', enc)


class DropsPickup(Trait):
    """Its death rolls for a pickup where it died."""
    NAME = 'dropsPickup'

    def on_death(self, c, enc):
        enc.drop_pickup(c.get_pos())


class BouncesOffWalls(Trait):
    """While it moves, a wall it touches bounces it back out."""
    NAME = 'bouncesOffWalls'

    def on_frame(self, c, enc, dt):
        if abs(c.vx) < config.MIN_VELOCITY_THRESHOLD and abs(c.vy) < config.MIN_VELOCITY_THRESHOLD:
            return
        maze = enc.maze
        walls = (maze.spatial_grid.get_nearby_walls((c.x, c.y), c.radius * 2.0) if maze.spatial_grid is not None
                 else maze.walls)
        for wall in walls:
            if hasattr(wall, 'get_segment'):
                if not wall.active:
                    continue
                start, end = wall.get_segment()
            else:
                start, end = wall
            if not circle_line_collision((c.x, c.y), c.radius, start, end):
                continue
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            if length > 0:
                normal_x, normal_y = -(end[1] - start[1]) / length, (end[0] - start[0]) / length
                if (c.x - start[0]) * normal_x + (c.y - start[1]) * normal_y < 0:
                    normal_x, normal_y = -normal_x, -normal_y
                vx, vy = c.vx, c.vy
                apply_wall_collision_physics(c, (normal_x, normal_y), config.COLLISION_RESTITUTION)
                c.wall_impulse = (c.vx - vx, c.vy - vy)
                c.x += normal_x * (c.radius + 1.0)
                c.y += normal_y * (c.radius + 1.0)
            c.emit('hit_wall', enc)
            return


class Gun(Trait):
    """Fires at where its brain aims when its brain says fire and it has reloaded. Numbers left out come from the
    strength of level `strengthLevel` (by default the creature's own). `startsLoaded` lets it fire on its first frame.
    The shot can be styled: `shotSpeed` scales its speed; `impact`, `glowColor`, `glowRadius` and `glowIntensity` are
    the projectile's. It keeps `gun_charge`, how far it is through reloading (0 to 1), for its Body to show, and
    `just_fired`, true on the frame it fires, for its neighbours. `cadence='burst'` takes its firing times from
    burstFire instead of reloading itself; `aimMode='nose'` emits a volley from the nose along the heading.
    `upgrades` enables crystal upgrades and expiry, including spread volleys and their fire-rate adjustment. Under
    the weapon rules a crystal gives a weapon instead (`weapon_pickups`): while the creature holds one (`c.weapon`),
    each round fires it, waiting on its own cooldown.
    `recoil` opts a gun into projectile momentum exchange while its body is unsupported."""
    NAME = 'gun'

    def __init__(self, intervalMin: Optional[int] = None, intervalMax: Optional[int] = None,
                 range: Optional[float] = None, strengthLevel: Optional[int] = None, startsLoaded: bool = False,
                 shotSpeed: float = 1.0, shotLifetime: float = 1.0, impact: float = 1.0, glowColor: Optional[list] = None,
                 glowRadius: float = 1.0, glowIntensity: float = 0.0,
                 cadence: str = 'reload', aimMode: str = 'target', shotStrength: float = 1.0,
                 upgrades: bool = False, recoil: bool = False):
        if cadence not in ('reload', 'burst') or aimMode not in ('target', 'nose'):
            raise ValueError('gun: unknown cadence or aimMode')
        self.cadence, self.aim_mode, self.shot_strength = cadence, aimMode, shotStrength
        self.upgrades = upgrades
        self.recoil = recoil
        self.REQUIRES = ('burstFire',) if cadence == 'burst' else ()
        self.interval_min, self.interval_max, self.range = intervalMin, intervalMax, range
        self.strength_level, self.starts_loaded = strengthLevel, startsLoaded
        self.shot_speed = shotSpeed
        self.shot_lifetime = shotLifetime
        self.styling = {'impact_force_multiplier': impact, 'glow_radius_multiplier': glowRadius,
                        'glow_intensity': glowIntensity}
        if glowColor is not None:
            self.styling['glow_color'] = tuple(glowColor)

    def _strength(self, c):
        return level_rules.get_enemy_strength(c.level if self.strength_level is None else self.strength_level)

    def on_spawn(self, c, enc):
        strength = self._strength(c)
        c.gun_range = strength.fire_range if self.range is None else self.range
        if self.cadence == 'reload':
            self._reload(c, strength)
        if self.starts_loaded:
            c.gun_cooldown = 0.0
        c.just_fired = False
        c.shots_fired = 0
        from weapons.catalog import PICKUP_RULESETS
        c.weapon_pickups = (self.upgrades
                            and getattr(enc.world, 'combat_ruleset', 'legacy') in PICKUP_RULESETS)
        if self.upgrades:
            c.gun_upgrade_level = 0
            c.upgrade_remaining = 0.0

    def on_frame(self, c, enc, dt):
        controls = c.controls
        if self.upgrades:
            if c.gun_upgrade_level:
                c.upgrade_remaining -= dt / config.FPS
                if c.upgrade_remaining <= 0:
                    c.gun_upgrade_level = 0
            if self.cadence == 'burst':
                c.fire_interval = (c.base_fire_interval * cooldown_for_upgrade(c.gun_upgrade_level)
                                   / config.SETTINGS.powerups.fireRateBaseCooldown)
        if self.cadence == 'reload':
            c.gun_cooldown = max(0.0, c.gun_cooldown - dt)
            ready = controls.fire and c.gun_cooldown <= 0
        else:
            ready = c.burst_round_due
        weapon = held_weapon(c)
        if weapon is not None:
            weapon.update(dt)
            ready = ready and weapon.cooldown <= 1e-8
        c.just_fired = bool(ready and (self.aim_mode == 'nose' or controls.aim is not None))
        if c.just_fired:
            if getattr(c, 'flocker_aim_enabled', False):
                from creatures.flocker_team import muzzle
                pos, angle = muzzle(c)
            elif self.aim_mode == 'nose':
                angle = c.angle
                rad = math.radians(angle)
                pos = (c.x + math.cos(rad) * (c.radius + 5), c.y + math.sin(rad) * (c.radius + 5))
            else:
                pos = c.get_pos()
                angle = get_angle_to_point(pos, controls.aim)
            level = c.gun_upgrade_level if self.upgrades else 0
            angles = [angle]
            if shots_per_volley(level) > 1:
                spread = config.UPGRADED_PROJECTILE_SPREAD_ANGLE
                angles.extend((angle - spread, angle + spread))
            extra = max(0, level - 3)
            colour = (tuple(c.kind.tuning['color']) if self.aim_mode == 'nose'
                      and 'color' in c.kind.tuning else None)
            launch_velocity=(c.vx,c.vy)
            launch_spin=getattr(c,'wall_angular_velocity',0.)
            source = 'enemy' if c.allegiance == 'enemy' else 'hunter'
            # Which pickup weapon this volley came from (None: the creature's own gun), for listeners to sound
            c.fired_weapon_id = weapon.weapon_id if weapon is not None else None
            if weapon is not None:
                shots = weapon.try_fire(pos, angle, source).projectiles
            else:
                shots = []
                for heading in angles:
                    shot = Projectile(pos, heading, is_enemy=c.allegiance == 'enemy', source=source,
                                      strength=self.shot_strength, dynamic_color=colour, is_upgraded=level >= 1,
                                      enhanced_size_multiplier=1.0 + extra * config.POWERUP_BEYOND_LEVEL_3_SIZE_INCREMENT,
                                      enhanced_speed_multiplier=1.0 + extra * config.POWERUP_BEYOND_LEVEL_3_SPEED_INCREMENT,
                                      **self.styling)
                    shot.vx *= self.shot_speed
                    shot.vy *= self.shot_speed
                    shot.lifetime *= self.shot_lifetime
                    shots.append(shot)
            for shot in shots:
                shot.aim_range = c.gun_range
                shot.actor_id = getattr(c, 'agent_id', None)
                if getattr(c, 'flocker_aim_enabled', False):
                    shot.flocker_target = getattr(c, 'flocker_target', None)
                    shot.flocker_frame = getattr(getattr(enc, 'flocker_team', None), 'frames', 0)
                if self.recoil:
                    from physics.recoil import apply_firing_recoil
                    if apply_firing_recoil(c,shot,launch_velocity,launch_spin):
                        c.emit('recoil',enc)
                enc.add_shot(shot)
            c.shots_fired += 1
            if self.cadence == 'reload':
                self._reload(c, self._strength(c))
            c.emit('fired', enc)
        if self.cadence == 'reload':
            c.gun_charge = 1.0 - c.gun_cooldown / c.gun_interval

    def on_event(self, c, name, enc):
        if name == 'pickup' and self.upgrades and not c.weapon_pickups:
            c.gun_upgrade_level += 1
            c.upgrade_remaining = config.POWERUP_DURATION_SECONDS
            if self.cadence == 'burst':
                c.fire_interval = (c.base_fire_interval * cooldown_for_upgrade(c.gun_upgrade_level)
                                   / config.SETTINGS.powerups.fireRateBaseCooldown)

    def _reload(self, c, strength):
        low = strength.fire_interval_min if self.interval_min is None else self.interval_min
        high = strength.fire_interval_max if self.interval_max is None else self.interval_max
        c.gun_interval = c.gun_cooldown = random.randint(low, high)


class Explodes(Trait):
    """Its death lights a blast: `reach` is in multiples of its radius, `damage` the hits it deals each enemy it
    reaches, `push` the speed it gives a thing at its centre, `wallHits` the hits on each wall block, and `fuse` the
    frames before it goes off. Its kills count for whoever killed this creature, so a chain counts for whoever lit
    the first."""
    NAME = 'explodes'

    def __init__(self, reach: float, damage: int = 1, push: float = 6.0, wallHits: int = 1, fuse: float = 0.0,
                 color: Optional[list] = None):
        self.reach, self.damage, self.push, self.wall_hits, self.fuse = reach, damage, push, wallHits, fuse
        self.color = tuple(color) if color is not None else None

    def blast(self, c, source: str) -> Blast:
        """The blast this creature makes; its kills count for `source`."""
        styling = {} if self.color is None else {'color': self.color}
        return Blast((c.x, c.y), c.radius * self.reach, damage=self.damage, push=self.push, wall_hits=self.wall_hits,
                     source=source, fuse=self.fuse, weapon_id=getattr(c, 'killed_by_weapon', None), **styling)

    def on_death(self, c, enc):
        enc.explode(self.blast(c, c.killed_by))


def flinch(c, frames: float, enc) -> None:
    """The creature flinches for `frames`, unless a longer flinch is already running, and its Body hears `flinch`."""
    if frames > getattr(c, 'flinch_timer', 0.0):
        c.flinch_timer = c.flinch_total = float(frames)
    c.emit('flinch', enc)


class FlinchesWhenHit(Trait):
    """It flinches for `frames` on each of the events `on` (by default, a shot's hit). Its Body decides how a flinch
    looks, a blink or clamping shut; other traits can hold back while `flinch_timer` runs. `flinch_total` is how long
    the flinch now running lasts."""
    NAME = 'flinchesWhenHit'

    def __init__(self, frames: float, on=('hit',)):
        self.frames, self.on = frames, tuple(on)

    def on_spawn(self, c, enc):
        c.flinch_timer = c.flinch_total = 0.0

    def on_frame(self, c, enc, dt):
        c.flinch_timer = max(0.0, c.flinch_timer - dt)

    def on_event(self, c, name, enc):
        if name in self.on:
            flinch(c, self.frames, enc)
