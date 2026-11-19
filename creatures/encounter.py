"""The Encounter: what happens to the creatures on a level in one frame.

It runs every live creature's traits in roster order, then each kind's crowd traits, and offers each friendly shot to
the creatures in roster order. It also gives traits what they act on, read live from its world (a Game, or the duel's
stand-in): the ship, the hunter, the maze, the scoring, the shots, the pickups and the collision handler.

Every enemy is a Creature; a shot no creature takes remains available to the rest of the level.
"""
from collections import Counter
from typing import Dict, List, Optional, Tuple

from creatures.creature import Creature


class Encounter:

    def __init__(self, roster, world):
        self.roster = roster
        self.world = world
        self.on_projectile_damage = None
        self.hits = Counter()          # shots that reached an enemy, by who fired them ('player', 'hunter', ...)

    @property
    def ship(self):
        return self.world.ship

    @property
    def hunter(self):
        return self.world.hunter

    @property
    def maze(self):
        return self.world.maze

    @property
    def scoring(self):
        return self.world.scoring

    @property
    def command_recorder(self):
        return self.world.command_recorder

    def creatures(self) -> List[Creature]:
        """Every creature on the roster, alive or not, in roster order."""
        return [enemy for enemy in self.roster.all() if isinstance(enemy, Creature)]

    def step(self, dt: float) -> None:
        """One frame: spawn hooks for newcomers, every live creature's traits and Body, then the crowd traits (once
        for each crowd: by default a kind, or the crowd a trait names)."""
        creatures = self.creatures()
        for creature in creatures:
            self._spawn(creature)
        if any(c.active and c.type == 'wallcrawler' for c in creatures):
            from creatures.wallcrawler_team import get_team
            get_team(self).before_frame()
        if any(c.active and any(t.NAME == 'neuralJelly' for t in c.traits) for c in creatures):
            from creatures.jelly_team import get_jelly_team
            get_jelly_team(self).before_frame()
        if any(c.active and any(t.NAME == 'flockerFlight' for t in c.traits) for c in creatures):
            from creatures.flocker_team import get_flocker_team
            get_flocker_team(self).before_frame()
        crowds: Dict[Tuple[str, str], Tuple[object, List[Creature]]] = {}
        for creature in creatures:
            if not creature.active:
                continue
            self.step_creature(creature, dt)
            if creature.active:
                for trait in creature.traits:
                    if trait.CROWD:
                        crowds.setdefault((trait.NAME, trait.crowd(creature)), (trait, []))[1].append(creature)
        for trait, members in crowds.values():
            live = [creature for creature in members if creature.active]
            if live:
                trait.on_crowd(live, self, dt)

    def step_creature(self, creature, dt):
        """Run one creature once; its gun fires before late propulsion moves it."""
        self._spawn(creature)
        if not creature.active:
            return
        creature.inflation.update(dt)
        for trait in creature.traits:
            trait.on_frame(creature, self, dt)
            if not creature.active:
                return
        for trait in creature.after_frame_traits:
            trait.on_after_frame(creature, self, dt)
            if not creature.active:
                return
        creature.body.update(creature, dt, self.target_for(creature))

    def close_creature(self, creature):
        """Release one creature's traits, even if it has died."""
        for trait in creature.traits:
            trait.on_close(creature, self)

    def ready(self) -> None:
        """Set up every creature on the roster that is not yet (its spawn hooks), without a frame passing."""
        for creature in self.creatures():
            self._spawn(creature)

    def close(self) -> None:
        """The encounter is over: every creature's traits let go of what they hold. Safe to call again."""
        for creature in self.creatures():
            self.close_creature(creature)

    def _spawn(self, creature) -> None:
        """Run a newcomer's spawn hooks, once."""
        if not creature.has_spawned and creature.active:
            creature.has_spawned = True
            creature.encounter = self
            for trait in creature.traits:
                trait.on_spawn(creature, self)

    def shoot(self, projectile) -> bool:
        """Offer a friendly shot to each live enemy creature in roster order; the first it reaches takes it, and the
        shot is spent (True). Return False when no creature takes it."""
        if not projectile.is_enemy and projectile.active:
            target = getattr(projectile, '_blast_target', None)
            if target is not None:
                if target.active:
                    self.apply_hit(target, projectile)
                    projectile.active = False
                    return True
                return False
            for creature in self.creatures():
                if (creature.active and creature.allegiance == 'enemy'
                        and creature.projectile_contact(projectile.get_pos(), (0., 0.), projectile.radius) is not None):
                    projectile.active = False
                    self.apply_hit(creature, projectile)
                    return True
        return False

    def apply_hit(self, creature, projectile, damage=1):
        """Trait-aware hit application shared by consuming and piercing projectiles."""
        if not creature.active:
            return
        self._spawn(creature)
        self.hits[projectile.source] += 1
        if creature.inflation.on_hit(creature,projectile,self):
            return
        health_before = creature.hit_points
        for _ in range(damage):
            if not creature.active:
                break
            for trait in creature.traits:
                trait.on_shot(creature, projectile, self)
                if not creature.active:
                    break

        if creature.hit_points < health_before and self.on_projectile_damage is not None:
            self.on_projectile_damage(creature, projectile)

    def target_for(self, creature) -> Optional[Tuple[float, float]]:
        """Where the creature's target is: the nearest friend or opposite-allegiance creature, or None."""
        target = self.target_entity_for(creature)
        return target.get_pos() if target is not None else None

    def target_entity_for(self, creature):
        """The creature's target itself: the nearest friend or opposite-allegiance creature, or None."""
        from game_handlers.combat_targets import nearest_friendly
        usual = nearest_friendly(creature,getattr(self.world,'ships',None) or self.world.ship,self.world.hunter)
        targets = [t for t in getattr(self.world,'additional_targets',()) if t.active]
        targets += [c for c in self.roster.alive() if c is not creature and c.allegiance != creature.allegiance]
        if usual is not None:
            targets.append(usual)
        return min(targets,key=lambda t:(t.x-creature.x)**2+(t.y-creature.y)**2,default=None)

    def kill(self, creature, shot=None) -> None:
        """The creature dies; a shot that killed it counts as a kill for whoever fired it."""
        if not creature.active:
            return
        creature.killed_by = shot.source if shot is not None else 'enemy'
        creature.killed_by_weapon = getattr(shot, 'weapon_id', None)
        creature.die()
        if creature.allegiance == 'enemy':
            self.world.sound_manager.play_enemy_destroy()
        if shot is not None:
            self.world.collision_handler.record_kill(shot, creature)
        for trait in creature.traits:
            trait.on_death(creature, self)

    def drop_pickup(self, pos) -> None:
        """Roll for what a dead creature leaves behind."""
        self.world.collision_handler.drop_pickup(pos, self.world.powerup_crystals, self.world.jev_beacon_allowed())

    def spawn(self, kind: str, pos, level: int):
        """Put a new enemy of this kind on the roster, and return it. A Creature spawned during a frame first acts on
        the next one (or when a shot reaches it)."""
        from game_handlers.spawn_manager import make_enemy
        enemy = make_enemy(kind, pos, level, self.world.command_recorder)
        self.roster.add(enemy)
        return enemy

    def explode(self, blast) -> None:
        """Light a blast; the game sets it off when its fuse has burnt."""
        self.world.collision_handler.explosions.trigger(blast)

    def add_shot(self, projectile) -> None:
        self.world.projectiles.append(projectile)
