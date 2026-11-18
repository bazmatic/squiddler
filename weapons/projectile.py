"""Projectile behaviours behind one simulation lifecycle interface."""
import math
import pygame
import config
from entities.projectile import Projectile
from entities.explosion import Blast
from weapons.geometry import circle_time


class WeaponProjectile(Projectile):
    """A consuming bolt; subclasses vary guidance and contact response."""

    def __init__(self, pos, angle, source, definition):
        super().__init__(pos, angle, source=source, dynamic_color=definition.color)
        self.definition = definition
        self.weapon_id = definition.id
        self.vx = math.cos(math.radians(angle))*definition.base_speed*definition.speed
        self.vy = math.sin(math.radians(angle))*definition.base_speed*definition.speed
        self.lifetime = definition.lifetime
        self.radius = definition.radius or definition.base_radius
        self.hit_targets = set()
        self.age = 0.0
        self.previous = pos

    def query_radius(self, dt, targets=()):
        return math.hypot(self.vx,self.vy)*dt+self.radius+1

    def extra_surfaces(self):
        return ()

    def prepare(self, dt, targets, segments):
        pass

    def target_contact(self, start, delta, target, segments):
        if hasattr(target, 'projectile_contact'):
            return target.projectile_contact(start, delta, self.radius)
        return circle_time(start,delta,target.get_pos(),getattr(target,'hit_radius',target.radius)+self.radius)

    def hit_target(self, target, apply_hit, trigger_blast):
        self.hit_targets.add(target)
        apply_hit(target,self,self.definition.damage)
        self.active = False

    def hit_surface(self, normal, wall, maze, obstacles):
        if wall is not None and hasattr(wall,'get_segment') and not self.is_enemy:
            maze.damage_wall(wall,self.get_pos(),source=self.source)
        if wall is not None and hasattr(wall,'body'):
            speed = max(1e-9,math.hypot(self.vx,self.vy))
            impulse = obstacles.settings.bulletImpulse
            wall.body.apply_impulse_at_world_point((self.vx/speed*impulse,self.vy/speed*impulse),self.get_pos())
        self.active = False

    def _advance_clock(self, dt):
        self.age += dt
        self.lifetime = max(0,self.lifetime-dt)

    def finish(self, dt, trigger_blast):
        self._advance_clock(dt)
        if self.lifetime <= 0:
            self.active = False


class RailProjectile(WeaponProjectile):
    BEAM_RADIUS = 14  # A wide beam for hitting creatures; walls still see the shot's own thin radius

    def query_radius(self, dt, targets=()):
        return super().query_radius(dt,targets)+self.BEAM_RADIUS

    def target_contact(self, start, delta, target, segments):
        if hasattr(target, 'projectile_contact'):
            return target.projectile_contact(start, delta, self.BEAM_RADIUS)
        return circle_time(start,delta,target.get_pos(),getattr(target,'hit_radius',target.radius)+self.BEAM_RADIUS)

    def hit_surface(self, normal, wall, maze, obstacles):
        if wall is not None and hasattr(wall, 'get_segment') and wall.destructible:
            if maze.damage_wall(wall, self.get_pos(), damage=max(1, wall.hit_points), source=self.source):
                return  # Keep the shot's speed and remaining travel through the destroyed block.
            self.active = False
            return
        super().hit_surface(normal, wall, maze, obstacles)

    def hit_target(self, target, apply_hit, trigger_blast):
        self.hit_targets.add(target)
        apply_hit(target,self,self.definition.damage)

    def draw(self, screen):
        if not self.active:
            return
        speed = max(1e-6,math.hypot(self.vx,self.vy))
        tail = (self.x-self.vx/speed*60,self.y-self.vy/speed*60)
        glow = tuple(c//3 for c in self.definition.color)
        pygame.draw.line(screen,glow,tail,self.get_pos(),self.BEAM_RADIUS*2)
        pygame.draw.line(screen,self.definition.color,tail,self.get_pos(),self.BEAM_RADIUS)
        pygame.draw.line(screen,(255,240,255),tail,self.get_pos(),4)


class SeekerProjectile(WeaponProjectile):
    def __init__(self, *args):
        super().__init__(*args)
        self.target = None

    def query_radius(self, dt, targets=()):
        return max(super().query_radius(dt,targets),self.definition.seek_range)

    def prepare(self, dt, targets, segments):
        from hunter.visibility import visible_point
        d = self.definition
        def eligible(target):
            if not target.active or target not in targets:
                return False
            angle = math.degrees(math.atan2(target.y-self.y,target.x-self.x))
            return (abs((angle-self.angle+180)%360-180) <= d.seek_cone
                    and visible_point(self.get_pos(),target.get_pos(),segments,d.seek_range))
        if self.target is None or not eligible(self.target):
            candidates = [(math.dist(self.get_pos(),target.get_pos()),index,target)
                          for index,target in enumerate(targets) if eligible(target)]
            self.target = min(candidates,key=lambda c:c[:2])[2] if candidates else None
        if self.target is not None:
            desired = math.degrees(math.atan2(self.target.y-self.y,self.target.x-self.x))
            turn = (desired-self.angle+180)%360-180
            self.angle += max(-d.turn_rate*dt,min(d.turn_rate*dt,turn))
            speed = math.hypot(self.vx,self.vy)
            self.vx = math.cos(math.radians(self.angle))*speed
            self.vy = math.sin(math.radians(self.angle))*speed

    def draw(self, screen):
        if not self.active:
            return
        speed = max(1e-6,math.hypot(self.vx,self.vy))
        dx,dy = self.vx/speed,self.vy/speed
        pygame.draw.polygon(screen,self.definition.color,[(self.x+dx*8,self.y+dy*8),
            (self.x-dx*5-dy*4,self.y-dy*5+dx*4),(self.x-dx*5+dy*4,self.y-dy*5-dx*4)])
        pygame.draw.line(screen,(255,210,100),(self.x-dx*5,self.y-dy*5),(self.x-dx*12,self.y-dy*12),2)


class GrenadeProjectile(WeaponProjectile):
    def __init__(self, *args):
        super().__init__(*args)
        self.detonated = False

    def query_radius(self, dt, targets=()):
        # Include walls between the grenade and nearby targets when testing visibility.
        reach = self.definition.proximity_radius + max((getattr(t,'hit_radius',t.radius) for t in targets),default=0)
        return super().query_radius(dt,targets)+reach

    def target_contact(self, start, delta, target, segments):
        proximity = self.definition.proximity_radius
        if proximity <= 0:
            return super().target_contact(start,delta,target,segments)
        t = circle_time(start,delta,target.get_pos(),getattr(target,'hit_radius',target.radius)+max(self.radius,proximity))
        if t is None:
            return None
        from hunter.visibility import visible_point
        trigger_pos = (start[0]+delta[0]*t,start[1]+delta[1]*t)
        if visible_point(trigger_pos,target.get_pos(),segments,float('inf')):
            return t
        return None

    def extra_surfaces(self):
        corners = ((config.UI_ZONE_WIDTH,0),(config.SCREEN_WIDTH,0),
                   (config.SCREEN_WIDTH,config.SCREEN_HEIGHT),(config.UI_ZONE_WIDTH,config.SCREEN_HEIGHT))
        return [(corners[i],corners[(i+1)%4],0,None) for i in range(4)]

    def detonate(self):
        if self.detonated:
            return None
        self.detonated, self.active = True, False
        d = self.definition
        return Blast(self.get_pos(), d.blast_radius, damage=d.damage, push=d.blast_push,
                     wall_hits=d.wall_hits, source=self.source, color=d.color, weapon_id=self.weapon_id)

    def hit_target(self, target, apply_hit, trigger_blast):
        trigger_blast(self.detonate())

    def hit_surface(self, normal, wall, maze, obstacles):
        dot = self.vx*normal[0]+self.vy*normal[1]
        self.vx -= (1+self.definition.restitution)*dot*normal[0]
        self.vy -= (1+self.definition.restitution)*dot*normal[1]
        self.x += normal[0]*.02
        self.y += normal[1]*.02

    def finish(self, dt, trigger_blast):
        self._advance_clock(dt)
        drag = .992**dt
        self.vx *= drag
        self.vy *= drag
        if self.active and self.lifetime <= 0:
            trigger_blast(self.detonate())

    def draw(self, screen):
        if not self.active:
            return
        d = self.definition
        center = (round(self.x),round(self.y))
        pygame.draw.circle(screen,(80,65,35),center,round(self.radius)+2)
        pygame.draw.circle(screen,d.color,center,round(self.radius),2)
        angle = self.age*.2
        tip = (self.x+math.cos(angle)*8,self.y+math.sin(angle)*8)
        pygame.draw.line(screen,d.color,center,tip,2)
        heat = int(140+115*(1-self.lifetime/d.lifetime))
        pygame.draw.circle(screen,(255,heat,80),(round(tip[0]),round(tip[1])),2)


class InflatorProjectile(WeaponProjectile):
    def draw(self, screen):
        if not self.active:
            return
        center = (round(self.x),round(self.y))
        radius = round(self.radius)
        pygame.draw.circle(screen,self.definition.color,center,radius,2)
        pygame.draw.circle(screen,(255,235,250),(center[0]-1,center[1]-1),2)
        pygame.draw.line(screen,self.definition.color,(self.x,self.y+radius),
                         (self.x-3,self.y+radius+5),1)


PROJECTILE_TYPES = {'bolt': WeaponProjectile, 'rail': RailProjectile,
                    'seeker': SeekerProjectile, 'grenade': GrenadeProjectile, 'inflator': InflatorProjectile}


def make_projectile(pos, angle, source, definition):
    return PROJECTILE_TYPES[definition.behaviour](pos,angle,source,definition)
