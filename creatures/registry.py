"""Every Trait and Body by the name config/enemies.json gives it, and the check that the catalogue's creatures can be
built."""
from typing import Dict, Mapping, Type

from creatures.traits.inchworm import Inchworms
from entities.inchworm import InchwormBody
from creatures.traits.sea_snake import SeaSnakes, StingsShip
from entities.sea_snake import SeaSnakeBody
from creatures.trait import Trait
from creatures.traits import brains, burst_fire, combat, crowd, energy, flock, growth, lighthouse, neural_jelly, piloted, propulsion, rooted, spawning, wallcrawl
from creatures.traits.flocker_flight import FlockerFlight
from entities.wallcrawler import WallcrawlerBody
from entities.anemone import AnemoneBody
from entities.belemnite import BelemniteBody
from entities.egg import EggBody
from entities.flocker import FlockerBody
from entities.flighthouse import FlighthouseBody
from entities.hunter_ship_body import HunterShipBody
from entities.jellyfish import Jellyfish
from entities.squid import Squid
from entities.patrol_crab import PatrolCrab
from entities.urchin import Urchin

TRAITS: Dict[str, Type[Trait]] = {trait.NAME: trait for trait in (
    brains.Still, brains.Patrol, brains.Chase, brains.ReplaysPlayer,
    propulsion.Coasts, propulsion.Drift, propulsion.Thrusters,
    combat.TakesHits, combat.ShovedByShots, combat.RamsShip, combat.DropsPickup, combat.BouncesOffWalls, combat.Gun, combat.Explodes, combat.FlinchesWhenHit,
    crowd.KeepsApart,
    spawning.SplitsOnDeath, spawning.LaysEggs,
    growth.Grows, growth.Hatches,
    rooted.Anchored, rooted.PullsShip, rooted.FlingsShip,
    flock.Flocks, flock.Tweets, FlockerFlight,
    lighthouse.Scans, lighthouse.LaunchesFlock,
    piloted.Pilot, burst_fire.BurstFire, energy.Energy,
    wallcrawl.WallCrawls, Inchworms, SeaSnakes, StingsShip,
    neural_jelly.NeuralJelly,
)}
BODIES: Dict[str, type] = {'urchin': Urchin, 'crab': PatrolCrab, 'jelly': Jellyfish, 'belemnite': BelemniteBody, 'squid': Squid, 'egg': EggBody, 'anemone': AnemoneBody, 'flocker': FlockerBody, 'flighthouse': FlighthouseBody, 'hunterShip': HunterShipBody}
BODIES['wallcrawler'] = WallcrawlerBody
BODIES['inchworm'] = InchwormBody
BODIES['sea_snake'] = SeaSnakeBody


def validate(catalogue: Mapping) -> None:
    """Raise ValueError, naming the kind, for an unknown body or trait, bad parameters, or a missing requirement."""
    for key, kind in catalogue.items():
        if kind.body is None:
            continue
        if kind.body not in BODIES:
            raise ValueError(f'enemy kind {key!r}: unknown body {kind.body!r}')
        names = [name for name, _ in kind.traits]
        for name, params in kind.traits:
            if name not in TRAITS:
                raise ValueError(f'enemy kind {key!r}: unknown trait {name!r}')
            try:
                trait = TRAITS[name].from_params(params)
            except ValueError as error:
                raise ValueError(f'enemy kind {key!r}: {error}') from None
            missing = [needed for needed in trait.REQUIRES if needed not in names]
            if missing:
                raise ValueError(f'enemy kind {key!r}: trait {name!r} requires {missing}')
            try:
                trait.validate_peers(dict(kind.traits))
            except ValueError as error:
                raise ValueError(f'enemy kind {key!r}: {error}') from None
