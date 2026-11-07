"""Game configuration loader that exposes typed settings backed by JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import enemy_kinds


def _as_color(value: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(value)


@dataclass
class ScreenSettings:
    width: int
    height: int
    fps: int
    fullscreen: bool


@dataclass
class ShipSettings:
    size: int
    rotationSpeed: float
    thrustForce: float
    friction: float
    maxSpeed: float
    spawnOffset: float
    collisionRestitution: float


@dataclass
class HunterShipSettings:
    rotationSpeed: float = 5.0
    thrustForce: float = 0.15
    friction: float = 0.998
    maxSpeed: float = 8.0


@dataclass
class ResourceSettings:
    initialFuel: int
    initialAmmo: int
    fuelPerThrust: int
    ammoPerShot: int
    shieldFuelPerFrame: int


@dataclass
class ScoringSettings:
    maxLevelScore: int
    timePenaltyRate: int
    collisionPenalty: int
    ammoPenaltyRate: float
    fuelPenaltyRate: float
    wallCollisionPenalty: int
    powerupCrystalBonus: int
    enemyBulletPenalty: int
    enemyDestructionBonus: int


@dataclass
class DifficultySettings:
    baseMazeSize: int
    mazeSizeIncrement: int
    maxMazeSize: int
    maxEnemyCount: int
    bossLevelInterval: int
    enemySpeedGrowth: float
    enemySpeedCeiling: float
    enemyDamageGrowth: float
    enemyDamageCeiling: float
    tutorialLevels: int


@dataclass
class MazeComplexityPreset:
    stepSize: int
    passageWidth: int
    clearRadius: int
    cornerClearSize: int
    extraPathsMultiplier: int
    gridSizeBase: int
    gridSizeIncrement: int


@dataclass
class CaveSettings:
    # Curved cave rock: a blurred, noisy field over the wall cells, traced at 0.5.
    # Shape tuning is in cell fractions (of sqrt(cell_w*cell_h)), so the curve scales with the cell.
    enabled: bool = True
    maxSampleSpacing: float = 6.0  # px between field samples, at most
    minSamplesPerCell: int = 6
    blurCells: float = 0.42
    noiseAmplitude: float = 0.7
    noiseScalesCells: Tuple[float, ...] = (2.5,)
    noiseWeights: Tuple[float, ...] = (1.0,)
    linkClearance: float = 2.0  # px kept open around every open-cell link
    wallCoreCells: float = 0.15
    simplifyTolerance: float = 0.75


@dataclass
class MazeSettings:
    wallThickness: int
    cellSize: Optional[int]
    minPassageWidth: int
    wallHitPoints: int
    shipSpawnOffset: float
    complexityPresets: Dict[str, MazeComplexityPreset]
    cave: CaveSettings = field(default_factory=CaveSettings)


@dataclass
class EnemySettings:
    damage: int
    stuckDetectionThreshold: float
    shiftAngleMin: int
    shiftAngleMax: int
    shiftDurationMin: int
    shiftDurationMax: int
    fireIntervalMin: int
    fireIntervalMax: int
    fireRange: float
    maxFireRange: float
    startClearance: float
    replayFireAngleTolerance: float


@dataclass
class ReplayEnemySettings:
    windowSize: int
    size: int
    color: Tuple[int, int, int]


@dataclass
class FlockerEnemySettings:
    size: int
    color: Tuple[int, int, int]
    clusterRadius: float   # a spawn rule: how closely a level's flockers start together


@dataclass
class FlighthouseEnemySettings:
    size: int
    color: Tuple[int, int, int]


@dataclass
class AnemoneSettings:
    size: int
    firstLevel: int
    maxCoverage: float
    color: Tuple[int, int, int]
    tipColor: Tuple[int, int, int]


@dataclass
class BabySettings:
    size: int


@dataclass
class MotherBossSettings:
    size: int
    lineGlowIntensityMax: float
    blinkFrequencyMultiplierMax: float
    blinkDurationMultiplierMax: float


@dataclass
class EggSettings:
    size: int
    color: Tuple[int, int, int]


@dataclass
class MomentumSettings:
    transferFactor: float
    frictionCoefficient: float
    minVelocityThreshold: float


@dataclass
class ProjectileSettings:
    speed: float
    size: int
    lifetime: int
    color: Tuple[int, int, int]
    impactForce: float


@dataclass
class VisualSettings:
    shipGlowIntensity: float
    shipGlowRadiusMultiplier: float
    enemyPulseSpeed: float
    enemyPulseAmplitude: float
    thrustPlumeLength: int
    thrustPlumeParticles: int


@dataclass
class ColorsSettings:
    background: Tuple[int, int, int]
    ship: Tuple[int, int, int]
    walls: Tuple[int, int, int]
    enemyStatic: Tuple[int, int, int]
    enemyDynamic: Tuple[int, int, int]
    projectile: Tuple[int, int, int]
    exit: Tuple[int, int, int]
    start: Tuple[int, int, int]
    text: Tuple[int, int, int]
    uiBackground: Tuple[int, int, int]
    shipNose: Tuple[int, int, int]
    shipRear: Tuple[int, int, int]
    shipDamagedNose: Tuple[int, int, int]
    shipDamagedRear: Tuple[int, int, int]


@dataclass
class ExitPortalSettings:
    attractionRadius: int
    attractionForceMultiplier: float
    glowMultiplier: float
    glowLayerOffset: int


@dataclass
class SoundSettings:
    enabled: bool
    sampleRate: int
    thrusterVolume: float
    shootVolume: float
    enemyDestroyVolume: float
    exitWarbleVolume: float
    powerupActivationVolume: float
    thrusterNoiseDuration: float
    shootBlipFrequency: int
    shootBlipDuration: float
    hitSoundVolume: float


@dataclass
class ControllerSettings:
    deadzone: float
    triggerThreshold: float


@dataclass
class PowerupUpgradedProjectile:
    spreadAngle: float
    sizeMultiplier: float
    speedMultiplier: float
    color: Tuple[int, int, int]
    glowColor: Tuple[int, int, int]


@dataclass
class PowerupBeyondLevel3:
    sizeIncrement: float
    speedIncrement: float
    glowIntensityIncrement: float
    hueRotation: float
    minFireCooldown: int


@dataclass
class PowerupSettings:
    crystalSize: int
    crystalSpawnChance: float
    jevBeaconSpawnChance: float
    crystalRotationSpeed: float
    crystalGlowIntensity: float
    crystalColor: Tuple[int, int, int]
    attractionRadius: int
    attractionSpeed: float
    flashDurationFrames: int
    flashTintStrength: float
    flashGlowMultiplier: float
    durationSeconds: float
    ammoRefill: int
    ammoCostMultiplier: float
    fireRateBaseCooldown: int
    firepowerPerCrystal: float
    upgradedProjectile: PowerupUpgradedProjectile
    beyondLevel3: PowerupBeyondLevel3
    rotationSpeedMultiplier: float
    inflationDurationSeconds: float = 5.0
    shieldDurationSeconds: float = 10.0
    inflationSizeMultiplier: float = 5.0


@dataclass
class StarAnimationSettings:
    appearDuration: float
    twinkleSpeed: float
    twinkleIntensity: float
    tinkleBasePitch: int
    tinklePitchIncrement: int
    levelCompleteStarSize: int


@dataclass
class UIAnimations:
    neonGlowIntensity: float
    neonGlowPulseSpeed: float
    buttonGlowIntensity: float
    buttonGlowPulseSpeed: float


@dataclass
class UIStarfield:
    starCount: int
    twinkleSpeed: float


@dataclass
class UIFonts:
    title: int
    subtitle: int
    button: int
    hint: int


@dataclass
class UISettings:
    splashEnabled: bool
    splashDisplayDuration: float
    splashFadeInDuration: float
    splashFadeOutDuration: float
    splashVideoSpeedMultiplier: float
    splashVideoEnabled: bool
    neonAsterStart: Tuple[int, int, int]
    neonAsterEnd: Tuple[int, int, int]
    neonVoidStart: Tuple[int, int, int]
    neonVoidEnd: Tuple[int, int, int]
    buttonAColor: Tuple[int, int, int]
    buttonBColor: Tuple[int, int, int]
    buttonGlowColor: Tuple[int, int, int]
    animations: UIAnimations
    starfield: UIStarfield
    menuParticles: int
    fonts: UIFonts


@dataclass
class GameSettings:
    criticalWarningThreshold: int
    hunterLevelInterval: int
    hunterFirstLevel: int
    # Which pilot flies the hunter: "jev", "rules" (the question criteria as plain code) or "neural"
    # (a network that flies with the basic controls only).
    hunterPilot: str = "jev"
    # The neural pilot decides every this many frames and holds its controls between, and its hunter
    # turns at this share of the player's rate.
    neuralDecisionFrames: int = 6
    neuralTurnRate: float = 1.0
    # The neural pilot's hunter can be destroyed: the hits it survives. The other pilots' hunters are indestructible.
    neuralHunterHealth: int = 10
    # Energy a wall takes from the neural pilot's hunter per full-speed impact (less for a gentler one).
    neuralWallEnergy: float = 4.0
    # Skips the drag physics of squid and jellyfish tentacles (they only look different: nothing reads them).
    # The lab turns it on for training and evaluation, where nobody watches.
    lowGraphics: bool = False
    # The pilot duel (FUN menu): hits each hunter survives, and the game seconds before it is called a draw.
    duelHealth: int = 5
    duelTimeoutSeconds: float = 120.0
    # The duel's maze: a complexity name (EMPTY, SIMPLE, NORMAL, COMPLEX or EXTREME) and its cells per side.
    # COMPLEX at 26 is how real level 13 is built.
    duelMazeComplexity: str = "COMPLEX"
    duelMazeGridSize: int = 26
    # Par time, for the par star: the maze route flown at this share of top speed, plus seconds for each enemy
    # on the level at its start (a Mother or a Squiddler Boss gets the boss allowance instead) and each crystal
    # dropped. A level whose exit opens on a timer gets at least the timer plus the margin.
    parSpeedShare: float = 0.5
    parSecondsPerEnemy: float = 1.2
    parSecondsPerSpawningBoss: float = 6.0
    parSecondsPerCrystal: float = 0.6
    parTimerMargin: float = 2.0


@dataclass
class AutoplaySettings:
    # Unattended runs (AUTOPLAY=1, or Demo): the Campaign Hunter flies the player ship through each level.
    firstLevel: int = 13
    lastLevel: int = 30
    # Safety nets for a level the ship never leaves: the most time it gets, and how long it may go without a kill
    # while enemies remain (0 turns that off).
    timeLimitSeconds: float = 120.0
    noKillSeconds: float = 20.0


@dataclass
class ObstacleSettings:
    # Physics gates: a plank on a centre pin, free or motor-driven.
    gateMass: float = 16.0  # the ship is r^2 = 64
    gateLengthCells: float = 3.0  # tip to tip, rounded ends included, in (average) cells
    gateThicknessCells: float = 0.16
    gateElasticity: float = 0.4
    motorRate: float = 1.2  # rad/s
    motorMaxForce: float = 30_000_000.0  # a torque limit: the push at radius r is max_force / r
    bulletImpulse: float = 300.0
    substeps: int = 8
    moverReachMargin: float = 24.0  # px


@dataclass
class FloraSettings:
    # Living weed on the cave rock. Times are frames at FPS, distances px.
    enabled: bool = False  # mainly for long-running scenarios; otherwise the rock's painted weed is shown
    seedFrames: float = 300.0
    growFrames: float = 1200.0
    podFrames: float = 600.0  # a pod swelling until it bursts
    restFrames: float = 600.0  # between one fruiting and the next
    fruitingsMin: int = 2
    fruitingsMax: int = 3
    stalksMin: int = 1
    stalksMax: int = 2
    witherFrames: float = 300.0
    lifeJitter: float = 0.2  # each stage lasts up to this share longer or shorter
    sporesMin: int = 3  # per pod
    sporesMax: int = 5
    sporeSpeed: float = 0.6  # px per frame, as it leaves the pod
    sporeDrag: float = 0.985  # speed kept each frame
    sporeWander: float = 0.03  # px per frame per frame of random push
    sporeFrames: float = 1200.0  # how long a spore drifts before it fades
    sporeRadius: float = 1.5
    plantSpacing: float = 50.0  # no spore takes root nearer a living plant than this
    startDensity: float = 4.0  # plants per 1000 px of rock face, when a cave is new
    startClear: float = 120.0  # kept clear of plants around the ship's start
    maxPlants: int = 400
    maxSpores: int = 200


@dataclass
class Settings:
    screen: ScreenSettings
    ship: ShipSettings
    resources: ResourceSettings
    scoring: ScoringSettings
    difficulty: DifficultySettings
    maze: MazeSettings
    enemies: EnemySettings
    replayEnemy: ReplayEnemySettings
    flockerEnemy: FlockerEnemySettings
    flighthouseEnemy: FlighthouseEnemySettings
    anemone: AnemoneSettings
    baby: BabySettings
    motherBoss: MotherBossSettings
    egg: EggSettings
    momentum: MomentumSettings
    projectile: ProjectileSettings
    visuals: VisualSettings
    colors: ColorsSettings
    exitPortal: ExitPortalSettings
    sound: SoundSettings
    controller: ControllerSettings
    powerups: PowerupSettings
    starAnimation: StarAnimationSettings
    ui: UISettings
    game: GameSettings
    hunterShip: HunterShipSettings = field(default_factory=HunterShipSettings)
    autoplay: AutoplaySettings = field(default_factory=AutoplaySettings)
    obstacles: ObstacleSettings = field(default_factory=ObstacleSettings)
    flora: FloraSettings = field(default_factory=FloraSettings)


def _load_settings_json() -> dict:
    from utils.resource_path import resource_path
    path = Path(resource_path("config/settings.json"))
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_settings() -> Settings:
    raw = _load_settings_json()
    # Each kind's own settings live with the kind, in config/enemies.json
    for key, section in (("replay", "replayEnemy"), ("flocker", "flockerEnemy"), ("flighthouse", "flighthouseEnemy"),
                         ("anemone", "anemone"), ("baby", "baby"),
                         ("mother_boss", "motherBoss"), ("egg", "egg")):
        raw[section] = enemy_kinds.kind(key).tuning

    def _preset_map(presets: dict) -> Dict[str, MazeComplexityPreset]:
        return {
            key: MazeComplexityPreset(**value)
            for key, value in presets.items()
        }

    starfield = UIStarfield(**raw["ui"]["starfield"])
    animations = UIAnimations(**raw["ui"]["animations"])
    fonts = UIFonts(**raw["ui"]["fonts"])

    powerup_raw = raw["powerups"]
    powerup_settings = PowerupSettings(
        crystalSize=powerup_raw["crystalSize"],
        crystalSpawnChance=powerup_raw["crystalSpawnChance"],
        jevBeaconSpawnChance=powerup_raw["jevBeaconSpawnChance"],
        crystalRotationSpeed=powerup_raw["crystalRotationSpeed"],
        crystalGlowIntensity=powerup_raw["crystalGlowIntensity"],
        crystalColor=_as_color(tuple(powerup_raw["crystalColor"])),
        attractionRadius=powerup_raw["attractionRadius"],
        attractionSpeed=powerup_raw["attractionSpeed"],
        flashDurationFrames=powerup_raw["flashDurationFrames"],
        flashTintStrength=powerup_raw["flashTintStrength"],
        flashGlowMultiplier=powerup_raw["flashGlowMultiplier"],
        durationSeconds=powerup_raw["durationSeconds"],
        ammoRefill=powerup_raw["ammoRefill"],
        ammoCostMultiplier=powerup_raw["ammoCostMultiplier"],
        fireRateBaseCooldown=powerup_raw["fireRateBaseCooldown"],
        firepowerPerCrystal=powerup_raw["firepowerPerCrystal"],
        upgradedProjectile=PowerupUpgradedProjectile(
            spreadAngle=powerup_raw["upgradedProjectile"]["spreadAngle"],
            sizeMultiplier=powerup_raw["upgradedProjectile"]["sizeMultiplier"],
            speedMultiplier=powerup_raw["upgradedProjectile"]["speedMultiplier"],
            color=_as_color(tuple(powerup_raw["upgradedProjectile"]["color"])),
            glowColor=_as_color(tuple(powerup_raw["upgradedProjectile"]["glowColor"]))
        ),
        beyondLevel3=PowerupBeyondLevel3(**powerup_raw["beyondLevel3"])
        ,
        rotationSpeedMultiplier=powerup_raw["rotationSpeedMultiplier"],
        inflationDurationSeconds=powerup_raw.get("inflationDurationSeconds", 5.0),
        shieldDurationSeconds=powerup_raw.get("shieldDurationSeconds", 10.0),
        inflationSizeMultiplier=powerup_raw.get("inflationSizeMultiplier", 5.0)
    )

    colors_raw = raw["colors"]

    return Settings(
        screen=ScreenSettings(**raw["screen"]),
        ship=ShipSettings(**raw["ship"]),
        resources=ResourceSettings(**raw["resources"]),
        scoring=ScoringSettings(**raw["scoring"]),
        difficulty=DifficultySettings(**raw["difficulty"]),
        maze=MazeSettings(
            wallThickness=raw["maze"]["wallThickness"],
            cellSize=raw["maze"]["cellSize"],
            minPassageWidth=raw["maze"]["minPassageWidth"],
            wallHitPoints=raw["maze"]["wallHitPoints"],
            shipSpawnOffset=raw["maze"]["shipSpawnOffset"],
            complexityPresets=_preset_map(raw["maze"]["complexityPresets"]),
            cave=CaveSettings(**{k: tuple(v) if isinstance(v, list) else v
                                 for k, v in raw["maze"].get("cave", {}).items()}),
        ),
        enemies=EnemySettings(**raw["enemies"]),
        replayEnemy=ReplayEnemySettings(
            windowSize=raw["replayEnemy"]["windowSize"],
            size=raw["replayEnemy"]["size"],
            color=_as_color(tuple(raw["replayEnemy"]["color"])),
        ),
        flockerEnemy=FlockerEnemySettings(
            size=raw["flockerEnemy"]["size"],
            color=_as_color(tuple(raw["flockerEnemy"]["color"])),
            clusterRadius=raw["flockerEnemy"]["clusterRadius"]
        ),
        flighthouseEnemy=FlighthouseEnemySettings(
            size=raw["flighthouseEnemy"]["size"],
            color=_as_color(tuple(raw["flighthouseEnemy"]["color"]))
        ),
        anemone=AnemoneSettings(**{
            **raw["anemone"],
            "color": _as_color(tuple(raw["anemone"]["color"])),
            "tipColor": _as_color(tuple(raw["anemone"]["tipColor"])),
        }),
        baby=BabySettings(**raw["baby"]),
        motherBoss=MotherBossSettings(**raw["motherBoss"]),
        egg=EggSettings(size=raw["egg"]["size"], color=_as_color(tuple(raw["egg"]["color"]))),
        momentum=MomentumSettings(**raw["momentum"]),
        projectile=ProjectileSettings(
            **{
                **raw["projectile"],
                "color": _as_color(tuple(raw["projectile"]["color"]))
            }
        ),
        visuals=VisualSettings(**raw["visuals"]),
        colors=ColorsSettings(
            background=_as_color(tuple(colors_raw["background"])),
            ship=_as_color(tuple(colors_raw["ship"])),
            walls=_as_color(tuple(colors_raw["walls"])),
            enemyStatic=_as_color(tuple(colors_raw["enemyStatic"])),
            enemyDynamic=_as_color(tuple(colors_raw["enemyDynamic"])),
            projectile=_as_color(tuple(colors_raw["projectile"])),
            exit=_as_color(tuple(colors_raw["exit"])),
            start=_as_color(tuple(colors_raw["start"])),
            text=_as_color(tuple(colors_raw["text"])),
            uiBackground=_as_color(tuple(colors_raw["uiBackground"])),
            shipNose=_as_color(tuple(colors_raw["shipNose"])),
            shipRear=_as_color(tuple(colors_raw["shipRear"])),
            shipDamagedNose=_as_color(tuple(colors_raw["shipDamagedNose"])),
            shipDamagedRear=_as_color(tuple(colors_raw["shipDamagedRear"]))
        ),
        exitPortal=ExitPortalSettings(**raw["exitPortal"]),
        sound=SoundSettings(**raw["sound"]),
        controller=ControllerSettings(**raw["controller"]),
        powerups=powerup_settings,
        starAnimation=StarAnimationSettings(**raw["starAnimation"]),
        ui=UISettings(
            splashEnabled=raw["ui"]["splashEnabled"],
            splashDisplayDuration=raw["ui"]["splashDisplayDuration"],
            splashFadeInDuration=raw["ui"]["splashFadeInDuration"],
            splashFadeOutDuration=raw["ui"]["splashFadeOutDuration"],
            splashVideoSpeedMultiplier=raw["ui"]["splashVideoSpeedMultiplier"],
            splashVideoEnabled=raw["ui"]["splashVideoEnabled"],
            neonAsterStart=_as_color(tuple(raw["ui"]["neonAsterStart"])),
            neonAsterEnd=_as_color(tuple(raw["ui"]["neonAsterEnd"])),
            neonVoidStart=_as_color(tuple(raw["ui"]["neonVoidStart"])),
            neonVoidEnd=_as_color(tuple(raw["ui"]["neonVoidEnd"])),
            buttonAColor=_as_color(tuple(raw["ui"]["buttonAColor"])),
            buttonBColor=_as_color(tuple(raw["ui"]["buttonBColor"])),
            buttonGlowColor=_as_color(tuple(raw["ui"]["buttonGlowColor"])),
            animations=animations,
            starfield=starfield,
            menuParticles=raw["ui"]["menuParticles"],
            fonts=fonts
        ),
        game=GameSettings(**raw["game"]),
        hunterShip=HunterShipSettings(**raw.get("hunterShip", {})),
        autoplay=AutoplaySettings(**raw.get("autoplay", {})),
        obstacles=ObstacleSettings(**raw.get("obstacles", {})),
        flora=FloraSettings(**raw.get("flora", {})),
    )


SETTINGS = load_settings()

STATES_DEFAULTS = {
    "splash": "splash",
    "menu": "menu",
    "profileSelection": "profile_selection",
    "settings": "settings",
    "levelSelect": "level_select",
    "fun": "fun",
    "editor": "editor",
    "duel": "duel",
    "playing": "playing",
    "levelComplete": "level_complete",
    "quitConfirm": "quit_confirm"
}

# Backwards-compatible constants
SCREEN_WIDTH = SETTINGS.screen.width
SCREEN_HEIGHT = SETTINGS.screen.height
# Width of the column on the left that holds the HUD dials; the maze starts where it ends
UI_ZONE_WIDTH = 200
FPS = SETTINGS.screen.fps
SCREEN_FULLSCREEN = SETTINGS.screen.fullscreen

SHIP_SIZE = SETTINGS.ship.size
SHIP_ROTATION_SPEED = SETTINGS.ship.rotationSpeed
SHIP_THRUST_FORCE = SETTINGS.ship.thrustForce
SHIP_FRICTION = SETTINGS.ship.friction
SHIP_MAX_SPEED = SETTINGS.ship.maxSpeed
COLLISION_RESTITUTION = SETTINGS.ship.collisionRestitution

INITIAL_FUEL = SETTINGS.resources.initialFuel
INITIAL_AMMO = SETTINGS.resources.initialAmmo
FUEL_CONSUMPTION_PER_THRUST = SETTINGS.resources.fuelPerThrust
AMMO_CONSUMPTION_PER_SHOT = SETTINGS.resources.ammoPerShot
SHIELD_FUEL_CONSUMPTION_PER_FRAME = SETTINGS.resources.shieldFuelPerFrame

MAX_LEVEL_SCORE = SETTINGS.scoring.maxLevelScore
TIME_PENALTY_RATE = SETTINGS.scoring.timePenaltyRate
COLLISION_PENALTY = SETTINGS.scoring.collisionPenalty
AMMO_PENALTY_RATE = SETTINGS.scoring.ammoPenaltyRate
FUEL_PENALTY_RATE = SETTINGS.scoring.fuelPenaltyRate
WALL_COLLISION_PENALTY = SETTINGS.scoring.wallCollisionPenalty
POWERUP_CRYSTAL_BONUS = SETTINGS.scoring.powerupCrystalBonus
ENEMY_BULLET_PENALTY = SETTINGS.scoring.enemyBulletPenalty
ENEMY_DESTRUCTION_BONUS = SETTINGS.scoring.enemyDestructionBonus

BASE_MAZE_SIZE = SETTINGS.difficulty.baseMazeSize
MAZE_SIZE_INCREMENT = SETTINGS.difficulty.mazeSizeIncrement
MAX_MAZE_SIZE = SETTINGS.difficulty.maxMazeSize
# Enemies on an endless level, not counting bosses or the hunter, never exceed this
MAX_ENEMY_COUNT = SETTINGS.difficulty.maxEnemyCount
# Every Nth level without a level file is a boss level
BOSS_LEVEL_INTERVAL = SETTINGS.difficulty.bossLevelInterval
# Speed and damage grow by this share per level after the tutorial levels, up to the ceiling
ENEMY_SPEED_GROWTH = SETTINGS.difficulty.enemySpeedGrowth
ENEMY_SPEED_CEILING = SETTINGS.difficulty.enemySpeedCeiling
ENEMY_DAMAGE_GROWTH = SETTINGS.difficulty.enemyDamageGrowth
ENEMY_DAMAGE_CEILING = SETTINGS.difficulty.enemyDamageCeiling
TUTORIAL_LEVELS = SETTINGS.difficulty.tutorialLevels

WALL_THICKNESS = SETTINGS.maze.wallThickness
CELL_SIZE = SETTINGS.maze.cellSize
MIN_PASSAGE_WIDTH = SETTINGS.maze.minPassageWidth
WALL_HIT_POINTS = SETTINGS.maze.wallHitPoints
SHIP_SPAWN_OFFSET = SETTINGS.maze.shipSpawnOffset
CAVE = SETTINGS.maze.cave
FLORA = SETTINGS.flora

STATIC_ENEMY_SIZE = enemy_kinds.kind("static").tuning["size"]
# The larger of the Crab and the Jelly
DYNAMIC_ENEMY_SIZE = max(enemy_kinds.kind("patrol").tuning["size"], enemy_kinds.kind("aggressive").tuning["size"])
ENEMY_PATROL_SPEED = enemy_kinds.kind("patrol").tuning["speed"]
ENEMY_AGGRESSIVE_SPEED = enemy_kinds.kind("aggressive").tuning["speed"]
ENEMY_DAMAGE = SETTINGS.enemies.damage
ENEMY_STUCK_DETECTION_THRESHOLD = SETTINGS.enemies.stuckDetectionThreshold
ENEMY_SHIFT_ANGLE_MIN = SETTINGS.enemies.shiftAngleMin
ENEMY_SHIFT_ANGLE_MAX = SETTINGS.enemies.shiftAngleMax
ENEMY_SHIFT_DURATION_MIN = SETTINGS.enemies.shiftDurationMin
ENEMY_SHIFT_DURATION_MAX = SETTINGS.enemies.shiftDurationMax
ENEMY_FIRE_INTERVAL_MIN = SETTINGS.enemies.fireIntervalMin
ENEMY_FIRE_INTERVAL_MAX = SETTINGS.enemies.fireIntervalMax
ENEMY_FIRE_RANGE = SETTINGS.enemies.fireRange
# Fire range grows with the level but never past this
ENEMY_MAX_FIRE_RANGE = SETTINGS.enemies.maxFireRange
# No enemy spawns within this distance of the player's start
ENEMY_START_CLEARANCE = SETTINGS.enemies.startClearance
REPLAY_ENEMY_FIRE_ANGLE_TOLERANCE = SETTINGS.enemies.replayFireAngleTolerance

REPLAY_ENEMY_WINDOW_SIZE = SETTINGS.replayEnemy.windowSize
REPLAY_ENEMY_SIZE = SETTINGS.replayEnemy.size
REPLAY_ENEMY_COLOR = SETTINGS.replayEnemy.color

FLOCKER_ENEMY_SIZE = SETTINGS.flockerEnemy.size
FLOCKER_ENEMY_COLOR = SETTINGS.flockerEnemy.color
FLOCKER_ENEMY_CLUSTER_RADIUS = SETTINGS.flockerEnemy.clusterRadius

FLIGHTHOUSE_ENEMY_SIZE = SETTINGS.flighthouseEnemy.size
FLIGHTHOUSE_ENEMY_COLOR = SETTINGS.flighthouseEnemy.color

# The level rules read these: where anemones first appear, how much of a maze their pull may cover, and how far it
# reaches (for that coverage, and to keep them clear of the player's start)
ANEMONE_REACH_CELLS = dict(enemy_kinds.kind('anemone').traits)['pullsShip']['reachCells']
ANEMONE_FIRST_LEVEL = SETTINGS.anemone.firstLevel
ANEMONE_MAX_COVERAGE = SETTINGS.anemone.maxCoverage
ANEMONE_COLOR = SETTINGS.anemone.color
ANEMONE_TIP_COLOR = SETTINGS.anemone.tipColor

BABY_SIZE = SETTINGS.baby.size



MOTHER_BOSS_LINE_GLOW_INTENSITY_MAX = SETTINGS.motherBoss.lineGlowIntensityMax
MOTHER_BOSS_BLINK_FREQUENCY_MULTIPLIER_MAX = SETTINGS.motherBoss.blinkFrequencyMultiplierMax
MOTHER_BOSS_BLINK_DURATION_MULTIPLIER_MAX = SETTINGS.motherBoss.blinkDurationMultiplierMax


# The sizes an egg is laid at and grows to; the hunter's spawn checks read them
EGG_INITIAL_SIZE = SETTINGS.egg.size
EGG_MAX_SIZE = dict(enemy_kinds.kind('egg').traits)['grows']['to']
COLOR_EGG = SETTINGS.egg.color

MOMENTUM_TRANSFER_FACTOR = SETTINGS.momentum.transferFactor
FRICTION_COEFFICIENT = SETTINGS.momentum.frictionCoefficient
MIN_VELOCITY_THRESHOLD = SETTINGS.momentum.minVelocityThreshold

PROJECTILE_SPEED = SETTINGS.projectile.speed
PROJECTILE_SIZE = SETTINGS.projectile.size
PROJECTILE_LIFETIME = SETTINGS.projectile.lifetime
COLOR_ENEMY_PROJECTILE = SETTINGS.projectile.color
PROJECTILE_IMPACT_FORCE = SETTINGS.projectile.impactForce

SHIP_GLOW_INTENSITY = SETTINGS.visuals.shipGlowIntensity
SHIP_GLOW_RADIUS_MULTIPLIER = SETTINGS.visuals.shipGlowRadiusMultiplier
ENEMY_PULSE_SPEED = SETTINGS.visuals.enemyPulseSpeed
ENEMY_PULSE_AMPLITUDE = SETTINGS.visuals.enemyPulseAmplitude
THRUST_PLUME_LENGTH = SETTINGS.visuals.thrustPlumeLength
THRUST_PLUME_PARTICLES = SETTINGS.visuals.thrustPlumeParticles

COLOR_BACKGROUND = SETTINGS.colors.background
COLOR_SHIP = SETTINGS.colors.ship
COLOR_WALLS = SETTINGS.colors.walls
COLOR_ENEMY_STATIC = SETTINGS.colors.enemyStatic
COLOR_ENEMY_DYNAMIC = SETTINGS.colors.enemyDynamic
COLOR_PROJECTILE = SETTINGS.colors.projectile
COLOR_EXIT = SETTINGS.colors.exit
COLOR_START = SETTINGS.colors.start
COLOR_TEXT = SETTINGS.colors.text
COLOR_UI_BG = SETTINGS.colors.uiBackground
COLOR_SHIP_NOSE = SETTINGS.colors.shipNose
COLOR_SHIP_REAR = SETTINGS.colors.shipRear
COLOR_SHIP_DAMAGED_NOSE = SETTINGS.colors.shipDamagedNose
COLOR_SHIP_DAMAGED_REAR = SETTINGS.colors.shipDamagedRear

EXIT_PORTAL_ATTRACTION_RADIUS = SETTINGS.exitPortal.attractionRadius
EXIT_PORTAL_ATTRACTION_FORCE_MULTIPLIER = SETTINGS.exitPortal.attractionForceMultiplier
EXIT_PORTAL_GLOW_MULTIPLIER = SETTINGS.exitPortal.glowMultiplier
EXIT_PORTAL_GLOW_LAYER_OFFSET = SETTINGS.exitPortal.glowLayerOffset

SOUND_ENABLED = SETTINGS.sound.enabled
SOUND_SAMPLE_RATE = SETTINGS.sound.sampleRate
THRUSTER_SOUND_VOLUME = SETTINGS.sound.thrusterVolume
SHOOT_SOUND_VOLUME = SETTINGS.sound.shootVolume
ENEMY_DESTROY_SOUND_VOLUME = SETTINGS.sound.enemyDestroyVolume
EXIT_WARBLE_SOUND_VOLUME = SETTINGS.sound.exitWarbleVolume
POWERUP_ACTIVATION_SOUND_VOLUME = SETTINGS.sound.powerupActivationVolume
THRUSTER_NOISE_DURATION = SETTINGS.sound.thrusterNoiseDuration
SHOOT_BLIP_FREQUENCY = SETTINGS.sound.shootBlipFrequency
SHOOT_BLIP_DURATION = SETTINGS.sound.shootBlipDuration
BAD_HIT_SOUND_VOLUME = SETTINGS.sound.hitSoundVolume

CONTROLLER_DEADZONE = SETTINGS.controller.deadzone
CONTROLLER_TRIGGER_THRESHOLD = SETTINGS.controller.triggerThreshold

POWERUP_CRYSTAL_SIZE = SETTINGS.powerups.crystalSize
POWERUP_CRYSTAL_SPAWN_CHANCE = SETTINGS.powerups.crystalSpawnChance
# Chance a kill drops a Jev beacon, which summons the hunter, in place of a crystal
JEV_BEACON_SPAWN_CHANCE = SETTINGS.powerups.jevBeaconSpawnChance
POWERUP_CRYSTAL_ROTATION_SPEED = SETTINGS.powerups.crystalRotationSpeed
POWERUP_CRYSTAL_GLOW_INTENSITY = SETTINGS.powerups.crystalGlowIntensity
COLOR_POWERUP_CRYSTAL = SETTINGS.powerups.crystalColor
POWERUP_CRYSTAL_ATTRACTION_RADIUS = SETTINGS.powerups.attractionRadius
POWERUP_CRYSTAL_ATTRACTION_SPEED = SETTINGS.powerups.attractionSpeed
POWERUP_FLASH_DURATION_FRAMES = SETTINGS.powerups.flashDurationFrames
POWERUP_FLASH_TINT_STRENGTH = SETTINGS.powerups.flashTintStrength
POWERUP_FLASH_GLOW_MULTIPLIER = SETTINGS.powerups.flashGlowMultiplier

POWERUP_DURATION_SECONDS = SETTINGS.powerups.durationSeconds
POWERUP_AMMO_REFILL = SETTINGS.powerups.ammoRefill
POWERUP_AMMO_COST_MULTIPLIER = SETTINGS.powerups.ammoCostMultiplier
UPGRADED_PROJECTILE_SPREAD_ANGLE = SETTINGS.powerups.upgradedProjectile.spreadAngle
UPGRADED_PROJECTILE_SIZE_MULTIPLIER = SETTINGS.powerups.upgradedProjectile.sizeMultiplier
UPGRADED_PROJECTILE_SPEED_MULTIPLIER = SETTINGS.powerups.upgradedProjectile.speedMultiplier
COLOR_UPGRADED_PROJECTILE = SETTINGS.powerups.upgradedProjectile.color
COLOR_UPGRADED_SHIP_GLOW = SETTINGS.powerups.upgradedProjectile.glowColor

POWERUP_BEYOND_LEVEL_3_SIZE_INCREMENT = SETTINGS.powerups.beyondLevel3.sizeIncrement
POWERUP_BEYOND_LEVEL_3_SPEED_INCREMENT = SETTINGS.powerups.beyondLevel3.speedIncrement
POWERUP_BEYOND_LEVEL_3_GLOW_INTENSITY_INCREMENT = SETTINGS.powerups.beyondLevel3.glowIntensityIncrement
POWERUP_BEYOND_LEVEL_3_HUE_ROTATION = SETTINGS.powerups.beyondLevel3.hueRotation
POWERUP_ROTATION_SPEED_MULTIPLIER = SETTINGS.powerups.rotationSpeedMultiplier

STAR_APPEAR_DURATION = SETTINGS.starAnimation.appearDuration
STAR_TWINKLE_SPEED = SETTINGS.starAnimation.twinkleSpeed
STAR_TWINKLE_INTENSITY = SETTINGS.starAnimation.twinkleIntensity
STAR_TINKLE_BASE_PITCH = SETTINGS.starAnimation.tinkleBasePitch
STAR_TINKLE_PITCH_INCREMENT = SETTINGS.starAnimation.tinklePitchIncrement
LEVEL_COMPLETE_STAR_SIZE = SETTINGS.starAnimation.levelCompleteStarSize

STATE_SPLASH = STATES_DEFAULTS["splash"]
STATE_MENU = STATES_DEFAULTS["menu"]
STATE_PROFILE_SELECTION = STATES_DEFAULTS["profileSelection"]
STATE_SETTINGS = STATES_DEFAULTS["settings"]
STATE_LEVEL_SELECT = STATES_DEFAULTS["levelSelect"]
STATE_FUN = STATES_DEFAULTS["fun"]
STATE_EDITOR = STATES_DEFAULTS["editor"]
STATE_DUEL = STATES_DEFAULTS["duel"]
STATE_PLAYING = STATES_DEFAULTS["playing"]
STATE_LEVEL_COMPLETE = STATES_DEFAULTS["levelComplete"]
STATE_QUIT_CONFIRM = STATES_DEFAULTS["quitConfirm"]

SPLASH_ENABLED = SETTINGS.ui.splashEnabled
SPLASH_DISPLAY_DURATION = SETTINGS.ui.splashDisplayDuration
SPLASH_FADE_IN_DURATION = SETTINGS.ui.splashFadeInDuration
SPLASH_FADE_OUT_DURATION = SETTINGS.ui.splashFadeOutDuration
SPLASH_VIDEO_SPEED_MULTIPLIER = SETTINGS.ui.splashVideoSpeedMultiplier
SPLASH_VIDEO_ENABLED = SETTINGS.ui.splashVideoEnabled

COLOR_NEON_ASTER_START = SETTINGS.ui.neonAsterStart
COLOR_NEON_ASTER_END = SETTINGS.ui.neonAsterEnd
COLOR_NEON_VOID_START = SETTINGS.ui.neonVoidStart
COLOR_NEON_VOID_END = SETTINGS.ui.neonVoidEnd
COLOR_BUTTON_A = SETTINGS.ui.buttonAColor
COLOR_BUTTON_B = SETTINGS.ui.buttonBColor
COLOR_BUTTON_GLOW = SETTINGS.ui.buttonGlowColor

NEON_GLOW_INTENSITY = SETTINGS.ui.animations.neonGlowIntensity
NEON_GLOW_PULSE_SPEED = SETTINGS.ui.animations.neonGlowPulseSpeed
BUTTON_GLOW_INTENSITY = SETTINGS.ui.animations.buttonGlowIntensity
BUTTON_GLOW_PULSE_SPEED = SETTINGS.ui.animations.buttonGlowPulseSpeed
STARFIELD_STAR_COUNT = SETTINGS.ui.starfield.starCount
STARFIELD_TWINKLE_SPEED = SETTINGS.ui.starfield.twinkleSpeed
MENU_PARTICLE_COUNT = SETTINGS.ui.menuParticles

FONT_SIZE_TITLE = SETTINGS.ui.fonts.title
FONT_SIZE_SUBTITLE = SETTINGS.ui.fonts.subtitle
FONT_SIZE_BUTTON = SETTINGS.ui.fonts.button
FONT_SIZE_HINT = SETTINGS.ui.fonts.hint

# Allied hunter: one appears on every Nth level (0 disables the schedule)
HUNTER_LEVEL_INTERVAL = SETTINGS.game.hunterLevelInterval
# No hunter flies before this level
HUNTER_FIRST_LEVEL = SETTINGS.game.hunterFirstLevel


def editor_enabled() -> bool:
    """Whether the level editor is switched on (environment variable LEVEL_EDITOR, any non-empty value)."""
    return bool(os.environ.get('LEVEL_EDITOR'))


def observer_mode_enabled() -> bool:
    """Testing switch: ASTROVOID_OBSERVER=1 makes the player an invulnerable observer, drawn faintly and unseen by enemies (off by default)."""
    return os.environ.get('ASTROVOID_OBSERVER', '').strip().lower() not in ('', '0', 'false', 'no')
