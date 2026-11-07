"""Main game loop and state management."""

import pygame
import math
import time
import random
import os
from typing import List, Optional, Tuple
import config
import enemy_kinds
from utils.neural_policy import neural_policy_path
from rendering.fonts import get_font
from rendering.announcements import Announcer
from rendering.exit_panel import draw_exit_panel
from rendering.last_enemy_ring import draw_last_enemy_ring
from rendering.star_row import STAR_ROW_SIZE, STAR_ROW_Y, draw_star_row
from entities.ship import Ship, LOW_POWER_THRESHOLD
from flora import Flora
from maze.generator import Maze
from physics.obstacle_world import ObstacleWorld
from rendering.gates import draw_gate
import level_rules
import level_config
from entities.projectile import Projectile
from entities.powerup_crystal import PowerupCrystal
from entities.jev_beacon import JevBeacon
from entities.command_recorder import CommandRecorder, CommandType
from input import InputHandler
from scoring.system import ScoringSystem
from policies.exit import DEFAULT_EXIT_POLICY, ExitPolicy, ExitState, OpenAfterTime
from scoring.par import par_seconds, route_cells
from scoring.stars import StarTally, describe, earned, star_count
from profiles import ProfileManager, LevelResult
from user_settings import UserSettings
from rendering import Renderer
from rendering.ui_elements import AnimatedStarRating, StarIndicator, UIElementRenderer
from rendering.menu_components import AnimatedBackground, NeonText, Button, ControllerIcon, render_hint_row
from rendering.visual_effects import draw_button_glow
from rendering.main_menu import MainMenu
from rendering.level_complete_menu import LevelCompleteMenu
from rendering.profile_selection_menu import ProfileSelectionMenu
from rendering.settings_menu import SettingsMenu
from rendering.level_select_menu import LevelSelectMenu
from rendering.fun_menu import FunMenu
from rendering.level_editor import LevelEditor
from rendering.duel_view import draw_duel
from rendering.battle_view import draw_battle
from rendering.quit_confirmation_menu import QuitConfirmationMenu
from sounds import SoundManager
from states.splash_screen import SplashScreenState
from game_handlers.autoplay import AutoplayDirector
from game_handlers.enemy_roster import EnemyRoster
from game_handlers.spawn_manager import SpawnManager, make_enemy
from game_handlers.collision_handler import CollisionHandler
from creatures.encounter import Encounter
from game_handlers.fire_rate_calculator import calculate_fire_cooldown
from game_handlers.state_handlers import StateHandlerRegistry
from utils.math_utils import get_angle_to_point


from creatures.creature import Creature
from hunter.build import PilotSetup, make_hunter
from hunter.flight import HunterFlight, NeuralPilot, WorkerPilot
from hunter.neural.loader import load_neural_policy
from hunter.settings_for import hunter_settings_for
from hunter.surroundings import GameSurroundings
from hunter.spawn import resolve_hunter_spawn, nearest_hunter_spawn, place_jev_beacon, reserve_hunter_clearance
from hunter.worker import PilotWorker
from hunter.duel import build_duel
from hunter.duel_enemies import DuelEnemies
from creatures.team_battle import TeamBattle, load_match
from game_handlers.combat_targets import nearest_friendly_pos
from hunter.rules_pilot import RulesClient
from maze.config import MazeComplexity
import logging



def _hunter_hotkey_enabled() -> bool:
    """Whether the H test key may call in the hunter: set ASTROVOID_HUNTER_HOTKEY to 1 (off by default)."""
    return os.environ.get('ASTROVOID_HUNTER_HOTKEY', '').strip().lower() not in ('', '0', 'false', 'no')


DUEL_BANNER_SECONDS = 3.0     # how long the winner is shown before the duel returns to the FUN list
DUEL_MAZE_LEVEL = 13          # the level number the duel's maze is generated as (its look and seed)

HUNTER_GAUGE_HEIGHT = 60  # The hunter's energy gauge under the EXIT block: label, bar and reading
PAR_WARNING_SECONDS = 5.0  # The par star pulses for this long before par runs out


class Game:
    """Main game class managing state and game loop."""
    autoplay = None  # An AutoplayDirector when the game runs unattended (AUTOPLAY=1)
    ship_pilot = None  # Anything with decide(game) -> ShipCommand; when set it flies the player ship
    clock = time.time  # Wall time for scoring and cooldowns; a training harness swaps in a frame clock
    clock_ms = pygame.time.get_ticks  # The same, in milliseconds, for the fire cooldown
    CRITICAL_WARNING_THRESHOLD = config.SETTINGS.game.criticalWarningThreshold
    
    def __init__(self, screen: pygame.Surface):
        """Initialize game."""
        self.screen = screen
        from weapons.catalog import RULESET
        self.combat_ruleset = RULESET
        self.grows_flora = config.FLORA.enabled     # living weed; training switches it off
        self.curved_walls = config.CAVE.enabled     # training runs recorded on square walls replay on square walls
        self.frame_limiter = pygame.time.Clock()
        self.font = get_font(36)
        self.small_font = get_font(24)
        self.renderer = Renderer(screen)
        
        self.state = config.STATE_SPLASH if config.SPLASH_ENABLED else config.STATE_MENU
        # Check for START_LEVEL environment variable
        start_level = os.getenv('START_LEVEL')
        if start_level:
            try:
                self.initial_start_level = max(1, int(start_level))
            except ValueError:
                self.initial_start_level = None
        else:
            self.initial_start_level = None
        self.profile_manager = ProfileManager()
        self.user_settings = UserSettings()
        self.ship: Optional[Ship] = None
        self.maze: Optional[Maze] = None
        self.obstacles: Optional[ObstacleWorld] = None
        self.hunter_flight: Optional[HunterFlight] = None
        self.hunter_worker = None
        self.hunter_surroundings = GameSurroundings(self)
        self.jev_beacon_level = False
        
        # Every enemy on the level
        self.roster = EnemyRoster()
        # Neural brains are the default wherever a trained one exists: an environment variable names another file (or,
        # empty, selects the scripted controller); otherwise the bundled assets/neural/<name>.npz is used if present.
        self.wallcrawler_policy = None
        crawler_weights = neural_policy_path('WALLCRAWLER_POLICY', 'wallcrawler.npz')
        if crawler_weights:
            from learning.adapters.wallcrawler import load_policy
            try:
                self.wallcrawler_policy = load_policy(crawler_weights)
            except (ValueError, OSError, KeyError) as exc:
                logging.getLogger(__name__).warning('Wallcrawler policy unavailable: %s', exc)
        # The `neural_jelly` kind replaces every campaign Jelly (game_handlers.spawn_manager.make_enemy); creatures/
        # jelly_team.py reads world.jelly_policy, and the game is its encounter's world. Without a brain, those Jellies
        # keep the scripted chase.
        self.jelly_policy = None
        jelly_weights = neural_policy_path('JELLY_POLICY', 'jelly.npz')
        if jelly_weights:
            from learning.adapters.jelly import load_policy as load_jelly_policy
            try:
                self.jelly_policy = load_jelly_policy(jelly_weights)
            except (ValueError, OSError, KeyError) as exc:
                logging.getLogger(__name__).warning('Jelly policy unavailable: %s', exc)

        
        self.flocker_policy = None
        flocker_weights = neural_policy_path('FLOCKER_POLICY', 'flocker.npz')
        if flocker_weights:
            from learning.adapters.flocker import load_policy as load_flocker_policy
            try:
                self.flocker_policy = load_flocker_policy(flocker_weights)
            except (ValueError, OSError, KeyError) as exc:
                logging.getLogger(__name__).warning('Flocker policy unavailable: %s', exc)

        self.projectiles: List[Projectile] = []
        self.powerup_crystals: List[PowerupCrystal] = []
        self.scoring = ScoringSystem()
        self.sound_manager = SoundManager()  # Game-level sound manager for enemy destruction
        self.command_recorder = CommandRecorder()  # Record player commands for replay enemy
        self.input_handler = InputHandler()  # Handle keyboard input and map to commands
        self.restart_hints = {
            connected: render_hint_row(get_font(24), [(key, "Restart")])
            for connected, key in ((False, "R"), (True, "R / Y"))
        }
        
        # Game handlers
        self.spawn_manager = SpawnManager(self.roster)
        self.collision_handler = CollisionHandler(
            self.sound_manager,
            self.scoring,
            self.command_recorder
        )
        self.encounter = Encounter(self.roster, self)
        self.state_handler_registry = StateHandlerRegistry()
        
        self.running = True
        self.start_time = time.time()
        self.level_complete_time = 0.0
        self.level_score_breakdown = {}
        self.completion_time_seconds = 0.0
        self.base_par_seconds = 0.0  # Par fixed at level start; dropped crystals add to it as they come
        self.finish_tally: Optional[StarTally] = None  # The star tally frozen when the ship enters the exit
        self.star_lines: Tuple[str, ...] = ()  # What each star on the level-complete screen was for
        self.total_score_before_level = 0  # Store score before level for replay
        self.replaying = False
        self.level_result: Optional[LevelResult] = None
        self.level_succeeded = False  # Track if level was completed successfully
        self.exit_explosion_active = False  # Track if exit explosion is playing
        self.exit_explosion_time = 0.0  # Time since explosion started
        self.exit_explosion_pos: Optional[Tuple[float, float]] = None  # Exit position for explosion
        self.exit_policy: ExitPolicy = DEFAULT_EXIT_POLICY
        self.announcer = Announcer()   # important news, in the middle of the playfield
        self.exit_opened = False  # Once the exit opens it stays open for the rest of the level
        self.play_seconds = 0.0  # Play time since the first move, which a timer exit counts
        self.star_animation: Optional[AnimatedStarRating] = None  # Animated star rating for level complete
        self.level_complete_quit_confirm = False  # Track quit confirmation on level complete screen
        self.player_has_moved = False  # Track if player has made their first move
        self.level_start_time = 0.0  # Time when level started (to ignore initial SPACE press)
        self.keys_pressed_at_start = set()  # Track keys that were pressed when level started
        self.star_indicator = StarIndicator(
            on_star_lost=self.sound_manager.play_star_lost,
            on_star_gained=self.sound_manager.play_star_gained
        )
        self.critical_warning_active = False
        self.game_over_active = False  # Track if game over sequence is active
        self.game_frozen = False  # Track if game action is frozen (score reached zero)
        self.game_over_start_time = 0.0  # Time when game over sequence started
        self.game_over_fade_duration = 0.5  # Duration of fade to black (seconds)
        self.game_over_death_duration = Ship.DEATH_DURATION / config.FPS
        self.game_over_text_delay = 0.0  # Delay before showing "GAME OVER" text (seconds)
        self.game_over_text_duration = 0.6  # Duration to show "GAME OVER" text (seconds)
        self.profile_selection_menu = ProfileSelectionMenu(screen, self.profile_manager)
        
        # Menu UI components
        self.main_menu = MainMenu(screen)
        self.settings_menu = SettingsMenu(screen, self.user_settings)
        self.level_select_menu = LevelSelectMenu(screen, self.profile_manager)
        self.fun_menu = FunMenu(screen)
        self.level_editor = LevelEditor(self, screen)
        self.duel = None                 # the pilot fight or team battle in progress (STATE_DUEL)
        self.duel_worker = None
        self.duel_banner_until = None
        self.level_complete_menu = LevelCompleteMenu(screen)
        self.quit_confirmation_menu = QuitConfirmationMenu(screen)
        self.splash_screen: Optional[SplashScreenState] = None
        if config.SPLASH_ENABLED:
            self._initialize_splash_screen()
        self.reset_scoring_to_profile_state()
        self.quit_confirmation_selection = 0
        if os.getenv('AUTOPLAY', '').lower() in ('1', 'true', 'yes', 'on'):
            self._start_autoplay()

    def _start_autoplay(self) -> None:
        """Run unattended: skip the menus and let the hunter fly level after level."""
        settings = config.SETTINGS.autoplay
        self.autoplay = AutoplayDirector(
            settings.firstLevel, settings.lastLevel, settings.timeLimitSeconds,
            no_kill_seconds=settings.noKillSeconds)
        self._begin_autoplay_level()

    def _begin_autoplay_level(self) -> None:
        """Start the director's level with the player parked and a hunter beside it."""
        self.level = self.autoplay.level
        self.state = config.STATE_PLAYING
        self.start_level()
        # No first move will come from a person: open the level the way one would.
        self.player_has_moved = True
        self.ship.game_started = True
        self._summon_hunter((self.ship.x, self.ship.y))
        self.autoplay.start_level(
            len(self.roster.alive()),
            hunter_present=self.hunter is not None)

    def _hunter_sees_player(self) -> bool:
        """Whether the hunter may perceive the player: not when autoplay hides the parked one."""
        if self.ship is not None and self.ship.observer:
            return False
        return not (self.autoplay is not None and config.SETTINGS.autoplay.hidePlayer)

    def _autoplay_level_over(self) -> bool:
        """Move on to the next level when the director says this one is over."""
        reason = self.autoplay.end_reason(
            enemies_left=len(self.roster.alive()),
            hunter_alive=self.hunter is not None and self.hunter.active)
        if reason is None:
            return False
        logging.getLogger(__name__).info('Autoplay level %d over (%s)', self.level, reason)
        self.autoplay.advance()
        self._begin_autoplay_level()
        return True
    
    def _initialize_splash_screen(self) -> None:
        """Initialize splash screen state."""
        self.splash_screen = SplashScreenState(None, self.screen)  # State machine not needed for simple splash
        self.splash_screen.enter()
    
    def _execute_ship_command(self, command_type: CommandType) -> None:
        """Execute a command on the player ship.
        
        Args:
            command_type: The command type to execute.
        """
        if command_type == CommandType.ROTATE_LEFT:
            self.ship.rotate_left()
        elif command_type == CommandType.ROTATE_RIGHT:
            self.ship.rotate_right()
        elif command_type == CommandType.APPLY_THRUST:
            self.ship.apply_thrust()
        elif command_type == CommandType.ACTIVATE_SHIELD:
            self.ship.activate_shield()
        # NO_ACTION and FIRE are handled separately
    
    def restart_level(self) -> None:
        """Retry immediately without keeping the current run's score."""
        replaying = self.replaying
        self.sound_manager.stop_all_sounds()
        self.scoring.total_score = self.total_score_before_level
        self.state = config.STATE_PLAYING
        self.start_level()
        # Retrying an ordinary clear replaces that attempt, even though the
        # first clear has already unlocked the next level in the profile.
        self.replaying = replaying

    def start_level(self, *, maze_seed: Optional[int] = None) -> None:
        """Start a new level."""
        self._close_hunter()
        self._close_enemy_hunters()
        self.collision_handler.last_player_kill = None
        self.collision_handler.level_kills = 0
        self.collision_handler.level_crystals_collected = 0
        self.collision_handler.weapon_kills.clear()
        self.collision_handler.weapon_pickups.clear()
        self.collision_handler.explosions.clear()
        self.finish_tally = None
        self.replaying = self.level < self.profile_manager.get_active_level()
        self.level_result = None
        # Store total score before starting level (for replay functionality)
        self.total_score_before_level = self.scoring.get_total_score()
        
        # Ensure warning sound is silenced when a level starts
        self.sound_manager.stop_critical_warning()
        self.critical_warning_active = False

        self.game_over_active = False
        self.game_frozen = False
        self.game_over_start_time = 0.0

        # Reset exit explosion state
        self.exit_explosion_active = False
        self.exit_explosion_time = 0.0
        self.exit_explosion_pos = None
        
        # Reset star animation
        self.star_animation = None
        self.level_complete_quit_confirm = False
        
        # Training can supply an episode seed; ordinary play uses the level's seed.
        seed = level_config.get_level_seed(self.level) if maze_seed is None else maze_seed
        random.seed(seed)
        # Note: Seed is set before maze generation to ensure reproducible mazes
        
        # Get maze complexity (from the level file, or calculated if it has none)
        maze_complexity = level_config.get_maze_complexity(self.level)
        
        # Get maze grid size (always returns a value, calculated if not in config)
        maze_grid_size = level_config.get_maze_grid_size(self.level)
        
        # Generate maze
        layout = level_config.get_level_layout(self.level)
        self.maze = Maze(self.level, complexity=maze_complexity, grid_size=maze_grid_size, layout=layout,
                         seed=seed, curved=getattr(self, 'curved_walls', None))
        self._plant_flora(seed)
        self.obstacles = ObstacleWorld(self.maze, layout.gates) if layout is not None and layout.gates else None
        self.maze.gate_segments = self.obstacles.segments() if self.obstacles else []
        
        # Create ship at start position
        self.ship = Ship(self.maze.start_pos)
        from weapons import EquippedWeapon, WeaponInventory, RULESET
        from weapons.catalog import PICKUP_RULESETS, default_catalog
        ruleset = getattr(self, 'combat_ruleset', 'legacy')
        catalog = getattr(self, 'weapon_catalog', None) or default_catalog(ruleset)
        if ruleset in PICKUP_RULESETS:
            state_type = WeaponInventory if ruleset == RULESET else EquippedWeapon
            self.ship.weapon = state_type(catalog)
        self.collision_handler.combat_ruleset = getattr(self, 'combat_ruleset', 'legacy')
        self.collision_handler.weapon_catalog = catalog
        data = level_config.load_level_config(self.level) or {}
        self.collision_handler.weapon_allowlist = data.get('weapons')
        self.ship.observer = config.observer_mode_enabled()
        # Explicitly reset velocity and previous position to prevent visual jumps
        self.ship.vx = 0.0
        self.ship.vy = 0.0
        self.ship.prev_x = self.ship.x
        self.ship.prev_y = self.ship.y
        # Set initial angle to point towards exit (opposite corner)
        exit_pos = (self.maze.exit.x, self.maze.exit.y)
        initial_angle = get_angle_to_point(self.maze.start_pos, exit_pos)
        self.ship.angle = initial_angle
        # Activate shield at level start (initial activation, no fuel consumed)
        self.ship.shield_active = True
        # Reset gun upgrade state
        self.ship.reset_gun_upgrade()
        self.exit_policy = level_config.get_level_exit_policy(self.level)
        # The mission is easy to miss in the side panel: announce it until the player starts flying
        self.announcer = Announcer()
        mission = self.exit_policy.mission()
        if mission is not None:
            self.announcer.show('MISSION', mission)
        self.ship.ammo_policy = level_config.get_level_ammo_policy(self.level)
        self.exit_opened = False
        self.play_seconds = 0.0
        
        # Reset player movement flag - game loop won't start until first move
        self.player_has_moved = False
        self.ship.game_started = False  # Prevent shield timer countdown until first move
        
        # Spawn all enemies using SpawnManager
        # Check for level config overrides, fall back to defaults if not present
        enemy_counts = level_config.get_level_enemy_counts(self.level)
        if enemy_counts is None:
            enemy_counts = level_rules.get_enemy_counts(self.level)
        enemy_counts.anemone = level_config.get_level_anemone_count(self.level)
        split_boss_count = level_config.get_level_split_boss_count(self.level)
        mother_boss_count = level_config.get_level_mother_boss_count(self.level)
        spawn_positions = [] if layout is not None else self.maze.get_valid_spawn_positions(
            enemy_counts.total + enemy_counts.replay + enemy_counts.flocker + enemy_counts.flighthouse + enemy_counts.vibehouse + enemy_counts.egg + enemy_counts.anemone + split_boss_count + mother_boss_count + level_config.get_level_rules_hunter_count(self.level) + level_config.get_level_neural_hunter_count(self.level) + level_config.get_level_boss_viber_count(self.level) + level_config.get_level_vibelet_count(self.level) + level_config.get_level_belemnite_count(self.level) + level_config.get_level_wallcrawler_count(self.level) + 5,  # Extra buffer for spawn positions
            start_clearance=config.ENEMY_START_CLEARANCE
        )
        # The hunter is summoned by a Jev beacon; some levels start with one lying in the maze
        self.jev_beacon_level = level_config.level_allows_jev_beacon(self.level)
        if self.jev_beacon_level and self.hunter_worker is None:
            self.hunter_worker = PilotWorker()
        beacon_pos = None
        if layout is not None:
            # A layout's beacon lies exactly where its author put it
            if layout.beacon is not None:
                beacon_pos = self.maze.position_calculator.grid_center_to_screen(*layout.beacon)
        elif level_config.level_has_hunter(self.level):
            placed = level_config.get_level_hunter_config(self.level)
            if placed is not None:
                try:
                    beacon_pos = resolve_hunter_spawn(placed, self.maze, self.ship)
                except ValueError as exc:
                    logging.getLogger(__name__).warning('Jev beacon placement rejected: %s; using automatic placement', exc)
            if beacon_pos is None:
                beacon_pos = place_jev_beacon(self.maze, self.ship)
        spawn_positions = reserve_hunter_clearance(spawn_positions, beacon_pos, config.SHIP_SIZE)

        # No anemone starts with the player already inside its pull
        anemone_reach = config.ANEMONE_REACH_CELLS * (self.maze.cell_size_x + self.maze.cell_size_y) / 2
        if layout is None:
            self.spawn_manager.spawn_all_enemies(
                self.level, spawn_positions, self.command_recorder, enemy_counts, split_boss_count, mother_boss_count,
                anemone_keep_clear=(self.maze.start_pos, anemone_reach), maze=self.maze,
                belemnite_count=level_config.get_level_belemnite_count(self.level),
                wallcrawler_count=level_config.get_level_wallcrawler_count(self.level)
            )
            self._spawn_enemy_hunters(spawn_positions)
        else:
            self._spawn_enemy_hunters_at(
                self.spawn_manager.spawn_from_layout(layout, self.maze, self.level, self.command_recorder))
        # Creatures are set up before the player first moves, so they show it from the start (an anemone its reach)
        self.encounter.ready()
        # Earn the ally by clearing most of the opening encounter; later spawns cannot lower this bar.
        self.hunter_unlock_kills = max(4, math.ceil(sum(self.roster.counts().values()) * 0.65))
        self.base_par_seconds = self._base_par_seconds()
        
        # Clear projectiles and crystals
        self.projectiles = []
        self.powerup_crystals = []
        if beacon_pos is not None:
            self.powerup_crystals.append(JevBeacon(beacon_pos))
        
        # Start scoring
        current_time = self.clock()
        self.scoring.start_level(current_time)
        self.level_start_time = current_time  # Record level start time to ignore initial SPACE press
        
        # Reset star indicator
        self.star_indicator.reset()
        
        # Start command recording
        self.command_recorder.start_recording()
        
        # Clear any pending input events FIRST to prevent keys from menu from affecting gameplay
        # This must happen BEFORE checking key state
        pygame.event.clear(pygame.KEYDOWN)
        pygame.event.clear(pygame.KEYUP)
        pygame.event.clear(pygame.JOYBUTTONDOWN)
        pygame.event.clear(pygame.JOYBUTTONUP)
        
        # Track which keys were pressed when level started - we'll ignore these until released
        # Check AFTER clearing events to see if keys are still held
        keys_at_start = pygame.key.get_pressed()
        self.keys_pressed_at_start = set()
        problematic_keys = [pygame.K_SPACE, pygame.K_UP, pygame.K_w]
        for key in problematic_keys:
            try:
                # Check if key is in range and pressed
                if key < len(keys_at_start) and keys_at_start[key]:
                    self.keys_pressed_at_start.add(key)
            except (IndexError, KeyError, TypeError):
                pass
        
    
    @property
    def hunter(self) -> Optional[Creature]:
        """The hunter's ship while a hunter flight is in play, else None."""
        return self.hunter_flight.ship if self.hunter_flight is not None else None

    def _close_hunter(self) -> None:
        if self.hunter_flight is not None:
            self.hunter_flight.close()
        self.hunter_flight = None

    def _summon_hunter(self, near) -> None:
        """Bring the hunter in beside a collected Jev beacon, replacing one that has died."""
        try:
            pos = nearest_hunter_spawn(near, self.maze, self.ship)
        except ValueError as exc:
            logging.getLogger(__name__).warning('Hunter not summoned: %s', exc)
            return
        self._close_hunter()
        mode = config.SETTINGS.game.hunterPilot
        if mode == 'neural':
            pilot = NeuralPilot(load_neural_policy(), config.SETTINGS.game.neuralDecisionFrames)
        else:
            if self.hunter_worker is None:
                self.hunter_worker = PilotWorker()
            pilot = WorkerPilot(self.hunter_worker)
        self.hunter_flight = make_hunter(pos, world=self, setup=PilotSetup(
            hunter_settings_for(mode), pilot, self.hunter_surroundings))
        self.hunter_flight.on_event(self._sound_hunter_event)

    def _sound_hunter_event(self, creature, name) -> None:
        """The hunter's shots are heard."""
        if name == 'fired':
            self.sound_manager.play_hunter_shot(creature.fired_weapon_id)

    def _hunter_energy_reading(self):
        """(energy, most energy) of the hunter flying now, or None when there is none (or it has been destroyed)."""
        if self.hunter is None or not self.hunter.active:
            return None
        return self.hunter.health, self.hunter.settings.health

    def _draw_hunter_energy(self, below: int = 755) -> None:
        """The hunter's energy gauge, in the HUD below the player's gauges and the EXIT block (never beside the ship)."""
        reading = self._hunter_energy_reading()
        if reading is None:
            return
        energy, most = reading
        y = min(below + 25, config.SCREEN_HEIGHT - 45)  # room for the label above the bar and the reading below
        UIElementRenderer.draw_energy_gauge(
            self.screen, config.UI_ZONE_WIDTH // 2, y, energy / max(most, 1), f"{math.ceil(energy)}/{most}")

    def _base_par_seconds(self) -> float:
        """Par from the maze route and the enemies on the level now, at its start."""
        to_grid = self.maze.position_calculator.screen_to_grid
        start, end = self.maze.start_pos, self.maze.exit.get_pos()
        cells = route_cells(self.maze.grid, to_grid(*start), to_grid(*end))
        route_px = cells * self.maze.cell_size if cells is not None else math.dist(start, end)
        timer = self.exit_policy.seconds if isinstance(self.exit_policy, OpenAfterTime) else None
        return par_seconds(route_px, self.roster.counts(), 0, timer)

    def star_tally(self) -> StarTally:
        """The stars as they stand: what the player would get by reaching the exit now."""
        collected = self.collision_handler.level_crystals_collected
        left = sum(1 for crystal in self.powerup_crystals
                   if crystal.active and not isinstance(crystal, JevBeacon))
        par = self.base_par_seconds + config.SETTINGS.game.parSecondsPerCrystal * (collected + left)
        return StarTally(self.collision_handler.level_kills, sum(self.roster.counts().values()),
                         collected, left, self.play_seconds, par, self.scoring.hits_taken())

    def par_pulse(self) -> float:
        """How brightly the par star throbs: 0 until its last few seconds, then a pulse, and 0 once par is past."""
        left = self.star_tally().par_seconds - self.play_seconds
        if left < 0 or left > PAR_WARNING_SECONDS:
            return 0.0
        return (math.sin(self.play_seconds * math.tau * 1.5) + 1) / 2

    def exit_state(self) -> ExitState:
        """What the exit policy may look at this frame."""
        return ExitState(self.play_seconds, self.roster.counts())

    def last_enemy(self):
        """The one enemy still keeping the exit shut, or None while it waits for more than one, or for none."""
        if self.exit_opened:
            return None
        waiting = [enemy for enemy in self.roster.alive() if self.exit_policy.waits_for(enemy.type)]
        return waiting[0] if len(waiting) == 1 else None

    def jev_beacon_allowed(self) -> bool:
        """Whether a kill may drop a Jev beacon: only where one is allowed, and not while a hunter flies."""
        hunter = level_config.get_level_hunter_config(self.level)
        earned = hunter is None or (isinstance(hunter, dict) and hunter.get('earned'))
        unlocked = not earned or self.collision_handler.level_kills >= self.hunter_unlock_kills
        return self.jev_beacon_level and unlocked and (self.hunter is None or not self.hunter.active)

    def _update_hunter(self, dt) -> None:
        if self.hunter_flight is None:
            return
        if self.hunter.is_dying:
            self.hunter.update_death(dt)
        if not self.player_has_moved:
            self.hunter_flight.pause()
            return
        self.hunter_flight.step(dt)

    def _hotkey_summon_hunter(self) -> None:
        """Test key (ASTROVOID_HUNTER_HOTKEY): call the hunter in now.

        A Jev beacon still lying in the level is the hunter waiting to be called, so it is collected as the
        player would, rather than a second hunter being added. With none left, the hunter appears beside the
        ship, replacing any that is flying.
        """
        beacon = next((c for c in self.powerup_crystals if isinstance(c, JevBeacon) and c.active), None)
        if beacon is not None:
            beacon.die()
            self.sound_manager.play_powerup_activation('beacon')
            self._summon_hunter(beacon.get_pos())
        else:
            self._summon_hunter((self.ship.x, self.ship.y))

    def start_fun_entry(self, key: str) -> None:
        """Start the FUN-list entry with this key."""
        if key == 'pilot_duel':
            self.start_duel()
        elif key == 'demo':
            self._start_autoplay()
        elif key == 'team_battle':
            self.start_team_battle()

    # How a Rules hunter's pilot worker is made. None: a threaded one answering on the real clock. The lab sets an
    # instance attribute to give a synchronous worker on its own clock.
    _rules_worker_factory = None

    @property
    def enemy_hunters(self):
        """Every enemy hunter on the level, Rules and Neural."""
        return self.roster.of_kind(*enemy_kinds.hunters())

    def protagonist(self):
        """Whom enemy pilots hunt: the player, or the Hunter while observing.

        The lab replaces this with its own hunter target.
        """
        if self.ship is not None and getattr(self.ship, 'observer', False):
            return self.hunter
        return self.ship

    def rules_worker(self):
        """A new pilot worker for a Terminator's rules pilot."""
        if self._rules_worker_factory is not None:
            return self._rules_worker_factory()
        return PilotWorker(client_factory=RulesClient)

    def enemy_policy(self):
        """The neural pilot's policy for Neural hunters, loaded once; None when no model is installed."""
        if not hasattr(self, '_enemy_policy'):
            self._enemy_policy = load_neural_policy()
        return self._enemy_policy

    def _spawn_enemy_hunters(self, spawn_positions):
        """Place the level's Rules and Neural hunters on open ground: as far from the player as the maze allows,
        clear of every other enemy. A level with no such room skips them (and says so) and stays playable."""
        wanted = [('rules_hunter', level_config.get_level_rules_hunter_count(self.level)),
                  ('neural_hunter', level_config.get_level_neural_hunter_count(self.level)),
                  ('boss_viber', level_config.get_level_boss_viber_count(self.level)),
                  ('vibelet', level_config.get_level_vibelet_count(self.level))]
        if not any(count for _, count in wanted) or self.ship is None:
            return
        taken = self.roster.positions()
        start = (self.ship.x, self.ship.y)
        clear = 2.0 * config.SHIP_SIZE * 2
        candidates = sorted(
            (p for p in spawn_positions
             if all(math.hypot(p[0] - x, p[1] - y) > clear for x, y in taken)),
            key=lambda p: -math.hypot(p[0] - start[0], p[1] - start[1]))
        placements = [kind for kind, count in wanted for _ in range(count)]
        self._spawn_enemy_hunters_at(list(zip(placements, candidates)))
        if len(candidates) < len(placements):
            logging.getLogger(__name__).warning(
                'Level %d: room for %d of %d enemy hunters', self.level, len(candidates), len(placements))

    def _spawn_enemy_hunters_at(self, placements):
        """Make an enemy hunter of each (kind, position). Without an installed neural model a kind with a neural
        pilot is left out (and said so)."""
        for kind, pos in placements:
            if enemy_kinds.kind(kind).pilot == 'neural' and self.enemy_policy() is None:
                logging.getLogger(__name__).warning('Level %d: no neural model, so no Neural hunter', self.level)
                continue
            self.roster.add(make_enemy(kind, pos, self.level, self.command_recorder))

    def _step_gates(self, dt):
        """Let the gates turn and push back on whatever flies into them; anemones and flighthouses jam them."""
        movers, anchored, seen = [self.ship], [], set()
        if self.hunter is not None and self.hunter.active:
            movers.append(self.hunter)
        seen.update(id(m) for m in movers)
        for enemy in self.roster.all():
            if not enemy.active or id(enemy) in seen:
                continue
            seen.add(id(enemy))
            if enemy_kinds.kind(enemy.type).anchored:
                anchored.append(enemy)
            elif hasattr(enemy, 'vx') and hasattr(enemy, 'vy'):
                movers.append(enemy)
        self.obstacles.step(dt, movers, anchored)
        self.maze.gate_segments = self.obstacles.segments()

    def _hit_enemies(self, projectile) -> bool:
        """A friendly shot meets the level's enemies; True when one of them took it."""
        return self.encounter.shoot(projectile)

    def _update_explosions(self, dt):
        """Set off the blasts whose fuse has burnt; what they kill may light more."""
        for blast in self.collision_handler.explosions.update(dt):
            self.collision_handler.handle_blast(
                blast, self.ship, self.hunter, self.roster.alive(), self._hit_enemies,
                self.maze)
            self._stop_dead_hunter()

    def _close_enemy_hunters(self):
        """Stop every enemy hunter's pilot and worker (a level ends, restarts, or the game closes)."""
        self.encounter.close()

    def _make_duel_worker(self):
        """The rules pilot's worker for a duel: its own thread, apart from the game's hunter worker."""
        return PilotWorker(client_factory=RulesClient)

    def start_duel(self) -> None:
        """Pilot fight: the rules pilot's hunter against the neural pilot's, in a dense maze, player watching."""
        policy = load_neural_policy()
        if policy is None:
            self.fun_menu.notice = 'Vibes unavailable: no neural policy found'
            self.state = config.STATE_FUN
            return
        game_settings = config.SETTINGS.game
        try:
            complexity = MazeComplexity[str(game_settings.duelMazeComplexity).upper()]
        except KeyError:
            self.fun_menu.notice = f'Duel unavailable: unknown maze complexity {game_settings.duelMazeComplexity!r}'
            self.state = config.STATE_FUN
            return
        self._close_hunter()
        seed = level_config.get_level_seed(DUEL_MAZE_LEVEL)
        random.seed(seed)
        self.maze = Maze(DUEL_MAZE_LEVEL, complexity=complexity, grid_size=game_settings.duelMazeGridSize, seed=seed,
                         curved=getattr(self, 'curved_walls', None))
        self.maze.exit.active = False            # nobody is flying out: do not draw the exit portal
        self._plant_flora(seed)
        self.ship = Ship(self.maze.start_pos)
        from weapons import EquippedWeapon, WeaponInventory, RULESET
        from weapons.catalog import PICKUP_RULESETS, default_catalog
        ruleset = getattr(self, 'combat_ruleset', 'legacy')
        catalog = getattr(self, 'weapon_catalog', None) or default_catalog(ruleset)
        if ruleset in PICKUP_RULESETS:
            state_type = WeaponInventory if ruleset == RULESET else EquippedWeapon
            self.ship.weapon = state_type(catalog)
        self.collision_handler.combat_ruleset = getattr(self, 'combat_ruleset', 'legacy')
        self.collision_handler.weapon_catalog = catalog
        data = level_config.load_level_config(self.level) or {}
        self.collision_handler.weapon_allowlist = data.get('weapons')    # the hidden spectator: only here so spawns keep clear of it
        self.projectiles = []
        worker = self._make_duel_worker()
        try:
            self.duel = build_duel(self.maze, self.ship, worker, policy, game_settings.duelHealth,
                                   game_settings.duelTimeoutSeconds, game_settings.neuralDecisionFrames)
        except ValueError as exc:
            worker.close()
            self.fun_menu.notice = f'Duel unavailable: {exc}'
            self.state = config.STATE_FUN
            return
        random.seed()                            # the maze was built from a fixed seed; the enemies are random
        enemies = DuelEnemies(self.maze, self.sound_manager)
        enemies.spawn([side.hunter.get_pos() for side in self.duel.sides])
        self.duel.enemies = enemies
        self.duel_worker = worker
        self.duel_banner_until = None
        self.state = config.STATE_DUEL

    def start_team_battle(self) -> None:
        """Flockers vs Wallcrawlers: the two teams fight in the match file's arena, player watching."""
        try:
            match = load_match()
        except ValueError as exc:
            self.fun_menu.notice = f'Battle unavailable: {exc}'
            self.state = config.STATE_FUN
            return
        self._close_hunter()
        self.projectiles = []
        self.duel = TeamBattle(match, self.sound_manager, (self.flocker_policy, self.wallcrawler_policy))
        self.maze = self.duel.maze
        self._plant_flora(match.seed, match.flora)
        self.duel_banner_until = None
        self.state = config.STATE_DUEL

    def _plant_flora(self, seed: Optional[int], grows: Optional[bool] = None) -> None:
        """Grow living weed in the new cave if `grows` says so, or else if the game grows it. It has its own
        randomness, so seeded caves and enemies come out as they would without it."""
        if seed is None:
            seed = random.SystemRandom().randrange(2 ** 32)
        if grows is None:
            grows = getattr(self, 'grows_flora', config.FLORA.enabled)
        self.maze.flora = Flora.seeded(self.maze, seed) if grows else None

    def _step_flora(self, dt: float) -> None:
        if self.maze is not None and self.maze.flora is not None:
            self.maze.flora.step(dt)

    def _update_duel(self, dt: float) -> None:
        if self.duel is None:
            return
        self.duel.update(dt)
        self._step_flora(dt)
        if self.duel.result is not None:
            if self.duel_banner_until is None:
                self.duel_banner_until = time.monotonic() + DUEL_BANNER_SECONDS
            elif time.monotonic() >= self.duel_banner_until:
                self.end_duel()

    def end_duel(self) -> None:
        """Leave the duel for the FUN list, closing its controllers and worker; the normal game is left clean."""
        if self.duel is not None:
            self.duel.close()
            self.duel = None
        if self.duel_worker is not None:
            self.duel_worker.close()
            self.duel_worker.join(2.0)
            self.duel_worker = None
        self.duel_banner_until = None
        self.projectiles = []
        self.fun_menu.refresh()
        self.state = config.STATE_FUN

    def _stop_dead_hunter(self) -> None:
        if self.hunter is not None and not self.hunter.active:
            self.hunter_flight.close()

    def handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            event = self.input_handler.dpad_button_as_hat(event)
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.JOYDEVICEADDED:
                # Controller connected
                self.input_handler.add_controller(event.device_index)
            elif event.type == pygame.JOYDEVICEREMOVED:
                # Controller disconnected
                self.input_handler.remove_controller(event.instance_id)
            elif event.type == pygame.JOYBUTTONDOWN:
                # Handle splash screen input
                if self.state == config.STATE_SPLASH and self.splash_screen:
                    self.splash_screen.handle_event(event)
                else:
                    # Use state handler for controller events
                    handler = self.state_handler_registry.get_handler(self.state)
                    handler.handle_controller(event, self)
            elif event.type == pygame.JOYHATMOTION or event.type == pygame.JOYAXISMOTION:
                # Handle controller hat (d-pad) and axis (stick) events for menu navigation
                if self.state in (config.STATE_MENU, config.STATE_PROFILE_SELECTION, config.STATE_LEVEL_SELECT, config.STATE_LEVEL_COMPLETE, config.STATE_SETTINGS, config.STATE_FUN, config.STATE_QUIT_CONFIRM, config.STATE_EDITOR):
                    handler = self.state_handler_registry.get_handler(self.state)
                    handler.handle_controller(event, self)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h and self.state == config.STATE_PLAYING and _hunter_hotkey_enabled():
                    self._hotkey_summon_hunter()
                # Handle splash screen input
                if self.state == config.STATE_SPLASH and self.splash_screen:
                    self.splash_screen.handle_event(event)
                else:
                    # Use state handler for keyboard events
                    handler = self.state_handler_registry.get_handler(self.state)
                    handler.handle_keyboard(event, self)
    
    def update(self, dt: float) -> None:
        """Update game state."""
        if self.state == config.STATE_DUEL:
            self._update_duel(dt)
            return
        if self.state == config.STATE_FUN:
            self.fun_menu.update(dt)
            return
        if self.state == config.STATE_EDITOR:
            self.level_editor.update(dt)
            return
        hunter_running = (self.state == config.STATE_PLAYING and self.player_has_moved
                          and not self.exit_explosion_active and not self.game_frozen
                          and not self.game_over_active)
        if self.hunter_flight is not None and not hunter_running:
            self.hunter_flight.pause()
        # Update splash screen
        if self.state == config.STATE_SPLASH:
            if self.splash_screen:
                self.splash_screen.update(dt)
                # Check if splash should transition to menu
                if self.splash_screen.should_transition:
                    self.state = config.STATE_MENU
        
        # Update menu UI animations
        if self.state == config.STATE_MENU:
            profile = self.profile_manager.get_active_profile()
            self.main_menu.set_profile_info(
                profile.name if profile else None,
                profile.level if profile else None
            )
            self.main_menu.update(dt)
        elif self.state == config.STATE_PROFILE_SELECTION:
            self.profile_selection_menu.update(dt)
            return
        elif self.state == config.STATE_LEVEL_SELECT:
            self.level_select_menu.update(dt)
            return
        elif self.state == config.STATE_SETTINGS:
            self.settings_menu.update(dt)
            return
        elif self.state == config.STATE_LEVEL_COMPLETE:
            self.level_complete_menu.update(dt)
        
        if self.state == config.STATE_LEVEL_COMPLETE:
            # Update star animation during level complete
            if self.star_animation:
                self.star_animation.update(dt)
            return
        
        if self.state != config.STATE_PLAYING:
            return  # Don't update if not playing (including quit confirmation)
        
        if self.game_over_active:
            if self.ship is not None and self.ship.is_dying:
                self.ship.update_death(dt)
            elapsed = self.clock() - self.game_over_start_time
            if elapsed >= self.game_over_death_duration + self.game_over_fade_duration + self.game_over_text_duration:
                self.complete_level(success=False)
            return

        # Update exit explosion animation
        if self.exit_explosion_active:
            self.exit_explosion_time += dt / 60.0  # Convert to seconds
            # Explosion lasts 1.5 seconds, then transition to level complete
            if self.exit_explosion_time >= 1.5:
                self.exit_explosion_active = False
                self.complete_level(success=True)
                return
        
        if not self.ship or not self.maze:
            return
        if self.autoplay is not None and self._autoplay_level_over():
            return
        
        self.maze.update(dt)
        self._step_flora(dt)
        if self.player_has_moved:
            self.announcer.dismiss()
        self.announcer.update(dt)

        if self.player_has_moved and not self.game_frozen:
            self.play_seconds += dt / config.FPS  # dt counts frames at config.FPS
        # The level's exit policy opens the exit; nothing that appears afterwards shuts it again
        if not self.exit_opened and self.exit_policy.is_open(self.exit_state()):
            self.exit_opened = True
        if self.maze.exit.active:
            self.maze.exit.set_activated(self.exit_opened, self.sound_manager)
        
        # Update exit animation and check player proximity
        if self.maze.exit.active:
            player_pos = (self.ship.x, self.ship.y) if self.ship else None
            self.maze.exit.update(dt, player_pos)
        
        # Apply exit portal attraction force to ship
        if self.ship and self.maze.exit.active:
            attraction_force = self.maze.exit.get_attraction_force((self.ship.x, self.ship.y))
            if attraction_force is not None:
                # Apply attraction force to ship velocity
                self.ship.vx += attraction_force[0]
                self.ship.vy += attraction_force[1]
                
                # Limit max speed (same as thrust)
                speed = math.sqrt(self.ship.vx * self.ship.vx + self.ship.vy * self.ship.vy)
                if speed > self.ship.max_speed:
                    scale = self.ship.max_speed / speed
                    self.ship.vx *= scale
                    self.ship.vy *= scale
        
        # Ignore input for a short period after level start to avoid counting menu key presses
        current_time = self.clock()
        ignore_input_duration = 0.5  # Ignore input for 0.5 seconds after level start
        ignore_input = (current_time - self.level_start_time) < ignore_input_duration
        
        # Process input and get commands
        keys = pygame.key.get_pressed()
        
        # Process keyboard input (match original process_input logic)
        keyboard_commands = []
        for key_codes, command_type in self.input_handler.key_mappings.items():
            # Check if any of the mapped keys are pressed
            if any(keys[key_code] for key_code in key_codes):
                keyboard_commands.append(command_type)
        
        # Process controller input only if not ignoring input
        controller_commands = []
        if not ignore_input:
            controller_commands = self.input_handler.process_controller_input()
        
        # Combine commands
        commands = keyboard_commands[:]
        for cmd in controller_commands:
            if cmd not in commands:
                commands.append(cmd)
        
        # Filter out problematic commands during ignore period OR if keys were pressed at start (until released)
        should_filter_commands = ignore_input
        
        if self.keys_pressed_at_start:
            # Check if problematic keys are still pressed
            keys_still_pressed = set()
            for key in self.keys_pressed_at_start:
                try:
                    if keys[key]:
                        keys_still_pressed.add(key)
                except (IndexError, KeyError):
                    pass
            
            # If ANY problematic key is still pressed, we should filter commands
            if keys_still_pressed:
                should_filter_commands = True
            
            # Update the set
            self.keys_pressed_at_start = keys_still_pressed
        
        if self.ship_pilot is not None:
            pilot_command = self.ship_pilot.decide(self)
            commands = []
            if pilot_command.turn < 0:
                commands.append(CommandType.ROTATE_LEFT)
            elif pilot_command.turn > 0:
                commands.append(CommandType.ROTATE_RIGHT)
            if pilot_command.thrust:
                commands.append(CommandType.APPLY_THRUST)
        elif self.autoplay is not None:
            commands = []  # Nobody is flying the player: it stays parked

        # Filter out ALL movement commands if we should ignore input
        # This prevents keys pressed at menu from affecting gameplay
        if should_filter_commands and self.ship_pilot is None:
            # Filter out all movement commands (thrust, rotate)
            commands = [cmd for cmd in commands if cmd not in (
                CommandType.APPLY_THRUST,
                CommandType.ROTATE_LEFT,
                CommandType.ROTATE_RIGHT
            )]
        
        # Down/S (or controller A) advances once per press in inventory games.
        cycle_pressed = (keys[pygame.K_DOWN] or keys[pygame.K_s]
                         or self.input_handler.is_controller_shield_pressed())
        from weapons import WeaponInventory
        inventory = isinstance(self.ship.weapon, WeaponInventory)
        if inventory:
            self.ship.cycle_weapon(cycle_pressed and not should_filter_commands
                                   and self.autoplay is None)
        elif self.ship.shield_initial_timer <= 0:
            self.ship.shield_active = bool(cycle_pressed)

        # Handle fire separately (has rate limiting)
        # Check both keyboard and controller for fire input
        # Ignore fire input during ignore period OR if SPACE was pressed at start and still pressed
        fire_key_pressed = False
        fire_controller_pressed = False
        
        # Check if we should ignore fire (same logic as command filtering)
        space_still_pressed = pygame.K_SPACE in self.keys_pressed_at_start
        should_ignore_fire = ignore_input or space_still_pressed or self.autoplay is not None
        
        if self.ship_pilot is not None:
            fire_key_pressed = pilot_command.fire
        elif not should_ignore_fire:
            try:
                fire_key_pressed = keys[pygame.K_SPACE]
            except (IndexError, KeyError):
                pass
            fire_controller_pressed = self.input_handler.is_controller_fire_pressed()
        
        fire_pressed = fire_key_pressed or fire_controller_pressed
        
        if not self.player_has_moved:
            # Check for explicit movement commands (not shield, not NO_ACTION)
            has_movement = False
            for cmd in commands:
                if cmd in (CommandType.ROTATE_LEFT, CommandType.ROTATE_RIGHT, CommandType.APPLY_THRUST):
                    has_movement = True
                    break
            
            # Fire also counts as first move, but ignore it if we just started the level
            if has_movement or (fire_pressed and not ignore_input):
                # First move detected - start the game loop
                self.player_has_moved = True
                self.ship.game_started = True  # Allow shield timer to countdown
        
        # Execute movement commands on ship and record them
        # Down/S is handled above as cycling (or shielding in historical rules).
        for cmd in commands:
            if cmd == CommandType.ACTIVATE_SHIELD:
                continue
            self._execute_ship_command(cmd)
            self.command_recorder.record_command(cmd)
        
        if fire_pressed:
            if not hasattr(self, 'last_shot_time'):
                self.last_shot_time = 0
            current_time = self.clock_ms()
            fire_cooldown = calculate_fire_cooldown(self.ship)
            
            if self.ship.weapon is not None or current_time - self.last_shot_time > fire_cooldown:
                projectiles = self.ship.fire()
                if projectiles:
                    # fire() now returns a list (single or multiple projectiles)
                    self.projectiles.extend(projectiles)
                    self.scoring.record_shot()
                    self.command_recorder.record_command(CommandType.FIRE)
                    self.last_shot_time = current_time
        
        # Record NO_ACTION once per loop when no input is detected
        # This allows the replay enemy to mirror periods of inactivity
        if not commands and not fire_pressed:
            self.command_recorder.record_command(CommandType.NO_ACTION)
        
        # Update ship
        self.ship.update(dt)
        
        # Check ship-wall collision (use spatial grid for optimization)
        # Walls always stop the ship; the shield only spares it the penalty
        if self.ship.check_wall_collision(self.maze.walls, self.maze.spatial_grid):
            if not self.ship.is_shield_active():
                self.scoring.record_wall_collision()
        
        # Only update enemies after player has made their first move
        if self.player_has_moved:
            if not self.exit_explosion_active and not self.game_frozen and not self.game_over_active:
                self._update_hunter(dt)
            self.encounter.step(dt)
            self.roster.update_dying(dt)

        if self.obstacles:
            self._step_gates(dt)

        if self.player_has_moved:
            self.collision_handler.handle_hunter_contacts(
                self.hunter, self.ship, self.roster.alive())
            self._stop_dead_hunter()

        # Update projectiles and handle collisions
        active_projectiles = []
        for projectile in self.projectiles:
            from weapons.projectile import WeaponProjectile
            if isinstance(projectile, WeaponProjectile):
                from weapons.simulation import advance
                targets = [c for c in self.encounter.creatures() if c.allegiance == 'enemy']
                advance(projectile, dt, self.maze, targets, self.encounter.apply_hit,
                        self.collision_handler.explosions.trigger, self.obstacles)
                if projectile.active:
                    active_projectiles.append(projectile)
                continue
            projectile.update(dt)
            
            if not projectile.active:
                continue
            
            # A plank stops a shot and takes a nudge from it
            if self.obstacles and self.obstacles.bullet_hit(projectile, dt):
                continue
            
            # Check projectile-wall collision (use spatial grid)
            # Friendly projectiles can damage walls
            if not projectile.is_enemy:
                hit_wall = projectile.check_wall_collision(self.maze.walls, self.maze.spatial_grid)
                if hit_wall:
                    # Damage the wall (hit_wall is already a WallSegment)
                    self.maze.damage_wall(hit_wall, (projectile.x, projectile.y))
            else:
                # Enemy projectiles just deactivate on wall collision
                projectile.check_wall_collision(self.maze.walls, self.maze.spatial_grid)
            
            # Check enemy projectile-ship collision
            if self.collision_handler.handle_projectile_ship_collision(projectile, self.ship, self.scoring):
                continue  # Projectile destroyed, skip adding to active list
            
            if self.collision_handler.handle_projectile_hunter_collision(projectile, self.hunter):
                self._stop_dead_hunter()
                continue

            # Friendly projectile-enemy collisions share normal destruction effects.
            if self._hit_enemies(projectile):
                continue  # Projectile destroyed, skip adding to active list
            
            # Only add to active list if projectile is still active after all collision checks
            if projectile.active:
                active_projectiles.append(projectile)
        
        # Replace projectiles list with active ones
        self.projectiles = active_projectiles

        self._update_explosions(dt)
        
        self._update_powerup_crystals(dt)
        
        # Check exit reached
        if (self.autoplay is None or self.ship_pilot is not None) and self.maze.check_exit_reached((self.ship.x, self.ship.y), self.ship.radius):
            if not self.exit_explosion_active:
                # Start exit explosion; nothing after this changes the stars
                self.finish_tally = self.star_tally()
                self.exit_explosion_active = True
                self.exit_explosion_time = 0.0
                self.exit_explosion_pos = self.maze.exit.get_pos()
                # Stop all sounds
                self.sound_manager.stop_all_sounds()
                # Stop ship's thruster sound specifically
                if self.ship:
                    self.ship.sound_manager.stop_thruster()
                # Play cosmic warble for exit
                self.sound_manager.play_exit_warble()
                self.sound_manager.stop_critical_warning()
                self.critical_warning_active = False
            return  # Don't update game during explosion
        
        # Check if score has reached zero (level failed)
        current_time = self.clock()
        potential = self.scoring.calculate_current_potential_score(
            current_time,
            self.ship.fuel,
            self.ship.ammo
        )
        potential_score = potential['potential_score']
        self._update_critical_warning(potential_score)
        self.ship.low_power = potential['score_percentage'] <= LOW_POWER_THRESHOLD

        # When score reaches zero, freeze all action and start fade
        if potential_score <= 0 and not self.game_frozen \
                and (self.autoplay is None or self.ship_pilot is not None):
            self.game_frozen = True
            self._start_game_over_sequence(current_time)
        
        # The stars the player would get by reaching the exit now; a change chimes
        self.star_indicator.update(earned(self.star_tally()))

    def _update_critical_warning(self, potential_score: float) -> None:
        """Start/stop critical warning sound based on remaining potential score."""
        # Only warn while there are still points/energy remaining
        if 0 < potential_score <= self.CRITICAL_WARNING_THRESHOLD:
            if not self.critical_warning_active:
                self.sound_manager.start_critical_warning()
                self.critical_warning_active = True
            else:
                # Periodic check: verify sound is still playing
                channel_busy = self.sound_manager.critical_warning_channel.get_busy() if self.sound_manager.critical_warning_channel else False
                if not channel_busy:
                    # Sound stopped unexpectedly, restart it
                    self.sound_manager.start_critical_warning()
        elif self.critical_warning_active:
            self.sound_manager.stop_critical_warning()
            self.critical_warning_active = False
    
    def _start_game_over_sequence(self, current_time: float) -> None:
        """Start the game over sequence: freeze action, fade to black, power down sound, then GAME OVER text."""
        self.game_over_active = True
        self.game_over_start_time = current_time
        if self.ship is not None and self.ship.active:
            self.ship.die()
        # Stop all sounds for dramatic effect
        self.sound_manager.stop_all_sounds()
        self.critical_warning_active = False
        # Play power down sound
        self.sound_manager.play_power_down()
    
    def complete_level(self, success: bool = True) -> None:
        """Handle level completion or failure.
        
        Args:
            success: True if level was completed successfully (reached exit),
                     False if level failed (score reached zero)
        """
        self._close_hunter()
        current_time = self.clock()
        completion_time = self.scoring.get_current_time(current_time)
        self.sound_manager.stop_critical_warning()
        self.critical_warning_active = False
        
        # Store success status and consume the power-out sequence.
        self.level_succeeded = success
        self.game_over_active = False
        self.game_frozen = False
        self.level_complete_menu.set_options(success, self.collision_handler.last_player_kill)
        
        # Calculate score
        self.level_score_breakdown = self.scoring.calculate_level_score(
            completion_time,
            self.ship.fuel,
            self.ship.ammo
        )
        
        # Store completion time for display
        self.completion_time_seconds = completion_time
        
        final_score = self.level_score_breakdown.get('final_score', 0)
        tally = self.finish_tally if self.finish_tally is not None else self.star_tally()
        star_flags = earned(tally)
        self.star_lines = describe(tally)
        self.level_result = None
        if success:
            self.level_result = self.profile_manager.record_level_result(
                self.level, final_score, completion_time, star_count(star_flags),
            )
            if not self.replaying:
                self.profile_manager.update_active_profile_progress(self.level, self.scoring.get_total_score())
        if self.replaying:
            self.scoring.total_score = self.total_score_before_level
            self.level_score_breakdown['total_score'] = self.total_score_before_level
        
        # Initialize animated star rating if level succeeded
        if success:
            # Calculate star spacing (default is star_size * 1.2)
            star_spacing = int(config.LEVEL_COMPLETE_STAR_SIZE * 1.2)
            # Center the stars: first star center should be at screen_center - 2 * star_spacing
            # (since we have 5 stars, the middle star at index 2 should be at screen center)
            star_x = config.SCREEN_WIDTH // 2 - 2 * star_spacing
            star_y = config.SCREEN_HEIGHT // 2 - 50
            self.star_animation = AnimatedStarRating(
                star_flags,
                star_x,
                star_y,
                star_size=config.LEVEL_COMPLETE_STAR_SIZE,
                previous_stars=(self.level_result.previous_best.stars
                                if self.level_result.previous_best else None)
            )
            # Set sound callback
            self.star_animation.set_sound_callback(self.sound_manager.play_tinkling)
        else:
            self.star_animation = None
        
        self.level_complete_time = current_time
        self.level_complete_quit_confirm = False
        self.state = config.STATE_LEVEL_COMPLETE
    
    def draw(self) -> None:
        """Draw game state."""
        self.screen.fill(config.COLOR_BACKGROUND)
        
        if self.state == config.STATE_SPLASH:
            if self.splash_screen:
                self.splash_screen.draw(self.screen)
        elif self.state == config.STATE_MENU:
            self.main_menu.draw()
        elif self.state == config.STATE_PROFILE_SELECTION:
            self.profile_selection_menu.draw()
        elif self.state == config.STATE_LEVEL_SELECT:
            self.level_select_menu.draw()
        elif self.state == config.STATE_SETTINGS:
            self.settings_menu.draw()
        elif self.state == config.STATE_FUN:
            self.fun_menu.draw()
        elif self.state == config.STATE_EDITOR:
            self.level_editor.draw()
        elif self.state == config.STATE_DUEL:
            if self.duel is not None:
                (draw_battle if isinstance(self.duel, TeamBattle) else draw_duel)(self.screen, self.duel)
        elif self.state == config.STATE_PLAYING:
            self.draw_game()
            if self.exit_explosion_active:
                self.draw_exit_explosion()
        elif self.state == config.STATE_QUIT_CONFIRM:
            self.draw_game()  # Draw game in background
            self.quit_confirmation_menu.draw_quit_confirmation(
                self.main_menu.menu_pulse_phase,
                self.quit_confirmation_selection
            )
        elif self.state == config.STATE_LEVEL_COMPLETE:
            self.level_complete_menu.draw(
                self.level,
                self.level_succeeded,
                self.completion_time_seconds,
                self.level_score_breakdown,
                self.star_animation,
                self.level_complete_quit_confirm,
                lambda: self.quit_confirmation_menu.draw_level_complete_quit_confirmation(
                    self.level_complete_menu.menu_pulse_phase,
                    self.quit_confirmation_selection
                ),
                level_result=self.level_result,
                star_lines=self.star_lines
            )
        
        pygame.display.flip()
    
    def _update_powerup_crystals(self, dt: float) -> None:
        """Update powerup crystals and collect any the ship or the hunter touches.
        
        The player has first claim on a crystal both touch. A crystal is drawn to
        whichever of the two is nearer (an observer is no one's target). A collected
        crystal stays in the list until its burst has played.
        
        Args:
            dt: Delta time since last update.
        """
        remaining = []
        player_pos = (self.ship.x, self.ship.y) if self.ship else None
        for crystal in self.powerup_crystals:
            if crystal.active:
                if isinstance(crystal, JevBeacon):
                    crystal.update(dt, player_pos)
                else:
                    # A crystal is drawn to whichever of the player and the hunter is nearer.
                    crystal.update(dt, nearest_friendly_pos(crystal, self.ship, self.hunter))
                if isinstance(crystal, JevBeacon):
                    if crystal.check_circle_collision((self.ship.x, self.ship.y), self.ship.radius):
                        self.sound_manager.play_powerup_activation('beacon')
                        self._summon_hunter(crystal.get_pos())
                else:
                    self.collision_handler.handle_ship_crystal_collision(self.ship, crystal, self.scoring)
                    self.collision_handler.handle_hunter_crystal_collision(self.hunter, crystal)
            elif crystal.is_dying:
                crystal.update_death(dt)
            
            if crystal.active or crystal.is_dying:
                remaining.append(crystal)
        
        self.powerup_crystals = remaining
    
    def draw_game(self) -> None:
        """Draw game play screen."""
        if not self.maze or not self.ship:
            return
        
        # Draw maze
        self.maze.draw(self.screen)
        if self.obstacles:
            for gate in self.obstacles.gates:
                draw_gate(self.screen, gate.pin, gate.angle, gate.half_length, gate.thickness, gate.motor)
        
        # Draw enemies, each kind on its layer, then the death animations of those just killed, then any blasts
        self.roster.draw(self.screen)
        last = self.last_enemy()
        if last is not None:
            draw_last_enemy_ring(self.screen, last.get_pos(), last.radius, self.play_seconds)
        self.collision_handler.explosions.draw(self.screen)
        
        if self.hunter is not None and self.hunter.active:
            self.hunter.draw(self.screen)
        elif self.hunter is not None and self.hunter.is_dying:
            self.hunter.draw_death(self.screen)

        # Draw powerup crystals
        for crystal in self.powerup_crystals:
            if crystal.active:
                crystal.draw(self.screen)
            elif crystal.is_dying:
                crystal.draw_death(self.screen)
        
        # Draw projectiles
        for projectile in self.projectiles:
            projectile.draw(self.screen)
        
        # Hide the parked player in autoplay; an observer otherwise shows faintly.
        hide_player = self.autoplay is not None and config.SETTINGS.autoplay.hidePlayer
        if not self.exit_explosion_active and not hide_player:
            if self.ship.observer:
                self.ship.draw_as_observer(self.screen)
            else:
                self.ship.draw(self.screen)
        
        # Draw UI (hide during exit explosion)
        if not self.exit_explosion_active:
            current_time = self.clock()
            potential = self.scoring.calculate_current_potential_score(
                current_time,
                self.ship.fuel,
                self.ship.ammo
            )
            elapsed = self.scoring.get_current_time(current_time)
            self.ship.draw_ui(
                self.screen,
                self.small_font,
                potential_score=potential['potential_score'],
                max_score=potential['max_score'],
                level=self.level,
                time_seconds=elapsed,
                level_score=potential['final_score'],
                progress_score=self.scoring.get_total_score()
            )
            # Room above the restart hint, less the hunter's gauge while one flies
            room = self.screen.get_height() - 6 - self.restart_hints[False].get_height()
            if self._hunter_energy_reading() is not None:
                room -= HUNTER_GAUGE_HEIGHT
            exit_panel = draw_exit_panel(
                self.screen, [] if self.exit_opened else self.exit_policy.describe(self.exit_state()), self.exit_opened,
                bottom=room)
            self._draw_hunter_energy(below=exit_panel.bottom)
            self.announcer.draw(self.screen)
            if self.star_indicator.flags:
                lit = self.star_indicator.star_count  # The row fills from the left, whichever goals are met
                draw_star_row(self.screen, tuple(i < lit for i in range(5)), self.par_pulse(),
                              (config.UI_ZONE_WIDTH // 2, STAR_ROW_Y), STAR_ROW_SIZE)
        
        # Draw game over sequence (fade to black and text)
        if self.game_over_active:
            self._draw_game_over_sequence()
            self.ship.draw_death(self.screen)

        if not self.exit_explosion_active:
            hint = self.restart_hints[bool(self.input_handler.get_controllers())]
            if self.game_over_active:
                # Keep the retry prompt visible over the fade, below GAME OVER.
                hint_rect = hint.get_rect(center=(self.screen.get_width() // 2,
                                                  self.screen.get_height() // 2 + 90))
            else:
                hint_rect = hint.get_rect(midbottom=(config.UI_ZONE_WIDTH // 2,
                                                     self.screen.get_height() - 6))  # under the hunter gauge
            self.screen.blit(hint, hint_rect)

    def _draw_game_over_sequence(self) -> None:
        """Draw game over sequence: fade to black and GAME OVER text."""
        current_time = self.clock()
        elapsed = max(0.0, current_time - self.game_over_start_time - self.game_over_death_duration)
        
        # Calculate fade progress (0.0 to 1.0)
        fade_progress = min(1.0, elapsed / self.game_over_fade_duration)
        
        # Create fade overlay (black with increasing alpha)
        fade_alpha = int(255 * fade_progress)
        if fade_alpha > 0:
            fade_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            fade_surface.fill((0, 0, 0, fade_alpha))
            self.screen.blit(fade_surface, (0, 0))
        
        # Show "GAME OVER" text after fade delay
        if elapsed >= self.game_over_fade_duration + self.game_over_text_delay:
            # Render "GAME OVER" text
            # Use large, bold font
            game_over_font = get_font(120)
            game_over_text = game_over_font.render("GAME OVER", True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(self.screen.get_width() // 2, 
                                                         self.screen.get_height() // 2))
            self.screen.blit(game_over_text, text_rect)
    
    def draw_exit_explosion(self) -> None:
        """Draw spectacular exit explosion effect that fills the screen with light."""
        if not self.exit_explosion_pos:
            return
        
        # Calculate explosion progress (0.0 to 1.0)
        progress = min(1.0, self.exit_explosion_time / 1.5)
        
        # Create expanding light effect
        # Multiple expanding rings of light
        max_radius = max(config.SCREEN_WIDTH, config.SCREEN_HEIGHT) * 1.5
        
        # Draw multiple expanding layers
        for layer in range(5):
            layer_progress = progress - (layer * 0.15)
            if layer_progress < 0:
                continue
            
            # Calculate radius for this layer
            layer_radius = max_radius * layer_progress * (1.0 + layer * 0.2)
            
            # Calculate intensity (fades out as it expands)
            intensity = 1.0 - (layer_progress * 0.8)
            intensity = max(0.0, intensity)
            
            # Color varies by layer for spectacular effect
            if layer == 0:
                # Core: bright white/yellow
                color = (255, 255, 200)
            elif layer == 1:
                # Inner: bright yellow/orange
                color = (255, 220, 100)
            elif layer == 2:
                # Mid: orange/red
                color = (255, 150, 50)
            elif layer == 3:
                # Outer: red/purple
                color = (200, 100, 255)
            else:
                # Outermost: purple/blue
                color = (150, 150, 255)
            
            # Draw expanding circle with glow
            alpha = int(255 * intensity * 0.6)
            if alpha > 0 and layer_radius > 0:
                # Create surface for this layer
                size = int(layer_radius * 2) + 100
                if size > 0:
                    glow_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                    center = size // 2
                    
                    # Draw multiple concentric circles for smooth glow
                    for i in range(10):
                        circle_radius = layer_radius - (i * layer_radius / 10)
                        if circle_radius > 0:
                            circle_alpha = int(alpha * (1.0 - i / 10))
                            if circle_alpha > 0:
                                glow_color = (*color, circle_alpha)
                                pygame.draw.circle(glow_surf, glow_color, (center, center), int(circle_radius))
                    
                    # Blit to screen
                    pos_x = self.exit_explosion_pos[0] - center
                    pos_y = self.exit_explosion_pos[1] - center
                    self.screen.blit(glow_surf, (int(pos_x), int(pos_y)))
        
        # Draw bright flash overlay that fills screen
        flash_intensity = 1.0 - progress
        if flash_intensity > 0:
            flash_alpha = int(255 * flash_intensity * 0.4)
            if flash_alpha > 0:
                flash_overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
                flash_color = (255, 255, 255, flash_alpha)
                flash_overlay.fill(flash_color)
                self.screen.blit(flash_overlay, (0, 0))
    
    def reset_scoring_to_profile_state(self) -> None:
        """Reset scoring state using the explicit START_LEVEL override first, then profile progress, else default."""
        self.scoring = ScoringSystem()
        self.collision_handler.scoring = self.scoring  # kills and crystals must count on the new one
        profile = self.profile_manager.get_active_profile()
        if profile:
            self.scoring.total_score = profile.total_score

        if self.initial_start_level is not None:
            resolved_level = self.initial_start_level
        elif profile and profile.level is not None:
            resolved_level = profile.level
        else:
            resolved_level = 1

        self.level = resolved_level

    def reset_quit_confirmation_selection(self) -> None:
        """Reset overlay selection back to the default option."""
        self.quit_confirmation_selection = 0

    def run(self) -> None:
        """Main game loop."""
        try:
            while self.running:
                dt_ms = self.frame_limiter.tick(config.FPS)
                # Normalize delta time: 1.0 = 60fps, scales movement for frame independence
                dt = dt_ms / (1000.0 / config.FPS)
            
                self.handle_events()
                self.update(dt)
                self.draw()


        finally:
            if self.duel is not None:
                self.end_duel()
            self._close_hunter()
            self._close_enemy_hunters()
            if self.hunter_worker is not None:
                self.hunter_worker.close()
                self.hunter_worker.join(2.0)
