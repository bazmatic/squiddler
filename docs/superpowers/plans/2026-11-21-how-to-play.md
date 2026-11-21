# How To Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A HOW TO PLAY entry on the main menu opens a four-page screen that teaches the basics, and the main menu's buttons get thinner to make room for it.

**Architecture:** Page copy lives in one pure function (`build_pages`) that reads every figure from `config`, so it can't drift from the rules and is testable without a screen. A `HowToPlayMenu` draws one page at a time (modelled on `FunMenu`), and a `HowToPlayStateHandler` turns pages and goes back (modelled on `FunStateHandler`).

**Tech Stack:** Python, pygame, pytest (`venv/bin/python -m pytest`).

**Spec:** [docs/superpowers/specs/2026-11-21-how-to-play-design.md](../specs/2026-11-21-how-to-play-design.md)

## Global Constraints

- Use `venv/bin/python`; system `python3` lacks pygame.
- Figures in the copy come from `config`/settings, never typed into the text: `config.ENEMY_DESTRUCTION_BONUS`, `config.POWERUP_CRYSTAL_BONUS`, `config.POWERUP_DURATION_SECONDS`, `config.POWERUP_AMMO_REFILL`, `config.POWERUP_AMMO_COST_MULTIPLIER`, `config.SETTINGS.powerups.shieldDurationSeconds`, `PowerupCrystal.LIFETIME_FRAMES / config.FPS`, `config.SETTINGS.game.criticalWarningThreshold`.
- Control labels come from `CONTROL_GROUPS` in [rendering/settings_menu.py](../../../rendering/settings_menu.py), not a second copy.
- Entry order: START GAME, LEVELS, **HOW TO PLAY**, FUN, SELECT PROFILE, SETTINGS, (EDITOR), QUIT.
- Tests cover behaviour, not layout internals (no pixel tests); a few valuable ones per task.
- The working tree has unrelated uncommitted edits in `game.py`, `game_handlers/state_handlers.py`, `rendering/main_menu.py` and `tests/test_menu_visuals.py`. Commit only your own hunks (`git add -p`); never commit the others' work.
- Rendering is not proven by the dummy SDL driver. The last task checks the real display (1352x878).

## Review Focus

- A small (1280x720) or large (1920x1080) screen: no page line runs off the screen. Pinned in Task 2.
- A controller connected or not: the controls page shows that device's labels. Pinned in Task 1.
- A changed setting (say the kill bonus) changes the text. Pinned in Task 1.
- Left on page one and Right on the last page stay put. Pinned in Task 3.
- The editor build (eight entries): all buttons still clear the title and the profile badge. Pinned in Task 4.

---

### Task 1: Page copy

**Files:**
- Create: `rendering/how_to_play_pages.py`
- Test: `tests/test_how_to_play.py`

**Interfaces:**
- Produces: `Page = NamedTuple('Page', [('title', str), ('lines', list[str])])` and `build_pages(controller: bool) -> list[Page]`, always four pages in this order: Goal and controls, Power, Crystals and weapons, Stars and enemies. Lines are plain sentences (no wrapping; the screen wraps).

- [ ] **Step 1: Write the failing tests** in `tests/test_how_to_play.py`:
  - `test_there_are_four_pages_in_order`: titles are `["GOAL AND CONTROLS", "POWER", "CRYSTALS AND WEAPONS", "STARS AND ENEMIES"]`.
  - `test_power_page_quotes_the_settings(monkeypatch)`: with `config.ENEMY_DESTRUCTION_BONUS` and `config.POWERUP_CRYSTAL_BONUS` monkeypatched to 37 and 41, the Power page text contains `"+37"` and `"+41"`; also contains the crystal-lifetime and critical-threshold figures read from config.
  - `test_crystal_page_quotes_the_settings`: contains `str(config.POWERUP_DURATION_SECONDS)`, `str(config.POWERUP_AMMO_REFILL)` and the shield seconds.
  - `test_controls_follow_the_device`: `build_pages(False)[0]` text contains every key label of the `KEYBOARD` group in `CONTROL_GROUPS` and none of the `CONTROLLER` labels that aren't also keyboard labels; `build_pages(True)[0]` the reverse.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_how_to_play.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `build_pages`** in `rendering/how_to_play_pages.py`. Copy for each page (rules text from [docs/gameplay.md](../../gameplay.md)):
  1. Goal: reach the green exit portal before power runs out; most levels keep it locked until the mission announced at the start is done; the ship keeps its momentum, so plan your braking; then one line per `CONTROL_GROUPS` entry for the chosen device, formatted `"LABEL: action"`.
  2. Power: the POWER gauge is the level's score counting down; time, thrust, shots, wall bumps and enemy bullet hits drain it; destroying an enemy gives `+{ENEMY_DESTRUCTION_BONUS}`, collecting a crystal `+{POWERUP_CRYSTAL_BONUS}`; at zero the level is lost; in the last `{threshold}` points the screen edge pulses red and a tick sounds.
  3. Crystals: enemies sometimes drop a crystal, fly close to pull it in, it vanishes after N seconds; a crystal adds a firepower level for `{POWERUP_DURATION_SECONDS}` seconds (faster fire, then a spread), `{POWERUP_AMMO_REFILL}` ammo, and ammo costs `{POWERUP_AMMO_COST_MULTIPLIER:g}` of normal while it lasts; weapon crystals add and select a weapon, and the cycle-weapon label (from `CONTROL_GROUPS`, the "Next weapon" row) cycles; a shield crystal makes you invulnerable for `{shieldDurationSeconds:g}` seconds.
  4. Stars: five stars, one per goal met at the exit (no hits, half the enemies, every enemy, every crystal, finishing within par time); a few lines naming Eggs, Urchins, Crabs, Jellies, Flockers and Wallcrawlers in one phrase each, and "bosses every sixth level".
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** `rendering/how_to_play_pages.py tests/test_how_to_play.py` with message "How to play: the four pages of copy, read from the settings".

---

### Task 2: The screen

**Files:**
- Create: `rendering/how_to_play.py`
- Test: `tests/test_how_to_play.py`

**Interfaces:**
- Consumes: `build_pages`, `Page` (Task 1); `AnimatedBackground`, `build_menu_backdrop`, `menu_scale`, `render_hint_row` from `rendering.menu_components`; `get_font` from `rendering.fonts`.
- Produces: `class HowToPlayMenu(screen: pygame.Surface)` with `page_index: int`, `page_count: int`, `turn(delta: int) -> None` (clamps to `0..page_count-1`), `refresh(controller: bool) -> None` (back to page 0, rebuilds pages for that device), `update(dt: float) -> None`, `draw() -> None`.

- [ ] **Step 1: Write the failing tests** (use the `screen` fixture and `is_opaque` helper pattern from [tests/test_menu_visuals.py](../../../tests/test_menu_visuals.py)):
  - `test_draws_every_page_and_leaves_the_screen_opaque` for both `controller` values.
  - `test_page_lines_stay_on_screen[size]` for sizes `(1352, 878)`, `(1280, 720)`, `(1920, 1080)` (monkeypatch `config.SCREEN_WIDTH/HEIGHT`): expose the laid-out line rects via a `line_rects() -> list[pygame.Rect]` method for the current page and assert each lies inside the screen and that none overlap.
  - `test_turn_clamps_at_both_ends`: from page 0, `turn(-1)` stays 0; after `page_count` forward turns, the index is `page_count-1`.
  - `test_refresh_returns_to_the_first_page`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement `HowToPlayMenu`** modelled on `FunMenu` ([rendering/fun_menu.py](../../../rendering/fun_menu.py)): title "HOW TO PLAY" at 18% height, the page title below it, body lines word-wrapped to about 70% of the screen width (wrap with `font.size`) and left-aligned in one column centred on the screen, a `"{n} / {count}"` indicator, and a bottom `render_hint_row` (keyboard: `("LEFT / RIGHT", "Page")`, `("ESC", "Back")`; controller: `("D-PAD", "Page")`, `("B", "Back")`). Font sizes scale with `menu_scale()`. Shrink the body font a step at a time until the longest page fits the space between the page title and the hint row.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** `rendering/how_to_play.py tests/test_how_to_play.py` with message "How to play: the paged screen".

---

### Task 3: State, input and the menu entry

**Files:**
- Modify: `config.py:634` (add `"howToPlay": "how_to_play"` to `STATES_DEFAULTS`, and `STATE_HOW_TO_PLAY = STATES_DEFAULTS["howToPlay"]` beside `STATE_FUN`)
- Modify: `rendering/main_menu.py:39` (insert `"HOW TO PLAY"` after `"LEVELS"`)
- Modify: `game_handlers/state_handlers.py` (the `MenuStateHandler` keyboard and controller branches near lines 65-100; a new `HowToPlayStateHandler` modelled on `FunStateHandler`; register it in `StateHandlerRegistry`)
- Modify: `game.py` (import and create `self.how_to_play_menu = HowToPlayMenu(screen)` beside `self.fun_menu`; `update` and `draw` branches beside the `STATE_FUN` ones; add `config.STATE_HOW_TO_PLAY` to the hat/axis event allowlist near line 1112)
- Test: `tests/test_how_to_play.py`; update `tests/test_menu_visuals.py` (`test_main_menu_no_longer_carries_the_key_mappings` lists the exact options)

**Interfaces:**
- Consumes: `HowToPlayMenu` (Task 2).
- Produces: `config.STATE_HOW_TO_PLAY`; `HowToPlayStateHandler` with `handle_keyboard` / `handle_controller`.

- [ ] **Step 1: Write the failing tests** using `make_game` and `key` from the settings-menu test class in `tests/test_menu_visuals.py` (copy the helper into the new test file rather than importing a test class):
  - `test_entry_sits_after_levels_and_opens_the_screen`: options contain `"HOW TO PLAY"` directly after `"LEVELS"`; selecting it and pressing Return gives `game.state == config.STATE_HOW_TO_PLAY` on page 0.
  - `test_right_and_left_turn_pages_and_escape_goes_back`: Right moves to page 1, Left back to 0, Left again stays 0, Escape returns `config.STATE_MENU`; the same via a controller cancel press.
  - `test_the_last_page_stops_turning`: pressing Right `page_count + 2` times leaves `page_index == page_count - 1`.
  - Update the options list in `test_main_menu_no_longer_carries_the_key_mappings` to the new order.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_how_to_play.py tests/test_menu_visuals.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement the wiring.** The menu handler opens the screen with `game.how_to_play_menu.refresh(bool(game.input_handler.get_controllers()))` then sets the state, in both the keyboard and controller branches. `HowToPlayStateHandler`: keyboard Left/Right turn by -1/+1, Escape or Q returns to `STATE_MENU`; controller cancel returns, hat or stick left/right turns with the same 0.15 s debounce as `FunStateHandler`.
- [ ] **Step 4: Run** the same tests, then the existing menu and state tests: `venv/bin/python -m pytest tests -k "menu or state or settings" -q`. Expected: PASS.
- [ ] **Step 5: Commit** only your hunks of those files, with message "How to play: a main-menu entry, state and input".

---

### Task 4: Thinner main-menu buttons

**Files:**
- Modify: `rendering/main_menu.py` (`BUTTON_HEIGHT`, the default spacing in `_button_column`, the button font size in `_initialize`)
- Test: `tests/test_menu_visuals.py` (the main-menu layout tests near line 363)

**Interfaces:**
- Produces: `BUTTON_HEIGHT = 50` (was 64), default spacing `_scaled(66)` (was 80), button font `get_font(self._scaled(34))` (was 40).

- [ ] **Step 1: Update the failing tests.** In `test_the_main_menu_buttons_fit_between_the_title_and_the_profile_badge`, expect `len(buttons) == (8 if editor else 7)`; in `test_seven_buttons_sit_a_little_higher_and_close_up_only_as_far_as_they_must`, rename it for seven entries and assert the default gap is `<= menu._scaled(66)`. Add `test_button_text_fits_inside_its_button`: for each button in the sized cases, the rendered label height is no more than the button's height.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Apply the three values** above. Keep `_button_column`'s packing logic as it is.
- [ ] **Step 4: Run** `venv/bin/python -m pytest tests/test_menu_visuals.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** only your hunks with message "Thinner main-menu buttons to make room for HOW TO PLAY".

---

### Task 5: Check on the real display and update the docs

**Files:**
- Modify: `docs/gameplay.md` (one line under "How to play": the same text is now in the game's HOW TO PLAY screen)

- [ ] **Step 1: Run the whole suite:** `venv/bin/python -m pytest tests -q`. Expected: PASS apart from anything already failing before this work.
- [ ] **Step 2: Look at it for real.** Launch the game (`./run.sh`, 1352x878 fullscreen), and screenshot the main menu (with and without `LEVEL_EDITOR=1`) and each of the four pages, with and without a controller if one is to hand. Check the text is readable, nothing overlaps or runs off-screen, and the thinner buttons still look right.
- [ ] **Step 3: Edit `docs/gameplay.md`** and commit it with message "Docs: the how-to-play screen".
