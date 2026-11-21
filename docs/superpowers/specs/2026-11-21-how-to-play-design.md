# How to play: design

## Intent
A new player can't learn the game from the game. [docs/gameplay.md](../../gameplay.md) explains it, but only to developers. Add a HOW TO PLAY screen, reached from the main menu, that teaches the basics, including how power drains and how to win it back, and what crystals and powerups do.

## Goals
- A main-menu entry, HOW TO PLAY, opening a paged screen. Left/Right turn pages, Back returns to the menu.
- Four pages:
  1. **Goal and controls.** Reach the green exit portal; most levels lock it until the announced mission is done. Rotate, thrust, shield, fire, restart. The ship keeps its momentum, so plan your braking.
  2. **Power.** The POWER gauge is the level's score counting down. Time, thrust, shots, wall bumps and enemy bullet hits drain it. Destroying an enemy gives +20 and collecting a crystal gives +20. Out of power loses the level. In the last few points the screen edge pulses red and a tick sounds.
  3. **Crystals and weapons.** What a crystal does: a firepower level for 10 seconds (faster fire, then a spread), 25 ammo back, half-price ammo while it lasts. Weapon crystals add a weapon (cycle with Down/S or controller A). The shield lasts 10 seconds. Crystals vanish after six seconds.
  4. **Stars and enemies.** The five stars, and a short note on enemies.
- Make the main menu's buttons thinner, so the extra entry doesn't crowd the screen (today `BUTTON_HEIGHT = 64` in [main_menu.py](../../../rendering/main_menu.py)). Spacing and the profile badge must still fit on small and large screens.
- Key and button labels follow the input device in use (keyboard or controller).

## Non-goals
- No interactive tutorial or practice level.
- No pre-level-1 popup. This is a screen you can come back to.
- No change to gameplay, scoring or settings.

## Anti-goals
- Copy that drifts from the rules. Numbers shown (+20, 10 s, 25 ammo, six seconds) must come from the settings, not be typed into the text.

## Constraints
- Follow the existing screen pattern: [fun_menu.py](../../../rendering/fun_menu.py), a state in `config`, a handler in [state_handlers.py](../../../game_handlers/state_handlers.py), and an entry in `MainMenu.menu_options` ([main_menu.py](../../../rendering/main_menu.py)).
- Use the menu components (backdrop, background, fonts, `menu_scale`).
- Few, behaviour-level tests: the entry opens the screen, pages turn and clamp, Back returns, the numbers shown match the settings. No pixel or layout-internals tests.

## Approaches
1. **Paged screen from the main menu.** Chosen. It reads well, works on a controller, and can be revisited.
2. One long scrolling page: simpler, but a wall of text and awkward on a controller.
3. A card before level 1 only: new players see it, but it can't be reopened.

## Left to the builder
Exact copy and layout, page transitions, where HOW TO PLAY sits in the menu order (suggested: after LEVELS), and how the setting-driven numbers are read.
