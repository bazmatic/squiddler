# Multiplayer: two players on one screen

## Intent

Astrovoid is single player today. We want two people to play it together, sitting at the same Mac:
each on a game controller, or one on a controller and one on the keyboard. There is no networking.

Three multiplayer modes are added alongside single player:

- **Level competition.** The players take turns on the same level; the higher score wins it.
- **Campaign (co-op).** Both ships fly at once through the campaign as a team.
- **Campaign race.** Both ships fly at once; the first through the portal wins the level.

## Choosing a mode

A multiplayer mode setting chooses between **Single player**, **Level competition**, **Campaign**
and **Campaign race**. START GAME and LEVELS keep working as they do now; the chosen mode decides how
the level is played.

- The setting is a main-menu item that cycles through the four modes (e.g. `MODE: CAMPAIGN RACE`),
  remembered between sessions in `config/settings.json`. (Placement and persistence are my call.)
- In any multiplayer mode, START GAME always begins at **level 1**, whatever the profile's progress.
- LEVELS lets you pick any level, as now, and plays it in the chosen mode.
- Multiplayer never changes profile progress, level bests or stars. (My call: profiles belong to one
  person, and two people sharing a run shouldn't move either profile.)

## The players

There are always exactly two players, called **Player 1** and **Player 2**, with distinct ship
colours used everywhere they're named (HUD, results, announcements).

Input devices (my call):

- With two controllers connected, controller 1 is Player 1 and controller 2 is Player 2. The
  keyboard also drives Player 1.
- With one controller, the keyboard is Player 1 and the controller is Player 2.
- With no controllers, the simultaneous modes can't start: the menu says a controller is needed.
  Level competition still works, since only one ship flies at a time.
- In Level competition, every device drives whichever ship is currently flying, so players can
  pass a single controller or share the keyboard.

Today `InputHandler` reads only the first controller, and `Game.update` merges keyboard and
controller into one command list. Both need to become per-player.

## Rules shared by the simultaneous modes (Campaign and Campaign race)

- Both ships start at the level's start position, side by side, at the same moment.
- Each ship has **its own power, ammo and fuel**. In the code, power is the running potential score
  (100, minus time and penalties, plus kill and crystal bonuses). Each ship's power works exactly as
  in single player, but is made only from **that ship's own** time drain, penalties, kills and
  crystals.
- Each ship's time drain starts when **that ship** first moves. Enemies wake (and the hunter
  starts) when **either** ship first moves. (My call.)
- Each ship has its own fire cooldown and its own "has moved yet" state.
- **Running out of power** drops that player out for the rest of the level: their ship dies as in
  single player but the level carries on for the other. A dropped player comes back at the start of
  the next level.
- If **both** players run out of power, the level fails and is replayed.
- Enemies, hazards and pickups treat both ships alike. Enemies go for the nearest living ship.
  Either ship can pick up a powerup; it applies to that ship.
- Ships **collide**: they bounce off each other exactly like a ship bouncing off rock, and each
  ship takes the normal wall-collision penalty to its own power.
- Ships can't shoot each other. (My call.)
- The level's mission and exit policy are shared: kills by either player count towards opening
  the exit.
- The portal's pull acts on each ship. A ship that touches the open portal goes through it and
  leaves play.
- The Jev hunter ally follows the **nearest living ship**, and either player can summon it by
  collecting the beacon. Its perception keeps a single friend, so the trained hunter is unchanged.
  (My call.)

## Campaign (co-op)

- The two players share **one team score** for the level: the two ships' remaining power added
  together at the end of the level. A ship that dropped out adds zero. A ship's power freezes when
  it goes through the portal.
- The restart key restarts the level for both players.
- A level ends once **every player still flying** has gone through the portal. A player who goes
  through first waits out of play.
- After a level is completed, play moves on to the next campaign level.

## Campaign race

- The **first player through the portal wins the level** and gets one point. The level ends there.
- Score plays no part beyond being each ship's power: kills and crystals still raise it, so they
  still help a ship survive, but no score is reported. (My call.)
- The restart key is switched off, so the losing player can't cancel a near-win. (My call.)
- If one player runs out of power, the other can still win by reaching the portal.
- If both run out, nobody gets the point and the level is replayed.
- There is no overall winner. A running tally of points (Player 1 vs Player 2) stays on screen,
  and play goes on through the campaign until the players stop.

## Level competition

- One ship flies at a time. The players take turns on the same level, each turn played exactly as a
  single-player level is today.
- Each turn is **a single attempt**: the restart key and the RETRY LEVEL option after failing are
  switched off.
- Both turns on a level face the **same enemy layout**: the enemy placement seed is fixed for the
  level's pair of turns. (My call.)
- Player 1 goes first on the first level, and the first turn alternates on each level after that.
- Between turns a short screen says whose turn it is and waits for a button press.
- When both have played, a **results screen** shows both scores and names the level's winner.
  A player who ran out of power scores zero. A tie gives nobody the point.
- **From LEVELS:** the pair play the chosen level, see the result, and return to the level list.
- **From START GAME:** the pair go through the campaign, one level at a time. Each level's winner
  gets a point, and a running tally is shown, as in the race. If both players ran out of power,
  nobody gets the point and the level is replayed. Otherwise play moves on to the next level.
  There's no overall winner; it continues until the players stop.

## HUD

The left HUD strip (`Ship.draw_ui`, 200 px) is single player today. In the simultaneous modes it
splits: Player 1's gauges on the top half, Player 2's on the bottom, each labelled in that player's
colour. (Layout is my call.)

- Each half shows that player's own power, ammo, and other per-ship gauges.
- Campaign shows the one shared team score.
- Campaign race shows the points tally in place of score.
- Level competition uses the normal single-player HUD for the ship flying, plus the player's name
  and, in a campaign, the tally.

A dropped-out player's half is dimmed. A player waiting beyond the portal shows as through.

## The Replay enemy

The Replay enemy mirrors the player's recent commands. With two ships, each player gets their own
command recording, and Replay enemies alternate between copying Player 1 and Player 2 in spawn
order. (My call.)

## Goals

- Two people can play any of the three modes on one Mac, with controllers or keyboard plus
  controller, from the campaign or from a chosen level.
- Single player plays exactly as it does today.

## Non-goals

- Networked or online play.
- More than two players.
- Split screen. Each level already fits on one fixed screen.
- Saving multiplayer progress or high scores.
- An overall match winner or a fixed match length.

## Anti-goals

- Single player getting worse or different in any way: feel, scoring, recordings, training
  harnesses (`experiments/harness.py` uses `game.ship`) and the hunter's training must keep working.
- One player being stuck watching for long in a simultaneous mode, beyond the rest of a level
  after dropping out.

## Constraints

- Code reads like the surrounding code. Behaviours are generic and transferable rather than
  special-cased per mode.
- A few valuable behaviour tests covering the modes' rules (who wins, when a level ends, drop-out,
  replay on double failure). No fragile tests of internals.
- No compatibility shims for old recordings or saved runs.
- Rendering must be checked on the real fullscreen display, not only the dummy SDL driver.

## Left to the builder

- How `Game` moves from one `self.ship` to a list of players. The hunter's model and
  `world.additional_targets` already show how enemies can target more than one friendly ship.
  Updating every trait and entity that reads the single ship (`creatures/encounter.py`,
  `creatures/traits/*`, `entities/*` listed in the survey) is part of the job.
- How `ScoringSystem` splits so each ship has its own power and clock.
- The exact wording and look of turn screens, results screens and the tally.
- How the side-by-side start positions are found in mazes with narrow start cells.
