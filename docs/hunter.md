# Hunter ally and the training manager

## The ally

On most levels from 10, killing enough enemies can drop an **ally beacon**: the hunter's arrowhead inside a turning amber ring. Fly into it and an allied hunter ship streams in beside you. Only one hunter flies at a time, and a beacon only drops while none is flying. The beacon itself is indestructible.

The hunter has three hit points, never hurts you, and starts each level with fresh memory. Enemies can target and destroy it. It hunts enemies it can see, follows you through the maze when none are in sight, and collects weapon crystals it passes. A crystal gives it a generic power boost (faster, bigger shots) rather than the weapon. Its kills and crystals count towards your stars, but it earns you no kill points. You get first claim on a crystal you both touch.

Level files control where beacons may appear; see [levels/README.md](../levels/README.md).

## Pilots

`game.hunterPilot` in `config/settings.json` chooses who flies it:

- **`"neural"`** (default): a trained network that flies with the same controls you have. It decides every `game.neuralDecisionFrames` frames and turns at `game.neuralTurnRate` times your rate. Its weights are `assets/neural/hunter.npz`. The same pilot flies the enemy Vibers and Vibelets.
- **`"rules"`**: hand-written flying rules in `hunter/rules_pilot.py`. No model or training is needed. The same pilot flies the enemy Terminator.

The neural pilot began as a student of the rules pilot (`experiments.teach`, `tools.train_neural`, `experiments.dagger`) and was then trained further with reinforcement learning through the training manager.

**Trying a pilot quickly:** `ASTROVOID_HUNTER_HOTKEY=1 ./run.sh`, then press **H** during a level to collect a waiting beacon or call the hunter in beside you. Add `ASTROVOID_OBSERVER=1` to become an invisible, invulnerable spectator.

**Pilot fight:** the **FUN** menu has *Rules vs Vibes*, where a rules hunter (green) and a neural hunter (magenta) fight in a rocky maze while you watch. Each takes `game.duelHealth` hits, and a fight lasting longer than `game.duelTimeoutSeconds` is a draw. It doesn't touch your profile.

## Training manager

Every creature brain (hunter, Jelly team, Flocker team, Wallcrawler team, Inchworm) is trained with one tool. It needs `pip install -r requirements-train.txt`.

```bash
./run-trainer.sh            # opens the manager in your browser (port 8765, or the next free one)
```

The manager shows what is training, with Pause, Resume and Stop. A **Next run** box starts from fresh weights or carries on from any run and checkpoint, with any setting changed first. It also charts any run, and **Replay an episode** lets you watch a finished iteration's episodes in the game window at normal, half or quarter speed (`python -m tools.watch_episode` does the same from the command line).

Runs live in `.astrovoid-local/neural/rl/runN/`, and one trains at a time. Pause lets the current iteration finish so the run can be resumed later; Stop ends it. Closing the manager doesn't stop training. The same controls work from the shell: `./run-trainer.sh status`, `runs`, `start [--from run51/iter0200] [--set key=value ...]`, `stop`, `pause`, `resume [RUN]` (`--help` for more).

Agents should use the `astrovoid-trainer` MCP tools (registered in `.mcp.json`) or `./run-trainer.sh`, never `tools.ppo_neural` directly.

To open the manager from another device on your Tailscale network, bind to this machine's Tailscale address and allow the hostnames browsers will use:

```bash
./run-trainer.sh serve --host "$(tailscale ip -4)" --port 8766 --allowed-host <hostname>
```

How the training system is built, and how to add a species to it: [Modular training system](modular-training.md).

## Unattended runs

`AUTOPLAY=1 ./run.sh` (or Demo in the FUN menu) skips the menus and lets the Campaign Hunter, a neural pilot, fly the player ship level after level with no one at the controls. It plays like a person: pickups and the Ally beacon work as they do for you, and a level ends when the ship leaves through the exit or scores out. As safety nets a level also ends on its time limit, or after `noKillSeconds` without a kill while enemies remain. Nothing is written to your profile. Until a trained `assets/neural/campaign_hunter.npz` is shipped, Demo flies the hunter pilot carried over to the campaign senses. The levels and limits are in the `autoplay` section of `config/settings.json`.
