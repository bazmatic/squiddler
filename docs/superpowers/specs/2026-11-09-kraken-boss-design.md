# Kraken Boss — design

(Working name "Kraken"; the name is the builder's call. The existing squid bosses are `split_boss` "Squiddler" and
`mother_boss` "Mother" and stay as they are.)

## Intent

A new boss: a huge squid whose tentacles are the fight. Each tentacle is its own creature with its own trained brain.
It reaches for the player, grabs, and hurts the player for as long as it holds on. The player gets free by shooting
the holding tentacle. A tentacle that takes enough damage tears off the squid and becomes a free-swimming, creepy,
organic sea snake that wriggles after the player on its own. Only once every tentacle is gone can the body itself be
hurt, so the fight is about stripping the tentacles and then punishing the exposed body.

## Goals

- A new boss kind, separate from `split_boss` and `mother_boss`, usable from level files, the editor palette and boss
  levels like any other catalogue kind.
- A huge **body** that moves slowly around the arena.
- **At most 4 tentacles in play at once.** The cap counts attached tentacles and loose sea snakes together. The squid
  starts with 4.
- Each tentacle is a physically simulated, jointed body (in the spirit of the Inchworm) driven by a **trained
  (PPO) brain**, one brain per tentacle, acting independently. Scripted fallback when no weights exist, as for the
  Inchworm, so tests and fresh checkouts run.
- **Grab:** a tentacle can grab the player ship and holds it, damaging it for as long as it holds. The player escapes
  by shooting that tentacle. It also lets go when the player dies or drops out. Follow the Inchworm's prey-grab
  rules in `docs/notes/inchworm-body.md` where they apply.
- **Tear-off:** a tentacle that has taken enough damage detaches from the body and becomes a **sea snake**: a free
  creature that swims and hunts the player. It can be killed. It is the same physical body with a different brain
  (trained to chase and wriggle, not to reach and grab), so the change is seamless. Whether a snake can also grab is
  the builder's call; the default is that it bites/damages on contact but does not hold.
- **Body invulnerable while any tentacle is in play** (attached or loose). Shots on the body do nothing in that state,
  with clear feedback so the player can tell.
- **Exposed window:** when the last tentacle is dead, the body becomes vulnerable for **5–10 seconds** (configurable;
  default 7). During the window it takes damage from shots. No tentacles regrow while it is open.
- **Regrowth:** if the window ends with the body still alive, it grows a fresh set of 4 tentacles and the cycle
  repeats. Regrowth never exceeds the cap.
- **One cycle is the intended kill:** the body's hit points are set so a player shooting steadily through one
  window kills it. If the window runs out, the cycle repeats. No escalation between cycles.
- Tentacles and sea snakes look **organic and creepy**: tapering, writhing, with a visible grip when holding.
- Works with two-ship local multiplayer: the tentacles target the nearest living ship (existing targeting), and a hold
  is released if its victim dies or leaves.

## Non-goals

- No changes to `split_boss`, `mother_boss`, the Inchworm, or other creatures' behaviour.
- No escalation, phases or multiple-cycle hit-point pools.
- No networking or determinism work beyond what the game already does.
- No per-level boss tuning beyond the existing level mechanisms.

## Anti-goals

- Don't blow the frame budget. An Inchworm-class body costs about 0.6 ms per frame, so 4 tentacles is about 2.4 ms.
  That is the budget; a design that needs more is a failure even if it works.
- Don't let the arena fill with snakes: the cap must hold in every code path, including the moment of tear-off (the
  snake takes the tentacle's slot) and regrowth.
- Don't make the body damageable "by accident" (splash, contact, other creatures' shots) while a tentacle lives,
  unless the builder finds that unavoidable and says so.
- No compatibility shims for old recordings or runs (project convention).

## Constraints

- Python/pygame; Pymunk for physics, as the Inchworm does. Creature = Body + Traits (see `ARCHITECTURE.md`); the
  Body draws and never decides; one trait per behaviour, transferable between creatures.
- Declare traits in `config/enemies.json`; new kinds are picked up by SpawnManager, editor palette and par time.
- Pymunk objects must be excluded from deepcopy for death portraits (the `Trait.LIVE` mechanism).
- The solver can blow up: follow the Inchworm's guard (reset after repeated failures, then die).
- Tests cover behaviour, not internals; few and valuable (`tests/README.md`). Scripted controllers and seeds.
- Training uses the existing PPO tooling and the `astrovoid-trainer` runs; one run trains at a time.

## Decided (by the user)

- Separate creature from Squiddler and Mother.
- Trained brains for the tentacles.
- Max 4 tentacles in play.
- Body is not damaged by shots unless all tentacles are gone.
- Exposed window of roughly 5–10 s; one cycle kills; body moves slowly around.
- Tentacles are all creatures; the body regrows tentacles.

## Left to the builder (my guesses, flagged)

- Name, size, hit points of body, tentacle hit points and the "enough damage" tear-off threshold, grab damage rate.
- Tentacle link count and joint layout within the 2.4 ms total budget.
- Whether a tentacle on cooldown re-grabs immediately after being shot free (guess: short cooldown).
- Reward design and observation set for training both brains (reach/grab and chase), and training order. Likely
  reach-and-grab first, then chase.
- Whether the sea snake can grab (guess: no).
- Which boss level(s) use it.
- Reusing `splitsOnDeath`/`laysEggs` ideas for tear-off and regrowth versus new traits.

## Phasing (decided by the user)

**Phase 1 — the Sea Snake as a standalone creature.** Build the free-swimming, organic, creepy sea snake as an ordinary
enemy kind: a jointed Pymunk body (sized for the tentacle budget), a chase-and-wriggle brain trained with PPO (scripted
fallback when no weights), hit points, drawing, level/editor support. It must be killable by the player's shots and
damage the player on contact. It is designed so its body can later be a tentacle: the root end must be able to be
anchored to something and to be released. Ship it when trained weights exist and it plays well in a test level.

**Phase 2 — the Kraken.** The body, 4 anchored tentacles that reuse the snake's physical body with a reach-and-grab
brain, the grab, tear-off (swap to the snake brain, release the root), the cap, the exposed window and regrowth.
Grab behaviour toward the player (see below) is decided at the start of this phase, not now.

## Calls made by Claude (not agreed by the user — check these)

- A tentacle's root is jointed to the moving body. Tear-off deletes that joint and swaps the brain; the brain observes
  the root pose relative to the body.
- Shots hit any link of a tentacle and are absorbed by it, so tentacles can screen the body. Only the player's
  weapons (either ship) damage tentacles and snakes.
- Loose snakes always hunt and never despawn; the builder adds a rule (e.g. lifespan) if stuck snakes can stall the
  fight.
- Scripted behaviour first, then training; the Kraken counts as shippable only once trained weights exist.

## Open for Phase 2

- How far a grab moves or constrains the player's ship: damage only (ship flies free), dragged but still
  controllable (Claude's lean), or dragged with partly locked controls.
