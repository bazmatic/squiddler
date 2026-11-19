# Squiddler glossary

The names we use for things in the game. Use them in code, docs and conversation, so that one word always means one thing.

## Language

### Ships

**Player**:
The human-flown ship.
_Avoid_: user, ship (on its own, when the hunter is also present)

**Hunter**:
The allied ship that flies itself, summoned by collecting an ally beacon. Its kills do not score for the player.
_Avoid_: ally ship, bot, AI ship

**Pilot**:
Whatever decides how a hunter ship turns, thrusts and fires: neural (a small trained network, the default) or rules (hand-written flying rules). It also flies the Terminator, the Vibers and any creature given the `pilot` trait.
_Avoid_: agent, controller; "brain" is the wider term for whatever decides a Creature's controls

**Burst**:
A queued group of rounds, scheduled by `burstFire`. A round is released when the gun is ready and aligned.

**Volley**:
The projectiles emitted together in one round: one projectile, or several from a weapon such as the Twin Blaster or Scattergun.

**Hunter flight**:
One hunter in play with its pilot, from the moment it is summoned until it dies, is replaced, or the level ends.
_Avoid_: hunter session, hunter instance

**Surroundings**:
What the hunter may perceive at the moment it looks: the maze, the enemies, the threats, the pickups, and the player when the player can be seen.
_Avoid_: snapshot, world state, entities

### Things in the maze

**Strong wall**:
A wall block that no shot can destroy, drawn as riveted metal plate. Only hand-made layouts have them.
_Avoid_: steel wall, indestructible block, rim (the outer ring is a different thing)

**Ally beacon**:
A pickup that summons the hunter. Only the player collects it; the hunter never perceives it as a pickup.
_Avoid_: hunter powerup, summon crystal

**Threat**:
A projectile that can harm the hunter.
_Avoid_: bullet (when allegiance matters)

**Pickup**:
A powerup crystal the hunter may collect.
_Avoid_: item, loot

**Flora**:
The living weed in one cave: every plant and spore in it. Off unless the `flora.enabled` setting is on, or a battle's
match file says `"flora": true`; without it the rock shows painted weed instead. Never in training.
_Avoid_: vegetation, greenery

**Plant**:
One living tuft of weed on breakable rock. It starts as a seed, grows, fruits pods that burst into spores, then
withers and dies. It goes when its rock does. Not a creature: nothing targets it and shots pass through it.
_Avoid_: weed (on its own, which also means the rock's painted tufts), seed (except for a plant's first stage)

**Spore**:
What a plant's pod bursts into. It drifts through open water and takes root as a seed where it touches breakable rock
with no plant too near.

### Enemies

**Roster**:
Every enemy on the level, of every kind, alive or dying, until the level is cleared.
_Avoid_: entity manager, enemy lists

**Enemy Kind**:
One kind of enemy as `config/enemies.json` describes it: its code name, the names the player reads, how it behaves in
the maze (anchored, replays the player, its par allowance, the room it needs) and its own tuning. Every kind can be
placed in a level.
_Avoid_: enemy type (in new code; level files still say "type")

The names the player reads. The friendly hunter keeps its own names ("Rules hunter", "Neural hunter"); these are for enemies only.

**Urchin**:
An enemy that sits still. Shots shove it, and it takes several hits.
_Code name_: static

**Crab**:
An enemy that moves back and forth.
_Code name_: patrol

**Jelly**:
A fragile enemy that hunts in packs, driven by a trained brain.
_Code name_: aggressive (scripted) and neural_jelly (the campaign's)

**Squiddler**:
An enemy that flies the player's own recorded moves.
_Code name_: replay

**Flocker**:
A bird that flies, aims and calls to others in a flock, driven by a trained brain.
_Code name_: flocker

**Flighthouse**:
A lighthouse that sweeps a searchlight and launches Flockers at whatever it sees.
_Code name_: flighthouse

**Vibehouse**:
A pink Flighthouse that launches Vibelets.
_Code name_: vibehouse

**Wallcrawler**:
A gun on twelve legs that walks on walls and leaps between them, driven by a trained brain.
_Code name_: wallcrawler

**Inchworm**:
A six-segment physical worm that crawls, rears and bites, and eats any creature except other inchworms. Driven by a trained brain.
_Code name_: inchworm

**Sea Snake**:
A free-swimming eight-link serpent that wriggles through open water after its adversaries and stings on contact. It swims by bending, with no thrust. Driven by a trained brain.
_Code name_: sea_snake

**Egg**:
An enemy that hatches into squidlets.
_Code name_: egg

**Squidlet**:
An enemy hatched from an egg, or placed in a level.
_Code name_: baby

**Anemone**:
A creature rooted to the rock that pulls the player's ship in.
_Code name_: anemone

**Squiddler Boss**:
A boss that splits when hit.
_Code name_: split_boss

**Mother**:
A boss that lays eggs.
_Code name_: mother_boss

**Terminator**:
An enemy ship flown by the rules pilot.
_Code name_: rules_hunter

**Viber**:
An enemy ship flown by the neural pilot.
_Code name_: neural_hunter

**Boss Viber**:
A Viber at double the size that takes double the hits.
_Code name_: boss_viber

**Vibelet**:
A Viber at half the size that one hit destroys; its shots are almost harmless.
_Code name_: vibelet

**Belemnite**:
A spiky squid in a pointed shell that chases the player as a Jelly does. It has no guns; when it dies it explodes, hurting everything near it, other enemies too, so one can set off the next.
_Code name_: belemnite

**Blast**:
An explosion's reach and what it does to everything inside it: the player, the hunter, enemies and wall blocks. A blast set off by another counts its kills for whoever lit the first.
_Avoid_: explosion (that is what the player sees of it), splash damage

### Creatures

How every enemy and the allied hunter is built: from its catalogue entry's `body` and `traits`.

**Creature**:
One enemy or ally assembled from its catalogue entry: a Body and a list of Traits. It holds what every creature has
(position, velocity, hit points, allegiance) and the state its traits keep.
_Avoid_: enemy class, entity (for this sense)

**Body**:
How a creature looks: it animates and draws the creature, and plays its death. It never decides what the creature does.
_Avoid_: sprite, skin, renderer

**Trait**:
One piece of a creature's behaviour, named and configured in its catalogue entry; its traits run in the entry's order.
_Avoid_: component, behaviour, strategy

**Brain**:
A Trait that decides a creature's controls each frame: where it heads and when it fires. Either scripted (chase, patrol, still) or a trained neural network.
_Avoid_: AI, strategy

**Propulsion**:
A Trait that turns a creature's controls into motion (drift steers toward a heading; coasts only keeps its momentum).
_Avoid_: movement strategy, engine

**Encounter**:
What happens to the creatures on a level in one frame: it runs their traits in roster order, offers each friendly shot
to them, and gives traits the ship, the hunter, the maze and the rest of what they act on.
_Avoid_: enemy updater, world, scene

### Modes

**Duel**:
Two hunters with different pilots fight each other in a maze while the player watches.
_Avoid_: versus, arena match
