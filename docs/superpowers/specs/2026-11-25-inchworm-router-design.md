# Inchworm specialists with a rule-based router

## Intent
One network learning every lesson forgets some of them: training Leap cost Crawl, Gap and About-face. Crawl must be
production-quality, Leap is a bonus. So keep the lessons in separate checkpoints and let a plain rule pick which one
drives the worm. Nothing new is trained: two checkpoints we already have are the specialists.

## Specialists (existing checkpoints)
- **Walker**: run336 `iter0999`. Crawl 1.00, Gap 0.92, About-face 0.90; Leap about 0.3.
- **Leaper**: run342 `iter0150`. Leap about 0.55; Crawl, Gap and About-face less steady.
The walker is the default. The leaper takes over only for a leap.

## Goals
- Router Crawl, Gap and About-face match the walker's on a large held-out set (150 trials each, not 48).
- Router Leap matches or beats the leaper's (about 0.55 on 240 held-out trials).
- A worm that is stuck (bored) tries a leap, and a held-out stuck case (a target it cannot walk to) shows it doing so.
- The router is a few readable lines of rule, driven only by what the worm already senses, so the game can run it unchanged.

## How far the leaper reaches
The worm sees 300 px; the 80–140 px band was only what Leap was trained on. Probed on markers out to 300 px, the leaper
(iteration 150) catches about 0.51 at 100–150 px, 0.32 at 150–200, 0.16 at 200–250 and 0.12 beyond; the walker 0.18, 0.19, 0.12, 0.10.
So the leaper does leap further but misses more, and a boredom leap would often miss, which is acceptable because it was stuck anyway.
Leaps that reach reliably out to about 250 px need a new leaper trained on farther markers: a follow-up after the router, not part of it.

## Non-goals
- A learned router. Training new specialists. Bundling the specialists into one shipped file.
- Putting it in the game, until it passes the held-out check (build step 2, below). Training a longer-reach leaper (follow-up).

## Approaches considered
- **A** router straight into the game driver: fast to see, but no numbers to say it is better.
- **B** router in the training harness first, measured on held-out trials, then the game: **chosen** (you picked it).
- **C** one bundled file: a packaging step, left for after the rule is settled.

## Design
- `RoutedPolicy` (new, `learning/router.py`) offers the same interface as a policy (`initial_state`, `choose`, normalisation),
  so `collect`, the evaluation scripts and the game driver need no change to use it.
- Both specialists see every observation and keep their own memory in step with the world; the rule only decides whose action is used.
- **Rule**: use the leaper when the worm is planted on a wall by one foot or more, the target is visible (it sees up to
  300 px) with no wall between them, and either (a) it is in leaping range, 80 px out to the leaper's useful reach, or
  (b) the worm is **bored**: it has been gripping the wall for about 5 s without getting meaningfully closer to the target
  (the thresholds are the builder's, tuned on training seeds), in which case it tries a leap from any visible distance. Once handed over it stays with the leaper until the worm has been
  in the air and landed, or 3 s have passed (a leap is one committed move, and switching mid-whip would drop it). After
  handing back it waits a short while before it can hand over again, so it cannot flip back and forth. Otherwise the walker.
- Evaluation reuses the existing held-out Crawl, Gap, About-face and Leap trials, with the router in place of a policy.

## Left to the builder (my calls, marked so they can be changed)
- Exact thresholds (distance band, timeout, wait before handing over again), tuned on training seeds and then
  confirmed once on separate held-out seeds, so the numbers are not fitted to the test.
- Which sensed features carry "planted", "wall between" and "target distance" (they exist in the observation).

## Success
Held-out numbers: Crawl, Gap and About-face within noise of the walker; Leap within noise of the leaper or better. If the
rule cannot beat the plain walker on Leap without costing Crawl, we say so and stop there rather than keep tuning.

## Measured result (2026-11-25)
Defaults after tuning on training seeds: leaping range 80-200 px, a stall of 20 decisions (2 s, under 20 px of progress)
before an in-range leap, 50 decisions (5 s) of the same before a bored leap from any distance, 30 decisions hold,
20 decisions cooldown. Two things the first version got wrong, found on training seeds: an in-range leap with no stall
handed over in 82-100% of every lesson (the router became the leaper), and boredom measured over the whole grip run
never fired because early progress counted for ever and a flailing worm's grips flicker.

Held-out seeds, looked at once (150 trials per lesson, 240 for Leap), success rate:

| | Crawl | Gap | About-face | Leap |
|---|---|---|---|---|
| walker (run336 iter 999) | 0.98 | 0.87 | 0.99 | 0.37 |
| leaper (run342 iter 150) | 0.94 | 0.85 | 0.97 | 0.54 |
| router | 0.97 | 0.91 | 0.97 | 0.42 |

The router keeps Crawl, Gap and About-face at the walker's level, and hands over in 95% of Leap trials. Its Leap success
is 0.42: above the walker (0.37) by about one standard error, below the leaper (0.54). The goal of Leap near the leaper's
was not met. The leaper alone is also as good as the walker on the other lessons (Crawl 0.94, Gap 0.85, About-face 0.97)
on these seeds. No further tuning was done on the held-out seeds, and the game step was not built.
