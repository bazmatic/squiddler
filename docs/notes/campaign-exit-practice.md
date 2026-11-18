# Campaign Hunter: exit with one crawler

Training level **1032** reproduces the decision at the end of level 32: the ship begins in the final passage, close to the already-open exit, while one Wallcrawler remains alive and out of sight elsewhere in the tunnel. The hunter may leave immediately; killing the crawler is not required. Its aim is to study how the pilot responds to enemy and exit cues when they compete.

This is a **Campaign Hunter training-only scenario**. It runs the real level 32 with its maze, permanent walls, enemy/ship physics, level-32 difficulty, unlimited ammo and open-exit policy. It replaces the layout's start with cell (20, 28) and its enemy roster with one Wallcrawler at (28, 17); the exit remains at (28, 28). It is built at episode setup and does not alter the campaign level file. Score starts as a one-enemy episode (around 100 potential points), not with the accumulated score, pickups, heading, speed or recurrent state of the original replay. The episode's reported level remains 1032 for evaluation and replay.

It is selectable in the trainer's level picker and in CLI level lists. A fixed-policy baseline:

```sh
venv/bin/python -m tools.campaign_eval assets/neural/campaign_hunter.npz \
  --levels 1032 --seeds 5 --first-seed 400 --workers 4 --target-recovery \
  --json .astrovoid-local/debug/exit-practice/baseline.json
```

For sampled actions, add `--sampled` and use a different JSON output path. Compare exit rate, time, wall crashes and the recorded decisions; mix 1032 with normal campaign levels during training to check for regressions and avoid specializing only to this position. Replay a selected episode with `tools.watch_episode` as usual. This scenario is not designed for other training tasks, ordinary gameplay, or the pursuit lesson.

Baseline on the deployed run291/iter0249 model: 0/5 greedy and 0/5 sampled episodes exited (seeds 400–404); all scored out, without killing the crawler. Greedy and sampled are separate comparisons, not estimates of overall campaign performance. The five attempts share a fixed authored layout; seeds change stochastic episode behavior but do not vary start position. Files are in `.astrovoid-local/debug/exit-practice/`.

Do not treat a higher exit rate here alone as proof that general navigation improved. Evaluate the original level 32 and additional levels as well, particularly clear-all missions. A new training session or model deployment was not started as part of creating this scenario.

## Run292: 40-iteration mixed training experiment

Run292 continued run291/iter0299 for 40 iterations at learning rate 0.0001, with 24 episodes per iteration cycling levels 2, 11, 16, 32 and 1032. It used the existing shaping and target-recovery rules. Final checkpoint: run292/iter0039. No model was deployed.

Paired greedy evaluations on seeds 400–404, recovery on, compare run291/iter0299 to run292/iter0039:

| Level | Before exits | After exits | Before Demo exits | After Demo exits |
| --- | ---: | ---: | ---: | ---: |
| 2 | 5/5 | 5/5 | 5/5 | 5/5 |
| 11 | 3/5 | 5/5 | 3/5 | 4/5 |
| 16 | 0/5 | 0/5 | 0/5 | 0/5 |
| 32 | 4/5 | 5/5 | 4/5 | 5/5 |
| 1032 | 0/5 | 0/5 | 0/5 | 0/5 |

The authored geometry and starting state are fixed; all five greedy 1032 attempts for each checkpoint were identical (not five independent starts). After training they all scored out at 44.67 seconds. A traced seed-400 attempt chose left in all 447 decisions, never fired or saw the crawler, and moved from about 280 to about 696 pixels from the exit. Enemy and exit cues remained present. On 10 matched sampled-action runs (seeds 400–409), both checkpoints exited only for seed 408 (1/10 each); on the final checkpoint that exit left the crawler alive. Five reconstructed iteration-39 training episodes on 1032 all scored out; these reconstructions are not saved-action recordings.

The regular-level sample is small and can vary: do not conclude general improvement solely from it. The clean conclusion is that this experiment has not solved the fixed near-exit failure. The current reward recorder directs incremental progress to enemies while any remain; exit-progress shaping begins only with no enemies alive. Leaving the enemy therefore has a terminal exit reward but no incremental exit-progress reward in this case. This is a plausible contributor, not a demonstrated sole cause.

Artifacts: `.astrovoid-local/debug/exit-practice/run292/compare-{before,after}.json`, `before-sampled.json`, `after-sampled.json`, `final-trace.json` and `final_training_samples.log`. Greedy data and the sampled data use different action-selection modes; compare within the same mode.
