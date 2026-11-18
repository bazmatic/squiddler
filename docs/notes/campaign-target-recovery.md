# Campaign target recovery

Demo and new Campaign Hunter training runs now use conservative recovery around persistent targeting. The deployed run288 weights and observation layout are unchanged; this is a targeting-rule change, not retraining.

A chase gets four simulated seconds to make meaningful route progress (a quarter of the larger maze cell dimension) or damage its target. Smaller route improvements accumulate. Damage resets the route baseline, including when a hit knocks the target away. After a stall, a visible, reachable alternative must save at least one cell or 25% of current route cost, whichever is larger. Equal-cost alternatives retain roster order.

A voluntary switch deprioritizes the old target for six seconds; each acquired target gets two seconds of commitment. Death, removal and unreachable routes allow immediate replacement. Cooldown cannot exclude the only reachable enemy. The original sustained-disadvantage rule remains, subject to commitment and cooldown. Single-enemy cases retain their target. Level reset clears the state.

`target_recovery` is a recorded lifetime rule. Missing values in historical runs mean false, even though newly started runs now record true. Replays/resumes use the recorded rule. Direct rollout and evaluation APIs default to false for compatibility; use the explicit option to evaluate current Demo behavior:

```sh
venv/bin/python -m tools.campaign_eval assets/neural/campaign_hunter.npz --levels 2 --target-recovery --workers 4
venv/bin/python -m tools.campaign_target_eval --out demo-out/target-recovery --workers 4
```

The second command runs the fixed acceptance set, saves per-decision target identities, switch reasons, route cost, damage, controls, kills and wall crashes, and writes `comparison.json`. Both treatments use the same policy and simulation clock. Stalled time counts only active play with enemies left and four seconds without progress/damage, sampled at the decision cadence.

## Acceptance result (2026-11-17)

74 episodes: level 2 seeds 102, 300–302; levels 3, 8, 11, 14, 17, 23, 28, 29, 31, 32, 34 seeds 400–402. Greedy policy, 180-second horizon. Policy SHA256: `71d318f3f26c63834bf767638364a52b3480685a4048a040bacc06c8e5fa1520` (run288/iter0299).

| Measure | Existing persistence | With recovery |
| --- | ---: | ---: |
| Level 2 total stalled seconds | 25.8 | 4.0 |
| Broader exits, out of 33 | 20 | 23 |
| Broader cleared exits | 19 | 21 |
| Broader mean own kills | 12.09 | 13.39 |
| Broader mean wall crashes | 7.58 | 7.42 |
| Broader mean stalled seconds | 14.38 | 12.87 |
| Broader mean target switches | 8.00 | 9.97 |

All predeclared performance gates passed, so recovery was enabled for Demo and new runs. Individual episodes can still regress; these are finite paired evaluations, not a guarantee of improvement everywhere. No thresholds were adjusted after the comparison.

Artifacts: `demo-out/target-recovery/comparison.json`, per-episode traces in the same folder, and `demo-out/target-recovery-level2.mp4` (45 seconds, corrected-clock Demo, candidate enabled). The full 15-minute recording was not regenerated.

The broader `tests/test_ppo_resume.py` check has four schema-expectation failures: it expects old top-level `rewards`, `episode`, or `shaping` fields. That test file and the hunter PPO/run-record implementation are unchanged by this work. Campaign replay/lifetime-rule and trainer-control regression tests cover the new rule separately.

## Follow-up: visible enemies near the curved outer boundary (2026-11-17)

The replacement Demo video still showed hesitation on level 2. A standalone, fixed-clock Demo reproduction (level 2, seed 102, run288, recovery enabled) killed two enemies, then spent 19.93 seconds without a kill while enemies remained visible. Recovery released a living crab at 6.8 seconds and again at 18.8 seconds as `unavailable`.

The coarse route grid treats the outer ring as blocked. Curved rock leaves open physical space inside those cells, so an enemy can be visible and shootable there even when `cheapest_route` returns no route. Dropping that target redirected pursuit behind the central rock and produced conflicting local-contact/compass inputs.

The fix is restricted to a target inside an outer-boundary cell and within sensor range. If the grid has no route, a clear segment through actual active wall/gate geometry supplies a direct distance for commitment/recovery costing. Real occlusion, interior unreachable cells, out-of-bounds positions and out-of-range targets still have infinite cost. Existing routing remains preferred whenever it supplies a route. No policy weights, observation fields, persistence thresholds or action sampling were changed.

Regression tests first reproduced the incorrect abandonment and now verify commitment, real-wall/gate occlusion, interior-cell rejection and sensor-range limits. The original standalone reproduction now kills all five enemies and exits at 15.43 seconds; its maximum visible-enemy no-kill gap is 5.87 seconds. Sampling actions was also tested diagnostically, but was not adopted.

An initial unrestricted direct-path fallback failed the wall-crash gate and was discarded. During the broader comparisons, unchanged later-level baselines varied across pooled runs. The comparison runner now seeds before Game construction and gives every episode a fresh worker (`maxtasksperchild=1`) to remove cached-game carryover. The isolated comparison below supersedes the earlier pooled comparisons for this follow-up.

| Isolated comparison measure | Existing persistence | Recovery with boundary fix |
| --- | ---: | ---: |
| Level 2 total stalled seconds | 25.8 | 8.0 |
| Broader exits, out of 33 | 21 | 24 |
| Broader cleared exits | 19 | 20 |
| Broader mean own kills | 13.03 | 13.09 |
| Broader mean wall crashes | 6.82 | 7.06 |
| Broader mean stalled seconds | 14.91 | 11.44 |

All five predeclared gates passed in the isolated 74-episode comparison. Note that the baseline is recovery disabled, as specified by the original acceptance protocol; it is not the previous recovery-enabled build. The standalone reproduction above directly compares that previous build against this fix.

Artifacts: `demo-out/target-recovery-boundary-isolated/comparison.json`, its decision traces, and `demo-out/level2-boundary-fix.mp4` (25-second corrected-clock Demo clip, beginning on level 2). The full 15-minute video predates this fix. Diagnostic scripts and rejected-probe outputs are retained under `.astrovoid-local/debug/attack-hesitation/`.

Final repeatability check: a fresh-process repeat of level 34 / seed 400 / recovery off differed from the isolated suite (3 kills over 67.95 seconds versus 5 kills over 88.98 seconds; both scored out with four wall crashes). Fresh workers and RNG seeding therefore do **not** eliminate all later-level simulation variability. The 74-episode gate passed for the recorded run, but its aggregate improvements should not be treated as a deterministic guarantee or proof of a universally better policy. Remaining simulation nondeterminism is unresolved. The standalone level 2 reproduction and the geometric unit regression isolate the boundary-abandonment bug independently of that broader comparison. Final focused verification: 122 tests passed; deployed policy checksum unchanged.
