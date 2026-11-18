# Campaign exit DAgger experiment

A targeted experiment teaches the Campaign Hunter to use the open exit in training level 1032, with a remaining unseen Wallcrawler. The teacher is experimental; it is not installed in the game. No reward changes, campaign level changes or policy deployment were made. Source policy: `run292/iter0039.npz` (SHA256 `71d8b884ca0975ee8c93538cb248f92a3dd2fb4014f835cdb96a5689df7863f5`).

## Teacher and collection

`experiments.campaign_exit_teacher.ExitTeacher` picks a visible waypoint on an open route to the exit and steers toward a velocity that closes on it, damping momentum. It declines when the exit is closed or an open route is unavailable. It never fires. It is a line-of-sight waypoint teacher, not a full collision-clearance planner: across 30 varied starts (seeds 100–129) it exited all 30 with ten total wall contacts.

`experiments.campaign_exit_dagger` runs the student in the real 1032 environment, records teacher turn/thrust labels for each observed state, and executes the teacher's action with probability 0.2. The student's recurrent state and previous-action observations reflect the executed action. Starting columns 8–24 in the final passage, headings 0–360 degrees and velocities −1.5 to +1.5 per axis vary deterministically with the seed. The teacher labels those states regardless of whether an intervention occurs; it is not used in evaluation. The regular level-1032 fixed start is evaluated separately.

Only turn and thrust are supervised. KL rehearsal anchors all three action distributions to the source model on six ordinary sampled flights from levels 2, 11 and 32; this does not guarantee combat retention. Recurrent trajectories are fitted with Adam and gradient clipping. The second trial saves each trajectory's observations, labels and executed controls plus the rehearsal data under `.astrovoid-local/neural/exit-dagger-trial-long/`.

The first small trial (three rounds, 12 episodes per round, three epochs, learning rate 0.00003, KL weight 3) failed the fixed and all ten varied greedy tests. The longer trial used three rounds, 16 episodes per round, 12 epochs, learning rate 0.0001, KL weight 2 and the same 0.2 intervention probability. Seeds 300–347 were used for data collection, and each round refitted from the original checkpoint with aggregated data. During collection, the first round exited 10/16 and the next two exited 16/16; these results include teacher interventions and are not student-only evaluation.

Reproduce the longer experiment (choose a new output directory):

```sh
venv/bin/python -m experiments.campaign_exit_dagger \
  --source .astrovoid-local/neural/rl/run292/iter0039.npz \
  --out .astrovoid-local/neural/exit-dagger-repro \
  --rounds 3 --episodes 16 --beta 0.2 --epochs 12 --lr 0.0001 --kl-weight 2
```

## Student-only evaluation

No teacher actions were injected. After the shorter trial failed, the longer trial's candidate 1 was checked on varied seeds 600–609: greedy exits rose from 1/10 to 8/10, sampled exits from 0/10 to 10/10; its fixed greedy attempt also exited. To reduce selection bias, the three longer-fit candidates were then checked on another set, varied seeds 700–709:

| Model | Fixed-start greedy exit | Varied greedy exits | Varied sampled exits | Mean wall contacts, varied greedy |
| --- | ---: | ---: | ---: | ---: |
| Source | 0/1 | 0/10 | 0/10 | 1.2 |
| Candidate 1 | 1/1 | 10/10 | 10/10 | 2.5 |
| Candidate 2 | 1/1 | 10/10 | 10/10 | 2.4 |
| Candidate 3 | 1/1 | 10/10 | 10/10 | 0.7 |

Those starts vary but all use the same final passage, one enemy and open exit. They are not a representative campaign sample. Fixed-start greedy is one deterministic attempt, not independent seeded trials.

On regular levels, use five seeds 500–504 per level, fresh process and random seeds per episode, greedy actions, 180-second horizon and target recovery. The following compares the source with candidate 3. `Demo exits` exclude exits occurring only after Demo's no-kill skip would have fired:

| Level | Source exits | Candidate exits | Source Demo exits | Candidate Demo exits | Source wall contacts (mean) | Candidate wall contacts (mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5/5 | 5/5 | 5/5 | 5/5 | 2.0 | 1.8 |
| 11 | 4/5 | 5/5 | 4/5 | 4/5 | 6.4 | 6.6 |
| 17 | 0/5 | 3/5 | 0/5 | 2/5 | 9.2 | 18.4 |
| 29 | 3/5 | 5/5 | 3/5 | 5/5 | 9.8 | 17.8 |
| 32 | 5/5 | 5/5 | 5/5 | 5/5 | 14.2 | 17.8 |

An earlier three-seed regular-level comparison gave mixed results, including 0/3 exits for candidate 3 on level 17 versus 1/3 for the source. Level 16 remains unsolved and is not addressed by this exit teacher. Later-level simulations have shown some run-to-run variability even with fixed seeds. These small evaluations show the lesson works on this practice passage but do not establish a safe general-campaign improvement; candidate 3 remains **experimental and not deployed**. It is `.astrovoid-local/neural/exit-dagger-trial-long/candidate3.npz` (SHA256 `2c7918302bb3b83acd0b11c96ef7b1ef81ba5ee354d51317c3c20f0fde535b4e`).

Results and per-episode metrics are in `evaluation-fresh.json`, `evaluation-candidate1.log` and `campaign-isolated.json` in that directory. A short clip of the candidate on the fixed practice state is `demo-out/exit-dagger-candidate3-preview.mp4`; it uses the candidate explicitly and does not install it. Diagnostic evaluation scripts are in `.astrovoid-local/debug/exit-practice/`.

Validation: `tests/test_campaign_exit_teacher.py`, `tests/test_campaign_training.py`, `tests/test_campaign_replay.py`, `tests/test_campaign_eval.py` and `tests/test_training_levels.py` passed (73 tests). A first standalone launch failed during display initialization before collecting data; the CLI now sets dummy drivers and was rerun successfully.
