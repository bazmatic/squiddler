# Demo videos of trained creatures

```sh
venv/bin/python -m tools.demo_video RUN --list          # which episode each clip uses
venv/bin/python -m tools.demo_video RUN --out DIR       # DIR/<task>-demo.mp4
```

`RUN` is a run directory or a name under `.astrovoid-local/neural/rl`. It works
for any task that replays through hooks (`learning/tasks.py`, `replay_mode='hooks'`):
Inchworm, Jelly team, Flocker team, Wallcrawler team. Hunter uses the builtin
replay path and is not supported; to film the Campaign Hunter, record the game's Demo mode with
[`tools/record_demo.py`](#recording-demo-mode) instead. The project skill `demo-video`
(`.claude/skills/demo-video/SKILL.md`) wraps these steps for Claude.

## How it works

Training records every episode's actions (`episodes/iterNNNN.json`). The tool
replays them through the task's replay hooks, the dashboard Watch button's path,
so each clip is exactly what training flew, and the replay is checked against the
recorded results. Frames are drawn offscreen (SDL dummy driver), optionally
cropped, and joined with ffmpeg. It groups episodes by lesson, takes the newest
good episodes of a readable length for each, and puts a title card before each.

## Replay-module hooks (all optional)

Add to `learning/adapters/<task>_replay.py`:

| Hook | Purpose | Default |
| --- | --- | --- |
| `episode(job)` | dict; its `lesson` string names the clip group | job's `lesson`, `practice_stage`, else `Stage N`, else `Demo` |
| `LESSONS` | tuple of labels giving the movie order | alphabetical |
| `CLIP_SECONDS` | `{label: (min, max)}` readable clip lengths | `--seconds` (3-25 s) |
| `showcase(row)` | score for an episode's recorded events, or `None` to skip it | `success`, else catches/kills/hits above zero |
| `focus(env)` | `(x, y, width)` to crop to, e.g. the subject | the whole arena |
| `draw(env, screen)` / `status` | already used by Watch; the video uses `draw` | the game's own drawing |

Inchworm sets all of them (`learning/adapters/inchworm_replay.py`): it crops to
the worm and its target and orders the clips Crawl, Rear, Curl, Turn, Reach,
Feed, Recover.

## Caveats

- Replay only works on code that matches the recording. Changing a scenario
  version, physics, controller or adapter makes the tool stop with "replay rules
  or schemas differ from the recorded run". Make the video before changing those,
  or from a run started after.
- A lesson with no successful episode in the length window is skipped; a run
  that has not reached a lesson has none. Runs without `episodes/` cannot be drawn.
- Clips come from different iterations: typical successes, not one final policy
  or a failure rate.
- Offscreen rendering can differ from the real window; check a frame, and use
  the dashboard Watch button for a fullscreen check.
- No sound. Titles come from `card()` in `tools/demo_video.py`.

## Recording Demo mode (the Campaign Hunter)

```
venv/bin/python -m tools.record_demo demo-out/demo.mp4 600     # OUT.mp4 SECONDS
```

Runs the game's Demo (`AUTOPLAY=1`: the Campaign Hunter flies the ship, level after level) offscreen and writes an mp4
with sound: 1352x878, 60 fps, stereo AAC. It needs `ffmpeg` on the PATH and uses the pilot installed in
`assets/neural/campaign_hunter.npz`, so it films whatever Demo would play now, not a particular training run.

- **Speed.** A fixed timestep with no real-time pacing: about 14 ms of wall clock per frame, so ten minutes of video
  renders in about eight. Drawing the game is most of that.
- **Sound.** pygame's mixer channels are replaced by a software mixer stepped once per video frame, so every sound
  lands on the exact frame it fired. No loopback device or real-time capture is needed. Overlapping sounds can peak at
  0 dB; lower the mix there if it sounds harsh.
- **Output.** The mp4 is about 10 MB per minute. Write it to `demo-out/` (untracked) and don't commit long ones.
- It prints the time each level began, so you can see how far the Hunter got.
