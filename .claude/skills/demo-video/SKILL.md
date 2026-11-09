---
name: demo-video
description: Make a demo movie of a trained creature (Inchworm, Jelly team, Flocker team, Wallcrawler team) from a training run's recorded episodes. Use when asked for a video, movie, clip or showcase of what a run learned, or one clip per lesson.
---

# Demo video of a trained creature

**The Campaign Hunter (newest run is often `campaign_hunter`) and "record the demo" or "a N minute demo":** use `venv/bin/python -m tools.record_demo OUT.mp4 SECONDS`, which films the game's Demo mode with sound (see `docs/demo-video.md`). The steps below are for the other creatures' training replays.

Run from the repo root with the repo `venv` (system `python3` lacks pygame; `ffmpeg` must be on the PATH).

1. **Pick the run.** Name under `.astrovoid-local/neural/rl` (e.g. `run208`) or a path. Use `mcp__astrovoid-trainer__training_status` or `list_runs` for the newest run of the creature. Training may continue while you render; use `--iteration N` to pin the newest episode drawn.
2. **Preview the choice:** `venv/bin/python -m tools.demo_video RUN --list` prints one line per clip (lesson, iteration, episode, length). Fix problems here before rendering.
3. **Render:** `venv/bin/python -m tools.demo_video RUN --out DIR` (about a minute; longer for big teams). Writes `DIR/<task>-demo.mp4`, 1280x832, 60 fps, a title card before each lesson.
4. **Look at it.** Pull frames with `ffmpeg -ss 20 -i DIR/*.mp4 -frames:v 1 f.png` and read them. Check the subject is visible and large enough. Rendering is offscreen, so for anything alpha- or compositing-sensitive also check the dashboard's Watch button in the real window.
5. **Report** the path, the length, what each clip is (lesson, iteration) and the caveats below. Don't publish or upload the video unless asked.

## Options
`--clips N` clips per lesson, `--seconds MIN MAX` length filter (default 3-25 s), `--iteration N`.

## Failures and what they mean
- `replay rules or schemas differ from the recorded run` — the code changed since the run (scenario version, physics, controller, adapter). Replays are only valid on the code that recorded them. Make the video from a newer run, or check out the old commit.
- `no good episodes found` / a lesson is missing — the run hasn't passed that lesson, or no episode of it succeeded within the length window. Widen `--seconds`, lower `--iteration`, or add a `showcase` hook.
- `recorded no episodes` — old run, nothing to replay. The Hunter task replays through a different (builtin) path and is not supported: use `tools.record_demo`.

## Caveats to state
Clips come from different iterations, so they show typical successes, not one final policy or a failure rate. A clip is a recorded episode replayed exactly; it is not re-flown.

## Tuning or adding a creature
The tool reads optional hooks from the task's replay module (`learning/adapters/<task>_replay.py`); see `docs/demo-video.md`. Without hooks it groups clips by the job's `lesson`, `practice_stage` or `stage`, treats `success` (or catches, kills, hits above zero) as a good episode, and shows the whole arena.
