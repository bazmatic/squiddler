# Astrovoid

Use the repo `venv` (`venv/bin/python`); system `python3` lacks pygame. Docs live in `docs/` (start at `README.md`).

## Demo videos

- **Demo mode / the Campaign Hunter flying the ship** (any "record a demo video of N minutes" request): `venv/bin/python -m tools.record_demo OUT.mp4 SECONDS`. Offscreen, with sound, faster than real time. See `docs/demo-video.md`.
- **A trained creature's training replays** (Inchworm, Jelly, Flocker, Wallcrawler teams): `venv/bin/python -m tools.demo_video RUN`. The `demo-video` skill covers it. It cannot film the Campaign Hunter.
- Write videos to `demo-out/` (untracked); never commit long ones.
