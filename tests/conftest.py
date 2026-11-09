"""Keep tests silent (no audio device output) and deterministic (inchworms use the scripted stride, not the sampled brain)."""
import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("INCHWORM_POLICY", "")      # tests that want the bundled brain unset it

# Under pytest-xdist each worker is its own process; one math thread each keeps torch and numpy from
# oversubscribing the cores the other workers are using. Must be set before torch is first imported.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _threads in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_threads, "1")
