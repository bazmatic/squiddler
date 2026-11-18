"""Record the game's Demo mode (the Campaign Hunter flying the ship) offscreen to an mp4, with audio.

    venv/bin/python -m tools.record_demo demo-out/demo.mp4 600      # OUT.mp4 SECONDS

1352x878 at 60 fps, a fixed timestep, no real-time pacing: about 14 ms of wall clock per frame, so ten minutes of
video renders in about eight. Audio: pygame's Channels are replaced by a software mixer stepped once per video frame,
so every sound lands on the exact frame it fired. Needs ffmpeg on the PATH. Demo never touches the profile."""
import os, sys, subprocess, time
import numpy as np
sys.path.insert(0, os.getcwd())
os.environ['AUTOPLAY'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
pygame.init()
W, H = 1352, 878
screen = pygame.display.set_mode((W, H), pygame.HIDDEN)
import config
config.SCREEN_WIDTH, config.SCREEN_HEIGHT = W, H
config.SPLASH_ENABLED = False
from game import Game

pygame.mixer.init(frequency=config.SOUND_SAMPLE_RATE, size=-16, channels=2, buffer=512)
RATE = pygame.mixer.get_init()[0]
SPF = RATE // 60            # samples per video frame (44100 -> 735)
_cache = {}
def pcm(sound):
    a = _cache.get(sound)
    if a is None:
        a = pygame.sndarray.array(sound).astype(np.float32) / 32768.0
        if a.ndim == 1: a = np.stack([a, a], 1)
        _cache[sound] = a
    return a

class Chan:
    all = {}
    def __new__(cls, i):
        if i not in cls.all:
            o = super().__new__(cls); o.i = i; o.sound = None; o.data = None; o.pos = 0; o.loops = 0; o.vol = 1.0
            cls.all[i] = o
        return cls.all[i]
    def play(self, sound, loops=0, maxtime=0, fade_ms=0):
        self.sound, self.data, self.pos, self.loops = sound, pcm(sound) * sound.get_volume(), 0, loops
    def stop(self): self.sound = self.data = None
    def fadeout(self, ms): self.stop()
    def get_busy(self): return self.data is not None
    def get_sound(self): return self.sound
    def set_volume(self, v, r=None): self.vol = v
    def get_volume(self): return self.vol
    def mix(self, n):
        out = np.zeros((n, 2), np.float32)
        if self.data is None: return out
        got = 0
        while got < n and self.data is not None:
            take = min(n - got, len(self.data) - self.pos)
            out[got:got + take] = self.data[self.pos:self.pos + take]; got += take; self.pos += take
            if self.pos >= len(self.data):
                if self.loops == 0: self.stop()
                else:
                    self.pos = 0
                    if self.loops > 0: self.loops -= 1
        return out * self.vol

pygame.mixer.Channel = Chan
pygame.mixer.stop = lambda: [c.stop() for c in Chan.all.values()]
pygame.mixer.set_num_channels = lambda n: None
pygame.mixer.get_num_channels = lambda: 16

def mix_frame():
    out = sum(c.mix(SPF) for c in Chan.all.values()) if Chan.all else np.zeros((SPF, 2), np.float32)
    return (np.clip(out, -1, 1) * 32767).astype('<i2').tobytes()

seconds = float(sys.argv[2]); out = sys.argv[1]
n = int(seconds * 60)
# Set the clock before Game starts its first level. Render speed must not
# change score drain, gun cooldowns, or Demo transition timers.
frame = 0
g = Game(screen, clock=lambda: 1000.0 + frame / 60)
aud = out + '.pcm'
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                       '-r', '60', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                       '-pix_fmt', 'yuv420p', out + '.v.mp4'], stdin=subprocess.PIPE)
af = open(aud, 'wb')
t0 = time.time(); levels = []; peak = 0
for i in range(n):
    frame = i
    pygame.event.pump(); g.handle_events(); g.update(1.0); g.draw()
    ff.stdin.write(pygame.image.tobytes(screen, "RGB"))
    b = mix_frame(); af.write(b)
    if not g.running: print('game stopped at frame', i); break
    if not levels or levels[-1][1] != g.level: levels.append((i / 60, g.level))
    if i % 3600 == 0: print(f'{i/60:.0f}s level {g.level} wall {time.time()-t0:.0f}s', flush=True)
ff.stdin.close(); ff.wait(); af.close()
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out + '.v.mp4', '-f', 's16le', '-ar', str(RATE), '-ac', '2',
                '-i', aud, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
os.remove(aud); os.remove(out + '.v.mp4')
print('levels (t, level):', levels)
