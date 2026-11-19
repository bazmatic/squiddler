"""How snake-like a sea snake swims: bends of a healthy size, travelling from head to tail, at a steady rhythm, that
carry it forward.

Two views of the same thing. frame_terms() is per physics frame and feeds reward shaping (learning.sea_snake_rewards
weighs it). score() looks back over a whole episode and grades it in [0, 1] for the dashboard and the deployment gate.
Bends are measured as a fraction of MAX_BEND and speeds in snake radii, so neither depends on the snake's size.

Joint i sits between links i and i+1; link 0 is the tail. A wave travels head to tail when each joint's bend leads the
bend of the joint behind it (nearer the tail).
"""
import math

import numpy as np

from physics.sea_snake import MAX_BEND

FPS = 60.
CURVE_BAND = (.15, .75)      # mean |bend| (fraction of MAX_BEND) that is a healthy wave: below rigid, above folded
CURVE_SOFT = .1              # the curvature score fades to 0 this far outside the band
RIGID = .08                  # mean |bend| below this is swimming rigid-straight (penalised)
COILED = .45                 # head-to-tail distance over body length below this is coiled or folded (penalised)
UNCOILED = .65               # ... and above this is clearly stretched out
RHYTHM_BAND = (.5, 2.5)      # undulations per second that score full rhythm
RHYTHM_SOFT = .4             # the rhythm score fades to 0 this many per second outside the band
STEERING = .25               # bends slower than this many per second are steering, not undulation
TWITCH_FREE = .04            # joint acceleration (fraction of MAX_BEND per frame per frame) a smooth wave stays under
JITTER_FREE = .1             # share of bend power above the rhythm band that still counts as smooth
JITTER_SCALE = .3            # ... and this much more zeroes the smoothness score
WAVE_REF = .02               # head-to-tail wave drive (frame_terms) that pays in full
WAVE_PHASE_REF = .5          # sine of the joint-to-joint phase lag (episode score) that scores full
MIN_SWIM = .5                # radii per second forward the snake must make for wriggling to pay
SWIM_REF = 1.                # radii per second forward that scores full swim
SPREAD_SHARE = .4            # a joint takes part in the wave when its swing is at least this share of the largest
SWING_RATE = 2*math.pi*1.5/FPS   # radians per frame of a 1.5-per-second undulation, to judge swing from bend and rate
COMPONENTS = ('curvature', 'spread', 'wave', 'rhythm', 'swim')


def bends(physics):
    links = physics.links
    return tuple(b.angle-a.angle for a, b in zip(links, links[1:]))


def bend_rates(physics):
    links = physics.links
    return tuple(b.angular_velocity-a.angular_velocity for a, b in zip(links, links[1:]))


def extension(physics):
    """Head-to-tail distance over the body's length: about 1 straight, small when coiled or folded."""
    return math.dist(physics.head, physics.tail)/(len(physics.links)*physics.length)


def forward_speed(physics):
    """Radii per frame along the heading."""
    vx, vy = physics.velocity
    h = physics.heading
    return (vx*math.cos(h)+vy*math.sin(h))/physics.radius


def _active(amplitudes):
    """How fully each joint takes part in the wave, 0-1: fully at SPREAD_SHARE of the most-bending joint's swing or
    more, in proportion below it. A tapered wave (a smaller swing toward the head) still counts nearly in full; a joint
    held rigid counts nothing."""
    amplitudes = np.asarray(amplitudes, float)
    peak = float(amplitudes.max()) if len(amplitudes) else 0.
    if peak <= 1e-9:
        return np.zeros(len(amplitudes))
    return np.clip(amplitudes/(SPREAD_SHARE*peak), 0., 1.)


def frame_spread(angles, rates):
    """Share of joints taking part in this frame's undulation. A joint's swing is judged from its bend and its rate
    together (sqrt(bend^2 + (rate/w)^2) is steady through a sine at w), so a joint passing through straight still
    counts; a rigid body flapping only its tail scores low."""
    a, r = np.asarray(angles)/MAX_BEND, np.asarray(rates)/MAX_BEND
    return float(np.mean(_active(np.hypot(a, r/SWING_RATE))))


def wave_drive(angles, rates):
    """Positive while bends travel head to tail. For a wave sin(w t + j lag) each joint pair contributes
    amplitude^2 w sin(lag), weighted by how fully both take part, averaged over every pair of the body, so a wave
    confined to a few joints drives little."""
    a, r = np.asarray(angles)/MAX_BEND, np.asarray(rates)/MAX_BEND
    active = _active(np.hypot(a, r/SWING_RATE))
    pairs = (a[1:]*r[:-1]-a[:-1]*r[1:])*np.minimum(active[1:], active[:-1])
    return float(np.mean(pairs))


def _band(value, low, high, soft):
    if low <= value <= high:
        return 1.
    return max(0., 1.-(low-value if value < low else value-high)/soft)


def frame_terms(angles, rates, previous_rates, extent, forward, closing):
    """This frame's shaping events. Wave and curvature pay only while the snake swims forward and closes on its
    target, so wriggling in place earns nothing, and in proportion to how much of the body takes part, so flapping
    only the tail earns little. Bad posture and twitching cost whatever the snake is doing."""
    curve = float(np.mean(np.abs(angles)))/MAX_BEND
    moving = forward*FPS >= MIN_SWIM and closing
    spread = frame_spread(angles, rates)
    accel = float(np.mean(np.abs(np.subtract(rates, previous_rates))))/MAX_BEND
    return dict(wave_seconds=min(1., max(0., wave_drive(angles, rates)/WAVE_REF))*spread/FPS if moving else 0.,
                curved_seconds=float(CURVE_BAND[0] <= curve <= CURVE_BAND[1])*spread/FPS if moving else 0.,
                posture_seconds=float(curve < RIGID or extent < COILED)/FPS,
                twitch=max(0., accel-TWITCH_FREE))


class Recorder:
    """Keeps an episode's bends and motion, frame by frame, for score()."""
    def __init__(self):
        self.angles, self.extension, self.forward = [], [], []

    def add(self, angles, extent, forward):
        self.angles.append(angles)
        self.extension.append(extent)
        self.forward.append(forward)

    def score(self):
        return score(self.angles, self.extension, self.forward)


def score(angles, extents, forwards):
    """{'wriggle': overall, and each component}, all in [0, 1]. The overall score is the geometric mean of
    curvature, wave, rhythm and swim, so failing any one of them (rigid, coiled, backward or standing wave, twitching,
    not swimming) fails it, multiplied by spread, so a wave carried by only part of the body scores in proportion.

    curvature: mean bend in the healthy band, times the share of time stretched out rather than coiled.
    spread: how much of the body swings at the dominant rhythm: each joint counts in full at SPREAD_SHARE of the
        largest joint's swing or more, in proportion below it.
    wave: bends lead from head to tail, by the phase of each joint at the dominant rhythm, every pair of neighbouring
        joints weighted alike and scoring only as far as both take part (a rigid stretch scores nothing).
    rhythm: that rhythm is a plausible undulation rate, times how smooth (untwitching) the bends are.
    swim: forward speed along the heading.
    """
    if len(angles) < 2:
        return dict(wriggle=0., **{k: 0. for k in COMPONENTS})
    a = np.asarray(angles, float)/MAX_BEND
    curve = float(np.mean(np.abs(a)))
    uncoiled = float(np.mean(np.clip((np.asarray(extents)-COILED)/(UNCOILED-COILED), 0., 1.)))
    curvature = _band(curve, *CURVE_BAND, CURVE_SOFT)*uncoiled
    spectrum = np.fft.rfft(a-a.mean(axis=0), axis=0)
    power = (np.abs(spectrum)**2).sum(axis=1)
    frequencies = np.fft.rfftfreq(len(a), 1/FPS)
    # The rhythm is the strongest undulation; slower drift is steering (turning toward the target), not the wave.
    undulating = frequencies >= STEERING
    k = int(np.argmax(np.where(undulating, power, -1.))) if undulating.any() else 0
    if k == 0 or power[k] <= 1e-12:
        wave = spread = frequency = 0.
    else:
        coefficients = spectrum[k]
        active = _active(np.abs(coefficients))
        spread = float(np.mean(active))
        # How far each joint's bend leads the joint behind it (nearer the tail), at the dominant rhythm.
        lags = np.angle(coefficients[1:]*np.conj(coefficients[:-1]))
        leads = np.clip(np.sin(lags)/WAVE_PHASE_REF, 0., 1.)*np.minimum(active[1:], active[:-1])
        wave = float(np.mean(leads))
        frequency = k*FPS/len(a)
    # Twitching puts bend power at rates no undulation has: the share above the plausible band.
    jitter = float(power[frequencies > RHYTHM_BAND[1]+RHYTHM_SOFT].sum()/max(1e-12, power.sum()))
    smooth = max(0., 1.-max(0., jitter-JITTER_FREE)/JITTER_SCALE)
    rhythm = _band(frequency, *RHYTHM_BAND, RHYTHM_SOFT)*smooth
    swim = float(np.clip(np.mean(forwards)*FPS/SWIM_REF, 0., 1.))
    parts = dict(curvature=curvature, wave=wave, rhythm=rhythm, swim=swim)
    overall = float(np.prod([max(0., v) for v in parts.values()])**(1/len(parts)))*spread
    return dict(wriggle=overall, spread=spread, **parts)
