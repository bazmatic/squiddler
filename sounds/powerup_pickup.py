"""Powerup pickup synthesis: each pickup sings a short motif in D Lydian, shaped by what the powerup does.

Lydian is the major scale with a raised fourth; that sharpened step gives every motif its bright, lifting,
slightly otherworldly sound. All the motifs share one key, so pickups heard in quick succession agree.
"""
import math
import numpy as np


TONIC = 293.66  # D4
LYDIAN = (0, 2, 4, 6, 7, 9, 11)  # Semitones above the tonic; 6 is the raised fourth


def pitch(degree):
    """The frequency of a scale degree counted from the tonic (0): 3 is the raised fourth, 7 the octave up, -7 down."""
    octave, step = divmod(degree, 7)
    return TONIC * 2 ** (octave + LYDIAN[step] / 12)


def _envelope(t, length, attack=.004, decay=None, release=.03):
    """Rise over attack, optionally fall away exponentially, and close smoothly at length so nothing clicks."""
    envelope = np.minimum(1, t / attack) * np.clip((length - t) / release, 0, 1)
    return envelope if decay is None else envelope * np.exp(-t / decay)


def _partials(frequency, t, partials):
    """A tone built from (harmonic ratio, level, decay time) partials; higher partials usually fade sooner."""
    return sum(level * np.sin(math.tau * frequency * ratio * t) * np.exp(-t / decay) for ratio, level, decay in partials)


def _notes(notes, duration, rate, voice):
    """Mix notes (start seconds, degree, length seconds, level, pan from -1 left to 1 right), each voiced by voice(frequency, t, length)."""
    out = np.zeros((round(rate * duration), 2))
    for start, degree, length, level, pan in notes:
        i = round(start * rate)
        t = np.arange(min(round(length * rate), len(out) - i)) / rate
        tone = voice(pitch(degree), t, length) * level
        out[i:i + len(t), 0] += tone * math.sqrt((1 - pan) / 2)
        out[i:i + len(t), 1] += tone * math.sqrt((1 + pan) / 2)
    return out


def _glide(path, duration, rate):
    """Time and frequency for a voice sliding smoothly in pitch through (seconds, degree) points."""
    t = np.arange(round(rate * duration)) / rate
    times, degrees = zip(*path)
    semitones = [12 * math.log2(pitch(d) / TONIC) for d in degrees]
    return t, TONIC * 2 ** (np.interp(t, times, semitones) / 12)


def _twin_blaster(rate):
    """Two bright voices climbing in parallel thirds, one in each ear: a pair of barrels firing as one."""
    def voice(f, t, length):
        # Odd harmonics give a hollow, square-ish zap
        return _partials(f, t, ((1, 1, 1), (3, .3, .05), (5, .15, .03))) * _envelope(t, length, decay=.09)
    steps = ((0, 2), (1, 3), (2, 4), (4, 6), (7, 9))
    notes = [(i * .06, d, .14 if i < 4 else .3, .5, pan) for i, pair in enumerate(steps) for d, pan in zip(pair, (-.7, .7))]
    return _notes(notes, .55, rate, voice)


def _scattergun(rate):
    """A wide Lydian chord struck almost at once and fanned across the stereo field, after a short crack: the spread of shot."""
    def voice(f, t, length):
        return _partials(f, t, ((1, 1, .25), (2, .5, .08), (3, .3, .04))) * _envelope(t, length, attack=.002)
    fan = (-7, -3, 2, 3, 6)  # D3 A3 F#4 G#4 C#5
    notes = [(.02 + i * .018, d, .5, .45, -1 + i * .5) for i, d in enumerate(fan)]
    out = _notes(notes, .62, rate, voice)
    t = np.arange(round(rate * .03)) / rate
    crack = np.random.default_rng(5).standard_normal(len(t)) * .5 * np.exp(-t / .006)
    out[:len(t)] += crack[:, None]
    return out


def _railgun(rate):
    """A two-octave run straight up the scale, a charge racing along the rail, holding on the raised fourth like a coil about to fire."""
    def voice(f, t, length):
        # A band-limited sawtooth for a hard electric edge
        saw = sum(np.sin(math.tau * f * k * t) / k for k in range(1, 9) if f * k < rate / 2)
        return .6 * saw * _envelope(t, length, attack=.002, decay=.04 if length < .1 else .35)
    run = [(i * .018, d, .05, .45, (d + 7) / 18 - .5) for i, d in enumerate(range(-7, 11))]
    hold = (18 * .018, 11, .5, .55, 0)
    return _notes(run + [hold], .85, rate, voice)


def _seeker(rate):
    """A tone that sweeps back and forth across the scale and the stereo field, hunting, then locks on with a sharpening pulse."""
    duration = .75
    t, frequency = _glide(((0, 4), (.1, 3), (.2, 5), (.3, 2), (.4, 6), (.48, 7), (duration, 7)), duration, rate)
    # The pitch shiver narrows as it locks on
    frequency = frequency * (1 + .012 * np.sin(math.tau * 14 * t) * np.clip((.5 - t) / .5, 0, 1))
    phase = math.tau * np.cumsum(frequency) / rate
    tone = np.sin(phase) + .25 * np.sin(2 * phase)
    # Once locked, the tone pulses quickly like a target lock beeping
    locked = t > .48
    tone[locked] *= .55 + .45 * np.sign(np.sin(math.tau * 16 * (t[locked] - .48)))
    tone *= _envelope(t, duration, attack=.01, release=.08) * .5
    pan = .8 * np.sin(math.tau * 2.5 * t) * np.clip((.48 - t) / .48, 0, 1)
    return np.stack([tone * np.sqrt((1 - pan) / 2), tone * np.sqrt((1 + pan) / 2)], axis=1)


def _inflator(rate):
    """A rubbery note that swells and stretches up the scale with a growing wobble, then pops."""
    duration = .75
    t, frequency = _glide(((0, -7), (.15, -5), (.3, -4), (.45, -3), (.55, 0), (.62, 0)), duration, rate)
    frequency = frequency * (1 + .03 * np.sin(math.tau * 9 * t) * t / duration)
    phase = math.tau * np.cumsum(frequency) / rate
    tone = np.sin(phase) + .3 * np.sin(2 * phase) + .12 * np.sin(3 * phase)
    # Growing louder as it fills, then cut off sharply where it bursts
    swell = (.25 + .75 * np.minimum(1, t / .58)) * np.clip((.62 - t) / .01, 0, 1) * np.minimum(1, t / .02)
    out = np.repeat((tone * swell * .45)[:, None], 2, axis=1)
    pop = _notes([(.62, 11, .12, .5, -.4), (.63, 14, .1, .35, .4)], duration, rate,
                 lambda f, t, length: np.sin(math.tau * f * t) * _envelope(t, length, attack=.001, decay=.025))
    return out + pop


def _grenade_launcher(rate):
    """A thump, then notes lobbed up in an arc that slows over the top and drops to a low landing."""
    def voice(f, t, length):
        # Wooden, marimba-like strikes
        return _partials(f, t, ((1, 1, .18), (4, .35, .02))) * _envelope(t, length, attack=.002)
    arc = ((0, 0), (.06, 4), (.13, 7), (.23, 11), (.36, 7), (.44, 3), (.5, 0))
    notes = [(start, d, .25, .45, -.6 + i * .2) for i, (start, d) in enumerate(arc)]
    notes.append((.58, -7, .35, .6, .6))
    out = _notes(notes, .95, rate, voice)
    t = np.arange(round(rate * .15)) / rate
    thump = np.sin(math.tau * np.cumsum(45 + 70 * np.exp(-t / .02)) / rate) * np.exp(-t / .05) * .8
    out[:len(t)] += thump[:, None]
    return out


def _shield(rate):
    """The whole Lydian chord blooming outward from the root and shimmering, held around the ship like an aura."""
    def voice(f, t, length):
        # Two slightly detuned bell-ish voices beat slowly against each other, with a quick tremolo shimmer on top
        chorus = _partials(f, t, ((1, 1, 2), (2.01, .25, .4))) + _partials(f * 1.003, t, ((1, .8, 2),))
        shimmer = 1 + .25 * np.sin(math.tau * 11 * t)
        return chorus * shimmer * _envelope(t, length, attack=.08, release=.35)
    chord = (-7, 0, 2, 3, 4, 6, 9)  # D3 D4 F#4 G#4 A4 C#5 F#5
    notes = [(i * .04, d, 1.1 - i * .04, .22, (-1) ** i * min(.9, i * .15)) for i, d in enumerate(chord)]
    return _notes(notes, 1.15, rate, voice)


def _upgrade(rate):
    """A quick bright arpeggio up to the raised fourth an octave above: a plain boost."""
    def voice(f, t, length):
        return _partials(f, t, ((1, 1, .4), (2, .4, .1), (3, .15, .05))) * _envelope(t, length)
    notes = [(i * .05, d, .3 if i < 4 else .4, .45, -.4 + i * .2) for i, d in enumerate((0, 2, 4, 7, 10))]
    return _notes(notes, .62, rate, voice)


def _beacon(rate):
    """A slow horn call up from the low root through the raised fourth, summoning the hunter."""
    def voice(f, t, length):
        # The tone brightens as it swells, like breath filling a horn
        bright = np.minimum(1, t / .15)
        tone = sum(np.sin(math.tau * f * k * t) * bright ** (k - 1) / k for k in range(1, 7))
        return tone * _envelope(t, length, attack=.06, release=.15)
    notes = ((0, -7, .32, .35, -.3), (.28, -4, .32, .35, .3), (.56, 0, .5, .4, 0), (.56, -3, .5, .25, 0))
    return _notes(notes, 1.06, rate, voice)


# Pickups not listed here (and the hunter's generic boost) play the plain boost.
PROFILES = {
    'twin_blaster': _twin_blaster,
    'scattergun': _scattergun,
    'railgun': _railgun,
    'seeker': _seeker,
    'inflator': _inflator,
    'grenade_launcher': _grenade_launcher,
    'shield': _shield,
    'upgrade': _upgrade,
    'beacon': _beacon,
}


def samples(kind, rate):
    """Stereo samples for a pickup, peaking at 0.85 so every pickup is as loud as the others."""
    out = PROFILES.get(kind, _upgrade)(rate)
    return out * (.85 / max(np.abs(out).max(), 1e-9))


if __name__ == '__main__':
    # Audition pickups: python -m sounds.powerup_pickup [kind ...], default all of them in turn.
    import sys, time, pygame
    pygame.mixer.init(44100, -16, 2)
    for kind in sys.argv[1:] or PROFILES:
        print(kind)
        pcm = (samples(kind, 44100) * 32767).astype(np.int16)
        pygame.sndarray.make_sound(pcm).play()
        time.sleep(len(pcm) / 44100 + .3)
