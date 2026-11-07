"""Sound manager for game audio effects.

This module provides programmatic sound generation and playback for game sounds
including white noise for thruster, click sounds for shooting, and explosion
sounds for enemy destruction.
"""

import pygame
import random
import math
import numpy as np
import numpy.random  # Loaded here, not lazily mid-game where it stalls a frame
import config
import wave
import os
from typing import Optional, Dict, Tuple, Any


HUNTER_SHOT_CHANNEL = 8

# The player's sound setting; shared by every SoundManager
_muted = False


def set_muted(muted: bool) -> None:
    """Silence or restore all game sound; muting also stops whatever is playing."""
    global _muted
    _muted = muted
    if muted and config.SOUND_ENABLED and pygame.mixer.get_init():
        pygame.mixer.stop()


def _audible() -> bool:
    """Whether sounds should play now: enabled in config and not muted by the player."""
    return config.SOUND_ENABLED and not _muted


class SoundManager:
    """Manages sound generation and playback for game audio effects.
    
    Generates sounds programmatically using pygame.sndarray and manages
    continuous playback for thruster sounds and one-shot playback for
    shooting sounds.
    """
    
    def __init__(self) -> None:
        """Initialize sound manager and generate sounds."""
        if not config.SOUND_ENABLED:
            self.thruster_sound: Optional[pygame.mixer.Sound] = None
            self.shoot_sound: Optional[pygame.mixer.Sound] = None
            self.enemy_destroy_sound: Optional[pygame.mixer.Sound] = None
            self.powerup_sounds: Dict[str, pygame.mixer.Sound] = {}
            self.thruster_channel: Optional[pygame.mixer.Channel] = None
            return
        
        # Initialize mixer if not already done
        if not pygame.mixer.get_init():
            pygame.mixer.init(
                frequency=config.SOUND_SAMPLE_RATE,
                size=-16,  # 16-bit signed samples
                channels=2,  # Stereo
                buffer=512  # Small buffer for low latency
            )
        
        # Generate sounds
        self.thruster_sound = self._generate_white_noise()
        self.shoot_sound = self._generate_click()
        self.enemy_destroy_sound = None  # Generated on-demand with pitch variation
        self.exit_warble_sound: Optional[pygame.mixer.Sound] = None  # Generated on first use
        self.upgraded_shoot_sound: Optional[pygame.mixer.Sound] = None  # Generated on first use
        self.powerup_sounds: Dict[str, pygame.mixer.Sound] = {}  # Pickup motifs by kind, generated on first use
        self.portal_power_up_sound: Optional[pygame.mixer.Sound] = None  # Generated on first use
        self.portal_power_down_sound: Optional[pygame.mixer.Sound] = None  # Generated on first use
        self.tinkling_sound_cache: Dict[float, Optional[pygame.mixer.Sound]] = {}  # Cache tinkling sounds by pitch
        self.critical_warning_sound: Optional[pygame.mixer.Sound] = None
        self.critical_warning_channel: Optional[pygame.mixer.Channel] = None
        
        # One channel past the eight the other sounds share, so the hunter's shots never cut off the player's
        pygame.mixer.set_num_channels(max(pygame.mixer.get_num_channels(), HUNTER_SHOT_CHANNEL + 1))

        # Set up dedicated channel for thruster (continuous sound)
        self.thruster_channel = pygame.mixer.Channel(0)
        self.thruster_channel.set_volume(config.THRUSTER_SOUND_VOLUME)
        
        # Set volume for shoot sound
        if self.shoot_sound:
            self.shoot_sound.set_volume(config.SHOOT_SOUND_VOLUME)
    
    @staticmethod
    def _moving_average(samples: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """Average each sample with its neighbours, over a window whose width varies per sample.

        Done with a running total rather than a Python loop over every sample,
        which took long enough to stall the game the first time a sound played.
        """
        num_samples = len(samples)
        index = np.arange(num_samples)
        start = np.maximum(0, index - widths // 2)
        end = np.minimum(num_samples, index + widths // 2 + 1)
        running_total = np.concatenate(([0.0], np.cumsum(samples, dtype=np.float64)))
        return ((running_total[end] - running_total[start]) / (end - start)).astype(np.float32)

    def _generate_white_noise(self) -> Optional[pygame.mixer.Sound]:
        """Generate white noise sound for thruster.
        
        Creates a short loop of random samples that can be played continuously.
        
        Returns:
            pygame.mixer.Sound object with white noise, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = config.THRUSTER_NOISE_DURATION
        num_samples = int(sample_rate * duration)
        
        # Generate random samples (white noise)
        # Use int16 range: -32768 to 32767
        samples = []
        for _ in range(num_samples):
            # Generate random value in int16 range
            sample = random.randint(-32768, 32767)
            samples.append(sample)
        
        # Convert to numpy array format expected by pygame.sndarray
        # pygame.sndarray expects a 2D array: (num_samples, channels)
        # For stereo, we duplicate the samples
        samples_array = np.array(samples, dtype=np.int16)
        stereo_samples = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_samples[:, 0] = samples_array
        stereo_samples[:, 1] = samples_array
        
        # Create sound from array
        sound = pygame.sndarray.make_sound(stereo_samples)
        return sound
    
    def _generate_click(self) -> Optional[pygame.mixer.Sound]:
        """Generate simple click sound for shooting.
        
        Creates a very short, sharp click sound - a brief impulse with quick decay.
        
        Returns:
            pygame.mixer.Sound object with click sound, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.01  # Very short duration for a click (10ms)
        num_samples = int(sample_rate * duration)
        
        # Generate click: sharp impulse with exponential decay
        samples = []
        for i in range(num_samples):
            progress = i / num_samples
            
            # Create a sharp click: immediate peak followed by exponential decay
            # Use a very short attack (first 5% of samples) and quick decay
            if progress < 0.05:
                # Quick attack to peak
                envelope = progress / 0.05
            else:
                # Exponential decay
                decay_progress = (progress - 0.05) / 0.95
                envelope = math.exp(-decay_progress * 15)  # Fast decay
            
            # Generate a brief tone at higher frequency for the "click" character
            t = i / sample_rate
            frequency = 2000  # Higher frequency for sharper click
            tone = math.sin(2 * math.pi * frequency * t)
            
            # Scale to int16 range and apply envelope
            sample_value = int(tone * 16383 * envelope)
            samples.append(sample_value)
        
        # Convert to numpy array format
        samples_array = np.array(samples, dtype=np.int16)
        stereo_samples = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_samples[:, 0] = samples_array
        stereo_samples[:, 1] = samples_array
        
        # Create sound from array
        sound = pygame.sndarray.make_sound(stereo_samples)
        return sound

    def _generate_bad_hit(self) -> Optional[pygame.mixer.Sound]:
        """Generate a short, unpleasant tone for when the player is hit."""
        if not config.SOUND_ENABLED:
            return None

        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.13
        num_samples = int(sample_rate * duration)
        t = np.arange(num_samples, dtype=np.float32) / sample_rate

        freq = random.uniform(260.0, 360.0)
        tone = np.sin(2 * math.pi * freq * t)
        envelope = np.exp(-9.0 * t)
        noise = np.random.uniform(-0.3, 0.3, size=num_samples) * np.exp(-12.0 * t)
        samples = (tone + noise) * envelope
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)

        stereo = np.stack([samples, samples], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)
        return pygame.sndarray.make_sound(stereo_int16)

    def play_bad_hit(self) -> None:
        """Play a quick 'bad hit' sound for when the player is struck."""
        if not _audible():
            return

        sound = self._generate_bad_hit()
        if not sound:
            return

        sound.set_volume(config.BAD_HIT_SOUND_VOLUME)
        hit_channel = pygame.mixer.Channel(4)
        hit_channel.play(sound)
    
    def _generate_upgraded_shoot(self) -> Optional[pygame.mixer.Sound]:
        """Generate machine gun sound for upgraded guns.
        
        Creates a rapid-fire machine gun effect with multiple rapid clicks.
        
        Returns:
            pygame.mixer.Sound object with machine gun sound, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.06  # Short duration for rapid-fire effect (60ms)
        num_samples = int(sample_rate * duration)
        
        # Machine gun: multiple rapid clicks
        num_clicks = 4  # Number of clicks in the burst
        click_duration = 0.008  # Each click is 8ms
        click_samples = int(sample_rate * click_duration)
        click_interval = num_samples // num_clicks  # Space between clicks
        
        samples = []
        for i in range(num_samples):
            sample_value = 0
            
            # Check if we're within any click
            for click_idx in range(num_clicks):
                click_start = click_idx * click_interval
                click_end = click_start + click_samples
                
                if click_start <= i < click_end:
                    # We're in a click - generate the click sound
                    click_local_sample = i - click_start
                    click_progress = click_local_sample / click_samples
                    click_time = click_local_sample / sample_rate
                    
                    # Slight pitch variation between clicks for realism
                    pitch_variation = 1.0 + (click_idx * 0.05)  # Slight upward pitch
                    frequency = 2000 * pitch_variation
                    
                    # Generate click tone
                    tone = math.sin(2 * math.pi * frequency * click_time)
                    
                    # Apply envelope: very quick attack and decay
                    if click_progress < 0.2:
                        # Quick attack
                        envelope = click_progress / 0.2
                    else:
                        # Fast exponential decay
                        decay_progress = (click_progress - 0.2) / 0.8
                        envelope = math.exp(-decay_progress * 20)  # Very fast decay
                    
                    # Add some high-frequency content for sharpness
                    tone += 0.3 * math.sin(2 * math.pi * frequency * 3 * click_time)
                    
                    sample_value += int(tone * 16383 * envelope * 0.8)  # Scale down slightly
                    break  # Only one click active at a time
            
            samples.append(sample_value)
        
        # Convert to numpy array format
        samples_array = np.array(samples, dtype=np.int16)
        stereo_samples = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_samples[:, 0] = samples_array
        stereo_samples[:, 1] = samples_array
        
        # Create sound from array
        sound = pygame.sndarray.make_sound(stereo_samples)
        if sound:
            sound.set_volume(config.SHOOT_SOUND_VOLUME * 1.1)  # Slightly louder
        return sound
    
    def _generate_sputter(self) -> pygame.mixer.Sound:
        """The thruster running out of fuel: its noise cuts out and coughs back in at irregular moments."""
        rate, _, channels = pygame.mixer.get_init()
        rng = np.random.default_rng(451)  # Local, so gameplay randomness is untouched
        duration = 1.2
        noise = rng.uniform(-1, 1, round(rate*duration))
        gate = np.empty(len(noise))
        start = 0
        while start < len(gate):
            length = round(rate*rng.uniform(.02, .09))
            # The loop starts and ends running, so it repeats without a click
            running = start == 0 or start+length >= len(gate) or rng.random() > .4
            gate[start:start+length] = rng.uniform(.75, 1) if running else .04
            start += length
        # A few milliseconds of smoothing softens each cut into a cough rather than a click
        width = round(rate*.004)
        gate = np.convolve(np.concatenate((gate[-width:], gate)), np.ones(width)/width, mode='valid')[:len(noise)]
        pcm = (noise*gate*32767).astype(np.int16)
        if channels > 1:
            pcm = np.repeat(pcm[:, None], channels, axis=1)
        return pygame.sndarray.make_sound(pcm)

    def start_thruster(self, sputtering: bool = False) -> None:
        """Play the thruster loop, sputtering when power is low; switches loops if the state changed."""
        if not _audible() or not self.thruster_sound or not self.thruster_channel:
            return
        if sputtering and getattr(self, 'sputter_sound', None) is None:
            self.sputter_sound = self._generate_sputter()
        sound = self.sputter_sound if sputtering else self.thruster_sound
        if not self.thruster_channel.get_busy() or self.thruster_channel.get_sound() is not sound:
            self.thruster_channel.play(sound, loops=-1)  # -1 = infinite loop
    
    def stop_thruster(self) -> None:
        """Stop playing thruster sound."""
        if not config.SOUND_ENABLED or not self.thruster_channel:
            return
        
        self.thruster_channel.stop()
    
    def stop_all_sounds(self) -> None:
        """Stop all currently playing sounds."""
        if not config.SOUND_ENABLED:
            return
        
        # Stop all mixer channels
        for i in range(pygame.mixer.get_num_channels()):
            channel = pygame.mixer.Channel(i)
            if channel.get_busy():
                channel.stop()
    
    def play_shoot(self, is_upgraded: bool = False) -> None:
        """Play shoot sound once.
        
        Uses a separate channel so it doesn't interfere with thruster sound.
        
        Args:
            is_upgraded: If True, plays an enhanced shooting sound for upgraded guns.
        """
        if not _audible():
            return
        
        # Use channel 1 for shoot sound (channel 0 is reserved for thruster)
        shoot_channel = pygame.mixer.Channel(1)
        
        if is_upgraded:
            # Play enhanced upgraded shoot sound
            if not hasattr(self, 'upgraded_shoot_sound') or self.upgraded_shoot_sound is None:
                self.upgraded_shoot_sound = self._generate_upgraded_shoot()
            if self.upgraded_shoot_sound:
                shoot_channel.play(self.upgraded_shoot_sound)
        else:
            # Play normal shoot sound
            if self.shoot_sound:
                shoot_channel.play(self.shoot_sound)
    
    def play_hunter_shot(self, weapon_id: Optional[str]) -> None:
        """The allied hunter's shot, on its own channel: its pickup weapon's sound, or the plain gun's click."""
        if not _audible():
            return
        if weapon_id is not None:
            self.play_weapon_fire(weapon_id, HUNTER_SHOT_CHANNEL)
        elif self.shoot_sound:
            pygame.mixer.Channel(HUNTER_SHOT_CHANNEL).play(self.shoot_sound)

    def play_powerup_activation(self, kind: str = 'upgrade') -> None:
        """Play the Lydian motif for a collected powerup: a weapon id, 'shield', 'beacon', or 'upgrade' for a plain boost."""
        if not _audible():
            return
        if kind not in self.powerup_sounds:
            rate, _, channels = pygame.mixer.get_init()
            from sounds.powerup_pickup import samples
            pcm = (samples(kind, rate) * 32767).astype(np.int16)
            if channels == 1:
                pcm = pcm.mean(axis=1).astype(np.int16)
            sound = pygame.sndarray.make_sound(pcm)
            sound.set_volume(config.POWERUP_ACTIVATION_SOUND_VOLUME)
            self.powerup_sounds[kind] = sound
        pygame.mixer.Channel(4).play(self.powerup_sounds[kind])

    def play_weapon_fire(self, weapon_id, channel=1):
        """Distinct deterministic tones, cached without consuming simulation randomness."""
        if not _audible():
            return
        from weapons.catalog import CATALOG
        d = CATALOG[weapon_id]
        cache = getattr(self, '_weapon_sounds', None)
        if cache is None:
            self._weapon_sounds = cache = {}
        if weapon_id not in cache:
            rate, _, channels = pygame.mixer.get_init()
            from sounds.weapon_fire import samples
            pcm=(np.clip(samples(d,rate),-1,1)*32767).astype(np.int16)
            # A profile gives one channel, or two for stereo
            if pcm.ndim == 1 and channels > 1:
                pcm = np.repeat(pcm[:, None], channels, axis=1)
            elif pcm.ndim == 2 and channels == 1:
                pcm = pcm.mean(axis=1).astype(np.int16)
            cache[weapon_id] = pygame.sndarray.make_sound(pcm)
        pygame.mixer.Channel(channel).play(cache[weapon_id])

    def play_inflation(self) -> None:
        """A rubbery rising swell when a creature balloons in size."""
        if not _audible():
            return
        sound = getattr(self, '_inflation_sound', None)
        if sound is None:
            rate, _, channels = pygame.mixer.get_init()
            duration = .65
            t = np.arange(round(rate*duration))/rate
            progress = t/duration
            frequency = 110*np.power(7, progress)*(1+.045*np.sin(math.tau*9*t))
            phase = math.tau*np.cumsum(frequency)/rate
            # Rounded harmonics give the upward slide a stretchy balloon character.
            tone = np.sin(phase)+.28*np.sin(2*phase)+.12*np.sin(3*phase)
            envelope = np.minimum(1,t/.025)*np.minimum(1,(duration-t)/.12)
            pcm = (tone*envelope*.42*32767).astype(np.int16)
            if channels > 1:
                pcm = np.repeat(pcm[:, None], channels, axis=1)
            self._inflation_sound = sound = pygame.sndarray.make_sound(pcm)
            sound.set_volume(config.POWERUP_ACTIVATION_SOUND_VOLUME)
        pygame.mixer.Channel(3).play(sound)

    def play_beacon_arrival(self) -> None:
        """A gathering shimmer, resolving into four bright notes as the beacon materializes."""
        if not _audible():
            return
        sound = getattr(self, 'beacon_arrival_sound', None)
        if sound is None:
            rate, _, channels = pygame.mixer.get_init()
            samples = np.zeros(round(rate * 1.95), dtype=np.float64)
            gather = np.arange(round(rate * 1.2)) / rate
            phase = math.tau * np.cumsum(180 + 650 * (gather / 1.2)**2) / rate
            samples[:len(gather)] = (np.sin(phase) + 0.3 * np.sin(phase * 2.01)) * np.sin(math.pi*gather/1.2)**2 * 0.09
            for i, frequency in enumerate((523.25, 659.25, 783.99, 1046.5)):
                start = round((0.96 + i * 0.08) * rate)
                t = np.arange(len(samples) - start) / rate
                envelope = np.minimum(1, t / 0.008) * np.exp(-t * 7)
                samples[start:] += (np.sin(math.tau * frequency * t)
                                    + 0.25 * np.sin(math.tau * frequency * 2 * t)) * envelope * 0.22
            pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
            if channels > 1:
                pcm = np.repeat(pcm[:, None], channels, axis=1)
            sound = pygame.sndarray.make_sound(pcm)
            sound.set_volume(config.POWERUP_ACTIVATION_SOUND_VOLUME)
            self.beacon_arrival_sound = sound
        pygame.mixer.Channel(6).play(sound)
    
    def _generate_enemy_destroy(
        self,
        pitch_multiplier: float = 1.0,
        return_layers: bool = False
    ) -> Optional[pygame.mixer.Sound]:
        """Generate layered arcade explosion sound for enemy destruction.
        
        Creates a satisfying, punchy explosion with multiple layers:
        - High-frequency crack (breaking glass/metallic snap)
        - Mid-range explosion (descending tone)
        - Bass rumble (low frequency weight)
        - Sparkle tail (high-pitched finishing flourish)
        
        Args:
            pitch_multiplier: Pitch variation multiplier (0.85 to 1.15 for ±15% variation).
        
        Returns:
            pygame.mixer.Sound object with explosion sound, or None if sounds disabled.
            When return_layers is True, returns (sound, layers_dict).
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.5  # 500ms for a satisfying explosion (middle of 0.3-0.7s range)
        num_samples = int(sample_rate * duration)
        
        # Create time array for vectorized operations
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        progress_array = np.arange(num_samples, dtype=np.float32) / num_samples
        
        # Pre-generate reusable noise sources
        noise_white = np.random.normal(0.0, 1.0, num_samples).astype(np.float32)
        # Slightly smoothed noise for body; short kernel to avoid dulling transients
        body_kernel = np.ones(9, dtype=np.float32) / 9.0
        filtered_body_noise = np.convolve(noise_white, body_kernel, mode='same')
        # High-passed noise for crack/sparkle by subtracting a wider blur
        blur_kernel = np.ones(25, dtype=np.float32) / 25.0
        blurred_noise = np.convolve(noise_white, blur_kernel, mode='same')
        noise_high = noise_white - blurred_noise
        
        # Initialize all layers
        crack_layer = np.zeros(num_samples, dtype=np.float32)
        explosion_layer = np.zeros(num_samples, dtype=np.float32)
        bass_layer = np.zeros(num_samples, dtype=np.float32)
        sparkle_layer = np.zeros(num_samples, dtype=np.float32)
        
        # Layer 1: High-frequency crack (breaking glass/metallic snap)
        # Very early, short, and noisy to avoid tonal chirp
        crack_mask = progress_array < 0.2
        crack_indices = np.where(crack_mask)[0]
        if len(crack_indices) > 0:
            crack_progress = progress_array[crack_mask] / 0.2
            crack_t = t_array[crack_mask]
            # Band-limited noisy crack with a slight FM wobble to avoid pure tone
            base_freq = 800 * pitch_multiplier
            fm = 400 * pitch_multiplier
            fm_mod = np.sin(2 * math.pi * fm * crack_t) * 0.1
            crack_carrier = np.sin(2 * math.pi * base_freq * crack_t * (1.0 + fm_mod))
            # Band-limit noise toward lower highs (~0.8–3 kHz) to remove squeak
            crack_noise_raw = noise_high[crack_mask]
            bp_mid = crack_noise_raw - np.convolve(crack_noise_raw, np.ones(19)/19.0, mode='same')
            bp_wide = crack_noise_raw - np.convolve(crack_noise_raw, np.ones(37)/37.0, mode='same')
            crack_noise = (bp_mid * 0.55 + bp_wide * 0.45) * 1.1
            # Add a micro click (single-cycle impulse) to emphasize the initial snap
            click_env = np.exp(-crack_progress * 220.0)
            click = click_env * 1.2
            # Reduce tonal part to avoid bell-like ring; noise dominates
            crack_signal = crack_carrier * 0.35 + crack_noise + click
            # Envelope: instantaneous attack, very fast decay to avoid tail
            crack_envelope = np.where(
                crack_progress < 0.02,
                crack_progress / 0.02,
                np.exp(-(crack_progress - 0.02) / 0.98 * 14.0)
            )
            crack_layer[crack_mask] = crack_signal * crack_envelope * 0.9  # strong but very brief
        
        # Layer 2: Mid-range explosion (main body)
        # Texture-dominant noise with slight downward bend to avoid tonal sweep
        explosion_mask = progress_array < 0.8
        explosion_indices = np.where(explosion_mask)[0]
        if len(explosion_indices) > 0:
            explosion_progress = progress_array[explosion_mask] / 0.8
            explosion_t = t_array[explosion_mask]
            # Lowpass cutoff sweeps downward for a satisfying drop
            start_cutoff = 180.0 * pitch_multiplier
            end_cutoff = 40.0 * pitch_multiplier
            cutoff = start_cutoff - (start_cutoff - end_cutoff) * explosion_progress
            # Envelope: fast attack, smooth but present decay across most of the duration
            explosion_envelope = np.where(
                explosion_progress < 0.03,
                explosion_progress / 0.03,
                np.exp(-(explosion_progress - 0.03) / 0.97 * 2.6)
            )
            # Crunchy click train that decelerates: start dense, then widen spacing
            # Create an impulse train whose period increases over time
            start_rate = 750.0   # clicks per second initially (lowered)
            end_rate = 90.0      # slow clicks near tail
            inst_rate = start_rate - (start_rate - end_rate) * explosion_progress
            phase = np.cumsum(inst_rate / sample_rate)
            # Create impulses at each wrap: diff of the floored phase
            impulse = np.diff(np.floor(phase), prepend=0).astype(np.float32)
            # Shape impulses with a short decay kernel
            click_kernel = np.exp(-np.arange(20) / 6.0).astype(np.float32)
            impulse = np.convolve(impulse, click_kernel, mode='same') * 1.25
            # Texture the impulses with filtered noise
            wide_kernel = np.ones(55, dtype=np.float32) / 55.0
            tight_kernel = np.ones(23, dtype=np.float32) / 23.0
            wide = np.convolve(noise_white, wide_kernel, mode='same')[explosion_mask]
            tight = np.convolve(noise_white, tight_kernel, mode='same')[explosion_mask]
            # Blend tighter filtering early, wider later
            blend = (explosion_progress).clip(0.0, 1.0)
            textured = tight * (1.0 - blend) + wide * blend
            explosion_signal = (textured * 0.4 + impulse * 1.35) * explosion_envelope
            # Add a downward sine/triangle hybrid for body without whistle, lower freq
            sweep_freq = (110.0 - 70.0 * explosion_progress) * pitch_multiplier
            tri_body = 2.0 * np.abs((sweep_freq * explosion_t) % 1.0) - 1.0
            tri_body = np.tanh(tri_body * 1.3) * 0.24
            sine_body = np.sin(2 * math.pi * sweep_freq * explosion_t) * 0.14
            explosion_signal += (sine_body + tri_body) * explosion_envelope
            # Post low-pass to tame residual highs
            lp_kernel = np.ones(9, dtype=np.float32) / 9.0
            explosion_signal = np.convolve(explosion_signal, lp_kernel, mode='same')
            explosion_layer[explosion_mask] = explosion_signal * 1.0  # Base amplitude
        
        # Layer 3: Bass rumble (low frequency weight)
        # Extends longer and overlaps to avoid gaps
        bass_mask = progress_array < 0.9
        bass_indices = np.where(bass_mask)[0]
        if len(bass_indices) > 0:
            bass_progress = progress_array[bass_mask] / 0.9
            bass_t = t_array[bass_mask]
            bass_freq = 52 * pitch_multiplier
            bass_signal = np.sin(2 * math.pi * bass_freq * bass_t)
            # Add weight via gentle distortion of a low sine + smoothed noise
            low_noise_kernel = np.ones(65, dtype=np.float32) / 65.0
            low_noise = np.convolve(noise_white * 0.4, low_noise_kernel, mode='same')[bass_mask]
            bass_signal = np.tanh(bass_signal * 1.8 + low_noise * 1.35)
            # Envelope: slower attack, long decay to keep weight under the tail
            bass_envelope = np.where(
                bass_progress < 0.12,
                bass_progress / 0.12,
                np.exp(-(bass_progress - 0.12) / 0.88 * 1.6)
            )
            bass_layer[bass_mask] = bass_signal * bass_envelope * 0.6  # 0.4-0.6+ range
        
        # Layer 4: Sparkle tail (high-pitched finishing flourish)
        # Start earlier with quicker onset to avoid gap
        sparkle_mask = progress_array >= 0.2
        sparkle_indices = np.where(sparkle_mask)[0]
        if len(sparkle_indices) > 0:
            sparkle_progress = (progress_array[sparkle_mask] - 0.2) / 0.8
            sparkle_t = t_array[sparkle_mask]
            sparkle_freq = 2600 * pitch_multiplier
            shimmer_noise = noise_high[sparkle_mask] * 0.16
            sparkle_signal = (np.sin(2 * math.pi * sparkle_freq * sparkle_t) * 0.45 +
                            0.28 * np.sin(2 * math.pi * sparkle_freq * 2.1 * sparkle_t) +
                            shimmer_noise)
            # Envelope: brisk attack, smooth but not-too-long fade
            sparkle_envelope = np.where(
                sparkle_progress < 0.08,
                sparkle_progress / 0.08,
                np.exp(-(sparkle_progress - 0.08) / 0.92 * 2.8)
            )
            sparkle_layer[sparkle_mask] = sparkle_signal * sparkle_envelope * 0.4  # 0.3-0.5 range
        
        # Mix all layers together
        samples = crack_layer + explosion_layer + bass_layer + sparkle_layer
        
        # Normalize to prevent clipping while preserving relative layer amplitudes
        max_amplitude = np.max(np.abs(samples))
        target_max = 0.9
        if max_amplitude > target_max:
            samples = samples * (target_max / max_amplitude)
        
        # Scale to int16 range
        samples_int16 = (samples * 16383).astype(np.int16)
        
        # Convert to stereo
        stereo_samples = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_samples[:, 0] = samples_int16
        stereo_samples[:, 1] = samples_int16
        
        # Create sound from array
        sound = pygame.sndarray.make_sound(stereo_samples)
        
        if not return_layers:
            return sound
        
        def _to_stereo_layer(layer: np.ndarray) -> np.ndarray:
            layer_copy = np.copy(layer)
            peak = np.max(np.abs(layer_copy))
            if peak > 0.0:
                layer_copy = layer_copy * (target_max / peak)
            layer_int16 = (layer_copy * 16383).astype(np.int16)
            stereo_layer = np.zeros((num_samples, 2), dtype=np.int16)
            stereo_layer[:, 0] = layer_int16
            stereo_layer[:, 1] = layer_int16
            return stereo_layer
        
        layers = {
            "combined": stereo_samples,
            "crack": _to_stereo_layer(crack_layer),
            "mid": _to_stereo_layer(explosion_layer),
            "bass": _to_stereo_layer(bass_layer),
            "sparkle": _to_stereo_layer(sparkle_layer),
        }
        return sound, layers
    
    def play_enemy_destroy(self) -> None:
        """Play enemy destruction explosion sound once with pitch variation.
        
        Generates the sound on-demand with random pitch variation (±15%) to prevent
        ear fatigue when destroying multiple enemies rapidly. Each destruction gets
        a fresh, slightly different sound.
        
        Uses a separate channel so it doesn't interfere with other sounds.
        """
        if not _audible():
            return
        
        # Generate sound with random pitch variation (±15%)
        # Pitch multiplier range: 0.85 to 1.15
        pitch_multiplier = random.uniform(0.85, 1.15)
        sound = self._generate_enemy_destroy(pitch_multiplier)
        
        if not sound:
            return
        
        # Set volume
        sound.set_volume(config.ENEMY_DESTROY_SOUND_VOLUME)
        
        # Use channel 2 for enemy destroy sound
        destroy_channel = pygame.mixer.Channel(2)
        destroy_channel.play(sound)
    
    def _generate_exit_warble(self) -> Optional[pygame.mixer.Sound]:
        """Generate a cosmic warble for level exit.
        
        Layered sine sweeps with vibrato and gentle stereo detune for a sci-fi flare.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 1.2  # seconds
        num_samples = int(sample_rate * duration)
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        progress = t_array / duration
        
        # Pitch glide up then ripple down slightly
        base_freq = 360.0 + 520.0 * progress
        downward_bend = 120.0 * np.square(np.maximum(progress - 0.6, 0.0))
        freq_curve = base_freq - downward_bend
        
        # Vibrato and accelerating phaser LFOs (phaser rate ramps up linearly)
        vibrato = 0.12 * np.sin(2 * math.pi * 6.5 * t_array)
        phaser_rate = 0.8 + 1.2 * progress  # Hz ramps from 0.8 to 2.0 across duration
        phaser_phase = np.cumsum(phaser_rate / sample_rate) * 2 * math.pi
        phaser = 0.35 * np.sin(phaser_phase)
        detune = 0.01
        
        phase_main = np.cumsum(freq_curve * (1.0 + vibrato) / sample_rate) * 2 * math.pi
        phase_detune = np.cumsum(freq_curve * (1.0 - vibrato + detune) / sample_rate) * 2 * math.pi
        
        # Two voices for richness
        voice_main = np.sin(phase_main + phaser * 0.6)
        voice_detune = np.sin(phase_detune * 0.98 + 0.4 - phaser * 0.4)
        shimmer = np.sin(phase_main * 2.1 + phaser * 1.1) * 0.15
        
        # Envelope: fast rise, lingering tail
        attack = np.clip(progress / 0.08, 0.0, 1.0)
        decay = np.exp(-np.maximum(progress - 0.1, 0.0) * 2.7)
        sustain = 0.9 - 0.4 * progress
        envelope = attack * decay * sustain
        
        samples = (voice_main * 0.7 + voice_detune * 0.5 + shimmer) * envelope
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        
        # Stereo with phase sweep for width
        left = samples
        right = np.sin(phase_detune + 0.22 + phaser * 0.8) * envelope
        right = np.clip(right, -1.0, 1.0).astype(np.float32)
        
        stereo = np.stack([left, right], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)
        return pygame.sndarray.make_sound(stereo_int16)
    
    def play_exit_warble(self) -> None:
        """Play cosmic warble when player reaches the exit."""
        if not _audible():
            return
        
        if self.exit_warble_sound is None:
            self.exit_warble_sound = self._generate_exit_warble()
            if self.exit_warble_sound:
                self.exit_warble_sound.set_volume(config.EXIT_WARBLE_SOUND_VOLUME)
        
        if not self.exit_warble_sound:
            return
        
        exit_channel = pygame.mixer.Channel(3)
        exit_channel.play(self.exit_warble_sound)
    
    def export_enemy_destroy_layers(
        self,
        output_dir: str = ".",
        pitch_multiplier: float = 1.0
    ) -> None:
        """Export combined and individual explosion layers to WAV for debugging."""
        if not config.SOUND_ENABLED:
            return
        
        result = self._generate_enemy_destroy(
            pitch_multiplier=pitch_multiplier,
            return_layers=True
        )
        
        if not result:
            return
        
        sound, layers = result  # type: ignore
        os.makedirs(output_dir, exist_ok=True)
        for name, stereo in layers.items():
            path = os.path.join(output_dir, f"enemy_destroy_{name}.wav")
            with wave.open(path, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(config.SOUND_SAMPLE_RATE)
                wf.writeframes(stereo.tobytes())
    
    def _generate_tinkling_sound(self, pitch: float) -> Optional[pygame.mixer.Sound]:
        """Generate bell-like tinkling sound with specified pitch.
        
        Creates a pleasant bell/chime sound with harmonics for richness.
        
        Args:
            pitch: Frequency in Hz for the base tone.
            
        Returns:
            pygame.mixer.Sound object with tinkling sound, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.15  # Short duration for tinkling effect (150ms)
        num_samples = int(sample_rate * duration)
        
        # Generate tinkling: bell-like tone with harmonics
        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            progress = i / num_samples
            
            # Base frequency
            base_freq = pitch
            
            # Create bell-like tone with harmonics
            tone = math.sin(2 * math.pi * base_freq * t)
            # Add second harmonic (octave) for bell character
            tone += 0.5 * math.sin(2 * math.pi * base_freq * 2 * t)
            # Add third harmonic for more richness
            tone += 0.25 * math.sin(2 * math.pi * base_freq * 3 * t)
            
            # Apply envelope: quick attack, then exponential decay
            if progress < 0.1:
                # Quick attack for bell-like character
                envelope = progress / 0.1
            else:
                # Exponential decay
                decay_progress = (progress - 0.1) / 0.9
                envelope = math.exp(-decay_progress * 5)  # Moderate decay
            
            # Normalize and scale to int16 range
            tone = max(-1.0, min(1.0, tone))  # Clamp to prevent clipping
            sample_value = int(tone * 16383 * envelope)
            samples.append(sample_value)
        
        # Convert to numpy array format
        samples_array = np.array(samples, dtype=np.int16)
        stereo_samples = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_samples[:, 0] = samples_array
        stereo_samples[:, 1] = samples_array
        
        # Create sound from array
        sound = pygame.sndarray.make_sound(stereo_samples)
        if sound:
            sound.set_volume(config.SHOOT_SOUND_VOLUME * 0.8)  # Slightly quieter than shoot
        return sound
    
    def play_tinkling(self, pitch: float) -> None:
        """Play tinkling sound at specified pitch.
        
        Uses cached sounds to avoid regeneration. Pitch increases with each star.
        
        Args:
            pitch: Frequency in Hz for the tinkling sound.
        """
        if not _audible():
            return
        
        # Check cache first
        if pitch not in self.tinkling_sound_cache:
            self.tinkling_sound_cache[pitch] = self._generate_tinkling_sound(pitch)
        
        sound = self.tinkling_sound_cache[pitch]
        if sound:
            # Use channel 3 for tinkling sound
            tinkling_channel = pygame.mixer.Channel(3)
            tinkling_channel.play(sound)
    
    def _generate_star_lost_sound(self) -> Optional[pygame.mixer.Sound]:
        """Generate a sad descending tone for when a star is lost.
        
        Creates a melancholic descending tone that conveys loss.
        
        Returns:
            pygame.mixer.Sound object with descending tone, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.5  # Half second for a clear descending tone
        num_samples = int(sample_rate * duration)
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        progress = t_array / duration
        
        # Descending frequency sweep (sad, going down)
        start_freq = 440.0  # A4 note
        end_freq = 220.0    # A3 note (octave lower)
        # Use exponential curve for more natural descent
        freq_curve = start_freq * np.power(end_freq / start_freq, progress)
        
        # Generate tone with slight vibrato for expressiveness
        vibrato = 0.08 * np.sin(2 * math.pi * 4.5 * t_array)
        phase = np.cumsum(freq_curve * (1.0 + vibrato) / sample_rate) * 2 * math.pi
        
        # Main tone with harmonics for richness
        tone = np.sin(phase)
        tone += 0.3 * np.sin(phase * 2)  # Second harmonic
        tone += 0.15 * np.sin(phase * 3)  # Third harmonic
        
        # Envelope: quick attack, smooth decay
        attack = np.clip(progress / 0.1, 0.0, 1.0)
        decay = np.exp(-progress * 3.0)
        envelope = attack * decay
        
        # Apply envelope and normalize
        samples = tone * envelope
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        
        # Convert to stereo
        stereo = np.stack([samples, samples], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_int16)
        if sound:
            sound.set_volume(config.SHOOT_SOUND_VOLUME * 0.7)  # Moderate volume
        return sound

    def _generate_critical_warning_sound(self) -> Optional[pygame.mixer.Sound]:
        """Generate a slow, pleasant sine wave to loop while score nears zero.
        
        Creates a gentle, slow sine wave that's pleasant to listen to as a warning.
        """
        if not config.SOUND_ENABLED:
            return None

        sample_rate = config.SOUND_SAMPLE_RATE
        # Create a slow, pleasant sine wave
        cycle_duration = 2.0  # Slow 2-second cycle for pleasant sound
        num_cycles = 1  # Single cycle for seamless loop
        total_duration = cycle_duration * num_cycles
        num_samples = int(sample_rate * total_duration)
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        
        # Pleasant sine wave parameters
        base_freq = 220.0  # A3 note - lower, more mellow
        # Slow modulation with more variation
        modulation_rate = 0.8  # Hz - slow but noticeable modulation
        modulation_depth = 0.15  # More pitch variation for interest
        
        # Generate slow sine wave with gentle modulation
        modulation = modulation_depth * np.sin(2 * math.pi * modulation_rate * t_array)
        freq_curve = base_freq * (1.0 + modulation)
        phase = np.cumsum(freq_curve / sample_rate) * 2 * math.pi
        
        # Pure sine wave (no harmonics for pleasant sound)
        samples = np.sin(phase).astype(np.float32)
        
        # No fade - let pygame handle seamless looping
        # Removing fade to ensure sound is audible and loops properly
        
        # Normalize and clip
        max_amp = np.max(np.abs(samples))
        if max_amp > 0:
            samples = samples / max_amp * 1.0  # Full amplitude for maximum audibility
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        
        # Convert to stereo
        stereo = np.stack([samples, samples], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)

        sound = pygame.sndarray.make_sound(stereo_int16)
        if sound:
            volume = 1.0  # Full volume for maximum audibility
            sound.set_volume(volume)
        return sound

    def start_critical_warning(self) -> None:
        """Start the looping critical warning tone."""
        if not _audible():
            return

        if self.critical_warning_sound is None:
            self.critical_warning_sound = self._generate_critical_warning_sound()

        if not self.critical_warning_sound:
            return

        if self.critical_warning_channel is None:
            self.critical_warning_channel = pygame.mixer.Channel(5)
            self.critical_warning_channel.set_volume(1.0)  # Ensure channel volume is at maximum

        channel_busy = self.critical_warning_channel.get_busy()
        if not channel_busy:
            # Play the warning sound with infinite loops
            self.critical_warning_channel.play(self.critical_warning_sound, loops=-1)

    def stop_critical_warning(self) -> None:
        """Stop the critical warning tone."""
        if not config.SOUND_ENABLED or not self.critical_warning_channel:
            return

        if self.critical_warning_channel.get_busy():
            self.critical_warning_channel.stop()
    
    def _generate_star_gained_sound(self) -> Optional[pygame.mixer.Sound]:
        """Generate an optimistic ascending tone for when a star is gained.
        
        Creates an uplifting ascending tone that conveys achievement.
        
        Returns:
            pygame.mixer.Sound object with ascending tone, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.5  # Half second for a clear ascending tone
        num_samples = int(sample_rate * duration)
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        progress = t_array / duration
        
        # Ascending frequency sweep (optimistic, going up)
        start_freq = 330.0  # E4 note
        end_freq = 495.0    # B4 note (perfect fifth up)
        # Use exponential curve for more natural ascent
        freq_curve = start_freq * np.power(end_freq / start_freq, progress)
        
        # Generate tone with slight vibrato for expressiveness
        vibrato = 0.06 * np.sin(2 * math.pi * 5.0 * t_array)
        phase = np.cumsum(freq_curve * (1.0 + vibrato) / sample_rate) * 2 * math.pi
        
        # Main tone with harmonics for brightness
        tone = np.sin(phase)
        tone += 0.35 * np.sin(phase * 2)  # Second harmonic
        tone += 0.18 * np.sin(phase * 3)  # Third harmonic
        
        # Envelope: quick attack, smooth decay with slight sustain
        attack = np.clip(progress / 0.08, 0.0, 1.0)
        decay = np.exp(-np.maximum(progress - 0.15, 0.0) * 2.5)
        sustain = 0.9 - 0.3 * progress
        envelope = attack * decay * sustain
        
        # Apply envelope and normalize
        samples = tone * envelope
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        
        # Convert to stereo
        stereo = np.stack([samples, samples], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_int16)
        if sound:
            sound.set_volume(config.SHOOT_SOUND_VOLUME * 0.7)  # Moderate volume
        return sound
    
    def play_star_lost(self) -> None:
        """Play sad descending tone when a star is lost."""
        if not _audible():
            return
        
        sound = self._generate_star_lost_sound()
        if sound:
            # Use channel 5 for star feedback sounds
            star_channel = pygame.mixer.Channel(5)
            star_channel.play(sound)
    
    def play_star_gained(self) -> None:
        """Play optimistic ascending tone when a star is gained."""
        if not _audible():
            return
        
        sound = self._generate_star_gained_sound()
        if sound:
            # Use channel 5 for star feedback sounds
            star_channel = pygame.mixer.Channel(5)
            star_channel.play(sound)
    
    def _generate_portal_power_up(self) -> Optional[pygame.mixer.Sound]:
        """Generate an ascending power-up sound for portal activation.
        
        Creates an optimistic, ascending tone with white noise for a whoosh effect.
        
        Returns:
            pygame.mixer.Sound object with power-up sound, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.6  # Slightly longer for a satisfying power-up
        num_samples = int(sample_rate * duration)
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        progress = t_array / duration
        
        # Ascending frequency sweep (optimistic, going up)
        start_freq = 200.0  # Lower start for more impact
        end_freq = 600.0    # Higher end for brightness
        # Use exponential curve for more natural ascent
        freq_curve = start_freq * np.power(end_freq / start_freq, progress)
        
        # Generate tone with slight vibrato for expressiveness
        vibrato = 0.08 * np.sin(2 * math.pi * 5.0 * t_array)
        phase = np.cumsum(freq_curve * (1.0 + vibrato) / sample_rate) * 2 * math.pi
        
        # Main tone with harmonics for brightness
        tone = np.sin(phase)
        tone += 0.4 * np.sin(phase * 2)  # Second harmonic
        tone += 0.2 * np.sin(phase * 3)  # Third harmonic
        tone += 0.1 * np.sin(phase * 4)  # Fourth harmonic for sparkle
        
        # Generate white noise for whoosh effect
        white_noise = np.random.normal(0.0, 1.0, num_samples).astype(np.float32)
        
        # Low-pass filter the noise to follow the frequency sweep (whoosh effect)
        # Filter cutoff follows the frequency curve
        noise_cutoff_base = freq_curve * 2.0  # Noise extends higher than tone
        # Apply low-pass filtering using convolution (simple moving average)
        # Filter width inversely related to cutoff frequency
        filter_widths = np.clip((sample_rate / (noise_cutoff_base * 4)).astype(np.int64), 3, 50)
        filtered_noise = self._moving_average(white_noise, filter_widths)
        
        # Mix tone and noise (50% tone, 50% noise for strong whoosh)
        tone_mix = 0.5
        noise_mix = 0.5
        mixed = tone * tone_mix + filtered_noise * noise_mix
        
        # Envelope: quick attack, smooth decay with sustain
        attack = np.clip(progress / 0.1, 0.0, 1.0)
        decay = np.exp(-np.maximum(progress - 0.2, 0.0) * 2.0)
        sustain = 0.95 - 0.3 * progress
        envelope = attack * decay * sustain
        
        # Apply envelope and normalize
        samples = mixed * envelope
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        
        # Convert to stereo
        stereo = np.stack([samples, samples], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_int16)
        if sound:
            sound.set_volume(config.POWERUP_ACTIVATION_SOUND_VOLUME * 0.9)
        return sound
    
    def _generate_portal_power_down(self) -> Optional[pygame.mixer.Sound]:
        """Generate a descending power-down sound for portal deactivation.
        
        Creates a descending, deactivation tone with white noise for a whoosh effect.
        
        Returns:
            pygame.mixer.Sound object with power-down sound, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 0.5  # Shorter for a quick deactivation
        num_samples = int(sample_rate * duration)
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        progress = t_array / duration
        
        # Descending frequency sweep (going down)
        start_freq = 500.0  # Higher start
        end_freq = 200.0    # Lower end
        # Use exponential curve for more natural descent
        freq_curve = start_freq * np.power(end_freq / start_freq, progress)
        
        # Generate tone with slight vibrato
        vibrato = 0.06 * np.sin(2 * math.pi * 4.0 * t_array)
        phase = np.cumsum(freq_curve * (1.0 + vibrato) / sample_rate) * 2 * math.pi
        
        # Main tone with fewer harmonics for a more muted sound
        tone = np.sin(phase)
        tone += 0.25 * np.sin(phase * 2)  # Second harmonic (reduced)
        tone += 0.1 * np.sin(phase * 3)   # Third harmonic (reduced)
        
        # Generate white noise for whoosh effect
        white_noise = np.random.normal(0.0, 1.0, num_samples).astype(np.float32)
        
        # Low-pass filter the noise to follow the frequency sweep (whoosh effect)
        # Filter cutoff follows the frequency curve
        noise_cutoff_base = freq_curve * 2.0  # Noise extends higher than tone
        # Apply low-pass filtering using convolution (simple moving average)
        filter_widths = np.clip((sample_rate / (noise_cutoff_base * 4)).astype(np.int64), 3, 50)
        filtered_noise = self._moving_average(white_noise, filter_widths)
        
        # Mix tone and noise (50% tone, 50% noise for strong whoosh)
        tone_mix = 0.5
        noise_mix = 0.5
        mixed = tone * tone_mix + filtered_noise * noise_mix
        
        # Envelope: quick attack, faster decay
        attack = np.clip(progress / 0.05, 0.0, 1.0)
        decay = np.exp(-progress * 4.0)  # Faster decay
        envelope = attack * decay
        
        # Apply envelope and normalize
        samples = mixed * envelope
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        
        # Convert to stereo
        stereo = np.stack([samples, samples], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_int16)
        if sound:
            sound.set_volume(config.POWERUP_ACTIVATION_SOUND_VOLUME * 0.8)
        return sound
    
    def play_portal_power_up(self) -> None:
        """Play ascending tone when portal activates."""
        if not _audible():
            return
        
        if self.portal_power_up_sound is None:
            self.portal_power_up_sound = self._generate_portal_power_up()
        
        if not self.portal_power_up_sound:
            return
        
        # Use channel 6 for portal sounds
        portal_channel = pygame.mixer.Channel(6)
        portal_channel.play(self.portal_power_up_sound)
    
    def play_portal_power_down(self) -> None:
        """Play descending tone when portal deactivates."""
        if not _audible():
            return
        
        if self.portal_power_down_sound is None:
            self.portal_power_down_sound = self._generate_portal_power_down()
        
        if not self.portal_power_down_sound:
            return
        
        # Use channel 6 for portal sounds
        portal_channel = pygame.mixer.Channel(6)
        portal_channel.play(self.portal_power_down_sound)
    
    def _generate_tweet_sound(self) -> Optional[pygame.mixer.Sound]:
        """Generate a bird-like tweeting/chirping sound.
        
        Creates a short, pleasant chirp with frequency variation to sound like a bird.
        
        Returns:
            pygame.mixer.Sound object with tweet sound, or None if sounds disabled.
        """
        if not config.SOUND_ENABLED:
            return None
        
        sample_rate = config.SOUND_SAMPLE_RATE
        duration = random.uniform(0.08, 0.15)  # Short, variable duration
        num_samples = int(sample_rate * duration)
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        progress = t_array / duration
        
        # Base frequency with variation (bird-like chirp range: 2000-4000 Hz)
        base_freq = random.uniform(2000.0, 4000.0)
        
        # Frequency modulation for chirp character (quick up-down)
        # Create a quick frequency sweep up then down using numpy where
        freq_mult = np.where(
            progress < 0.3,
            1.0 + (progress / 0.3) * 0.3,  # Quick rise: Rise 30%
            np.where(
                progress < 0.7,
                1.3 - ((progress - 0.3) / 0.4) * 0.2,  # Peak and slight fall: Fall from 1.3 to 1.1
                1.1 - ((progress - 0.7) / 0.3) * 0.2  # Final fall: Fall to 0.9
            )
        )
        
        freq_curve = base_freq * freq_mult
        
        # Add vibrato for natural bird sound
        vibrato_rate = random.uniform(8.0, 15.0)  # Fast vibrato
        vibrato_depth = random.uniform(0.05, 0.12)
        vibrato = vibrato_depth * np.sin(2 * math.pi * vibrato_rate * t_array)
        
        # Generate phase with frequency modulation
        phase = np.cumsum(freq_curve * (1.0 + vibrato) / sample_rate) * 2 * math.pi
        
        # Main tone with harmonics for richness
        tone = np.sin(phase)
        tone += 0.3 * np.sin(phase * 2)  # Second harmonic
        tone += 0.15 * np.sin(phase * 3)  # Third harmonic
        
        # Add slight noise for texture (bird sounds aren't pure tones)
        noise = np.random.normal(0.0, 0.15, num_samples).astype(np.float32)
        tone = tone + noise
        
        # Envelope: quick attack, smooth decay
        attack = np.clip(progress / 0.15, 0.0, 1.0)
        decay = np.exp(-np.maximum(progress - 0.2, 0.0) * 4.0)
        envelope = attack * decay
        
        # Apply envelope and normalize
        samples = tone * envelope
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        
        # Convert to stereo
        stereo = np.stack([samples, samples], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_int16)
        if sound:
            sound.set_volume(config.SHOOT_SOUND_VOLUME * 0.4)  # Quieter than shoot
        return sound
    
    def play_tweet(self) -> None:
        """Play a random bird-like tweet/chirp sound.
        
        Generates a new tweet sound each time with random characteristics.
        """
        if not _audible():
            return
        
        sound = self._generate_tweet_sound()
        if sound:
            # Use channel 7 for tweet sounds (or find an available channel)
            # Try to use a channel that won't interfere with other sounds
            tweet_channel = pygame.mixer.Channel(7)
            tweet_channel.play(sound)

    def _generate_power_down_sound(self) -> Optional[pygame.mixer.Sound]:
        """Generate a descending power-down sound like an electric motor spinning down."""
        if not config.SOUND_ENABLED:
            return None

        sample_rate = config.SOUND_SAMPLE_RATE
        duration = 2.0  # 2 seconds for dramatic power-down
        num_samples = int(sample_rate * duration)
        t_array = np.arange(num_samples, dtype=np.float32) / sample_rate
        progress = t_array / duration

        # Descending frequency sweep (motor spinning down)
        start_freq = 400.0  # Start at mid-range
        end_freq = 80.0     # End at low rumble
        freq_curve = start_freq * np.power(end_freq / start_freq, progress)

        # Add slight vibrato for motor-like character
        vibrato_rate = 8.0  # Hz
        vibrato_depth = 0.05
        vibrato = vibrato_depth * np.sin(2 * math.pi * vibrato_rate * t_array)

        # Generate phase with frequency modulation
        phase = np.cumsum(freq_curve * (1.0 + vibrato) / sample_rate) * 2 * math.pi

        # Main tone with harmonics for motor character
        tone = np.sin(phase)
        tone += 0.4 * np.sin(phase * 2)   # Second harmonic
        tone += 0.2 * np.sin(phase * 3)   # Third harmonic
        tone += 0.1 * np.sin(phase * 4)   # Fourth harmonic

        # Add filtered noise for motor texture
        white_noise = np.random.normal(0.0, 1.0, num_samples).astype(np.float32)
        filter_widths = np.clip((sample_rate / (freq_curve * 2)).astype(np.int64), 5, 100)
        filtered_noise = self._moving_average(white_noise, filter_widths)

        # Mix tone and noise (motor has both tonal and noisy components)
        tone_mix = 0.7
        noise_mix = 0.3
        mixed = tone * tone_mix + filtered_noise * noise_mix

        # Envelope: quick attack, then slow decay (motor takes time to spin down)
        attack = np.clip(progress / 0.05, 0.0, 1.0)  # Quick start
        decay = np.exp(-progress * 1.5)
        envelope = attack * decay

        # Apply envelope and normalize
        samples = mixed * envelope
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)

        # Convert to stereo
        stereo = np.stack([samples, samples], axis=1)
        stereo_int16 = (stereo * 16383).astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_int16)
        if sound:
            sound.set_volume(config.SHOOT_SOUND_VOLUME * 0.8)  # Moderate volume
        return sound

    def play_power_down(self) -> None:
        """Play power-down sound when ship runs out of energy."""
        if not _audible():
            return

        sound = self._generate_power_down_sound()
        if sound:
            # Use channel 2 (enemy destroy channel), safe during game over
            power_down_channel = pygame.mixer.Channel(2)
            power_down_channel.play(sound)


