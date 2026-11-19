"""Scripted swimming for the sea snake: a travelling wave plus a steering bias on the front joints."""
import math

from physics.sea_snake import MUSCLES, SeaSnakeCommand


WAVE_LAG = 0.8        # radians of wave phase between neighbouring joints
STEER_JOINTS = 2      # the joints nearest the head carry the steering bias
IDLE_AMPLITUDE = 0.3  # gentle wave when there is nothing to chase
FULL_TURN = math.pi/2  # heading error at which the wave is at its smallest
MIN_AMPLITUDE = 0.7


class SeaSnakeController:
    def __init__(self, wavelength_frames=32.):
        self.phase_speed = 2*math.pi/wavelength_frames
        self.frame = 0.

    def command(self, physics, dt, target):
        self.frame += dt
        phase = -self.frame*self.phase_speed
        if target is None:
            amplitude, bias = IDLE_AMPLITUDE, 0.
        else:
            hx, hy = physics.head
            error = math.atan2(target[1]-hy, target[0]-hx)-physics.heading
            error = math.atan2(math.sin(error), math.cos(error))
            bias = max(-1., min(1., error/FULL_TURN*2))
            amplitude = 1.-(1.-MIN_AMPLITUDE)*min(1., abs(error)/FULL_TURN)
        muscles = []
        for i in range(MUSCLES):
            steer = bias if i >= MUSCLES-STEER_JOINTS else 0.
            muscles.append(max(-1., min(1., amplitude*math.sin(phase-i*WAVE_LAG)+steer)))
        return SeaSnakeCommand(tuple(muscles))
