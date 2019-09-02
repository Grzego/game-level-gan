import random
import torch

from utils import device
from .race import Race, RaceCar


# ---------------------------------------------------------
# Default params for cars across experiments
# ---------------------------------------------------------


class RaceConfig(object):
    max_segments = 128
    cars = [RaceCar(max_speed=60., acceleration=4., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]


race_game = Race(timeout=40., framerate=1. / 20., cars=RaceConfig.cars)


def predefined_tracks():
    tracks = torch.zeros(6, RaceConfig.max_segments, 2, device=device)

    # zig-zag
    t0 = random.randint(0, 15)
    for i in range(t0, RaceConfig.max_segments, 16):
        tracks[0, i: i + 16, 0] = 2. * ((i // 16) % 2) - 1.

    # one sharp turn
    t1 = random.randint(0, 100)
    tracks[1, t1: t1 + 20, 0] = 1.

    # S track
    t2 = random.randint(0, 70)
    tracks[2, t2: t2 + 12, 0] = 1.
    tracks[2, t2 + 30: t2 + 42, 0] = -1.

    # large U turn
    t3 = [random.randint(0, 5) for _ in range(7)]
    offs = [0, 10, 20, 40, 50, 70, 100]
    for o, t in zip(offs, t3):
        tracks[3, t + o: t + o + 5, 0] = 1.

    # small bumpy turn
    t4 = random.randint(0, 90)
    tracks[4, t4: t4 + 6, 0] = 1.
    tracks[4, t4 + 6: t4 + 18, 0] = -1.
    tracks[4, t4 + 18: t4 + 24, 0] = 1.

    # immediate turn
    t5 = random.randint(30, 60)
    tracks[5, :20, 0] = 1.
    tracks[5, -t5: -t5 + 20, 0] = -1.

    return tracks
