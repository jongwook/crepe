import numpy as np
from random import Random
import pyrubberband as pyrb

random = Random(42)


def add_noise(audio, pitch):
    if random.random() < 0.5:
        return audio, pitch

    level = random.uniform(0, 0.1)
    noise = np.random.randn(*audio.shape) * level
    return audio + noise, pitch


def pitch_shift(audio, pitch):
    if random.random() < 0.2:
        return audio, pitch

    amount = random.uniform(-2, 2)
    audio = pyrb.pitch_shift(audio, 16000, amount)
    pitch = pitch * 2.0 ** (amount / 12)

    return audio, pitch
