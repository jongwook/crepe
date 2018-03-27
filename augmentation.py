import numpy as np
from random import Random

random = Random(42)


def add_noise(audio, pitch):
    if random.random() < 0.5:
        return audio, pitch

    level = random.uniform(0, 0.1)
    noise = np.random.randn(*audio.shape) * level
    return audio + noise, pitch
