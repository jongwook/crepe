import numpy as np
from random import Random

try:
    import pyrubberband.rubberband
    print('native rubberband library loaded')
    def shift(audio, amount):
        return pyrubberband.rubberband.pitch_shift(audio, 16000, amount, rbargs=['--realtime'])
except ImportError:
    import pyrubberband as pyrb
    def shift(audio, amount):
        return pyrb.pitch_shift(audio, 16000, amount)

random = Random(42)


def normalize(audio, pitch):
    audio = audio - np.mean(audio)
    audio /= np.std(audio)
    return audio, pitch


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
    audio = shift(audio, amount)
    pitch = pitch * 2.0 ** (amount / 12)

    return audio, pitch
