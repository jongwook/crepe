import numpy as np


def raw_pitch_accuracy(true_cents, predicted_cents, cent_tolerence=50):
    from mir_eval.melody import raw_pitch_accuracy
    assert true_cents.shape == predicted_cents.shape

    voicing = np.ones(true_cents.shape)
    return raw_pitch_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerence)
