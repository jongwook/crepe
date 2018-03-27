import numpy as np


def accuracies(true_cents, predicted_cents, cent_tolerence=50):
    from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy
    assert true_cents.shape == predicted_cents.shape

    voicing = np.ones(true_cents.shape)
    rpa = raw_pitch_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerence)
    rca = raw_chroma_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerence)
    return rpa, rca
