import numpy as np
import librosa
import torch
from .constants import *


def to_local_average_cents(salience, center=None, thred=0.03):
    if not hasattr(to_local_average_cents, 'cents_mapping'):  # bin-number -> cents mapping (function-attr cache)
        to_local_average_cents.cents_mapping = 20 * np.arange(N_CLASS) + CONST
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i], None, thred) for i in range(salience.shape[0])])
    if center is None: center = int(np.argmax(salience))
    start, end = max(0, center - 4), min(len(salience), center + 5)
    salience = salience[start:end]
    return np.sum(salience * to_local_average_cents.cents_mapping[start:end]) / np.sum(salience) if np.max(salience) > thred else 0


def to_viterbi_cents(salience, thred=0.03):
    if not hasattr(to_viterbi_cents, 'transition'):  # viterbi transition matrix (function-attr cache)
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        to_viterbi_cents.transition = transition / transition.sum(axis=1, keepdims=True)
    prob = salience.T / salience.T.sum(axis=0)
    path = librosa.sequence.viterbi(prob, to_viterbi_cents.transition).astype(np.int64)
    return np.array([to_local_average_cents(salience[i], path[i], thred) for i in range(len(path))])


def to_local_average_f0(hidden, center=None, thred=0.03):
    # weighted average of the 9 cents bins around the argmax (idx*20+CONST)
    idx_cents = torch.arange(N_CLASS, device=hidden.device)[None, None, :] * 20 + CONST
    if center is None: center = torch.argmax(hidden, dim=2, keepdim=True)
    idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]
    idx_mask = (idx >= torch.clip(center - 4, min=0)) & (idx < torch.clip(center + 5, max=N_CLASS))
    w = hidden * idx_mask
    product_sum, weight_sum = torch.sum(w * idx_cents, dim=2), torch.sum(w, dim=2)
    f0 = 10 * 2 ** (product_sum / (weight_sum + (weight_sum == 0)) / 1200)  # /0 guarded, then uv-masked
    return (f0 * (hidden.max(dim=2)[0] >= thred)).squeeze(0).cpu().numpy()


def to_viterbi_f0(hidden, thred=0.03):
    if not hasattr(to_viterbi_cents, 'transition'):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        to_viterbi_cents.transition = transition / transition.sum(axis=1, keepdims=True)
    prob = hidden.squeeze(0).cpu().numpy().T
    prob = prob / prob.sum(axis=0)
    path = librosa.sequence.viterbi(prob, to_viterbi_cents.transition).astype(np.int64)
    return to_local_average_f0(hidden, torch.from_numpy(path).unsqueeze(0).unsqueeze(-1).to(hidden.device), thred)