import pathlib

import torch
from .mel_extractor import Wav2MelModule


class DotDict(dict):
    def __getattr__(*a):
        v = dict.get(*a)
        return DotDict(v) if type(v) is dict else v
    __setattr__, __delattr__ = dict.__setitem__, dict.__delitem__


def _opt(x, default, what):  # warn + default when config key missing
    if x is None:
        print(f'  [WARN] {what} is None; use default {default}')
        return default
    return x


def _must(x, what):
    if x is None:
        print(f'  [ERROR] {what} is None')
        raise ValueError(f'  [ERROR] {what} is None')
    return x


def get_device(device, func_name):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'  [INFO]: Using {device} automatically.')
    else:
        print(f'  [INFO]: device is not None, use {device}')
    if (device == 'cuda' and not torch.cuda.is_available()) or (device == 'mps' and not torch.backends.mps.is_available()):
        print(f'  [WARN]: Specified device ({device}) is not available, switching to cpu.')
        device = 'cpu'
    return device


def spawn_wav2mel(args, device=None):
    m_type = args.mel.type
    lt = str(m_type).lower()
    if lt in ('none', 'default'):
        m_type = 'default'
    elif lt == 'stft':
        m_type = 'stft'
    else:
        raise ValueError(f'  [ERROR] torchfcpe.tools.args.spawn_wav2mel: {m_type} is not a supported args.mel.type')
    wav2mel = Wav2MelModule(
        sr=_opt(args.mel.sr, 16000, 'args.mel.sr'),
        n_mels=_opt(args.mel.num_mels, 128, 'args.mel.num_mels'),
        n_fft=_opt(args.mel.n_fft, 1024, 'args.mel.n_fft'),
        win_size=_opt(args.mel.win_size, 1024, 'args.mel.win_size'),
        hop_length=_opt(args.mel.hop_size, 160, 'args.mel.hop_size'),
        fmin=_opt(args.mel.fmin, 0, 'args.mel.fmin'),
        fmax=_opt(args.mel.fmax, 8000, 'args.mel.fmax'),
        clip_val=1e-05,
        mel_type=m_type,
    )
    return wav2mel.to(torch.device(_opt(device, 'cpu', '.device')))