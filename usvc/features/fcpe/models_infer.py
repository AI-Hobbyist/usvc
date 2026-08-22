import pathlib

import torch

from .models import CFNaiveMelPE
from .tools import DotDict, _must, _opt, get_device, spawn_wav2mel
from .torch_interp import batch_interp_with_replacement_detach


def ensemble_f0(f0s, key_shift_list, tta_uv_penalty):
    # DP over key-shift candidates per frame: uv penalty p=tta_uv_penalty**2
    # (v->uv costs extra 2p), L2 gap between adjacent voiced candidates (ignored
    # when < 0.5); backtrack picks the min-cost path; f0 normalized by keyshift.
    device = f0s.device
    f0s = f0s / torch.pow(2, torch.tensor(key_shift_list, device=device).unsqueeze(0).unsqueeze(0) / 12)
    notes = torch.log2(f0s / 440) * 12 + 69
    notes[notes < 0] = 0
    uv_penalty = tta_uv_penalty ** 2
    dp = torch.zeros_like(notes)
    backtrack = torch.zeros_like(notes).long()
    dp[:, 0, :] = (notes[:, 0, :] <= 0) * uv_penalty
    for t in range(1, notes.size(1)):
        t_uv, t1_uv = notes[:, t, :] <= 0, notes[:, t - 1, :] <= 0
        penalty = uv_penalty * t_uv.unsqueeze(1)
        l2 = torch.pow((notes[:, t - 1, :].unsqueeze(-1) - notes[:, t, :].unsqueeze(1)) * (~t1_uv).unsqueeze(-1) * (~t_uv).unsqueeze(1), 2)
        l2 = l2 - 0.5
        l2 = l2 * (l2 > 0)  # mask AFTER subtracting 0.5 (original op order, bitwise contract)
        penalty = penalty + l2 + t1_uv.unsqueeze(-1) * (~t_uv).unsqueeze(1) * uv_penalty * 2
        dp[:, t, :], backtrack[:, t, :] = torch.min(dp[:, t - 1, :].unsqueeze(-1) + penalty, dim=1)
    t = f0s.size(1) - 1
    f0_result = torch.zeros_like(f0s[:, :, 0])
    min_indices = torch.argmin(dp[:, t, :], dim=-1)
    for i in range(0, t + 1):
        f0_result[:, t - i] = f0s[:, t - i, min_indices]
        min_indices = backtrack[:, t - i, min_indices]
    return f0_result.unsqueeze(-1)


class InferCFNaiveMelPE(torch.nn.Module):
    def __init__(self, args, state_dict):
        super().__init__()
        self.wav2mel = spawn_wav2mel(args, device="cpu")
        self.model = spawn_model(args)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.args = args
        self.register_buffer("tensor_device_marker", torch.tensor(1.0).float(), persistent=False)  # device/dtype markers (wav2mel stays fp32)
        self.register_buffer("tensor_dtype_marker", torch.tensor(1.0).float(), persistent=False)

    def forward(self, wav, sr, decoder_mode="local_argmax", threshold=0.006, key_shifts=[0]):
        with torch.no_grad():
            wav = wav.to(self.tensor_device_marker.device)
            mels = torch.stack([self.wav2mel(wav, sr, keyshift=keyshift) for keyshift in key_shifts], -1)  # B T C K -> (B K) T C
            mels = mels.permute(0, 3, 1, 2).reshape(-1, mels.shape[1], mels.shape[2]).to(self.tensor_dtype_marker.dtype)
            f0s = self.model.infer(mels, decoder=decoder_mode, threshold=threshold)
            return f0s.view(f0s.shape[0] // len(key_shifts), len(key_shifts), *f0s.shape[1:]).transpose(1, 2).squeeze(-1)  # (B K) T 1 -> B T K

    def infer(self, wav, sr, decoder_mode="local_argmax", threshold=0.006, f0_min=None, f0_max=None, interp_uv=False,
              output_interp_target_length=None, return_uv=False, test_time_augmentation=False, tta_uv_penalty=12.0,
              tta_key_shifts=[0, -12, 12], tta_use_origin_uv=False):
        if test_time_augmentation:
            assert len(tta_key_shifts) > 0
            flag = 0
            if tta_use_origin_uv and 0 not in tta_key_shifts:
                flag = 1
                tta_key_shifts.append(0)
            tta_key_shifts.sort(key=lambda x: (x if x >= 0 else -x / 2))
            f0s = self.__call__(wav, sr, decoder_mode, threshold, tta_key_shifts)
            f0 = ensemble_f0(f0s[:, :, flag:], tta_key_shifts[flag:], tta_uv_penalty)
            f0_for_uv = f0s[:, :, [0]] if tta_use_origin_uv else f0
        else:
            f0 = self.__call__(wav, sr, decoder_mode, threshold)
            f0_for_uv = f0
        if f0_min is None: f0_min = self.args.model.f0_min
        uv = (f0_for_uv < f0_min).type(f0_for_uv.dtype)
        f0 = f0 * (1 - uv)
        if interp_uv:
            f0 = batch_interp_with_replacement_detach(uv.squeeze(-1).bool(), f0.squeeze(-1)).unsqueeze(-1)
        if f0_max is not None:
            f0[f0 > f0_max] = f0_max
        if output_interp_target_length is not None:  # nan-gap linear interp to target length
            f0 = torch.where(f0 == 0, float("nan"), f0)
            f0 = torch.nn.functional.interpolate(f0.transpose(1, 2), size=int(output_interp_target_length), mode="linear").transpose(1, 2)
            f0 = torch.where(f0.isnan(), float(0.0), f0)
        if return_uv:
            uv = torch.nn.functional.interpolate(uv.transpose(1, 2), size=int(output_interp_target_length), mode="nearest").transpose(1, 2)
            return f0, uv
        return f0

    def get_hop_size(self): return self.args.mel.hop_size
    def get_hop_size_ms(self): return self.args.mel.hop_size / self.args.mel.sr * 1000
    def get_model_sr(self): return self.args.mel.sr
    def get_mel_config(self): return dict(self.args.mel)
    def get_device(self): return self.tensor_device_marker.device
    def get_model_f0_range(self): return {"f0_min": self.args.model.f0_min, "f0_max": self.args.model.f0_max}


class InferCFNaiveMelPEONNX:
    def __init__(self, args, onnx_path, device):
        raise NotImplementedError


def spawn_bundled_infer_model(device=None, dtype=None):
    return spawn_infer_model_from_pt(str(pathlib.Path(__file__).parent / "assets" / "fcpe_c_v001.pt"), device, bundled_model=True, dtype=dtype)


def spawn_infer_model_from_pt(pt_path, device=None, bundled_model=False, dtype=None):
    device = get_device(device, "torchfcpe.tools.spawn_infer_cf_naive_mel_pe_from_pt")
    ckpt = torch.load(pt_path, map_location=torch.device(device))
    if bundled_model:
        ckpt["config_dict"]["model"]["conv_dropout"] = 0.0
        ckpt["config_dict"]["model"]["atten_dropout"] = 0.0
    args = DotDict(ckpt["config_dict"])
    if args.is_onnx is True:
        raise ValueError("  [ERROR] spawn_infer_model_from_pt: this model is an onnx model.")
    if args.model.type != "CFNaiveMelPE":
        raise ValueError(f"  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE")
    infer_model = InferCFNaiveMelPE(args, ckpt["model"]).to(device)
    if dtype is not None:
        # only the CFNaiveMelPE model runs at low precision; wav2mel keeps FP32
        infer_model.model = infer_model.model.to(dtype)
        infer_model.tensor_dtype_marker = infer_model.tensor_dtype_marker.to(dtype)
    infer_model.eval()
    return infer_model


def spawn_model(args):
    if args.model.type != "CFNaiveMelPE":
        raise ValueError(f"  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE")
    mel, m = args.mel, args.model
    return CFNaiveMelPE(
        input_channels=_must(mel.num_mels, "args.mel.num_mels"),
        out_dims=_must(m.out_dims, "args.model.out_dims"),
        hidden_dims=_must(m.hidden_dims, "args.model.hidden_dims"),
        n_layers=_must(m.n_layers, "args.model.n_layers"),
        n_heads=_must(m.n_heads, "args.model.n_heads"),
        f0_max=_must(m.f0_max, "args.model.f0_max"),
        f0_min=_must(m.f0_min, "args.model.f0_min"),
        use_fa_norm=_must(m.use_fa_norm, "args.model.use_fa_norm"),
        conv_only=_opt(m.conv_only, False, "args.model.conv_only"),
        conv_dropout=_opt(m.conv_dropout, 0.0, "args.model.conv_dropout"),
        atten_dropout=_opt(m.atten_dropout, 0.0, "args.model.atten_dropout"),
        use_harmonic_emb=_opt(m.use_harmonic_emb, False, "args.model.use_harmonic_emb"),
    )