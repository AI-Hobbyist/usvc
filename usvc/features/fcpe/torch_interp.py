import torch
'''use from https://github.com/autumn-DL/TorchInterp (MIT license)'''


def torch_interp(x, xp, fp):
    sort_idx = torch.argsort(xp)
    xp, fp = xp[sort_idx], fp[sort_idx]
    right_idxs = torch.searchsorted(xp, x).clamp(max=len(xp) - 1)
    left_idxs = (right_idxs - 1).clamp(min=0)
    interp_vals = fp[left_idxs] + ((x - xp[left_idxs]) * (fp[right_idxs] - fp[left_idxs]) / (xp[right_idxs] - xp[left_idxs]))
    interp_vals[x < xp[0]] = fp[0]
    interp_vals[x > xp[-1]] = fp[-1]
    return interp_vals


def batch_interp_with_replacement_detach(uv, f0):  # uv B T, f0 B T -> interp f0 over voiced frames
    result = f0.clone()
    for i in range(uv.shape[0]):
        x = torch.where(uv[i])[-1]
        voiced = ~uv[i]
        result[i][uv[i]] = torch_interp(x, torch.where(voiced)[-1], f0[i][voiced]).detach()
    return result