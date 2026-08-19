"""跨 torch/mlx 共享的后端配置默认表(was 各 module 内嵌副本)。

只放纯数据表 + 纯函数,不 import 任何 usvc 子包(防循环)。
表中的 _* 目标是:改一处配置,torch 与 mlx 训练/推理路径同步生效。
"""

DDSP6_VERSION_DEFAULTS = {
    "6.1": dict(velocity_net="LYNXNet", n_aux_layers=3, n_aux_chans=256,
                n_layers=6, n_chans=512, t_start=0.7, use_norm=False),
    "6.2": dict(velocity_net="LYNXNet", n_aux_layers=6, n_aux_chans=512,
                n_layers=6, n_chans=1024, t_start=0.0, use_norm=True),
    "6.3": dict(velocity_net="LYNXNet2", n_aux_layers=6, n_aux_chans=512,
                n_layers=6, n_chans=1024, t_start=0.0, use_norm=True),
}


def resolve_ddsp6_opts(opts: dict) -> dict:
    """Version defaults overlaid by explicit backend_opts.

    Returns a dict containing the version defaults PLUS every other key present
    in ``opts`` (n_units / num_speaker / out_dims / sampling_rate / hop_length
    / etc.), so downstream ``opts.get(...)`` calls can actually read user
    overrides. Previously only keys already in the version-defaults dict
    survived, which silently dropped multi-speaker / units-dim overrides.
    """
    version = str(opts.get("version", "6.3"))
    if version not in DDSP6_VERSION_DEFAULTS:
        raise ValueError(
            f"ddsp6 version must be one of {list(DDSP6_VERSION_DEFAULTS)}, got {version!r}"
        )
    resolved = dict(DDSP6_VERSION_DEFAULTS[version])
    # Merge ALL user opts on top of the version defaults (user wins). This keeps
    # the version-specific knobs (velocity_net / n_layers / ...) overridable while
    # also passing through architecture keys not listed in DDSP6_VERSION_DEFAULTS.
    for k, v in opts.items():
        if k != "version":
            resolved[k] = v
    return resolved


# RVC generator upsample 默认表(镜像 torch RVCModule._RVC_UPSAMPLE)。
# 查表语义(ValueError 包装 / 直接 KeyError)由各调用方保留,这里只存数据。
RVC_UPSAMPLE = {
    ("v1", 32000): ([10, 4, 2, 2, 2], [16, 16, 4, 4, 4]),
    ("v1", 40000): ([10, 10, 2, 2], [16, 16, 4, 4]),
    ("v1", 48000): ([10, 6, 2, 2, 2], [16, 16, 4, 4, 4]),
    ("v2", 32000): ([10, 8, 2, 2], [20, 16, 4, 4]),
    ("v2", 48000): ([12, 10, 2, 2], [24, 20, 4, 4]),
}