"""MLX 训练栈:函数式训练循环 + MLX 优化器 + MLX objective
(与 usvc/train/ torch Lightning 版并行)。"""
from .gan import MlxGANTrainer
from .objective import FlowMatchingObjective, DDPMObjective
from .optimizer import build_optimizer, build_optimizer_gan, build_lr_schedule
from .trainer import MlxTrainer
__all__ = ["FlowMatchingObjective", "DDPMObjective", "build_optimizer",
           "build_optimizer_gan", "build_lr_schedule", "MlxTrainer",
           "MlxGANTrainer"]
