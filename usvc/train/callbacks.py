"""ScheduleFree optimizer mode-switching callback.

AdamWScheduleFree requires train()/eval() calls to switch between
full-param (train) and averaged-param (eval) modes. This callback
handles the switching automatically.
"""
from __future__ import annotations

import lightning.pytorch as pl


class ScheduleFreeModeCallback(pl.Callback):
    """Switch ScheduleFree optimizer between train/eval modes."""

    def on_train_epoch_start(self, trainer, pl_module):
        opt = pl_module.optimizers()
        if hasattr(opt, "train"):
            opt.train()

    def on_validation_start(self, trainer, pl_module):
        opt = pl_module.optimizers()
        if hasattr(opt, "eval"):
            opt.eval()

    def on_validation_end(self, trainer, pl_module):
        opt = pl_module.optimizers()
        if hasattr(opt, "train"):
            opt.train()

    def on_test_start(self, trainer, pl_module):
        opt = pl_module.optimizers()
        if hasattr(opt, "eval"):
            opt.eval()

    def on_test_end(self, trainer, pl_module):
        opt = pl_module.optimizers()
        if hasattr(opt, "train"):
            opt.train()
