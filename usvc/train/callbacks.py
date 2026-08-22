"""ScheduleFree optimizer train/eval mode switching (AdamWScheduleFree needs full-param train() / averaged-param eval())."""
import lightning.pytorch as pl


class ScheduleFreeModeCallback(pl.Callback):
    def _sf(self, trainer, pl_module, mode):
        opt = pl_module.optimizers()
        if hasattr(opt, mode):
            getattr(opt, mode)()

    def on_train_epoch_start(self, trainer, pl_module): self._sf(trainer, pl_module, "train")
    def on_validation_start(self, trainer, pl_module): self._sf(trainer, pl_module, "eval")
    def on_validation_end(self, trainer, pl_module): self._sf(trainer, pl_module, "train")
    def on_test_start(self, trainer, pl_module): self._sf(trainer, pl_module, "eval")
    def on_test_end(self, trainer, pl_module): self._sf(trainer, pl_module, "train")