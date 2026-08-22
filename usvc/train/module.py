"""Base LightningModule for non-GAN SVC backends (rift_dit, ddsp6, diffusion); subclass implements forward_loss(batch)->(loss, sub) and sample(batch)->mel."""
from lightning.pytorch import LightningModule
import torch

from .config import TrainConfig
from .metrics import compute_all_metrics


class BaseSVCModule(LightningModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = True
        self.vocoder = None  # lazily loaded for validation
        self._val_metrics = []

    def training_step(self, batch, batch_idx):
        loss, sub = self.forward_loss(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=self._bs(batch))
        for name, val in sub.items():
            self.log(f"train/{name}", val, batch_size=self._bs(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.trainer.is_global_zero: return
        with torch.no_grad(): mel_gen = self.sample(batch)
        mel_gt = batch.mel
        if mel_gt is None or mel_gen is None: return
        mel_gen, mel_gt = mel_gen.float(), mel_gt.float()
        fl = batch.frame_len
        for i in range(mel_gen.shape[0]):
            n = fl[i].item() if fl is not None else mel_gen.shape[1]  # true (unpadded) frames
            self._val_metrics.append({k: v.item() for k, v in
                compute_all_metrics(mel_gen[i, :n], mel_gt[i, :n]).items()})

    def on_validation_epoch_end(self):
        if not self._val_metrics: return
        import numpy as np
        for k in self._val_metrics[0]:
            self.log(f"val/{k}", float(np.mean([m[k] for m in self._val_metrics])))
        self._val_metrics.clear()

    def _sf(self, mode):  # ScheduleFree optimizer train/eval switching
        opt = self.optimizers()
        if hasattr(opt, mode): getattr(opt, mode)()

    def on_train_epoch_start(self): self._sf("train")
    def on_validation_start(self): self._sf("eval")
    def on_validation_end(self): self._sf("train")
    def on_test_start(self): self._sf("eval")
    def on_test_end(self): self._sf("train")

    def configure_optimizers(self):
        from .optimizer import build_optimizer
        return build_optimizer(self.model.parameters(), self.cfg.optimizer, self.cfg.schedule, self.cfg.max_steps)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # torch-native clip: lightning's AMP plugin unscales gradients internally, fused AdamW does it itself
        if gradient_clip_val is not None:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], gradient_clip_val)

    def forward_loss(self, batch): raise NotImplementedError
    def sample(self, batch): raise NotImplementedError

    @staticmethod
    def _unpack(batch):
        volume = batch.rms
        if volume is None:  # volume falls back to mel energy when rms is absent
            volume = (batch.mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)
        return batch.units, batch.f0, volume, batch.spk_id, batch.mel, batch.aug_shift

    @staticmethod
    def _bs(batch): return batch.units.shape[0]