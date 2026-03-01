"""
PyTorch Lightning module for cardiac segmentation training.

Handles training/validation loops, per-class Dice monitoring,
differential learning rates (encoder vs decoder), and cosine annealing.
"""

import torch
import pytorch_lightning as pl


class CardiacSegModule(pl.LightningModule):
    def __init__(self, model, criterion, lr=3e-4, encoder_lr_factor=0.1,
                 weight_decay=1e-4, epochs=100, num_classes=4):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.encoder_lr_factor = encoder_lr_factor
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, y)

        main_out = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = main_out.argmax(dim=1)
        dice = self._mean_fg_dice(preds, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dice', dice, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        preds = outputs.argmax(dim=1)

        dice = self._mean_fg_dice(preds, y)
        class_dices = self._per_class_dice(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
        for c in range(1, self.num_classes):
            self.log(f'val_dice_c{c}', class_dices[c], prog_bar=False)
        return loss

    def _mean_fg_dice(self, preds, targets):
        dice_sum, count = 0.0, 0
        for c in range(1, self.num_classes):
            pred_c = (preds == c).float()
            tgt_c = (targets == c).float()
            inter = (pred_c * tgt_c).sum()
            union = pred_c.sum() + tgt_c.sum()
            if union > 0:
                dice_sum += 2.0 * inter / (union + 1e-8)
                count += 1
        return dice_sum / max(count, 1)

    def _per_class_dice(self, preds, targets):
        dices = {}
        for c in range(self.num_classes):
            pred_c = (preds == c).float()
            tgt_c = (targets == c).float()
            inter = (pred_c * tgt_c).sum()
            union = pred_c.sum() + tgt_c.sum()
            dices[c] = (2.0 * inter / (union + 1e-8)) if union > 0 else torch.tensor(1.0)
        return dices

    def configure_optimizers(self):
        encoder_params = list(self.model.base.encoder.parameters())
        decoder_params = (
            list(self.model.base.decoder.parameters()) +
            list(self.model.base.segmentation_head.parameters()) +
            list(self.model.aux_heads.parameters())
        )

        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.lr * self.encoder_lr_factor},
            {'params': decoder_params, 'lr': self.lr},
        ], weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]
