import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from tqdm import tqdm
from typing import Optional

from ..models import SegmentationModel, ModalityDiscriminator


class BiLevelMetaTrainer:
    """
    inner loop: updates only adapter parameters on partial-modality data.
    outer loop: updates base network and meta-generator on full-modality data
                with gradients flowing through the inner-adapted model.
    both loops: update discriminator for modality-invariant representations.
    """
    def __init__(
        self,
        segmentation_model: SegmentationModel,
        discriminator_model: ModalityDiscriminator,
        dataloader_Dm: DataLoader,
        dataloader_Df: DataLoader,
        device: torch.device,
        lr_meta_G: float = 1e-4,
        lr_inner_adapter: float = 1e-3,
        lr_D: float = 1e-4,
        lambda_seg: float = 1.0,
        lambda_adv_g: float = 0.1,
        lambda_reg_adapter: float = 0.01,
        inner_loop_k_steps: int = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1
    ):
        """
        args:
            segmentation_model: main segmentation model (base parameters)
            discriminator_model: adversarial modality discriminator
            dataloader_Dm: partial-modality training data loader
            dataloader_Df: full-modality training data loader
            device: torch device (cuda/cpu)
            lr_meta_G: learning rate for base network and meta-generator
            lr_inner_adapter: learning rate for inner loop adapter updates
            lr_D: learning rate for discriminator
            lambda_seg: weight for segmentation loss
            lambda_adv_g: weight for adversarial generator loss
            lambda_reg_adapter: weight for adapter l2 regularization
            inner_loop_k_steps: number of inner loop gradient steps (K)
            use_amp: enable automatic mixed precision
            accumulation_steps: gradient accumulation steps
        """
        self.segmentation_model = segmentation_model.to(device)
        self.discriminator_model = discriminator_model.to(device)
        self.dataloader_Dm = dataloader_Dm
        self.dataloader_Df = dataloader_Df
        self.device = device

        self.lr_inner_adapter = lr_inner_adapter
        self.lambda_seg = lambda_seg
        self.lambda_adv_g = lambda_adv_g
        self.lambda_reg_adapter = lambda_reg_adapter
        self.inner_loop_k_steps = inner_loop_k_steps
        self.use_amp = use_amp if device.type == 'cuda' else False
        self.accumulation_steps = accumulation_steps

        self.criterion_seg = DiceLoss(
            sigmoid=True,
            include_background=True,
            reduction='mean'
        )
        self.criterion_bce = nn.BCEWithLogitsLoss()

        self.optimizer_meta = Adam(
            self.segmentation_model.parameters(),
            lr=lr_meta_G
        )
        self.optimizer_D = Adam(
            self.discriminator_model.parameters(),
            lr=lr_D
        )

        self.scaler_meta = GradScaler(enabled=self.use_amp)
        self.scaler_D = GradScaler(enabled=self.use_amp)
        self.scaler_inner = GradScaler(enabled=self.use_amp)

        self.epoch_loss_history = {
            "outer_gen": [],
            "outer_seg": [],
            "outer_adv": [],
            "inner_seg": [],
            "inner_adv": [],
            "discriminator": []
        }
        self.epoch_dice_history = []
        self.best_dsc = -1.0

        print(
            f"[BiLevelMetaTrainer] initialized. "
            f"amp={self.use_amp}, k_steps={inner_loop_k_steps}"
        )

    def train_epoch(self, epoch: int):
        """
        - inner loop: adapt Δθ on partial-modality data D_m
        - outer loop: update θ, φ on full-modality data D_f
        - both loops: update discriminator D_ψ
        """
        self.segmentation_model.train()
        self.discriminator_model.train()

        iter_Df = iter(self.dataloader_Df)
        total_batches = len(self.dataloader_Dm)

        losses = {
            "inner_seg": [], "inner_adv": [],
            "outer_seg": [], "outer_adv": [],
            "discrim": []
        }
        metrics = {"dice": []}

        print(f"\n> Epoch {epoch + 1}")

        with tqdm(
            enumerate(self.dataloader_Dm),
            total=total_batches,
            desc=f"Epoch {epoch + 1}"
        ) as pbar:
            for batch_idx, (batch_Dm, mask_Dm, seg_Dm) in pbar:
                batch_Dm = {k: v.to(self.device)
                           for k, v in batch_Dm.items()}
                mask_Dm = mask_Dm.to(self.device)
                seg_Dm = seg_Dm.to(self.device)

                inner_model = copy.deepcopy(
                    self.segmentation_model
                )
                inner_model = inner_model.to(self.device)

                if self.segmentation_model.use_meta_generator:
                    inner_params = list(
                        inner_model.generator.meta_generator.parameters()
                    )
                else:
                    inner_params = list(inner_model.parameters())

                optimizer_inner = Adam(
                    inner_params,
                    lr=self.lr_inner_adapter
                )

                for k in range(self.inner_loop_k_steps):
                    optimizer_inner.zero_grad()

                    with torch.amp.autocast(
                        self.device.type,
                        enabled=self.use_amp
                    ):
                        seg_logits_inner, bottleneck_inner, _ = (
                            inner_model(batch_Dm, modality_mask=mask_Dm)
                        )

                        d_logits_inner = (
                            self.discriminator_model(bottleneck_inner)
                        )

                        loss_seg_inner = self.criterion_seg(
                            seg_logits_inner,
                            seg_Dm
                        )

                        target_all_present = torch.ones_like(
                            d_logits_inner,
                            device=self.device
                        )
                        loss_adv_inner = self.criterion_bce(
                            d_logits_inner,
                            target_all_present
                        )

                        loss_inner = (
                            self.lambda_seg * loss_seg_inner +
                            self.lambda_adv_g * loss_adv_inner
                        )

                    self.scaler_inner.scale(loss_inner).backward()
                    self.scaler_inner.step(optimizer_inner)
                    self.scaler_inner.update()

                    losses["inner_seg"].append(loss_seg_inner.item())
                    losses["inner_adv"].append(loss_adv_inner.item())

                self.optimizer_D.zero_grad()
                with torch.no_grad(), torch.amp.autocast(
                    self.device.type,
                    enabled=self.use_amp
                ):
                    _, bottleneck_for_D, _ = (
                        inner_model(batch_Dm, modality_mask=mask_Dm)
                    )

                with torch.amp.autocast(
                    self.device.type,
                    enabled=self.use_amp
                ):
                    d_logits_for_D = (
                        self.discriminator_model(bottleneck_for_D.detach())
                    )
                    loss_D_inner = self.criterion_bce(
                        d_logits_for_D,
                        mask_Dm
                    )

                self.scaler_D.scale(loss_D_inner).backward()
                self.scaler_D.step(self.optimizer_D)
                self.scaler_D.update()

                try:
                    batch_Df, mask_Df, seg_Df = next(iter_Df)
                except StopIteration:
                    iter_Df = iter(self.dataloader_Df)
                    batch_Df, mask_Df, seg_Df = next(iter_Df)

                batch_Df = {k: v.to(self.device)
                           for k, v in batch_Df.items()}
                mask_Df = mask_Df.to(self.device)
                seg_Df = seg_Df.to(self.device)

                self.optimizer_meta.zero_grad()

                with torch.amp.autocast(
                    self.device.type,
                    enabled=self.use_amp
                ):
                    seg_logits_outer, bottleneck_outer, adapter_params = (
                        inner_model(batch_Df, modality_mask=mask_Df)
                    )

                    d_logits_outer = (
                        self.discriminator_model(bottleneck_outer)
                    )

                    loss_seg_outer = self.criterion_seg(
                        seg_logits_outer,
                        seg_Df
                    )

                    loss_adv_outer = self.criterion_bce(
                        d_logits_outer,
                        target_all_present
                    )

                    loss_reg = torch.tensor(0.0, device=self.device)
                    if (self.segmentation_model.use_meta_generator and
                        adapter_params is not None):
                        for adapter_dict in adapter_params:
                            for param_tensor in adapter_dict.values():
                                loss_reg = loss_reg + (
                                    self.lambda_reg_adapter *
                                    torch.norm(param_tensor, p=2)
                                )

                    loss_outer = (
                        self.lambda_seg * loss_seg_outer +
                        self.lambda_adv_g * loss_adv_outer +
                        loss_reg
                    )

                loss_outer_scaled = loss_outer / self.accumulation_steps
                self.scaler_meta.scale(loss_outer_scaled).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler_meta.step(self.optimizer_meta)
                    self.scaler_meta.update()
                    self.optimizer_meta.zero_grad()

                self.optimizer_D.zero_grad()
                with torch.no_grad(), torch.amp.autocast(
                    self.device.type,
                    enabled=self.use_amp
                ):
                    _, bottleneck_for_D_outer, _ = (
                        inner_model(batch_Df, modality_mask=mask_Df)
                    )

                with torch.amp.autocast(
                    self.device.type,
                    enabled=self.use_amp
                ):
                    d_logits_for_D_outer = (
                        self.discriminator_model(
                            bottleneck_for_D_outer.detach()
                        )
                    )
                    loss_D_outer = self.criterion_bce(
                        d_logits_for_D_outer,
                        mask_Df
                    )

                loss_D_outer_scaled = loss_D_outer / self.accumulation_steps
                self.scaler_D.scale(loss_D_outer_scaled).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler_D.step(self.optimizer_D)
                    self.scaler_D.update()
                    self.optimizer_D.zero_grad()

                losses["outer_seg"].append(loss_seg_outer.item())
                losses["outer_adv"].append(loss_adv_outer.item())
                losses["discrim"].append(
                    (loss_D_inner.item() + loss_D_outer.item()) / 2
                )

                with torch.no_grad():
                    seg_probs = torch.sigmoid(seg_logits_outer)
                    seg_preds = (seg_probs > 0.5).float()

                    dice = DiceMetric(include_background=False)(
                        seg_preds,
                        seg_Df
                    )
                    metrics["dice"].append(dice)

                if losses["outer_seg"]:
                    mean_seg_loss = sum(
                        losses["outer_seg"]
                    ) / len(losses["outer_seg"])
                    mean_dice = (
                        sum(m.item() if torch.is_tensor(m) else m
                            for m in metrics["dice"]) /
                        len(metrics["dice"])
                        if metrics["dice"] else 0.0
                    )
                    pbar.set_postfix({
                        "seg_loss": f"{mean_seg_loss:.4f}",
                        "adv_loss": f"{losses['outer_adv'][-1]:.4f}",
                        "dice": f"{mean_dice:.4f}"
                    })

        if losses["outer_seg"]:
            epoch_seg = sum(losses["outer_seg"]) / len(losses["outer_seg"])
            epoch_adv = sum(losses["outer_adv"]) / len(losses["outer_adv"])
            epoch_dice = (
                sum(m.item() if torch.is_tensor(m) else m
                    for m in metrics["dice"]) /
                len(metrics["dice"])
                if metrics["dice"] else 0.0
            )

            self.epoch_loss_history["outer_seg"].append(epoch_seg)
            self.epoch_loss_history["outer_adv"].append(epoch_adv)
            self.epoch_dice_history.append(epoch_dice)

            if epoch_dice > self.best_dsc:
                old_best = self.best_dsc
                self.best_dsc = epoch_dice
                torch.save({
                    "model": self.segmentation_model.state_dict(),
                    "discriminator": self.discriminator_model.state_dict(),
                    "epoch": epoch,
                    "best_dice": self.best_dsc
                }, "checkpoints/best_model.pth")

                if old_best > 0:
                    improvement = 100 * (self.best_dsc - old_best) / old_best
                    print(f"[Epoch {epoch + 1}] Dice improved "
                          f"{improvement:.2f}% "
                          f"({old_best:.4f} → {self.best_dsc:.4f})")

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self.train_epoch(epoch)

        print("\n[Training Complete] Bi-level meta-learning finished.")
