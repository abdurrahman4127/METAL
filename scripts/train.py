import os
import torch
from torch.utils.data import DataLoader

from src.models import SegmentationModel, ModalityDiscriminator
from src.training import BiLevelMetaTrainer
from src.data import BraTSDataset
from src.config import (
    BATCH_SIZE_DM, BATCH_SIZE_DF, NUM_WORKERS, SHUFFLE,
    BASE_FEATURES, BOTTLENECK_SCALE, DISCRIMINATOR_HIDDEN_DIM,
    INPUT_CHANNELS, OUTPUT_CHANNELS, ALL_MODALITY_NAMES,
    INNER_LOOP_LR_ALPHA, INNER_LOOP_STEPS_K, META_LR_G, DISCRIMINATOR_LR,
    LAMBDA_SEG, LAMBDA_ADV_G, LAMBDA_REG_ADAPTER, ACCUMULATION_STEPS,
    USE_AMP, NUM_EPOCHS, DATASET_DIR
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}")

    if not os.path.exists(DATASET_DIR):
        print(f"[Error] Dataset directory not found: {DATASET_DIR}")
        return

    print("[Data] Loading BraTSDataset...")
    dataset_Dm = BraTSDataset(
        data_dir=DATASET_DIR,
        modality_names=ALL_MODALITY_NAMES,
        mode="train",
        partial_modality_prob=0.5,
        num_partial_modalities=3
    )

    dataset_Df = BraTSDataset(
        data_dir=DATASET_DIR,
        modality_names=ALL_MODALITY_NAMES,
        mode="train",
        partial_modality_prob=0.0
    )

    dataloader_Dm = DataLoader(
        dataset_Dm,
        batch_size=BATCH_SIZE_DM,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS
    )

    dataloader_Df = DataLoader(
        dataset_Df,
        batch_size=BATCH_SIZE_DF,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS
    )

    print("[Models] Initializing architectures...")
    segmentation_model = SegmentationModel(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_features=BASE_FEATURES,
        all_modality_names=ALL_MODALITY_NAMES,
        use_meta_generator=True,
        adapter_bottleneck_scale=BOTTLENECK_SCALE
    )

    bottleneck_dim = segmentation_model.generator.encoder_feature_dims[-1]
    discriminator_model = ModalityDiscriminator(
        bottleneck_dim=bottleneck_dim,
        num_modalities=len(ALL_MODALITY_NAMES),
        hidden_dim=DISCRIMINATOR_HIDDEN_DIM
    )

    print("[Training] Initializing bi-level trainer...")
    trainer = BiLevelMetaTrainer(
        segmentation_model=segmentation_model,
        discriminator_model=discriminator_model,
        dataloader_Dm=dataloader_Dm,
        dataloader_Df=dataloader_Df,
        device=device,
        lr_meta_G=META_LR_G,
        lr_inner_adapter=INNER_LOOP_LR_ALPHA,
        lr_D=DISCRIMINATOR_LR,
        lambda_seg=LAMBDA_SEG,
        lambda_adv_g=LAMBDA_ADV_G,
        lambda_reg_adapter=LAMBDA_REG_ADAPTER,
        inner_loop_k_steps=INNER_LOOP_STEPS_K,
        use_amp=USE_AMP,
        accumulation_steps=ACCUMULATION_STEPS
    )

    print(f"[Training] Starting {NUM_EPOCHS} epochs...")
    trainer.train(NUM_EPOCHS)

    print("[Complete] Training finished successfully")


if __name__ == "__main__":
    main()
