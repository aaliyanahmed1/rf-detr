# train_rfdetr_minimal.py

import os
from rfdetr import RFDETRBase
import torch

def main():
    # 1. Install prerequisites:
    #    pip install rfdetr

    # 2. Configure your dataset path (must follow COCO format)
    dataset_dir = "dataset"  # contains train/, valid/, test/ subfolders with images + _annotations.coco.json

    # 3. Choose the model variant: "nano", "small", "base", "large"
    model_variant = "nano"  # <-- paste your desired model name here

    # 4. Load the selected RF-DETR model with pretrained weights
    model = RFDETRBase(variant=model_variant)  # loads chosen variant with pretrained weights

    # 5. Fine-tune the model on your dataset
    model.train(
        dataset_dir=dataset_dir,
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=f"rf_detr_finetuned_{model_variant}"
    )

    print(f"Training completed. Model and checkpoints saved to 'rf_detr_finetuned_{model_variant}'.")

if __name__ == "__main__":
    main()
