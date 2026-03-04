"""
train.py
--------
Complete training script for CocktailNet on Google Colab T4.

Before running:
  1. Mount Drive
  2. Run prepare_data.py to build processed_segments/
  3. Update DRIVE_BASE below
  4. Run:  !python train.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models.fusion_net import CocktailNet
from dataset import CocktailDataset


# ------------------------------------------------------------------ #
# config — adjust these paths for your Drive layout                    #
# ------------------------------------------------------------------ #

DRIVE_BASE       = "/content/drive/MyDrive/PE2_Project"
DATA_DIR         = f"{DRIVE_BASE}/processed_segments"
CHECKPOINT_DIR   = f"{DRIVE_BASE}/checkpoints"
CHECKPOINT_PATH  = f"{CHECKPOINT_DIR}/cocktail_net_latest.pth"
BEST_MODEL_PATH  = f"{CHECKPOINT_DIR}/cocktail_net_best.pth"

BATCH_SIZE  = 4      # safe for T4 with 112x112 video
LR          = 1e-4
NUM_EPOCHS  = 60
NUM_WORKERS = 2
RESUME      = True   # set False to train from scratch


# ------------------------------------------------------------------ #
# helpers                                                              #
# ------------------------------------------------------------------ #

def freeze_visual_backbone(model):
    """
    Freeze everything in r3d_18 except layer3, layer4, and fc.
    With only ~20 videos we don't have enough data to retrain the whole backbone.
    """
    for name, param in model.visual_frontend.named_parameters():
        if not any(k in name for k in ("layer3", "layer4", "fc")):
            param.requires_grad = False


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"] + 1, ckpt.get("best_val_loss", float("inf"))


# ------------------------------------------------------------------ #
# main training loop                                                   #
# ------------------------------------------------------------------ #

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # dataset
    full_ds = CocktailDataset(DATA_DIR)
    val_n   = max(1, int(0.1 * len(full_ds)))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {train_n} samples | Val: {val_n} samples")

    # model
    model = CocktailNet().to(device)
    freeze_visual_backbone(model)
    count_params(model)

    optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                       patience=5, factor=0.5,
                                                       min_lr=1e-6)
    criterion  = nn.L1Loss()
    scaler     = torch.amp.GradScaler("cuda")

    start_epoch    = 0
    best_val_loss  = float("inf")

    if RESUME and os.path.exists(CHECKPOINT_PATH):
        start_epoch, best_val_loss = load_checkpoint(CHECKPOINT_PATH, model, optimizer, device)
        print(f"Resumed from epoch {start_epoch} | best val loss so far: {best_val_loss:.4f}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        # ---- training ----
        model.train()
        train_loss = 0.0
        for video, noisy_mag, irm in train_loader:
            video, noisy_mag, irm = video.to(device), noisy_mag.to(device), irm.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                pred_mask = model(video, noisy_mag)
                loss = criterion(pred_mask, irm)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for video, noisy_mag, irm in val_loader:
                video, noisy_mag, irm = video.to(device), noisy_mag.to(device), irm.to(device)
                with torch.amp.autocast("cuda"):
                    pred_mask = model(video, noisy_mag)
                    val_loss += criterion(pred_mask, irm).item()

        avg_val = val_loss / len(val_loader)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  {elapsed:.1f}s")

        scheduler.step(avg_val)

        # always save latest so training can resume after Colab disconnect
        ckpt = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }
        save_checkpoint(ckpt, CHECKPOINT_PATH)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(ckpt, BEST_MODEL_PATH)
            print(f"  ✓ best model saved  (val={best_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best model at: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
