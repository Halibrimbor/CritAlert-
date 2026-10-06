"""
╔══════════════════════════════════════════════════════════════════╗
║  train_ecg_model.py  —  CritAlert ECG CNN Trainer              ║
║                                                                  ║
║  Trains a Convolutional Neural Network on ECG images.           ║
║  Output: ecg_model.h5  (used by app.py automatically)          ║
║                                                                  ║
║  FOLDER STRUCTURE REQUIRED:                                      ║
║                                                                  ║
║    ecg_dataset/                                                  ║
║      Normal/                                                     ║
║        ecg_001.jpg   ← normal ECG images here                   ║
║        ecg_002.jpg                                               ║
║        ...                                                       ║
║      Abnormal/                                                   ║
║        ecg_101.jpg   ← abnormal ECG images here                 ║
║        ecg_102.jpg                                               ║
║        ...                                                       ║
║                                                                  ║
║  SUPPORTED FORMATS: JPG, JPEG, PNG                              ║
║                                                                  ║
║  HOW TO RUN (step-by-step at bottom of this file)              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import numpy as np

# ── Check TensorFlow ──────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    print(f"TensorFlow {tf.__version__} loaded")
except ImportError:
    print("✗ TensorFlow not found!")
    print("  Install it:  pip install tensorflow")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  — change these if needed
# ══════════════════════════════════════════════════════════════════
DATASET_DIR  = "D:\\PROJECTS\\Critalert\\Dataset\\ECG_dataset"   # folder with Normal/ and Abnormal/ subfolders
OUTPUT_MODEL = "ecg_model.h5"  # saved model — used by app.py
IMG_SIZE     = (150, 150)       # resize all images to this
BATCH_SIZE   = 16
EPOCHS       = 25               # increase for better accuracy


def check_dataset(path):
    """Validate dataset structure before training."""
    if not os.path.exists(path):
        print(f"\n✗ Dataset folder '{path}' not found!")
        print("  Create the folder structure:")
        print(f"    {path}/")
        print(f"      Normal/    ← put normal ECG images here")
        print(f"      Abnormal/  ← put abnormal ECG images here")
        return False

    classes = [d for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d))]
    if len(classes) < 2:
        print(f"\n✗ Need at least 2 class folders. Found: {classes}")
        print("  Expected folders: Normal  and  Abnormal")
        return False

    print(f"\n✓ Dataset found: {path}")
    total = 0
    for cls in sorted(classes):
        cls_path = os.path.join(path, cls)
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg",".jpeg",".png"))]
        print(f"   {cls:12s}  →  {len(imgs):4d} images")
        total += len(imgs)

    print(f"   {'TOTAL':12s}  →  {total:4d} images")

    if total < 20:
        print(f"\n⚠  Only {total} images found. More images = better accuracy.")
        print("   Recommended: 100+ images per class for good results.")
    return True


def build_ecg_cnn(num_classes: int) -> keras.Model:
    """
    MobileNetV2-based transfer learning model.
    Works well even with small datasets (50+ images).
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze base initially

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        # Output: binary (Normal vs Abnormal)
        layers.Dense(1 if num_classes == 2 else num_classes,
                     activation="sigmoid" if num_classes == 2 else "softmax"),
    ])
    return model, base


def train():
    print("\n" + "═"*55)
    print("  CritAlert — ECG CNN Trainer")
    print("═"*55)

    # ── Validate dataset ──────────────────────────────────────
    if not check_dataset(DATASET_DIR):
        sys.exit(1)

    # ── Data augmentation pipeline ────────────────────────────
    print("\n► Setting up data pipeline with augmentation…")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=False,    # ECGs should not be flipped horizontally
        brightness_range=[0.9, 1.1],
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )
    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)
    print(f"✓ Classes: {train_gen.class_indices}")
    print(f"✓ Train batches: {len(train_gen)}  |  Val batches: {len(val_gen)}")

    # Save class mapping so app.py knows which index = Abnormal
    with open("ecg_class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print("✓ Class indices saved → ecg_class_indices.json")

    # ── Build model ───────────────────────────────────────────
    print("\n► Building MobileNetV2 transfer learning model…")
    model, base = build_ecg_cnn(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # ── Phase 1: train head only (base frozen) ────────────────
    print("\n► Phase 1: Training classifier head (base model frozen)…")
    cb_phase1 = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]
    model.fit(train_gen, validation_data=val_gen,
              epochs=min(10, EPOCHS), callbacks=cb_phase1, verbose=1)

    # ── Phase 2: fine-tune top layers ─────────────────────────
    print("\n► Phase 2: Fine-tuning top layers of base model…")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),   # lower LR for fine-tuning
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    cb_phase2 = [
        callbacks.EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.3, verbose=1),
        callbacks.ModelCheckpoint(OUTPUT_MODEL, save_best_only=True, verbose=1),
    ]
    model.fit(train_gen, validation_data=val_gen,
              epochs=EPOCHS, callbacks=cb_phase2, verbose=1)

    # ── Final evaluation ──────────────────────────────────────
    print("\n► Evaluating on validation set…")
    loss, acc, auc = model.evaluate(val_gen, verbose=0)
    print(f"\n  Accuracy : {acc*100:.1f}%")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Loss     : {loss:.4f}")

    # Save final model
    model.save(OUTPUT_MODEL)
    print(f"\n✓ Model saved → {OUTPUT_MODEL}")
    print("✓ Place ecg_model.h5 in the same folder as app.py")
    print("✓ Restart app.py — it will auto-load the ECG model\n")

    return model


if __name__ == "__main__":
    train()


# ══════════════════════════════════════════════════════════════════
#
#  STEP-BY-STEP GUIDE TO RUN THIS FILE
#  ─────────────────────────────────────────────────────────────────
#
#  STEP 1 — Install Python packages (run once)
#  ─────────────────────────────────────────────────────────────────
#  Open terminal / command prompt and run:
#
#    pip install tensorflow pillow numpy
#
#  If you have a GPU (recommended for faster training):
#    pip install tensorflow[and-cuda]
#
#
#  STEP 2 — Prepare your ECG dataset
#  ─────────────────────────────────────────────────────────────────
#  Create this folder structure next to train_ecg_model.py:
#
#    ecg_dataset/
#      Normal/
#        001.jpg    ← normal ECG images (JPG or PNG)
#        002.jpg
#        ...
#      Abnormal/
#        101.jpg    ← abnormal ECG images
#        102.jpg
#        ...
#
#  WHERE TO GET FREE ECG DATASETS:
#    • Kaggle ECG Image Dataset:
#      https://www.kaggle.com/datasets/erhmad/ecg-image-dataset
#    • PhysioNet ECG:
#      https://physionet.org/content/ecg-arrhythmia/
#    • PTB Diagnostic ECG:
#      https://www.kaggle.com/datasets/shayanfazeli/heartbeat
#
#  Minimum recommended: 50 images per class
#  Good accuracy:       200+ images per class
#
#
#  STEP 3 — Run training
#  ─────────────────────────────────────────────────────────────────
#  In terminal, navigate to the Critalert folder:
#
#    cd path/to/Critalert
#    python train_ecg_model.py
#
#  Training takes:
#    • CPU only:     15–45 minutes
#    • GPU (CUDA):   3–8 minutes
#
#  You will see output like:
#    Epoch 1/25 — loss: 0.68 — accuracy: 0.61 — val_accuracy: 0.68
#    Epoch 2/25 — loss: 0.54 — accuracy: 0.74 — val_accuracy: 0.79
#    ...
#
#
#  STEP 4 — Model is saved automatically
#  ─────────────────────────────────────────────────────────────────
#  After training completes, you will see:
#    ✓ Model saved → ecg_model.h5
#
#  The file ecg_model.h5 is created in the same folder.
#
#
#  STEP 5 — Run the app
#  ─────────────────────────────────────────────────────────────────
#    python app.py
#
#  Now upload an ECG image at http://127.0.0.1:5000
#  Select "ECG Report" from the dropdown — it will use ecg_model.h5
#
#
#  TROUBLESHOOTING
#  ─────────────────────────────────────────────────────────────────
#  Error: "CUDA out of memory"
#    → Reduce BATCH_SIZE from 16 to 8 (line 45 in this file)
#
#  Error: "No module named tensorflow"
#    → Run:  pip install tensorflow
#
#  Low accuracy (<70%)
#    → Add more training images (200+ per class recommended)
#    → Increase EPOCHS to 40
#    → Make sure images are labelled correctly in correct folders
#
#  Model not loading in app.py
#    → Make sure ecg_model.h5 is in same folder as app.py
#    → Restart app.py after training
#
# ══════════════════════════════════════════════════════════════════
