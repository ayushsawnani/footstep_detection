# ==============================
# File: train.py
# Purpose: Load parquet features, build a CNN (64, 32, 32 conv layers)
#          with regularization, and train a binary classifier
#          (footstep vs not-footstep) in TensorFlow/Keras.
# ==============================

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

keras = tf.keras  # type: ignore


def load_parquet(path: Path):
    df = pd.read_parquet(path)
    X = []
    for feat, fb, tfm in zip(df["feature"], df["freq_bins"], df["time_frames"]):
        arr = np.array(feat, dtype=np.float32).reshape((fb, tfm))
        X.append(arr)
    X = np.stack(X, axis=0)
    # Ensure a pure NumPy 1D array (avoid pandas ExtensionArray causing Pylance/typing issues)
    y = df["label"].astype(int).to_numpy(copy=True).reshape(-1)
    return X, y, df


def standardize_train_val(X: np.ndarray, X_val: np.ndarray):
    mean = X.mean()
    std = X.std() + 1e-6
    return (X - mean) / std, (X_val - mean) / std, float(mean), float(std)


def build_model(input_shape):
    l2 = keras.regularizers.l2(1e-4)
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Reshape((*input_shape, 1))(inputs)

    # Conv block 1 - 64
    x = keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    # Conv block 2 - 32
    x = keras.layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    # Conv block 3 - 32
    x = keras.layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    X, y, df = load_parquet(Path(args.parquet))
    # Train/val split (stratified if both classes present)
    classes = np.unique(y)
    strat = y if classes.size > 1 else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=strat
    )

    # Standardize
    X_tr, X_va, mean, std = standardize_train_val(X_tr, X_va)

    # Add channel dim now
    input_shape = X_tr.shape[1:]

    model = build_model(input_shape)
    model.summary()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor="val_auc", mode="max"
        ),
        keras.callbacks.ModelCheckpoint(
            str(out_dir / "footstep_cnn.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training artifacts
    model.save(str(out_dir / "final_cnn.keras"))
    with open(out_dir / "norm.json", "w") as f:
        json.dump({"mean": mean, "std": std, "input_shape": list(input_shape)}, f)

    # Quick evaluation
    eval_metrics = model.evaluate(X_va, y_va, verbose=0)
    print({k: float(v) for k, v in zip(model.metrics_names, eval_metrics)})


if __name__ == "__main__":
    main()
