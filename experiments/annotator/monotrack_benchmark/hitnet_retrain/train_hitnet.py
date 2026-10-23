"""Train one released HitNet architecture and calibrate its best validation checkpoint."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from inputs import read_array, read_json, write_json
from recipe import (
    BALANCE_SEED,
    BATCH_SIZE,
    FEATURES_PER_FRAME,
    balanced,
    fetch_data,
    num_consec,
    role_windows,
)

LEARNING_RATE = 1e-3
MAX_EPOCHS = 600


def build_model(dropout: float, l2_reg: float) -> Any:
    import tensorflow as tf  # pyrefly: ignore[missing-import]
    from tensorflow.keras import Model, layers  # pyrefly: ignore[missing-import]

    inputs = layers.Input(shape=(num_consec * FEATURES_PER_FRAME,))
    values = layers.Reshape((num_consec, FEATURES_PER_FRAME))(inputs)
    for _ in range(2):
        values = layers.Bidirectional(layers.GRU(
            64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), dropout=dropout,
        ))(values)
    values = layers.GlobalMaxPool1D()(values)
    values = layers.Dense(3)(values)
    model = Model(inputs, layers.Softmax()(values))
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(clipnorm=2.0, learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def temperature_scaling(logits: Any, labels: np.ndarray) -> float:
    """Fit the notebook's scalar temperature on validation logits for 150 steps."""
    import tensorflow as tf  # pyrefly: ignore[missing-import]

    temperature = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

    def loss() -> Any:
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            tf.convert_to_tensor(tf.keras.utils.to_categorical(labels)), logits / temperature,
        ))

    optimiser = tf.optimizers.Adam()
    for _ in range(150):
        optimiser.minimize(loss, var_list=[temperature])
    return float(temperature.numpy())


def save_model(model: Any, path: Path, temperature: float) -> None:
    import h5py  # pyrefly: ignore[missing-import]

    model.save(str(path), include_optimizer=True)
    with h5py.File(path, "r+") as stream:
        stream.attrs["temperature"] = temperature


def train(features: Path, arm: str, dropout: float, l2_reg: float, output: Path,
          smoke: bool = False, seed: int = 0) -> dict[str, Any]:
    import tensorflow as tf  # pyrefly: ignore[missing-import]

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rallies = read_json(features / "rallies.json.gz")
    frames = read_array(features / arm / "frames.npy.xz")
    labels = read_array(features / "labels.npy.xz")
    x_train, y_train = balanced(*role_windows(frames, labels, rallies, "train"), frames, BALANCE_SEED)
    x_val, y_val = balanced(*role_windows(frames, labels, rallies, "validation"), frames, BALANCE_SEED + 1)
    del frames
    model = build_model(dropout, l2_reg)
    if model.count_params() != 130179:
        raise ValueError("HitNet architecture differs from the released 130,179-parameter model")
    dataset = tf.data.Dataset.from_generator(
        lambda: fetch_data(x_train, y_train),
        output_signature=(tf.TensorSpec((None, 936), tf.float32), tf.TensorSpec((None,), tf.float32)),
    ).prefetch(tf.data.AUTOTUNE)
    # Keep the notebook's floor division and partial-batch generator; only the
    # approved 10x multiplier is removed (approximately one pass per epoch).
    steps = x_train.shape[0] // BATCH_SIZE
    if steps < 1:
        raise ValueError("Training split has fewer than one batch after balancing")
    best = output / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15),
        tf.keras.callbacks.ReduceLROnPlateau(patience=12),
        tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_loss", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.CSVLogger(str(output / "history.csv")),
    ]
    history = model.fit(
        dataset, steps_per_epoch=min(steps, 2) if smoke else steps, epochs=1 if smoke else MAX_EPOCHS,
        validation_data=(x_val[:128], y_val[:128]) if smoke else (x_val, y_val), callbacks=callbacks, verbose=2,
    )
    model.load_weights(str(best))
    logits_model = tf.keras.Model(model.inputs, model.layers[-2].output)
    calibration_labels = y_val[:128] if smoke else y_val
    logits = logits_model.predict(x_val[:128] if smoke else x_val, batch_size=4096, verbose=0)
    temperature = temperature_scaling(logits, calibration_labels)
    save_model(model, output / "hitnet.h5", temperature)
    with (output / "history.csv").open("rb") as source, gzip.open(output / "history.csv.gz", "wb") as destination:
        shutil.copyfileobj(source, destination)
    (output / "history.csv").unlink()
    best_epoch = int(np.argmin(history.history["val_loss"]))
    summary = {
        "arm": arm, "dropout": dropout, "l2": l2_reg, "learning_rate": LEARNING_RATE, "seed": seed,
        "parameters": model.count_params(), "steps_per_epoch": min(steps, 2) if smoke else steps,
        "train_windows": len(y_train), "validation_windows": len(y_val),
        "train_classes": np.bincount(y_train.astype(int), minlength=3).tolist(),
        "validation_classes": np.bincount(y_val.astype(int), minlength=3).tolist(),
        "epochs_run": len(history.history["loss"]), "best_epoch": best_epoch + 1,
        "best_val_loss": float(history.history["val_loss"][best_epoch]),
        "best_val_accuracy": float(history.history["val_accuracy"][best_epoch]),
        "temperature": temperature, "seconds": time.perf_counter() - started,
        "tensorflow": tf.__version__, "numpy": np.__version__, "smoke": smoke,
    }
    write_json(output / "summary.json.gz", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--arm", choices=("kept", "zeroed"), required=True)
    parser.add_argument("--dropout", type=float, choices=(0, 0.2), required=True)
    parser.add_argument("--l2", type=float, choices=(0, 1e-6, 1e-4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="training seed; balanced data splits stay fixed")
    parser.add_argument("--smoke", action="store_true", help="two training batches, one validation and H5 export")
    args = parser.parse_args()
    train(args.features, args.arm, args.dropout, args.l2, args.output, args.smoke, args.seed)


if __name__ == "__main__":
    main()
