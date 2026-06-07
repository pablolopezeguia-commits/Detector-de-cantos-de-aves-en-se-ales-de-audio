"""YAMNet embedding extraction and final MLP training helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


SAMPLE_RATE = 16000
EMBEDDING_DIM = 1024


def load_yamnet(source: str | Path = "https://tfhub.dev/google/yamnet/1"):
    """Load YAMNet either from a local SavedModel folder or from TensorFlow Hub."""
    source = str(source)
    if Path(source).exists():
        return tf.saved_model.load(source)

    import tensorflow_hub as hub

    return hub.load(source)


def call_yamnet(yamnet, waveform: np.ndarray) -> np.ndarray:
    """Run YAMNet and return frame-level embeddings."""
    tensor = tf.convert_to_tensor(np.asarray(waveform, dtype=np.float32), dtype=tf.float32)
    try:
        outputs = yamnet(tensor)
    except TypeError:
        serving = yamnet.signatures["serving_default"]
        input_name = next(iter(serving.structured_input_signature[1]))
        outputs = serving(**{input_name: tensor})

    if isinstance(outputs, dict):
        for key in ("output_1", "embeddings", "embedding"):
            if key in outputs:
                embeddings = outputs[key]
                break
        else:
            raise RuntimeError(f"Could not find YAMNet embeddings in keys: {list(outputs)}")
    else:
        embeddings = outputs[1]

    embeddings_np = np.asarray(embeddings.numpy() if hasattr(embeddings, "numpy") else embeddings, dtype=np.float32)
    if embeddings_np.ndim != 2 or embeddings_np.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(f"Unexpected YAMNet embedding shape: {embeddings_np.shape}")
    return embeddings_np


def extract_mean_embedding(audio_path: str | Path, yamnet, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Convert one audio file into the mean 1024-dimensional YAMNet embedding."""
    waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    embeddings = call_yamnet(yamnet, waveform.astype(np.float32))
    return embeddings.mean(axis=0).astype(np.float32)


def extract_embedding_table(
    metadata: pd.DataFrame,
    yamnet,
    path_column: str = "filepath",
    label_column: str = "label",
    weight_column: str | None = "sample_weight",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings, labels, and sample weights from a metadata table."""
    X, y, weights = [], [], []
    for _, row in metadata.iterrows():
        X.append(extract_mean_embedding(row[path_column], yamnet))
        y.append(int(row[label_column]))
        weights.append(float(row[weight_column]) if weight_column and weight_column in row else 1.0)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), np.asarray(weights, dtype=np.float32)


def build_yamnet_mlp(input_dim: int = EMBEDDING_DIM) -> keras.Model:
    """Build the final binary MLP head used on top of frozen YAMNet embeddings."""
    inputs = keras.Input(shape=(input_dim,), name="yamnet_embedding")
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.35)(x)
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="song_probability")(x)
    model = keras.Model(inputs, outputs, name="yamnet_mlp_final")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None = None,
    output_path: str | Path | None = None,
    epochs: int = 80,
    batch_size: int = 64,
) -> keras.Model:
    """Train the final MLP head with early stopping on validation PR-AUC."""
    model = build_yamnet_mlp(X_train.shape[1])
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=5, min_lr=1e-5),
    ]
    if output_path is not None:
        callbacks.insert(
            0,
            keras.callbacks.ModelCheckpoint(str(output_path), monitor="val_pr_auc", mode="max", save_best_only=True),
        )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model
