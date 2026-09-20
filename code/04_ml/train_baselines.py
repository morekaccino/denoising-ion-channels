"""Retrain the three thesis Keras models on the frozen synth_v1 splits.

The committed models were trained on random data with val == test; this script
retrains the identical architectures on the frozen train/val splits and evaluates
on the frozen test set so the KI-HMM comparison is apples-to-apples.

Artefacts: `code/04_ml/models/*_benchmark.keras`,
`code/04_ml/results/baselines_synth_v1.json`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark as B  # noqa: E402

MODELS = ROOT / "code" / "04_ml" / "models"
RESULTS = ROOT / "code" / "04_ml" / "results"
N_POINTS = 50


def one_hot_n(n_values: np.ndarray) -> np.ndarray:
    y = np.zeros((len(n_values), 4), dtype=np.float32)
    y[np.arange(len(n_values)), n_values] = 1.0
    return y


def sample_windows(X, y, n_windows, rng, include_center=False):
    centers = rng.integers(N_POINTS, X.shape[1] - N_POINTS, size=n_windows)
    traces = rng.integers(0, len(X), size=n_windows)
    out = np.empty((n_windows, 2 * N_POINTS + (1 if include_center else 0)), dtype=np.float32)
    tgt = np.empty(n_windows, dtype=np.int64)
    for j, (i, c) in enumerate(zip(traces, centers)):
        row = X[i]
        out[j] = row[c - N_POINTS : c + N_POINTS + 1] if include_center else np.concatenate(
            [row[c - N_POINTS : c], row[c + 1 : c + N_POINTS + 1]]
        )
        tgt[j] = y[i, c]
    return out, tgt


def dense_windows(X, include_center=False):
    _, T = X.shape
    k = N_POINTS
    centers = np.arange(k, T - k)
    left = np.stack([X[:, c - k : c] for c in centers], axis=1)
    right = np.stack([X[:, c + 1 : c + k + 1] for c in centers], axis=1)
    if include_center:
        mid = X[:, centers][:, :, None]
        return np.concatenate([left, mid, right], axis=-1)
    return np.concatenate([left, right], axis=-1)


class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_layer = keras.layers.LSTM(units=100)
        self.lstm_dense_layer = keras.layers.Dense(units=4, activation="softmax")
        self.MD_layer = keras.layers.Dense(units=101, activation="relu")
        self.MD_middle_layer = keras.layers.Dense(units=50, activation="relu")
        self.MD_output_layer = keras.layers.Dense(units=3, activation="softmax")
        self.concat = keras.layers.Concatenate()
        self.output_layer = keras.layers.Dense(units=4, activation="softmax")

    def call(self, inputs, **kwargs):
        x = self.lstm_dense_layer(self.lstm_layer(inputs[0]))
        y = self.MD_output_layer(self.MD_middle_layer(self.MD_layer(inputs[1])))
        return self.output_layer(self.concat([x, y]))


def train_ccnn(tr, va, epochs, batch):
    Xtr = B.histogram_features(tr["X_s1"])
    Xva = B.histogram_features(va["X_s1"])
    ytr, yva = one_hot_n(tr["N"]), one_hot_n(va["N"])
    model = keras.Sequential(
        [keras.layers.Input(shape=(100,)), keras.layers.Dense(100, activation="relu"), keras.layers.Dense(4, activation="softmax")]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, epochs=epochs, batch_size=batch, validation_data=(Xva, yva), verbose=2)
    return model


def train_lstm(tr, va, epochs, batch, n_windows=1000):
    m_tr = tr["N"] == 3
    m_va = va["N"] == 3
    rng = np.random.default_rng(0)
    Xtr, ytr = sample_windows(tr["X_s1"][m_tr], tr["y"][m_tr], n_windows, rng)
    Xva, yva = sample_windows(va["X_s1"][m_va], va["y"][m_va], n_windows, rng)
    model = keras.Sequential([keras.layers.LSTM(100, input_shape=(100, 1)), keras.layers.Dense(4, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr[..., None], ytr, epochs=epochs, batch_size=batch, validation_data=(Xva[..., None], yva), verbose=2)
    return model


def train_combined(tr, va, epochs, batch):
    Xh_tr = B.histogram_features(tr["X_s1"])
    Xh_va = B.histogram_features(va["X_s1"])
    rng = np.random.default_rng(0)
    Xw_tr, y_tr = sample_windows(tr["X_s1"], tr["y"], len(tr["X_s1"]), rng, include_center=True)
    Xw_va, y_va = sample_windows(va["X_s1"], va["y"], len(va["X_s1"]), rng, include_center=True)
    model = CustomModel()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        [Xw_tr[..., None], Xh_tr],
        one_hot_n(y_tr),
        epochs=epochs,
        batch_size=batch,
        validation_data=([Xw_va[..., None], Xh_va], one_hot_n(y_va)),
        verbose=2,
    )
    return model


def evaluate_tracewise(model, X, y, hist=None, include_center=False, batch=512):
    preds = []
    for i in range(len(X)):
        w = dense_windows(X[i : i + 1], include_center=include_center)[0]
        if hist is None:
            probs = model.predict(w[..., None], batch_size=batch, verbose=0)
        else:
            h = np.repeat(hist[i : i + 1], len(w), axis=0)
            probs = model.predict([w[..., None], h], batch_size=batch, verbose=0)
        preds.append(probs.argmax(axis=1))
    return np.stack(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-ccnn", type=int, default=30)
    parser.add_argument("--epochs-lstm", type=int, default=100)
    parser.add_argument("--epochs-combined", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    tr, va = B.load_split("train"), B.load_split("val")
    test = B.load_split("test")
    report = {}
    k = N_POINTS
    y_centered = test["y"][:, k : test["y"].shape[1] - k]
    n_centered = test["N"]

    ccnn_path = MODELS / "state_estimator_benchmark.keras"
    ccnn = keras.models.load_model(ccnn_path) if ccnn_path.exists() else train_ccnn(tr, va, args.epochs_ccnn, args.batch)
    ccnn.save(MODELS / "state_estimator_benchmark.keras")
    n_hat = np.argmax(ccnn.predict(B.histogram_features(test["X_s1"]), verbose=0), axis=1)
    report["ccnn"] = B.n_estimation_metrics(test["N"], n_hat)
    report["ccnn"]["per_n"] = {}
    for n in np.unique(test["N"]):
        m = test["N"] == n
        report["ccnn"]["per_n"][str(int(n))] = float((n_hat[m] == test["N"][m]).mean())
    print("CCNN:", report["ccnn"], flush=True)

    lstm_path = MODELS / "LSTM_benchmark.keras"
    lstm = keras.models.load_model(lstm_path) if lstm_path.exists() else train_lstm(tr, va, args.epochs_lstm, args.batch)
    lstm.save(MODELS / "LSTM_benchmark.keras")
    m3 = test["N"] == 3
    for s in ["1", "2", "4"]:
        pred = evaluate_tracewise(lstm, test[f"X_s{s}"][m3], test["y"][m3])
        report[f"lstm_scale_{s}"] = B.summarize(y_centered[m3], pred, N_true=test["N"][m3])
        print(f"LSTM scale {s}:", round(report[f'lstm_scale_{s}']['accuracy'], 4), flush=True)

    combined_path = MODELS / "state_estimator_and_LSTM_benchmark.keras"
    if combined_path.exists():
        combined = keras.models.load_model(combined_path, custom_objects={"CustomModel": CustomModel})
    else:
        combined = train_combined(tr, va, args.epochs_combined, args.batch)
    combined.save(MODELS / "state_estimator_and_LSTM_benchmark.keras")
    for s in ["1", "2", "4"]:
        hist = B.histogram_features(test[f"X_s{s}"])
        pred = evaluate_tracewise(combined, test[f"X_s{s}"], test["y"], hist=hist, include_center=True)
        report[f"combined_scale_{s}"] = B.summarize(y_centered, pred, N_true=n_centered)
        print(f"Combined scale {s}:", round(report[f'combined_scale_{s}']['accuracy'], 4), flush=True)

    (RESULTS / "baselines_synth_v1.json").write_text(json.dumps(report, indent=2))
    print("saved", RESULTS / "baselines_synth_v1.json")


if __name__ == "__main__":
    main()
