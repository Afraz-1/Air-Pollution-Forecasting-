"""Reproducible one-hour-ahead Beijing PM2.5 forecasting experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


LOOKBACK = 24
TRAIN_END = pd.Timestamp("2013-01-01")
VALIDATION_END = pd.Timestamp("2014-01-01")
WEATHER_COLUMNS = ["dew", "temp", "press", "wnd_spd", "snow", "rain"]
WIND_DIRECTIONS = ["SE", "cv", "NW", "NE"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {"date", "pollution", "wnd_dir", *WEATHER_COLUMNS}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("The input contains missing values; choose a documented imputation policy.")
    if df["date"].duplicated().any() or not df["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Expected unique, consecutive hourly timestamps.")
    if unknown := set(df["wnd_dir"]).difference(WIND_DIRECTIONS):
        raise ValueError(f"Unexpected wind directions: {sorted(unknown)}")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[["pollution", *WEATHER_COLUMNS]].astype(np.float32).copy()
    for direction in WIND_DIRECTIONS:
        features[f"wind_{direction}"] = (df["wnd_dir"] == direction).astype(np.float32)
    hour = df["date"].dt.hour.to_numpy()
    day = df["date"].dt.dayofyear.to_numpy()
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["year_sin"] = np.sin(2 * np.pi * day / 365.25)
    features["year_cos"] = np.cos(2 * np.pi * day / 365.25)
    return features


def make_windows(values: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
    # Window k contains rows k..k+lookback-1; its target is row k+lookback.
    return np.stack([values[i - lookback : i] for i in range(lookback, len(values))])


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/LSTM-Multivariate_pollution.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    args.output.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    features = make_features(df)
    dates = df["date"].to_numpy()[LOOKBACK:]
    target = df["pollution"].to_numpy(dtype=np.float32)[LOOKBACK:]
    train = dates < np.datetime64(TRAIN_END)
    validation = (dates >= np.datetime64(TRAIN_END)) & (dates < np.datetime64(VALIDATION_END))
    test = dates >= np.datetime64(VALIDATION_END)
    if not all(mask.any() for mask in (train, validation, test)):
        raise ValueError("Dataset must contain training, 2013 validation, and 2014 test periods.")

    # Learn all preprocessing parameters only from timestamps before 2013.
    feature_scaler = StandardScaler().fit(features.loc[df["date"] < TRAIN_END])
    target_scaler = StandardScaler().fit(df.loc[df["date"] < TRAIN_END, ["pollution"]].to_numpy())
    scaled_features = feature_scaler.transform(features).astype(np.float32)
    windows = make_windows(scaled_features)
    scaled_target = target_scaler.transform(target.reshape(-1, 1)).ravel().astype(np.float32)
    assert windows.shape[0] == len(target) == len(dates)
    assert int(train.sum() + validation.sum() + test.sum()) == len(target)

    results: dict[str, dict[str, dict[str, float]]] = {}
    predictions: dict[str, np.ndarray] = {}

    predictions["persistence"] = df["pollution"].to_numpy(dtype=np.float32)[LOOKBACK - 1 : -1]
    predictions["seasonal_persistence"] = df["pollution"].to_numpy(dtype=np.float32)[:-LOOKBACK]

    ridge = Ridge(alpha=1.0)
    ridge.fit(windows[train].reshape(int(train.sum()), -1), scaled_target[train])
    ridge_scaled = ridge.predict(windows.reshape(len(windows), -1))
    predictions["ridge"] = target_scaler.inverse_transform(ridge_scaled.reshape(-1, 1)).ravel()
    joblib.dump(ridge, args.output / "ridge.joblib")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=windows.shape[1:]),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber())
    checkpoint = args.output / "best_lstm.keras"
    history = model.fit(
        windows[train],
        scaled_target[train],
        validation_data=(windows[validation], scaled_target[validation]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=False,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        ],
    )
    best_model = tf.keras.models.load_model(checkpoint)
    lstm_scaled = best_model.predict(windows, batch_size=args.batch_size, verbose=0)
    predictions["lstm"] = target_scaler.inverse_transform(lstm_scaled).ravel()

    for name, values in predictions.items():
        results[name] = {
            "train": score(target[train], values[train]),
            "validation": score(target[validation], values[validation]),
            "test": score(target[test], values[test]),
        }

    test_predictions = pd.DataFrame({"date": dates[test], "actual_pm25": target[test]})
    for name, values in predictions.items():
        test_predictions[name] = values[test]
    test_predictions.to_csv(args.output / "test_predictions.csv", index=False)
    joblib.dump({"features": feature_scaler, "target": target_scaler, "columns": list(features.columns)}, args.output / "scalers.joblib")

    report = {
        "task": "Predict pollution (PM2.5, micrograms per cubic meter) one hour ahead using the prior 24 hours.",
        "data_source": "https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate",
        "split": {
            "train": {"targets": int(train.sum()), "end_exclusive": str(TRAIN_END.date())},
            "validation": {"targets": int(validation.sum()), "start": str(TRAIN_END.date()), "end_exclusive": str(VALIDATION_END.date())},
            "test": {"targets": int(test.sum()), "start": str(VALIDATION_END.date()), "end": str(df["date"].iloc[-1])},
        },
        "features_per_hour": list(features.columns),
        "lookback_hours": LOOKBACK,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "versions": {"tensorflow": tf.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "metrics": results,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
