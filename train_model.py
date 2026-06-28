import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils.data_loader import get_stock_data
from utils.feature_engineer import add_technical_indicators
from utils.preprocess import create_target, prepare_data_for_lstm
from utils.model_builder import create_lstm_model
from utils.sentiment import build_daily_sentiment_features


def _ensure_models_dir():
    os.makedirs("models", exist_ok=True)


def compute_error_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute std dev of residuals on a holdout set.
    Used as a simple uncertainty proxy.
    """
    residuals = (y_true - y_pred).astype(float)
    return float(np.std(residuals))


def save_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_training_history(history, ticker: str, save_path: str):
    plt.figure(figsize=(12, 5))

    # MAE
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("mae", []), label="train_mae")
    plt.plot(history.history.get("val_mae", []), label="val_mae")
    plt.title(f"{ticker} - MAE")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title(f"{ticker} - Loss (MSE)")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, ticker: str, save_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2)
    plt.title(f"{ticker} - Predicted vs Actual (Return)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def train_stock_model(
    ticker: str,
    prediction_date=None,
    future_days: int = 5,
    n_steps: int = 60,
    test_size: float = 0.2,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    verbose: int = 1,
):
    """
    End-to-end training pipeline:
      data -> indicators -> sentiment features -> target -> sequences -> train -> evaluate -> save artifacts

    Returns:
      model, metrics_dict
    """
    if not ticker:
        raise ValueError("ticker is required (e.g., 'AAPL').")

    _ensure_models_dir()

    if prediction_date is None:
        prediction_date = pd.to_datetime("today").date()
    else:
        prediction_date = pd.to_datetime(prediction_date).date()

    print(f"\nTraining {ticker} | predict {future_days}-day return | cutoff={prediction_date}")
    print("=" * 72)

    # 1) Load data
    print("1) Loading data...")
    raw = get_stock_data(ticker, prediction_date)
    print(f"   Raw rows: {len(raw)} | cols: {list(raw.columns)}")

    # 2) Technical features
    print("2) Engineering technical features...")
    feats = add_technical_indicators(raw)
    print(f"   Rows after indicators: {len(feats)} | feature cols: {list(feats.columns)}")

    # 2.1) Sentiment features aligned by date
    print("2.1) Building sentiment features...")
    sent_daily, scored_headlines = build_daily_sentiment_features(
        ticker=ticker,
        market_index=feats.index,
        max_days_back_news_pull=120,
        company_name=None,
    )

    feats = feats.join(sent_daily, how="left")
    feats["Sentiment_MeanCompound"] = feats["Sentiment_MeanCompound"].fillna(0.0)
    feats["Sentiment_HeadlineCount"] = feats["Sentiment_HeadlineCount"].fillna(0.0)
    feats["Sentiment_Vote"] = feats["Sentiment_Vote"].fillna(0.0)

    print(
        "   Added sentiment columns: "
        "Sentiment_MeanCompound, Sentiment_HeadlineCount, Sentiment_Vote"
    )
    print(f"   Headlines fetched/scored: {len(scored_headlines)}")

    # 3) Target
    print("3) Creating target...")
    df = create_target(feats, future_days=future_days)
    print(f"   Rows after target drop: {len(df)}")

    # 4) Sequences + leakage-free scaling
    print("4) Preparing sequences...")
    X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data_for_lstm(
        df, n_steps=n_steps, test_size=test_size
    )

    # 5) Model
    print("5) Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape, learning_rate=learning_rate, show_summary=True)

    early = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1,
    )

    print("6) Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, reduce_lr],
        verbose=verbose,
    )

    print("7) Evaluating...")
    train_loss, train_mae, train_mse = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)

    y_pred_test = model.predict(X_test, verbose=0).flatten().astype(float)
    error_std = compute_error_std(y_test, y_pred_test)
    ci95 = 1.96 * error_std

    metrics = {
        "ticker": ticker,
        "prediction_date": str(prediction_date),
        "future_days": int(future_days),
        "n_steps": int(n_steps),
        "test_size": float(test_size),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "test_mse": float(test_mse),
        "error_std": float(error_std),
        "ci95": float(ci95),
        "num_train_sequences": int(len(X_train)),
        "num_test_sequences": int(len(X_test)),
        "feature_columns": feature_columns,
        "num_headlines_used_source": int(len(scored_headlines)),
    }

    print(f"   Train MAE: {train_mae:.5f}")
    print(f"   Test  MAE: {test_mae:.5f}")
    print(f"   Test  MSE: {test_mse:.5f}")
    print(f"   Residual STD (test): {error_std:.5f} | 95% CI: ±{ci95:.5f}")

    # 8) Save artifacts
    print("8) Saving artifacts...")
    model_path = f"models/{ticker}_lstm_model.keras"
    scaler_path = f"models/{ticker}_scaler.pkl"
    cols_path = f"models/{ticker}_feature_columns.json"
    meta_path = f"models/{ticker}_error_std.json"
    hist_path = f"models/{ticker}_training_history.png"
    scatter_path = f"models/{ticker}_pred_vs_actual.png"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    save_json(cols_path, {"feature_columns": feature_columns})
    save_json(meta_path, {"error_std": error_std, "ci95": ci95})

    plot_training_history(history, ticker, hist_path)
    plot_pred_vs_actual(y_test, y_pred_test, ticker, scatter_path)

    print(f"   Saved model:   {model_path}")
    print(f"   Saved scaler:  {scaler_path}")
    print(f"   Saved cols:    {cols_path}")
    print(f"   Saved meta:    {meta_path}")
    print(f"   Saved plots:   {hist_path}, {scatter_path}")

    return model, metrics


if __name__ == "__main__":
    # Default test run
    train_stock_model(
        ticker="AAPL",
        prediction_date="2023-12-01",
        future_days=5,
        n_steps=60,
        test_size=0.2,
        epochs=15,
        batch_size=32,
        learning_rate=1e-3,
        verbose=1,
    )