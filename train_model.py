import pandas as pd
import numpy as np
from utils.data_loader import get_stock_data
from utils.feature_engineer import add_technical_indicators
from utils.preprocess import create_target, prepare_data_for_lstm
from utils.model_builder import create_lstm_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

def calculate_confidence_interval(model, X_train, y_train, X_test, y_test):
    """
    Calculates prediction intervals for regression results.
    """
    # Make predictions
    train_predictions = model.predict(X_train, verbose=0).flatten()
    test_predictions = model.predict(X_test, verbose=0).flatten()
    
    # Calculate errors
    train_errors = y_train - train_predictions
    test_errors = y_test - test_predictions
    
    # Calculate standard deviation of errors (this will be our uncertainty measure)
    error_std = np.std(test_errors)
    
    print(f"   Error Standard Deviation: {error_std:.4f}")
    print(f"   95% Confidence Interval: ±{1.96 * error_std:.4f}")
    
    return error_std, test_predictions, test_errors, train_predictions, train_errors

def plot_predictions_vs_actual(y_true, y_pred, error_std, ticker):
    """
    Creates a comprehensive visualization of prediction performance.
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Scatter plot: Predicted vs Actual
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
    plt.xlabel('Actual Percentage Change')
    plt.ylabel('Predicted Percentage Change')
    plt.title(f'{ticker} - Predicted vs Actual\n(Perfect prediction = red line)')
    plt.grid(True, alpha=0.3)
    
    # 2. Error distribution histogram
    plt.subplot(2, 2, 2)
    errors = y_true - y_pred
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.axvline(x=error_std, color='orange', linestyle='--', label=f'+1 STD ({error_std:.3f})')
    plt.axvline(x=-error_std, color='orange', linestyle='--', label=f'-1 STD ({error_std:.3f})')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Time series of predictions with confidence intervals
    plt.subplot(2, 1, 2)
    plt.plot(y_true, 'b-', label='Actual', alpha=0.7)
    plt.plot(y_pred, 'r-', label='Predicted', alpha=0.7)
    # Add confidence interval band
    plt.fill_between(range(len(y_pred)), 
                    y_pred - 1.96*error_std, 
                    y_pred + 1.96*error_std, 
                    color='orange', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Percentage Change')
    plt.title('Time Series: Actual vs Predicted with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'models/{ticker}_predictions_analysis.png')
    plt.show()
    
    # Print some statistics
    print("\nPrediction Analysis:")
    print(f"Correlation coefficient: {np.corrcoef(y_true, y_pred)[0,1]:.3f}")
    print(f"Percentage within 95% CI: {np.mean((y_true >= y_pred - 1.96*error_std) & (y_true <= y_pred + 1.96*error_std)) * 100:.1f}%")
    print(f"Max overprediction: {np.max(errors):.3f}")
    print(f"Max underprediction: {np.min(errors):.3f}")
    
def train_stock_model(ticker=None, prediction_date=None, future_days=5, n_steps=60):
    if ticker is None:
        raise ValueError("Please provide a ticker (e.g., MSFT, NVDA)")
    """
    Complete training pipeline for the stock prediction model.
    """
    # Use current date if none provided
    if prediction_date is None:
        prediction_date = pd.to_datetime('today').date()
    else:
        prediction_date = pd.to_datetime(prediction_date).date()
    
    print(f"Training model for {ticker} predicting {future_days} days from {prediction_date}")
    print("=" * 60)
    
    # Step 1: Load data
    print("1. Loading data...")
    data = get_stock_data(ticker, prediction_date)
    print(f"   Raw data shape: {data.shape}")
    
    # Step 2: Feature engineering
    print("2. Engineering features...")
    featured_data = add_technical_indicators(data)
    print(f"   Featured data shape: {featured_data.shape}")
    
    # Step 3: Create target and prepare for LSTM
    print("3. Creating target and preparing sequences...")
    data_with_target = create_target(featured_data, future_days=future_days)
    print(f"   Data with target shape: {data_with_target.shape}")
    print(f"   Target distribution: {data_with_target['Target'].value_counts().to_dict()}")
    
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(
        data_with_target, n_steps=n_steps, test_size=0.2
    )
    
    # Step 4: Create and train model
    print("4. Building and training model...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = create_lstm_model(input_shape)
    
    # Use early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Step 5: Evaluate the model
    print("5. Evaluating model...")
    print("5. Evaluating model and calculating confidence intervals...")
    train_loss, train_mae, train_mse = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    
    error_std, test_predictions, test_errors, train_predictions, train_errors = calculate_confidence_interval(
        model, X_train, y_train, X_test, y_test
    )
    
    # Plot comprehensive predictions analysis
    plot_predictions_vs_actual(y_test, test_predictions, error_std, ticker)
    
    print(f"   Training MAE: {train_mae:.4f}")
    print(f"   Test MAE: {test_mae:.4f}")
    print(f"   Test MSE: {test_mse:.4f}")
    # Step 6: Save the model and scaler
    print("6. Saving model and artifacts...")
    model.save(f'models/{ticker}_lstm_model.keras')
    joblib.dump(scaler, f'models/{ticker}_scaler.pkl')
    
    print(f"   Model saved as: models/{ticker}_lstm_model.keras")
    print(f"   Scaler saved as: models/{ticker}_scaler.pkl")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'models/{ticker}_training_history.png')
    plt.show()
    
    return model, history, test_mae  # Return test_mae instead of test_accuracy


if __name__ == "__main__":
    # Install required package if not already installed
    # pip install joblib matplotlib
    
    # Train the model
    model, history, test_mae = train_stock_model(
        ticker='AAPL',
        prediction_date='2023-12-01',
        future_days=5,
        n_steps=60
    )
    
    print(f"\nTraining completed! Final Test MAE: {test_mae:.4f}")
    
    