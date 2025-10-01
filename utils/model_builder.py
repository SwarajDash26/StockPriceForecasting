from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_lstm_model(input_shape):
    """
    Creates and compiles a Stacked LSTM model for REGRESSION.

    Parameters:
    input_shape (tuple): Shape of input data (timesteps, features)

    Returns:
    model: Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer with return_sequences=True to pass sequences to next layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout for regularization
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Third LSTM layer (don't return sequences for the last LSTM layer)
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Dense layer to process LSTM outputs
    model.add(Dense(units=25, activation='relu'))
    
    # OUTPUT LAYER FOR REGRESSION: 1 neuron with linear activation
    model.add(Dense(units=1, activation='linear'))
    
    # Compile the model for REGRESSION
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss='mean_squared_error',  # MSE loss for regression
                  metrics=['mae', 'mse'])     # Mean Absolute Error & Mean Squared Error
    
    # Display model architecture
    model.summary()
    
    return model

# Example usage for testing:
if __name__ == "__main__":
    # Test the model creation with a sample input shape
    # This would typically be (n_steps, n_features) = (60, number_of_features)
    sample_input_shape = (60, 14)  # 60 days, 14 features (approx based on our features)
    
    print("Testing model creation...")
    model = create_lstm_model(sample_input_shape)
    print("Model created successfully!")