from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def create_lstm_model(input_shape, learning_rate: float = 1e-3, show_summary: bool = True):
    """
    Creates and compiles a compact LSTM model for REGRESSION.

    Parameters
    ----------
    input_shape : tuple
        (timesteps, features), e.g. (60, 14)
    learning_rate : float
        Adam learning rate
    show_summary : bool
        If True, prints model.summary()

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model
    """
    model = Sequential()

    # Compact stacked LSTM (enough for daily data)
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    # Small dense head
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="linear"))

    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mae", "mse"],
    )

    if show_summary:
        model.summary()

    return model


if __name__ == "__main__":
    # Quick self-test
    m = create_lstm_model((60, 14), show_summary=True)
    print("Model build OK.")