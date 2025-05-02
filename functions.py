import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense


def preprocess_data(scaled_data, k):
    # Prepare sequences
    x, y = [], []
    for i in range(len(scaled_data) - k):
        x.append(scaled_data[i : i + k])
        y.append(scaled_data[i + k])

    x = np.array(x)
    y = np.array(y)

    return x, y


def train(scaled_data, k, split_data: bool = False):
    # Prepare sequences
    x, y = preprocess_data(scaled_data, k)

    if split_data:
        # Do a 80/20 split for training and testing
        split_index = int(len(x) * 0.8)
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
    else:
        x_train, y_train = x, y
        x_test, y_test = x, y

    # Build the RNN model
    model = Sequential([SimpleRNN(50, activation="relu", input_shape=(k, 1)), Dense(1)])

    # Compile the model with Mean Squared Error loss and Adam optimizer
    model.compile(optimizer="adam", loss="mse")

    # Train the model for 200 epochs on the training data
    history = model.fit(x_train, y_train, epochs=200, verbose=0)

    # Plot training loss over epochs for visual evaluation
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.title(f"Model Loss Over Epochs (k={k}, Optimizer=adam)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.show()

    return x_test, y_test, model


def predict(x_test, y_test, model, scaler):
    # Predict
    predicted_scaled = model.predict(x_test)

    # Inverse transform predictions and targets back to original scale for evaluation
    y_predicted = scaler.inverse_transform(predicted_scaled)
    y_true = scaler.inverse_transform(y_test)

    mse_per_prediction = [
        mean_squared_error(y_true[i : i + 1], y_predicted[i : i + 1])
        for i in range(len(y_true))
    ]

    # Plot MSE for each prediction on the original scale
    plt.figure(figsize=(10, 5))
    plt.plot(mse_per_prediction, label="MSE per Prediction on Original Scale")
    plt.title("Mean Squared Error per Prediction (on Original Scale)")
    plt.xlabel("Sample Index")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_true, y_predicted


# Create a plot to visually assess the model's performance
def plot_data(y_true, y_pred, n_parts: int):
    size = len(y_true)
    step = size // n_parts

    for i in range(n_parts):
        start = i * step
        end = (i + 1) * step if i != n_parts - 1 else size

        plt.figure(figsize=(10, 5))
        plt.plot(y_true[start:end], label="Actual", alpha=1, linestyle="-")
        plt.plot(y_pred[start:end], label="Predicted", alpha=0.7, linestyle="--")
        plt.legend()
        plt.title(f"Model Predictions vs Actual Values (Part {i + 1})")
        plt.xlabel("Time Step")
        plt.ylabel("Laser Measurement")
        plt.show()
