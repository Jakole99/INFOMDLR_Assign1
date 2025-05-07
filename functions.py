import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(scaled_data, k):
    # Prepare sequences
    x, y = [], []
    for i in range(len(scaled_data) - k):
        x.append(scaled_data[i : i + k])
        y.append(scaled_data[i + k])

    x = np.array(x)
    y = np.array(y)

    return x, y


# Create a plot to visually assess the model's performance
def plot_actual_vs_predictions(y_true, y_pred, n_parts: int):
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


def plot_MSE(data):
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=data,
        x="k",
        y="avg_val_mse",
        hue="rnn_type",
        style="rnn_type",
        markers=True,
        dashes=True,
        palette="tab10",
    )
    plt.title("Validation MSE for Different k and RNN Types")
    plt.xlabel("Window Size (k)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_MAE(data):
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=data,
        x="k",
        y="avg_val_mae",
        hue="rnn_type",
        style="rnn_type",
        markers=True,
        dashes=False,  # solid lines
        palette="tab10",
    )
    plt.title("Validation MAE for Different k and RNN Types")
    plt.xlabel("Window Size (k)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_validation_and_training_loss(history, k, rnn_type):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Loss Curve (k={k}, RNN={rnn_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
