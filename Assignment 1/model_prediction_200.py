from keras.models import load_model
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from functions import preprocess_data
import numpy as np
from functions import plot_predictions

matrixVar = loadmat("Xtrain.mat")
test_data = matrixVar["Xtrain"]

model = load_model("model_k10.h5")
k = 10  # <-- Same k value as the final model

scaler = MinMaxScaler()
scaled = scaler.fit_transform(test_data)

x_scaled, y_scaled = preprocess_data(scaled, k)

initial_window = y_scaled[-k:]
x_scaled = initial_window.reshape(1, k, 1)

val_pred_scaled_total = []
for _ in range(200):
    val_pred_scaled = model.predict(x_scaled)
    val_pred_scaled_total.append(val_pred_scaled[0])
    x_scaled = np.concatenate((x_scaled[0][1:], [[val_pred_scaled[0][0]]]), axis=0)
    x_scaled = x_scaled.reshape(1, k, 1)

val_pred_scaled_total = np.array(val_pred_scaled_total)
print("Scaled predictions:" + str(val_pred_scaled_total))


def replace_nan_and_inf(array):
    array = array.flatten()
    cleaned = np.copy(array)

    # Start with the first valid number
    last_valid = None
    for i in range(len(array)):
        if np.isfinite(array[i]):
            last_valid = array[i]
        else:
            cleaned[i] = last_valid  # Replace inf and nan with last valid value

    return cleaned.reshape(-1, 1)  # Restore original shape


val_pred_scaled_total = replace_nan_and_inf(val_pred_scaled_total)

val_pred_total = scaler.inverse_transform(val_pred_scaled_total)
print("Next 200 predictions:" + str(val_pred_total))


# plot the predictions
plot_predictions(val_pred_total, k)
