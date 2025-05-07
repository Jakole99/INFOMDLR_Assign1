from keras.models import load_model
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from functions import preprocess_data
import numpy as np


matrixVar = loadmat("Xtrain.mat")  # <-- Load the data from a .mat file May 9th
test_data = matrixVar["Xtrain"]

model = load_model("model_k3.h5")
k = 3  # <-- Same k value as the final model

scaler = MinMaxScaler()
scaled = scaler.fit_transform(test_data)

x, y = preprocess_data(scaled, k)

# Not scaling again, but giving its own scaler to the y values
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

initial_window = scaled[-k:]  # last k scaled values from the original input data
x = initial_window.reshape(1, k, 1)

val_pred_scaled_total = []
for _ in range(200):
    val_pred_scaled = model.predict(x)
    val_pred_scaled_total.append(val_pred_scaled[0])
    # Slide the window: keep the last 2 timesteps and append the new prediction
    x = np.concatenate((x[0][1:], [[val_pred_scaled[0][0]]]), axis=0)
    x = x.reshape(1, 3, 1)

val_pred_scaled_total = np.array(val_pred_scaled_total)
print("Scaled predictions:" + str(val_pred_scaled_total))
val_pred_total = scaler_y.inverse_transform(val_pred_scaled_total)
print("Next 200 predictions:" + str(val_pred_total))
