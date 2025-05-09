from keras.models import load_model
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from functions import (
    plot_actual_vs_predictions,
    preprocess_data,
)


matrixVar = loadmat("Xtest.mat")  # <-- Load the data from a .mat file May 9th
test_data = matrixVar["Xtest"]

# load the model from a .h5 file, this file has a specific k value in the name, try to keyword mach the begin just
model = load_model("model_k10.h5")
k = 10  # <-- Same k value as the final model

# Scale the data to [0, 1] using the MinMaxScaler()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(test_data)

x_scaled, y_scaled = preprocess_data(scaled, k)


y_val_pred_scaled = model.predict(x_scaled)
y_val_true = scaler.inverse_transform(y_scaled)
y_val_pred = scaler.inverse_transform(y_val_pred_scaled)

print("MSE:", mean_squared_error(y_val_true, y_val_pred))
print("MAE:", mean_absolute_error(y_val_true, y_val_pred))
plot_actual_vs_predictions(y_val_true, y_val_pred, n_parts=1)
