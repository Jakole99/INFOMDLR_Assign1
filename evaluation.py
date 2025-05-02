from keras.models import load_model
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from functions import predict, plot_data, preprocess_data

matrixVar = loadmat("Xtest.mat")  # <-- Load the data from a .mat file May 9th
test_data = matrixVar["Xtest"]

# load the model from a .h5 file, this file has a specific k value in the name, try to keyword mach the begin just
model = load_model("model_k7.h5")

# scale the data to [0, 1] using the MinMaxScaler() - assignment requirement
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(test_data)

x_test, y_test = preprocess_data(scaled_data, 7)  # <-- Same k value as the final model
y_true, y_pred = predict(x_test, y_test, model, scaler)
print("MSE:", mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
plot_data(y_true, y_pred, n_parts=1)
