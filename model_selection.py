import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functions import train, predict, plot_data

from scipy.io import loadmat

matrixVar = loadmat("Xtrain.mat")
train_data = matrixVar["Xtrain"]
train_data.shape

# scale the data to [0, 1] using the MinMaxScaler() - assignment requirement
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(train_data)

# Set lookback window
k_values = [i for i in range(1, 11)]
results = []

# Loop through different k values to find the best one
for k in k_values:
    # Train the model on 80%, so we can test on the remaining 20% unseen data
    x_test, y_test, model = train(scaled_data, k, split_data=True)

    # Predict the remaining 20% of the data
    y_true, y_pred = predict(x_test, y_test, model, scaler)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    results.append((k, mse, mae, y_true, y_pred))


# plot the results for the different k values
plt.figure(figsize=(10, 5))
plt.plot(k_values, [result[1] for result in results], label="MSE")
plt.plot(k_values, [result[2] for result in results], label="MAE")
plt.title("MSE and MAE vs. Lookback Window Size")
plt.xlabel("Lookback Window Size (k)")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()

# Get the best k values based on MSE and MAE
min_mse = min(results, key=lambda x: x[1])
min_mae = min(results, key=lambda x: x[2])

if min_mse[0] == min_mae[0]:
    print(f"Best k value based on both MSE and MAE: {min_mse[0]}")
    print(f"MSE is: {min_mse[1]}")
    print(f"MAE is: {min_mae[2]}")
else:
    print(
        f"Best k value based on MAE is: {min_mae[0]}, we care more about average performance"
    )
    print(f"MSE is: {min_mse[1]}")
    print(f"MAE is: {min_mae[2]}")

best_k = min_mae[0]
best_k_index = k_values.index(best_k)


# Compare the predictions of the best k value with the true values
plot_data(
    results[best_k_index][3], results[best_k_index][4], n_parts=4
)  # y_true, y_pred, n_parts


# Retrain now on the entire dataset with the best k value
x_test, y_test, final_model = train(scaled_data, best_k, split_data=False)
y_true, y_pred = predict(x_test, y_test, final_model, scaler)

mse = mean_squared_error(y_true, y_pred)
print(f"Training Mean Squared Error on Original Scale: {mse:.4f}")

# Evaluate model performance using Mean Absolute Error on original scale
mae = mean_absolute_error(y_true, y_pred)
print(f"Training Mean Absolute Error on Original Scale: {mae:.4f}")

plot_data(y_true, y_pred, n_parts=4)  # y_true, y_pred, n_parts

# save the model
final_model.save(f"model_k{best_k}.h5")
