import numpy as np
import pandas as pd
from itertools import product
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Dense
from keras.optimizers import Adam, RMSprop, SGD
from functions import preprocess_data, plot_validation_and_training_loss


###-----  Model builder function  -----###
def build_model(rnn_type, units, activation, dropout_rate, input_shape):
    model = Sequential()
    RNNClass = {"SimpleRNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}[rnn_type]

    model.add(
        RNNClass(
            units,
            activation=activation,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            input_shape=input_shape,
        )
    )

    model.add(Dense(1))
    return model


###-----  Hyperparameter combination testing  -----###


def tune_hyperparameters(
    data_path,
    k_values,
    rnn_types,
    unit_options,
    activations,
    dropout_rates,
    optimizers,
    learning_rates,
    batch_sizes,
    epochs_list,
):
    raw = loadmat(data_path)["Xtrain"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)
    results = []

    # Go over all parameter combinations
    for (
        k,
        rnn_type,
        units,
        activation,
        dropout_rate,
        opt_name,
        lr,
        batch_size,
        epochs,
    ) in product(
        k_values,
        rnn_types,
        unit_options,
        activations,
        dropout_rates,
        optimizers,
        learning_rates,
        batch_sizes,
        epochs_list,
    ):
        # prepare data
        x, y = preprocess_data(scaled, k)

        n_folds = 5
        train_window = int(len(x) * 0.4)  # Fixed-size training window 40%
        val_window = int((len(x) - train_window) / n_folds)

        mse_folds = []
        mae_folds = []
        histories = []

        for fold in range(n_folds):
            train_start = fold * val_window
            train_end = train_start + train_window
            val_end = train_end + val_window

            if val_end > len(x):
                break

            x_train, y_train = x[train_start:train_end], y[train_start:train_end]
            print("train_start: ", train_start)
            print("train_end: ", train_end)
            x_val, y_val = x[train_end:val_end], y[train_end:val_end]
            print("val_end: ", val_end)

            x_train, y_train = x[:train_end], y[:train_end]
            x_val, y_val = x[train_end:val_end], y[train_end:val_end]

            model = build_model(rnn_type, units, activation, dropout_rate, (k, 1))
            optimizer = {
                "adam": Adam(learning_rate=lr),
                "rmsprop": RMSprop(learning_rate=lr),
                "sgd": SGD(learning_rate=lr),
            }[opt_name]
            model.compile(optimizer=optimizer, loss="mse")

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

            histories.append(history.history)
            print(histories)

            # correct inverse transform using the y scaler
            y_val_pred_scaled = model.predict(x_val)
            y_val_true = scaler.inverse_transform(y_val)
            y_val_pred = scaler.inverse_transform(y_val_pred_scaled)

            mse = mean_squared_error(y_val_true, y_val_pred)
            mae = mean_absolute_error(y_val_true, y_val_pred)

            print("MSE: ", mse)
            print("MAE: ", mae)

            mse_folds.append(mse)
            mae_folds.append(mae)

            plot_validation_and_training_loss(history.history, k, rnn_type)

        # Get list of lists for each metric
        all_train_losses = [h["loss"] for h in histories]
        all_val_losses = [h["val_loss"] for h in histories]

        # Convert to numpy arrays to compute mean
        avg_train_loss = np.mean(all_train_losses, axis=0)
        avg_val_loss = np.mean(all_val_losses, axis=0)

        # Combine into one averaged history dictionary
        average_history = {
            "loss": avg_train_loss.tolist(),
            "val_loss": avg_val_loss.tolist(),
        }
        print(average_history)

        # plot_validation_and_training_loss(average_history, k, rnn_type)
        avg_mse = np.mean(mse_folds)
        avg_mae = np.mean(mae_folds)

        results.append(
            {
                "k": k,
                "rnn_type": rnn_type,
                "units": units,
                "activation": activation,
                "dropout": dropout_rate,
                "optimizer": opt_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "avg_val_mse": avg_mse,
                "avg_val_mae": avg_mae,
            }
        )

    results_data = pd.DataFrame(results)
    results_data = results_data.sort_values(by="avg_val_mae").reset_index(drop=True)
    results_data.to_csv("hyperparam_tuning_results.csv", index=False)

    print("best parameter combinations:")
    print(results_data.head())

    return results_data


def train_model(
    data_path,
    k,
    rnn_type,
    units,
    activation,
    dropout_rate,
    optimizer_name,
    learning_rate,
    batch_size,
    epochs,
):
    raw = loadmat(data_path)["Xtrain"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)

    # prepare data
    x, y = preprocess_data(scaled, k)

    x_train, y_train = x, y

    # build model
    model = build_model(rnn_type, units, activation, dropout_rate, (k, 1))
    optimizer = {
        "adam": Adam(learning_rate=learning_rate),
        "rmsprop": RMSprop(learning_rate=learning_rate),
        "sgd": SGD(learning_rate=learning_rate),
    }[optimizer_name]
    model.compile(optimizer=optimizer, loss="mse")

    # train
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    # plot_validation_and_training_loss(history, k, rnn_type)

    return model, history
