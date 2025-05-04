import numpy as np
import pandas as pd
from itertools import product
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

# Import helper functions (adjust the import path as needed)
from functions import preprocess_data  




###-----  Model builder function  -----###

def build_model(rnn_type, units, activation, dropout_rate, input_shape):
    model = Sequential()
    RNNClass = {
        'SimpleRNN': SimpleRNN,
        'LSTM': LSTM,
        'GRU': GRU
    }[rnn_type]

    model.add(RNNClass(
        units,
        activation=activation,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        input_shape=input_shape
    ))
    
    model.add(Dense(1))
    return model



###-----  Hyperparameter combination testing  -----###

def tune_hyperparameters(data_path, k_values, rnn_types, unit_options, activations, dropout_rates, optimizers, learning_rates, batch_sizes, epochs_list, train_val_split = 0.8):

    raw     = loadmat(data_path)['Xtrain']
    scaler  = MinMaxScaler()
    scaled  = scaler.fit_transform(raw)
    results = []

    # Go over all parameter combinations 
    for (k, rnn_type, units, activation, dropout_rate, opt_name, lr, batch_size, epochs) in product(
            k_values, rnn_types, unit_options, activations, dropout_rates, optimizers, learning_rates, batch_sizes, epochs_list):
        
        # prepare data
        x, y            = preprocess_data(scaled, k)
        split_index     = int(len(x) * train_val_split)
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # build model
        model     = build_model(rnn_type, units, activation, dropout_rate, (k, 1))
        optimizer = {
            'adam':    Adam(learning_rate=lr),
            'rmsprop': RMSprop(learning_rate=lr),
            'sgd':     SGD(learning_rate=lr)
        }[opt_name]
        model.compile(optimizer=optimizer, loss='mse')

        # training
        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # prediction and evaluation 
        y_test_pred_scaled = model.predict(x_test)
        y_test_true = scaler.inverse_transform(y_test)
        y_test_pred = scaler.inverse_transform(y_test_pred_scaled)

        mse = mean_squared_error(y_test_true, y_test_pred)
        mae = mean_absolute_error(y_test_true, y_test_pred)

        # show results 
        results.append({
            'k':             k,
            'rnn_type':      rnn_type,
            'units':         units,
            'activation':    activation,
            'dropout':       dropout_rate,
            'optimizer':     opt_name,
            'learning_rate': lr,
            'batch_size':    batch_size,
            'epochs':        epochs,
            'val_mse':       mse,
            'val_mae':       mae
        })

    results_data = pd.DataFrame(results)
    results_data = results_data.sort_values(by='val_mse').reset_index(drop=True)
    results_data.to_csv('hyperparam_tuning_results.csv', index=False)

    print("best parameter combinations:")
    print(results_data.head())
    return results_data