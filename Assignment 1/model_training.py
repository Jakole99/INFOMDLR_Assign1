from model_parameter_tester import train_model


### Final model
HYPERPARAMETERS = {
    "k": 10,
    "rnn_type": "SimpleRNN",
    "units": 20,
    "activation": "relu",
    "dropout_rate": 0.0,
    "optimizer_name": "adam",
    "learning_rate": 1e-3,
    "batch_size": 16,
    "epochs": 50,
}

model, history = train_model("Xtrain.mat", **HYPERPARAMETERS)

# Save the model
model.save("model_k10.h5")
