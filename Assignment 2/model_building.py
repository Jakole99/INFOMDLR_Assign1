import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation, Add, Dropout, GlobalAveragePooling1D, Dense, SpatialDropout1D)



def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, 248)),
        tf.keras.layers.Conv1D(128, 7, strides=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 5, strides=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(512, 3, strides=2, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model