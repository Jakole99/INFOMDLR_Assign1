import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv2D,
    DepthwiseConv2D,
    SeparableConv2D,
    MaxPooling1D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Add,
    Dropout,
    AveragePooling2D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    SpatialDropout2D,
    Dense,
    SpatialDropout1D,
    Flatten,
)
from tensorflow.keras.constraints import max_norm


def Simple2DConvNet(nb_classes=4, Chans=248, Samples=None, dropout_rate=0.5):
    input_layer = Input(shape=(Chans, Samples, 1))

    x = Conv2D(16, (3, 11), padding="same", activation="relu")(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((1, 4))(x)

    x = Conv2D(32, (3, 11), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((1, 4))(x)

    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(nb_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# MEGNet architecture implemented as described in:
# Nguyen, T.-L., & Chau, T. T. (2024). MEGNet: A Compact Convolutional Neural Network
# for MEG-based Brain–Computer Interfaces. IEEE Access.
# https://github.com/Charliebond125/MEGNet.git.
#
# Original paper: https://ieeexplore.ieee.org/document/10385695
def MEGNet(
    nb_classes=4,
    Chans=248,
    Samples=None,
    dropoutRate=0.5,
    kernLength=64,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
):
    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(Chans, Samples, 1),
        use_bias=False,
    )(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)
    model = Model(inputs=input1, outputs=softmax)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
