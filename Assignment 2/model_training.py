import random
from pathlib import Path
from model_building import Simple2DConvNet, MEGNet
import math
import h5py
import numpy as np
import tensorflow as tf
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# --------------------------------- CONFIGURATION ---------------------------------

# set up testing type as "intra" or "cross"
SETUP = "intra"
MODEL = "Simple2DConvNet"  # "MEGNet" or "Simple2DConvNet"

DOWNSAMPLE_LIST = [4, 8, 16]  # ≥ 1
NORMALISE_LIST = ["z"]  # "z" or "minmax" or None
BATCH_SIZE_LIST = [8]
EPOCHS_LIST = [20]
EARLY_STOPPING = [10]
WINDOW_LIST = [None, 250, 500]
STRIDE_FRAC_LIST = [0.25, 0.5]
DROP_OUT_LIST = [0.1, 0.25, 0.5]
LEARNING_RATE_LIST = [1e-5, 1e-4, 1e-3]

# non-tunable hyperparameters
VAL_SPLIT = 0.2
CHANNELS = 248  # number of MEG sensors
TIMEPOINTS = 35624  # known fixed length of all MEG recordings
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
# BASE_DIR = BASE_DIR = Path("./data/Final Project data") Google Cola
DATA_ROOT = BASE_DIR / ("Intra" if SETUP == "intra" else "Cross")


TASK_LABELS = {
    "rest": 0,
    "task_story_math": 1,
    "task_working_memory": 2,
    "task_motor": 3,
}


# --------------------------------- FILE OBTAINING ---------------------------------


def get_files():
    # get 'Intra' files
    if SETUP == "intra":
        train = sorted((DATA_ROOT / "train").glob("*.h5"))
        test = sorted((DATA_ROOT / "test").glob("*.h5"))

    # get 'Cross' files
    elif SETUP == "cross":
        train = sorted((DATA_ROOT / "train").glob("*.h5"))
        test = []
        for i in range(1, 4):
            test += sorted((DATA_ROOT / f"test{i}").glob("*.h5"))

    # raise error if there are no files found in the folder
    if not train or not test:
        raise SystemExit(
            f"!! No .h5 files found in {DATA_ROOT} !!. Make sure 'Cross' & 'Intra' folders are in the 'Assignment 2' folder"
        )

    return train, test


# --------------------------------- FILE PARSING ---------------------------------


# Get the label associated with the file
def infer_label(fp: Path):
    name = fp.name.replace(" ", "_")
    for key, label in TASK_LABELS.items():
        if name.startswith(key):
            return label
    raise ValueError(f"Unknown label for {fp.name}")


# Read one dataset from a given .h5 file && apply downsampling + normalisation
def read_h5(fp: Path):
    with h5py.File(fp, "r") as f:
        keys = list(f.keys())
        if len(keys) != 1:
            raise ValueError(
                f"!!Expected exactly 1 dataset in {fp.name}, found: {keys}"
            )
        x = f[keys[0]][()]
    if DOWNSAMPLE > 1:
        x = x[:, ::DOWNSAMPLE]
    TARGET_SAMPLES = TIMEPOINTS // DOWNSAMPLE
    x = x[:, :TARGET_SAMPLES]  # crop to exact length
    if NORMALISE == "z":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-7
        x = (x - mean) / std
    elif NORMALISE == "minmax":
        min_, max_ = x.min(axis=0, keepdims=True), x.max(axis=0, keepdims=True)
        x = (x - min_) / (max_ - min_ + 1e-7)
    return x.astype("float32")


# ----------------------------------- WINDOWING ------------------------------------


def create_windows(data, window_size, stride):
    windows = []
    length = data.shape[1]
    for start in range(0, length - window_size + 1, stride):
        end = start + window_size
        if end > length:
            window = data[
                :, -window_size:
            ]  # last valid window, ensure maximum coverage
        else:
            window = data[:, start:end]
        windows.append(window)
    return windows


# --------------------------------- MAKE DATASETS ---------------------------------


def make_dataset(files, shuffle=False, window_size=None, stride=None, repeat=False):
    # make a list of labels for the given data  (rest= 0,  task_story_math= 1,  task_working_memory= 2,  task_motor= 3)
    labels = [infer_label(f) for f in files]

    def gen():
        for f, y in zip(files, labels):
            x = read_h5(f)  # x shape: (248, SAMPLES)

            if window_size:
                x_windws = create_windows(x, window_size, stride)
                for window in x_windws:
                    yield tf.expand_dims(window, axis=-1), y  # shape: (248, WINDOW, 1)
            else:
                yield tf.expand_dims(x, axis=-1), y  # shape: (248, SAMPLES, 1)

    input_shape = (
        (CHANNELS, TIMEPOINTS // DOWNSAMPLE, 1)
        if not window_size
        else (CHANNELS, window_size, 1)
    )
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if shuffle:
        if window_size:
            # If windowing is applied, shuffle the dataset based on the number of windows
            ds = ds.shuffle(
                len(files) * (TIMEPOINTS // DOWNSAMPLE - window_size) // stride,
                reshuffle_each_iteration=True,
            )
        else:
            ds = ds.shuffle(len(files), reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    return ds.batch(BATCH_SIZE)


def count_samples(files, window_size=None, stride=None):
    if window_size:
        windows_per_file = (TIMEPOINTS // DOWNSAMPLE - window_size) // stride + 1
        return len(files) * windows_per_file
    else:
        return len(files)


# --------------------------------- AGGREGATE RESULTS ---------------------------------


def evaluate_on_trials(model, files, window_size, stride):
    y_true, y_pred = [], []

    for f in files:
        label = infer_label(f)
        x = read_h5(f)

        # Generate windows
        windows = create_windows(x, window_size, stride)
        windows = [
            tf.expand_dims(w, axis=-1) for w in windows
        ]  # shape: (248, window_size, 1)
        windows = tf.stack(windows)  # shape: (num_windows, 248, window_size, 1)

        # Predict per window
        preds = model.predict(windows, verbose=0)  # shape: (num_windows, num_classes)
        majority_vote = np.argmax(
            np.sum(preds, axis=0)
        )  # majority voting over softmax scores

        y_true.append(label)
        y_pred.append(majority_vote)

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    return acc


# ---------------------------------------------------------------------------
#                              RUN TRAINING
# ---------------------------------------------------------------------------

# Create a directory to save results
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = BASE_DIR / "results" / f"run_{run_id}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

results = []

for (
    DOWNSAMPLE,
    NORMALISE,
    BATCH_SIZE,
    EPOCHS,
    WINDOW_SIZE,
    STRIDE_FRAC,
    DROP_OUT,
    LEARNING_RATE,
) in product(
    DOWNSAMPLE_LIST,
    NORMALISE_LIST,
    BATCH_SIZE_LIST,
    EPOCHS_LIST,
    WINDOW_LIST,
    STRIDE_FRAC_LIST,
    DROP_OUT_LIST,
    LEARNING_RATE_LIST,
):
    print("==========================================================")
    print("Running experiment with:")
    print(f"  SETUP             = {SETUP}")
    print(f"  MODEL             = {MODEL}")
    print(f"  DOWNSAMPLE        = {DOWNSAMPLE}")
    print(f"  NORMALISE         = {NORMALISE!r}")
    print(f"  BATCH_SIZE        = {BATCH_SIZE}")
    print(f"  EPOCHS            = {EPOCHS}")
    print(f"  WINDOW_SIZE       = {WINDOW_SIZE}")
    print(f"  STRIDE_FRAC       = {STRIDE_FRAC}")
    print(f"  DROP_OUT          = {DROP_OUT}")
    print(f"  LEARNING_RATE     = {LEARNING_RATE}")
    print("==========================================================\n")

    # Create a unique directory for this combination of hyperparameters
    combo_id = f"combo_{len(results):03d}"
    combo_dir = RUN_DIR / combo_id
    combo_dir.mkdir(parents=True, exist_ok=True)

    if WINDOW_SIZE is not None:
        STRIDE = int(WINDOW_SIZE * STRIDE_FRAC)
    else:
        STRIDE = None

    try:
        # 1. set random seed
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # 2. get all the raw data
        train_files, test_files = get_files()

        # 3. shuffle the raw data and split into validation and train parts
        val_split_idx = int(VAL_SPLIT * len(train_files))
        random.shuffle(train_files)
        val_files = train_files[:val_split_idx]
        train_files_subset = train_files[val_split_idx:]

        # 4. make datasets with the filepaths obtained above
        train_dataset = make_dataset(
            train_files_subset,
            shuffle=True,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            repeat=True,
        )
        val_dataset = make_dataset(val_files, window_size=WINDOW_SIZE, stride=STRIDE)
        test_dataset = make_dataset(test_files, window_size=WINDOW_SIZE, stride=STRIDE)

        if WINDOW_SIZE:
            # If windowing is applied, adjust the model input shape accordingly
            samples = WINDOW_SIZE
        else:
            # If no windowing, use the full timepoints length
            samples = TIMEPOINTS // DOWNSAMPLE

        # 5. build the model using Keras
        if MODEL == "MEGNet":
            model = MEGNet(
                Samples=samples, dropoutRate=DROP_OUT, learning_rate=LEARNING_RATE
            )
        else:
            model = Simple2DConvNet(
                Samples=samples, dropout_rate=DROP_OUT, learning_rate=LEARNING_RATE
            )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING[0],  # or a looped value if you're tuning this
            min_delta=0.001,
            restore_best_weights=True,
        )

        NUM_VAL_BATCHES = math.ceil(
            count_samples(val_files, WINDOW_SIZE, STRIDE) / BATCH_SIZE
        )
        NUM_TEST_BATCHES = math.ceil(
            count_samples(test_files, WINDOW_SIZE, STRIDE) / BATCH_SIZE
        )
        NUM_TRAIN_BATCHES = math.ceil(
            count_samples(train_files_subset, WINDOW_SIZE, STRIDE) / BATCH_SIZE
        )
        print(NUM_TEST_BATCHES)
        print(NUM_VAL_BATCHES)
        print(NUM_TRAIN_BATCHES)

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            validation_steps=NUM_VAL_BATCHES,
            epochs=EPOCHS,
            callbacks=[early_stopping],
            steps_per_epoch=NUM_TRAIN_BATCHES,
            verbose=1,
        )

        # 6. Plot Training History, validation Loss, and Accuracy
        plt.plot(history.history["loss"], label="train loss")
        plt.plot(history.history["val_loss"], label="val loss")
        plt.legend()
        plt.title("Loss over Epochs")
        plt.savefig(combo_dir / "loss.png")
        plt.close()

        plt.plot(history.history["accuracy"], label="train acc")
        plt.plot(history.history["val_accuracy"], label="val acc")
        plt.legend()
        plt.title("Accuracy over Epochs")
        plt.savefig(combo_dir / "accuracy.png")
        plt.close()

        # 7. evaluate the model
        print("\nEvaluating…")
        if WINDOW_SIZE:
            test_acc = evaluate_on_trials(
                model, test_files, window_size=WINDOW_SIZE, stride=STRIDE
            )
        else:
            test_acc = model.evaluate(test_dataset, steps=NUM_TEST_BATCHES, verbose=0)[
                1
            ]
        print(f"Test accuracy: {test_acc:.3f}\n")

        # 8. record and save results
        config = {
            "combo_id": combo_id,
            "test_acc": float(test_acc),
            "setup": SETUP,
            "model_name": MODEL,
            "downsample": DOWNSAMPLE,
            "normalise": NORMALISE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "window_size": WINDOW_SIZE,
            "stride_frac": STRIDE_FRAC,
            "stride": STRIDE,
            "dropout_rate": DROP_OUT,
            "learning_rate": LEARNING_RATE,
        }

        # fill all null with "None"
        config = {k: (v if v is not None else "None") for k, v in config.items()}
        with open(combo_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        results.append(config)
        # Append the current result to a partial CSV after each success
        pd.DataFrame([config]).to_csv(
            RUN_DIR / "summary_partial.csv",
            mode="a",
            header=not (RUN_DIR / "summary_partial.csv").exists(),
            index=False,
        )

    except Exception as e:
        error_path = combo_dir / "error.txt"
        with open(error_path, "w") as f:
            f.write(f"Error: {str(e)}\n")
        print(f"!! COMBINATION FAILED !! due to error: {e}\nLogged to {error_path}")
        continue


# after all combinations, save results to the CSV and print the top 5
df = pd.DataFrame(results)
df = df.sort_values(by="test_acc", ascending=False).reset_index(drop=True)
df.to_csv(RUN_DIR / "summary.csv", index=False)

print("o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o")
print("Top 5 combinations:")
print(df.head(5))
