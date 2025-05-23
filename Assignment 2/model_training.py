import random
from pathlib import Path
from model_building import build_model
import h5py
import numpy as np
import tensorflow as tf


# --------------------------------- CONFIGURATION ---------------------------------

SETUP = "intra"         # "intra" or "cross"
DOWNSAMPLE = 4          # ≥ 1
NORMALISE = "z"         # "z", "minmax", or None
BATCH_SIZE = 8
EPOCHS = 15
VAL_SPLIT = 0.2
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
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
        raise SystemExit(f"!! No .h5 files found in {DATA_ROOT} !!. Make sure 'Cross' & 'Intra' folders are in the 'Assignment 2' folder")
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
            raise ValueError(f"!!Expected exactly 1 dataset in {fp.name}, found: {keys}")
        x = f[keys[0]][()]
    if DOWNSAMPLE > 1:
        x = x[:, ::DOWNSAMPLE]
    if NORMALISE == "z":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-7
        x = (x - mean) / std
    elif NORMALISE == "minmax":
        min_, max_ = x.min(axis=0, keepdims=True), x.max(axis=0, keepdims=True)
        x = (x - min_) / (max_ - min_ + 1e-7)
    return x.astype("float32")






# --------------------------------- MAKE DATASETS ---------------------------------

def make_dataset(files, shuffle=False):

    # make a list of labels for the given data  (rest= 0,  task_story_math= 1,  task_working_memory= 2,  task_motor= 3)
    labels = [infer_label(f) for f in files]

    # generator that streams (data, label) pairs
    def gen():
        for f, y in zip(files, labels):
            x = read_h5(f) # returns an array of shape (channels, timepoints)
            yield x.T, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 248), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if shuffle:
        ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
    return ds.padded_batch(BATCH_SIZE)









# ---------------------------------------------------------------------------
#                              RUN TRAINING
# ---------------------------------------------------------------------------

# 1. set random seed
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 2. get all the raw data 
train_files, test_files = get_files()

# 3. shuffle the raw data and split into validation and train parts
val_split = int(VAL_SPLIT * len(train_files))
random.shuffle(train_files)
val_files = train_files[:val_split]
train_files = train_files[val_split:]

# 4. make datasets with the filepaths obtained above 
train_dataset = make_dataset(train_files, shuffle=True)
val_dataset = make_dataset(val_files)
test_dataset = make_dataset(test_files)

# 5. build the model using Keras
model = build_model()
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=2)

# 6. evaluate the model
print("\nEvaluating…")
_, test_acc = model.evaluate(test_dataset, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")