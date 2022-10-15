import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from keras.utils import dataset_utils


##
## Functions - Kera
##
ALLOWED_FORMATS = (".wav",)

def audio_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    sampling_rate=None,
    output_sequence_length=None,
    ragged=False,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
):
 
    if labels not in ("inferred", None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                "The `labels` argument should be a list/tuple of integer "
                "labels, of the same size as the number of audio files in "
                "the target directory. If you wish to infer the labels from "
                "the subdirectory names in the target directory,"
                ' pass `labels="inferred"`. '
                "If you wish to get a dataset that only contains audio samples "
                f"(no labels), pass `labels=None`. Received: labels={labels}"
            )
        if class_names:
            raise ValueError(
                "You can only pass `class_names` if "
                f'`labels="inferred"`. Received: labels={labels}, and '
                f"class_names={class_names}"
            )
    if label_mode not in {"int", "categorical", "binary", None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", '
            '"binary", '
            f"or None. Received: label_mode={label_mode}"
        )

    if ragged and output_sequence_length is not None:
        raise ValueError(
            "Cannot set both `ragged` and `output_sequence_length`"
        )

    if sampling_rate is not None:
        if not isinstance(sampling_rate, int):
            raise ValueError(
                "`sampling_rate` should have an integer value. "
                f"Received: sampling_rate={sampling_rate}"
            )

        if sampling_rate <= 0:
            raise ValueError(
                "`sampling_rate` should be higher than 0. "
                f"Received: sampling_rate={sampling_rate}"
            )

        if tfio is None:
            raise ImportError(
                "To use the argument `sampling_rate`, you should install "
                "tensorflow_io. You can install it via `pip install "
                "tensorflow-io`."
            )

    if labels is None or label_mode is None:
        labels = None
        label_mode = None

    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed
    )

    if seed is None:
        seed = np.random.randint(1e6)

    file_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels,
        formats=ALLOWED_FORMATS,
        class_names=class_names,
        shuffle=shuffle,
        seed=seed,
        follow_links=follow_links,
    )

    if label_mode == "binary" and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary"`, there must be exactly 2 '
            f"class_names. Received: class_names={class_names}"
        )

    if subset == "both":
        train_dataset, val_dataset = get_training_and_validation_dataset(
            file_paths=file_paths,
            labels=labels,
            validation_split=validation_split,
            directory=directory,
            label_mode=label_mode,
            class_names=class_names,
            sampling_rate=sampling_rate,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )

        train_dataset = prepare_dataset(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            class_names=class_names,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )
        val_dataset = prepare_dataset(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            class_names=class_names,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )
        return train_dataset, val_dataset

    else:
        dataset = get_dataset(
            file_paths=file_paths,
            labels=labels,
            directory=directory,
            validation_split=validation_split,
            subset=subset,
            label_mode=label_mode,
            class_names=class_names,
            sampling_rate=sampling_rate,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )

        dataset = prepare_dataset(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            class_names=class_names,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )
        return dataset

def prepare_dataset(
    dataset,
    batch_size,
    shuffle,
    seed,
    class_names,
    output_sequence_length,
    ragged,
):
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)

        if output_sequence_length is None and not ragged:
            dataset = dataset.padded_batch(
                batch_size, padded_shapes=([None, None], [])
            )
        else:
            dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)

    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    return dataset

def get_training_and_validation_dataset(
    file_paths,
    labels,
    validation_split,
    directory,
    label_mode,
    class_names,
    sampling_rate,
    output_sequence_length,
    ragged,
):
    (
        file_paths_train,
        labels_train,
    ) = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, "training"
    )
    if not file_paths_train:
        raise ValueError(
            f"No training audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    file_paths_val, labels_val = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, "validation"
    )
    if not file_paths_val:
        raise ValueError(
            f"No validation audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    train_dataset = paths_and_labels_to_dataset(
        file_paths=file_paths_train,
        labels=labels_train,
        label_mode=label_mode,
        num_classes=len(class_names),
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
    )

    val_dataset = paths_and_labels_to_dataset(
        file_paths=file_paths_val,
        labels=labels_val,
        label_mode=label_mode,
        num_classes=len(class_names),
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
    )

    return train_dataset, val_dataset

def get_dataset(
    file_paths,
    labels,
    directory,
    validation_split,
    subset,
    label_mode,
    class_names,
    sampling_rate,
    output_sequence_length,
    ragged,
):
    file_paths, labels = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, subset
    )
    if not file_paths:
        raise ValueError(
            f"No audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    dataset = paths_and_labels_to_dataset(
        file_paths=file_paths,
        labels=labels,
        label_mode=label_mode,
        num_classes=len(class_names),
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
    )

    return dataset

def read_and_decode_audio(
    path, sampling_rate=None, output_sequence_length=None
):
    """Reads and decodes audio file."""
    audio = tf.io.read_file(path)

    if output_sequence_length is None:
        output_sequence_length = -1

    audio, default_audio_rate = tf.audio.decode_wav(
        contents=audio, desired_samples=output_sequence_length,desired_channels=1
    )
    if sampling_rate is not None:
        # default_audio_rate should have dtype=int64
        default_audio_rate = tf.cast(default_audio_rate, tf.int64)
        audio = tfio.audio.resample(
            input=audio, rate_in=default_audio_rate, rate_out=sampling_rate
        )
    return audio

def paths_and_labels_to_dataset(
    file_paths,
    labels,
    label_mode,
    num_classes,
    sampling_rate,
    output_sequence_length,
    ragged,
):
    """Constructs a fixed-size dataset of audio and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    audio_ds = path_ds.map(
        lambda x: read_and_decode_audio(
            x, sampling_rate, output_sequence_length
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if ragged:
        audio_ds = audio_ds.map(
            lambda x: tf.RaggedTensor.from_tensor(x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(
            labels, label_mode, num_classes
        )
        audio_ds = tf.data.Dataset.zip((audio_ds, label_ds))
    return audio_ds

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = 'data/instruments/IRMAS-TrainingData'

data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']

train_ds, val_ds =audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=32000,
    subset='both')

label_names = np.array(train_ds.class_names)

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(commands)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Flatten(),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 20
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

model.evaluate(test_spectrogram_ds, return_dict=True)

y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=commands,
            yticklabels=commands,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')


def precision(mx):
    precision_1_base = 0
    precision_2_base = 0
    precision_3_base = 0
    for i in range(3):
        precision_1_base += int(mx[i][0])
    
    if precision_1_base != 0:
        precision_1 = int(mx[0][0])/precision_1_base
    else:
        precision_1 = -1

    for i in range(3):
        precision_2_base += int(mx[i][1])
    
    if precision_2_base != 0:
        precision_2 = int(mx[1][1])/precision_2_base
    else:
        precision_2 = -1

    for i in range(3):
        precision_3_base += int(mx[i][2])
    if precision_3_base != 0:
        precision_3 = int(mx[2][2])/precision_3_base
    else:
        precision_3 = -1

    return (precision_1,precision_2,precision_3)

def recuperacion(mx):
    recuperacion_1_base = 0
    recuperacion_2_base = 0
    recuperacion_3_base = 0
    for i in range(3):
        recuperacion_1_base += int(mx[0][i])
    recuperacion_1 = int(mx[0][0])/recuperacion_1_base

    for i in range(3):
        recuperacion_2_base += int(mx[1][i])
    recuperacion_2 = int(mx[1][1])/recuperacion_2_base

    for i in range(3):
        recuperacion_3_base += int(mx[2][i])
    recuperacion_3 = int(mx[2][2])/recuperacion_3_base
    return (recuperacion_1,recuperacion_2,recuperacion_3)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Exactitud = accuracy
print("Exactitud: "+str(acc[-1]))
# Precision
p = precision(confusion_mtx)
print("Precisión")
print("acoustic-guitar:"+str(p[0]))
print("piano:"+str(p[1]))
print("trumpet:"+str(p[2]))

# Recuperacion
r = recuperacion(confusion_mtx)
print("Recuperacion")
print("acoustic-guitar:"+str(r[0]))
print("piano:"+str(r[1]))
print("trumpet:"+str(r[2]))

plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()