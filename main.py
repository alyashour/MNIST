import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from gui import run_gui

# NOTE: had to run
# `set TF_ENABLE_ONEDNN_OPTS=1`
# in windows console before running
# src=https://pypi.org/project/tensorflow-intel/

# check installation
print(tf.__version__)

# data preprocessing
# load data
mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# check for nans
print(np.isnan(train_images).any()) # expect false
print(np.isnan(test_images).any()) # expect false

# normalization
# "first layer will expect a single 60_0000x28x28x1 tensor instead of 60_000 28x28x1 tensors."
# We will normalize each value between 0.0 and 1.0 to make the model run faster.
# since each pixel is black-white byte (0-255) we can normalize by dividing by 255.0
input_shape = (28, 28, 1)
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
train_images = train_images / 255.0
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
test_images = test_images / 255.0

# label encoding
# since our data is categorical we need to convert them into "one-hot encodings"
# what is a one hot encoding? https://www.geeksforgeeks.org/ml-one-hot-encoding/
# basically convert the label (1, 2, 3, etc.) to ([1, 0, 0], [0, 1, 0], [0, 0, 1], etc.)
train_labels = tf.one_hot(train_labels.astype(np.int32), depth=10)
test_labels = tf.one_hot(test_labels.astype(np.int32), depth=10)

# visualize one of the images for fun
index = random.randint(0, train_images.shape[0] - 1)
plt.imshow(train_images[index][:,:,0])
plt.title("Sample Image")
plt.show()

# building the cnn model
# define the module
batch_size = 64
num_classes = 10
epochs = 2

layers = tf.keras.layers
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
    layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D(strides=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# compile the model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-8),
    loss='categorical_crossentropy',
    metrics=['acc']
)

# define the callback
class C(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc') > 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = C()

# save the model, weights, and optimizer state
model.save('mnist_cnn_model.keras')

# test
history = model.fit(train_images, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])

# evaluate
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label='Training Loss')
ax[0].plot(history.history['val_loss'], color='r', label='Validation Loss')
ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label='Training Accuracy')
ax[1].plot(history.history['val_acc'], color='r', label='Validation Accuracy')
ax[1].legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

model.summary()

run_gui(model)