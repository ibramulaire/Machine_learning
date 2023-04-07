# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
matplotlib.use('Agg')
tf.random.set_seed(42)
np.random.seed(42)

# %%
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# %%
X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. - 1. # reshape and rescale

# %%
codings_size = 100

# %% [markdown]
# generateur

# %%
generator = keras.models.Sequential([
keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
keras.layers.Reshape([7, 7, 128]),
keras.layers.BatchNormalization(),
keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",
activation="selu"),
keras.layers.BatchNormalization(),
keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME",
activation="tanh"),
])

# %% [markdown]
# discriminateur

# %%
discriminator = keras.models.Sequential([
keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
activation=keras.layers.LeakyReLU(0.2),
input_shape=[28, 28, 1]),
keras.layers.Dropout(0.4),
keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
activation=keras.layers.LeakyReLU(0.2)),
keras.layers.Dropout(0.4),
keras.layers.Flatten(),
keras.layers.Dense(1, activation="sigmoid")])

# %%
gan = keras.models.Sequential([generator, discriminator])

# %%
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False

# %%
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# %%
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

# %% [markdown]
# affichage des images 

# %%
def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

# %% [markdown]
# enregistrement des images 

# %%
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join("image", fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
     plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# %%


# %%
def save_img(images):
    path = os.path.join("images", fig_id + "." + fig_extension)
    images = np.squeeze(images, axis=-1)

# %% [markdown]
# entrainement 

# %%
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            with tf.device('/gpu:0'):
                gan.train_on_batch(noise, y2)
    plot_multiple_images(generated_images, 8)
    save_fig("fashion_mnist_plot"+str(epoch))

# %%
import time

start_time = time.time()

train_gan(gan, dataset, batch_size, codings_size,1)

end_time = time.time()

elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time)


# %%
noise = tf.random.normal(shape=[batch_size, codings_size])
generated_images = generator(noise)
#plot_multiple_images(generated_images, 8)
save_fig("dcgan_generated_images_plot", tight_layout=False)

# %%
images = np.squeeze(generated_images, axis=-1)
for index, image in enumerate(images):
    plt.imsave("images/"+str(index)+".png",image, cmap="binary")
    



