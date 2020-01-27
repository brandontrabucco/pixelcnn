"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


import tensorflow_datasets as tfds
import tensorflow as tf


if __name__ == "__main__":

    model = tf.keras.models.load_model(
        'models/model.h5', custom_objects={'tf': tf})

    test_ds = tfds.load("cifar10", split="test")
    test_ds = test_ds.shuffle(1024).batch(32).repeat(1)
    test_ds = test_ds.prefetch(10)

    for i, example in enumerate(test_ds):

        images = tf.cast(tf.cast(
            example["image"], tf.float32) / 25.6, tf.int32)

        images = (images[:, :, :, 0] +
                  images[:, :, :, 1] * 10 +
                  images[:, :, :, 2] * 100)

        labels = tf.cast(
            example["label"], tf.int32)[:, tf.newaxis, tf.newaxis]

        bits_per_dim = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                images, model([images, labels]), from_logits=True))

        tf.print("Iteration:", i, "Bits Per Dim:", bits_per_dim)
