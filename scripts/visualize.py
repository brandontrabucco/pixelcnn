"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == "__main__":

    model = tf.keras.models.load_model(
        'models/model.h5', custom_objects={'tf': tf})

    images = tf.zeros([10, 32, 32], dtype=tf.int32)
    labels = tf.range(10, dtype=tf.int32)[:, tf.newaxis, tf.newaxis]

    for i in range(32 * 32):
        logits = model([images, labels])
        images = tf.math.argmax(logits, output_type=tf.int32, axis=3)
        print("{} % [{} / {}]".format(
            float(i) / float(32 * 32) * 100.0, i, 32 * 32))

    r_channel = tf.cast(images % 10, tf.float32) / 9.0
    g_channel = tf.cast((images // 10) % 10, tf.float32) / 9.0
    b_channel = tf.cast((images // 100) % 10, tf.float32) / 9.0

    images = tf.stack([r_channel, g_channel, b_channel], axis=3)

    for i in range(10):
        plt.imshow(images[i, :, :, :].numpy())
        plt.show()
        plt.clf()
