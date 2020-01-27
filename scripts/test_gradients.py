"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from pixelcnn import PixelCNNPlusPlus
import tensorflow as tf


if __name__ == "__main__":

    model = PixelCNNPlusPlus(
        1000,
        image_height=32,
        image_width=32,
        image_is_discrete=False,
        num_modules=3,
        num_layers_per_module=6,
        filters=64,
        dropout_rate=0.1)

    with tf.GradientTape(persistent=True) as tape:

        images = tf.random.normal([1, 32, 32, 64])
        tape.watch(images)
        logits = model(images)
        slice = logits[0, 1, 5, :]

    print(tape.gradient(slice, images)[0, :, :, 0].numpy().tolist())
