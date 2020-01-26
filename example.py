"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from pixelcnn import ConditionalPixelCNNPlusPlus
import tensorflow as tf


if __name__ == "__main__":

    model = ConditionalPixelCNNPlusPlus(
        256,    # output_size
        32,     # conditional_vector_size
        image_height=32,
        image_width=32,
        conditional_height=1,
        conditional_width=1,
        num_preprocess_layers=5,
        num_modules=3,
        num_layers_per_module=6,
        filters=256,
        dropout_rate=0.1)

    inputs = tf.random.normal([12, 1, 1, 32])
    images = tf.random.uniform(
        [12, 32, 32],
        maxval=256,
        dtype=tf.dtypes.int32)

    logits = model([images, inputs])

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        images,
        logits,
        from_logits=True)
    loss = tf.reduce_mean(loss)

    print("loss {}: logits shape: {}".format(
        loss.numpy(), logits.shape))
