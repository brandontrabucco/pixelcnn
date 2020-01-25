"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from pixelcnn import pixelcnn
import tensorflow as tf


if __name__ == "__main__":

    model = pixelcnn(
        32,     # input_size
        5,      # num_upconv_layers
        5,      # num_gated_masked_conv_layers
        1024,   # filters
        5)      # kernel_size

    inputs = tf.random.normal([1, 32])
    image = tf.random.uniform(
        [1, 32, 32],
        minval=0,
        maxval=256,
        dtype=tf.dtypes.int32)

    logits = model(inputs)

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        image,
        logits,
        from_logits=True)
    loss = tf.reduce_mean(loss)

    print("loss {}: logits shape: {}".format(
        loss.numpy(), logits.shape))
