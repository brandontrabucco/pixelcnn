"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from pixelcnn.ops import down_shifted_conv2d
from pixelcnn.ops import concat_elu
from tensorflow.keras import layers
import tensorflow as tf


def gated_resnet(
        x,
        a=None,
        h=None,
        conv2d=down_shifted_conv2d,
        nonlinearity=concat_elu,
        kernel_size=(2, 3),
        dropout_rate=0.1,
        **kwargs
):
    """Build a single Gated Masked Conv2D module.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]
    - a: a 4-Tensor with shape [batch_dim, height, width, channels]
        that represents features from earlier hierarchies
    - h: a 2-Tensor with shape [batch_dim, height, width, channels]
        that conditions image generation

    - conv2d: The type of convolution operation to use in this module,
        must be Keras compatible.
    - nonlinearity: The type of nonlinearity to use in this module,
        must be Keras compatible.
    - kernel_size: An integer or tuple/list of 2 integers, specifying
        the height and width of the 2D convolution window. Can be a single
        integer to specify the same value for all spatial dimensions.

    - dropout_rate: Float between 0 and 1. Fraction of the input
        units to drop.

    Returns:
    - out_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    filters = int(x.shape[-1])

    c = nonlinearity(x)
    c = conv2d(c, filters, kernel_size, **kwargs)

    if a is not None:
        a = nonlinearity(a)
        c = layers.add([c, layers.Conv2D(
            filters,
            1,
            padding='valid',
            data_format='channels_last',
            **kwargs)(a)])

    c = nonlinearity(c)
    c = layers.SpatialDropout2D(
        dropout_rate, data_format='channels_last')(c)
    c = conv2d(c, filters * 2, kernel_size, **kwargs)

    if h is not None:
        h = nonlinearity(h)
        c = layers.add([c, layers.Conv2D(
            filters * 2,
            1,
            padding='valid',
            data_format='channels_last',
            **kwargs)(h)])

    c_a, c_b = layers.Lambda(lambda z: tf.split(z, 2, axis=3))(c)
    return layers.add([x, layers.multiply([
        c_a, layers.Activation(tf.math.sigmoid)(c_b)])])
