"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from tensorflow.keras import layers
import tensorflow as tf


def concat_elu_backend(x):
    """Concatenated ELU activation function.

    Args:
    - x: a Tensor with arbitrary shape

    Returns:
    - y: a Tensor the same shape as x
    """
    return tf.nn.elu(tf.concat([x, -x], -1))


def concat_elu(x):
    """Concatenated ELU activation function.

    Args:
    - x: a Tensor with arbitrary shape

    Returns:
    - y: a Tensor the same shape as x
    """
    return layers.Lambda(concat_elu_backend)(x)


def down_shift_backend(x):
    """Down shift the image, an alternative to masked convolutions.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]

    Returns:
    - shifted_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    cropped_x = x[:, :x.shape[1] - 1, :, :]
    padding = tf.zeros([
        tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
    return tf.concat([
        padding, cropped_x], 1)


def down_shift(x):
    """Down shift the image, an alternative to masked convolutions.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]

    Returns:
    - shifted_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    return layers.Lambda(down_shift_backend)(x)


def right_shift_backend(x):
    """Right shift the image, an alternative to masked convolutions.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]

    Returns:
    - shifted_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    cropped_x = x[:, :, :x.shape[2] - 1, :]
    padding = tf.zeros([
        tf.shape(x)[0], x.shape[1], 1, x.shape[3]], dtype=x.dtype)
    return tf.concat([
        padding, cropped_x], 2)


def right_shift(x):
    """Right shift the image, an alternative to masked convolutions.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]

    Returns:
    - shifted_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    return layers.Lambda(right_shift_backend)(x)


def down_shifted_conv2d(
        x,
        filters,
        kernel_size,
        **kwargs
):
    """Perform a down shifted Conv2D.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]

    - filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    - kernel_size: An integer or tuple/list of 2 integers, specifying
        the height and width of the 2D convolution window. Can be a single
        integer to specify the same value for all spatial dimensions.

    Returns:
    - out_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    dy = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    dx = kernel_size if isinstance(kernel_size, int) else kernel_size[1]

    x = layers.Lambda(lambda z: tf.pad(
        z, [[0, 0], [dy - 1, 0], [int((
            dx - 1) / 2), int((dx - 1) / 2)], [0, 0]]))(x)

    return layers.Conv2D(
        filters,
        kernel_size,
        padding='valid',
        data_format='channels_last',
        **kwargs)(x)


def down_shifted_conv2d_transpose(
        x,
        filters,
        kernel_size,
        **kwargs
):
    """Perform a down shifted Conv2DTranspose.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]

    - filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    - kernel_size: An integer or tuple/list of 2 integers, specifying
        the height and width of the 2D convolution window. Can be a single
        integer to specify the same value for all spatial dimensions.

    Returns:
    - out_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        padding='valid',
        output_padding=(1, 1),
        data_format='channels_last',
        **kwargs)(x)

    dy = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    dx = kernel_size if isinstance(kernel_size, int) else kernel_size[1]

    return layers.Lambda(
        lambda z: z[:, :(x.shape[1] - dy + 1), int((
            dx - 1) / 2):(x.shape[2] - int((dx - 1) / 2)), :])(x)


def down_right_shifted_conv2d(
        x,
        filters,
        kernel_size,
        **kwargs
):
    """Perform a down and right shifted Conv2D.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]

    - filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    - kernel_size: An integer or tuple/list of 2 integers, specifying
        the height and width of the 2D convolution window. Can be a single
        integer to specify the same value for all spatial dimensions.

    Returns:
    - out_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    dy = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    dx = kernel_size if isinstance(kernel_size, int) else kernel_size[1]

    x = layers.Lambda(lambda z: tf.pad(
        z, [[0, 0], [dy - 1, 0], [dx - 1, 0], [0, 0]]))(x)

    return layers.Conv2D(
        filters,
        kernel_size,
        padding='valid',
        data_format='channels_last',
        **kwargs)(x)


def down_right_shifted_conv2d_transpose(
        x,
        filters,
        kernel_size,
        **kwargs
):
    """Perform a down and right shifted Conv2DTranspose.

    Args:
    - x: a 4-Tensor with shape [batch_dim, height, width, channels]

    - filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    - kernel_size: An integer or tuple/list of 2 integers, specifying
        the height and width of the 2D convolution window. Can be a single
        integer to specify the same value for all spatial dimensions.

    Returns:
    - out_x: a 4-Tensor with shape [batch_dim, height, width, channels]
    """
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        padding='valid',
        output_padding=(1, 1),
        data_format='channels_last',
        **kwargs)(x)

    dy = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    dx = kernel_size if isinstance(kernel_size, int) else kernel_size[1]

    return layers.Lambda(
        lambda z: z[:, :(
            x.shape[1] - dy + 1):, :(x.shape[2] - dx + 1), :])(x)
