"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from tensorflow.keras import layers
import tensorflow as tf


def gated_masked_conv(
        vertical_x,
        horizontal_x,
        filters,
        kernel_size,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
):
    """Perform a Gated Masked Convolution forward pass.
    
    Args:
    - vertical_x: input features with shape [batch_dim, height, width, channels]
    - horizontal_x: input features with shape [batch_dim, height, width, channels]

    - filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
    - kernel_size: An integer or tuple/list of 2 integers, specifying the height
        and width of the 2D convolution window. Can be a single integer to specify
        the same value for all spatial dimensions.

    - activation: Activation function to use. If you don't specify anything, no
        activation is applied (ie. "linear" activation: a(x) = x).
    - use_bias: Boolean, whether the layer uses a bias vector.

    - kernel_initializer: Initializer for the kernel weights matrix.
    - bias_initializer: Initializer for the bias vector.

    - kernel_regularizer: Regularizer function applied to the kernel weights matrix.
    - bias_regularizer: Regularizer function applied to the bias vector.
    - activity_regularizer: Regularizer function applied to the output of the layer
        (its "activation").

    - kernel_constraint: Constraint function applied to the kernel matrix.
    - bias_constraint: Constraint function applied to the bias vector.

    Returns:
    - vertical_x: output features with shape [batch_dim, height, width, channels]
    - horizontal_x: output features with shape [batch_dim, height, width, channels]
    """
    dy = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    tf.debugging.assert_equal(
        dy % 2,
        1,
        message="filter height must be an odd number")

    dx = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
    tf.debugging.assert_equal(
        dx % 2,
        1,
        message="filter width must be an odd number")

    vertical_dy = dy // 2
    vertical_dx = dx

    horizontal_dy = 1
    horizontal_dx = dx // 2

    #######################################
    # Mask the vertical convolution stack #
    #######################################

    vertical_activations = layers.ZeroPadding2D(
        padding=[[vertical_dy, 0], [horizontal_dx, horizontal_dx]],
        data_format='channels_last')(vertical_x)

    vertical_activations = layers.Cropping2D(
        cropping=[[0, 1], [0, 0]],
        data_format='channels_last')(vertical_activations)

    vertical_activations = layers.Conv2D(
        filters * 2,
        (vertical_dy, vertical_dx),
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint)(vertical_activations)

    #########################################
    # Mask the horizontal convolution stack #
    #########################################

    horizontal_activations = layers.ZeroPadding2D(
        padding=[[0, 0], [horizontal_dx, 0]],
        data_format='channels_last')(horizontal_x)

    horizontal_activations = layers.Cropping2D(
        cropping=[[0, 0], [0, 1]],
        data_format='channels_last')(horizontal_activations)

    horizontal_activations = layers.Conv2D(
        filters * 2,
        (horizontal_dy, horizontal_dx),
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint)(horizontal_activations)

    #########################################
    # Gate the horizontal convolution stack #
    #########################################

    horizontal_activations = layers.add([
        horizontal_activations,
        layers.Conv2D(
            filters * 2,
            (1, 1),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)(vertical_activations)])

    horizontal_activations_a, horizontal_activations_b = layers.Lambda(
        lambda x: tf.split(x, 2, axis=3))(horizontal_activations)

    horizontal_activations = layers.multiply([
        layers.Activation(tf.math.tanh)(horizontal_activations_a),
        layers.Activation(tf.math.sigmoid)(horizontal_activations_b)])

    horizontal_activations = layers.add([
        horizontal_x,
        layers.Conv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)(horizontal_activations)])

    #######################################
    # Gate the vertical convolution stack #
    #######################################

    vertical_activations_a, vertical_activations_b = layers.Lambda(
        lambda x: tf.split(x, 2, axis=3))(vertical_activations)

    vertical_activations = layers.multiply([
        layers.Activation(tf.math.tanh)(vertical_activations_a),
        layers.Activation(tf.math.sigmoid)(vertical_activations_b)])

    return vertical_activations, horizontal_activations
