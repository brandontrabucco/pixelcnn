"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from pixelcnn.gated_masked_conv import gated_masked_conv
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf


def pixelcnn_plus_plus(
        input_size,
        output_size,
        num_upconv_layers,
        num_gated_masked_conv_layers_per_block=6,
        num_blocks=6,
        filters=64,
        kernel_size=5,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        dropout_rate=0.1,
):
    """Stack many Gated Masked Convolution layers into a Conditional Gated PixelCNN.

    Args:
    - input_size: the cardinality of the vector space of the inputs
    - output_size: the cardinality of the vector space of the outputs
    - num_upconv_layers: the number of Transpose Convolution layers to upscale
        the input vector
    - num_gated_masked_conv_layers_per_block: the number of Gated Masked
        Convolution layers per block
    - num_blocks: the number of blocks each dealing with a different resolution

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

    - dropout_rate: Float between 0 and 1. Fraction of the input units to drop.

    Returns:
    - model: a keras model that accepts input vectors with shape [batch_dim, input_size]
       and returns image logits with shape [batch_dim, height, width, 256]
    """
    tf.debugging.assert_equal(
        num_blocks % 2,
        0,
        message="num_blocks must be an even number")

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

    inputs = layers.Input(shape=[None, None, input_size])
    image = layers.Input(shape=[None, None])

    #########################################
    # Embed image pixels into feature space #
    #########################################

    embeddings = layers.TimeDistributed(
        layers.Embedding(output_size, filters))(image)

    ###########################################
    # Build the Transpose Convolutional stack #
    ###########################################

    x = inputs
    for i in range(num_upconv_layers):
        x = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=(2, 2),
            padding='same',
            output_padding=None,
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
            bias_constraint=bias_constraint)(x)

    x = layers.concatenate([embeddings, x])
    vertical_x = horizontal_x = layers.Conv2D(
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
        bias_constraint=bias_constraint)(x)

    ##############################################
    # Build the Gated Masked Convolutional Stack #
    ##############################################

    for block in range(num_blocks // 2):

        for i in range(num_gated_masked_conv_layers_per_block):
            vertical_x, horizontal_x = gated_masked_conv(
                vertical_x,
                horizontal_x,
                filters * (2 ** block),
                kernel_size,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                dropout_rate=dropout_rate)

        if block < num_blocks // 2 - 1:

            vertical_x = layers.ZeroPadding2D(
                padding=[[vertical_dy, 0], [horizontal_dx, horizontal_dx]],
                data_format='channels_last')(vertical_x)
            vertical_x = layers.Cropping2D(
                cropping=[[0, 1], [0, 0]],
                data_format='channels_last')(vertical_x)
            vertical_x = layers.Conv2D(
                filters * (2 ** (block + 1)),
                (vertical_dy, vertical_dx),
                strides=(2, 2),
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
                bias_constraint=bias_constraint)(vertical_x)

            horizontal_x = layers.ZeroPadding2D(
                padding=[[0, 0], [horizontal_dx, 0]],
                data_format='channels_last')(horizontal_x)
            horizontal_x = layers.Cropping2D(
                cropping=[[0, 0], [0, 1]],
                data_format='channels_last')(horizontal_x)
            horizontal_x = layers.Conv2D(
                filters * (2 ** (block + 1)),
                (horizontal_dy, horizontal_dx),
                strides=(2, 2),
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
                bias_constraint=bias_constraint)(horizontal_x)

    for block in reversed(range(num_blocks // 2)):

        if block < num_blocks // 2 - 1:

            vertical_x = layers.Conv2DTranspose(
                filters * (2 ** block),
                (vertical_dy, vertical_dx),
                strides=(2, 2),
                padding='valid',
                output_padding=(1, 1),
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
                bias_constraint=bias_constraint)(vertical_x)
            vertical_x = layers.Cropping2D(
                cropping=[[vertical_dy, 0], [horizontal_dx, horizontal_dx]],
                data_format='channels_last')(vertical_x)
            vertical_x = layers.ZeroPadding2D(
                padding=[[0, 1], [0, 0]],
                data_format='channels_last')(vertical_x)

            horizontal_x = layers.Conv2DTranspose(
                filters * (2 ** block),
                (horizontal_dy, horizontal_dx),
                strides=(2, 2),
                padding='valid',
                output_padding=(1, 1),
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
                bias_constraint=bias_constraint)(horizontal_x)
            horizontal_x = layers.Cropping2D(
                cropping=[[0, 0], [horizontal_dx, 0]],
                data_format='channels_last')(horizontal_x)
            horizontal_x = layers.ZeroPadding2D(
                padding=[[0, 0], [0, 1]],
                data_format='channels_last')(horizontal_x)

        for i in range(num_gated_masked_conv_layers_per_block):
            vertical_x, horizontal_x = gated_masked_conv(
                vertical_x,
                horizontal_x,
                filters * (2 ** block),
                kernel_size,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                dropout_rate=dropout_rate)

    #######################################
    # Predict image logits for each pixel #
    #######################################

    x = layers.add([
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
            bias_constraint=bias_constraint)(vertical_x)])

    logits = layers.Conv2D(
        output_size,
        (1, 1),
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint)(x)

    model = models.Model(inputs=[inputs, image], outputs=logits)

    return model
