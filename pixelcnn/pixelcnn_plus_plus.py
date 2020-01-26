"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from pixelcnn.gated_resnet import gated_resnet
from pixelcnn.ops import down_shifted_conv2d
from pixelcnn.ops import down_right_shifted_conv2d
from pixelcnn.ops import down_shifted_conv2d_transpose
from pixelcnn.ops import down_right_shifted_conv2d_transpose
from pixelcnn.ops import down_shift
from pixelcnn.ops import right_shift
from pixelcnn.ops import concat_elu
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf


def PixelCNNPlusPlus(
        output_size,
        image_height=32,
        image_width=32,
        num_modules=3,
        num_layers_per_module=6,
        filters=256,
        dropout_rate=0.1,
        **kwargs
):
    """Build a Pixel CNN ++ model in Keras.

    Args:
    - output_size: the cardinality of the output vector space.

    - image_height: the height of the images to generate.
    - image_width: the width of the images to generate.

    - num_modules: the number of Residual Modules.
    - num_layers: the number of Gated Masked Conv2D layers
        per module.

    - filters: the number of filters iun each Conv2D layer.
    - dropout_rate: the fraction of units to drop.

    Returns:
    - model: a Keras model that accepts one tf.int32 tensor
        with shape [batch_dim, image_height, image_width]
    """
    images = layers.Input(shape=[image_height, image_width])

    #####################################################
    # Embed the discrete image pixels in a vector space #
    #####################################################

    images_embedding = layers.TimeDistributed(
        layers.Embedding(output_size, filters))(images)
    images_embedding = layers.concatenate([
        images_embedding,
        layers.Lambda(lambda z: tf.ones([
            tf.shape(z)[0],
            tf.shape(z)[1],
            tf.shape(z)[2],
            1]))(images_embedding)])

    ##############################################
    # Prepare the image for shifted convolutions #
    ##############################################

    top_streams = [down_shift(
        down_shifted_conv2d(
            images_embedding, filters, kernel_size=(2, 3)))]

    initial_top_left_stream_a = down_shift(
        down_shifted_conv2d(
            images_embedding, filters, kernel_size=(1, 3)))
    initial_top_left_stream_b = right_shift(
        down_right_shifted_conv2d(
            images_embedding, filters, kernel_size=(2, 1)))
    top_left_streams = [
        layers.add([initial_top_left_stream_a, initial_top_left_stream_b])]

    ######################################################
    # Downsample with Residual Gated Masked Convolutions #
    ######################################################

    for block in range(num_modules):

        for layer in range(num_layers_per_module):

            top_streams.append(gated_resnet(
                top_streams[-1],
                conv2d=down_shifted_conv2d,
                nonlinearity=concat_elu,
                kernel_size=(2, 3),
                dropout_rate=dropout_rate,
                **kwargs))

            top_left_streams.append(gated_resnet(
                top_left_streams[-1],
                a=top_streams[-1],
                conv2d=down_right_shifted_conv2d,
                nonlinearity=concat_elu,
                kernel_size=(2, 2),
                dropout_rate=dropout_rate,
                **kwargs))

        if block < num_modules - 1:

            top_streams.append(down_shifted_conv2d(
                top_streams[-1],
                (2, 3),
                strides=(2, 2),
                **kwargs))

            top_left_streams.append(down_right_shifted_conv2d(
                top_left_streams[-1],
                (2, 2),
                strides=(2, 2),
                **kwargs))

    ####################################################
    # Upsample with Residual Gated Masked Convolutions #
    ####################################################

    top = top_streams.pop()
    top_left = top_left_streams.pop()

    for block in reversed(range(num_modules)):

        if block < num_modules - 1:

            top = down_shifted_conv2d_transpose(
                top,
                (2, 3),
                strides=(2, 2),
                **kwargs)

            top_left = down_right_shifted_conv2d_transpose(
                top_left,
                (2, 2),
                strides=(2, 2),
                **kwargs)

        for layer in range(num_layers_per_module):

            top = gated_resnet(
                top,
                a=top_streams.pop(),
                conv2d=down_shifted_conv2d,
                nonlinearity=concat_elu,
                kernel_size=(2, 3),
                dropout_rate=dropout_rate,
                **kwargs)

            top_left = gated_resnet(
                top_left,
                a=layers.concatenate([top, top_left_streams.pop()]),
                conv2d=down_right_shifted_conv2d,
                nonlinearity=concat_elu,
                kernel_size=(2, 2),
                dropout_rate=dropout_rate,
                **kwargs)

    #################################################
    # Compute logits for every image pixel location #
    #################################################

    top_left = concat_elu(top_left)
    logits = layers.Conv2D(
        output_size,
        (1, 1),
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        **kwargs)(top_left)

    return models.Model(inputs=[images], outputs=logits)


def ConditionalPixelCNNPlusPlus(
        output_size,
        conditional_vector_size,
        image_height=32,
        image_width=32,
        conditional_height=1,
        conditional_width=1,
        num_preprocess_layers=5,
        num_modules=3,
        num_layers_per_module=6,
        filters=256,
        dropout_rate=0.1,
        class_conditional=True,
        num_classes=None,
        **kwargs
):
    """Build a Conditional Pixel CNN ++ model in Keras.

    Args:
    - output_size: the cardinality of the output vector space.
    - conditional_vector_size: the cardinality of the vector space
        for conditioning image generation.

    - image_height: the height of the images to generate.
    - image_width: the width of the images to generate.

    - conditional_height: the height of the conditional input.
    - conditional_width: the width of the conditional input.

    - num_preprocess_layers: the number of Conv2DTranspose layers
        for upsampling the conditional input.
    - num_modules: the number of Residual Modules.
    - num_layers: the number of Gated Masked Conv2D layers
        per module.

    - filters: the number of filters iun each Conv2D layer.
    - dropout_rate: the fraction of units to drop.

    - class_conditional: a boolean that indicates that
        the conditional inputs are class labels.
    - num_classes: an integer that determines the number
        of unique classes to condition on.

    Returns:
    - model: a Keras model that accepts one tf.int32 tensor
        with shape [batch_dim, image_height, image_width] and
        with shape [batch_dim, conditional_height,
            conditional_width, conditional_vector_size]
    """
    images = layers.Input(shape=[image_height, image_width])
    if condition_on_classes:
        inputs = layers.Input(shape=[conditional_height, conditional_width])
    else:
        inputs = layers.Input(shape=[
            conditional_height, conditional_width, conditional_vector_size])

    #####################################################
    # Upsample the conditional inputs to the image size #
    #####################################################

    conditional_embedding = [inputs]
    if condition_on_classes:
        conditional_embedding[-1] = layers.TimeDistributed(
            layers.Embedding(
                num_classes, conditional_vector_size))(conditional_embedding[-1])

    for i in range(num_preprocess_layers):
        x = conditional_embedding[-1]
        if i > 0:
            x = concat_elu(x)
        conditional_embedding.append(layers.Conv2DTranspose(
            filters,
            (5, 5),
            strides=(2, 2),
            padding='same',
            data_format='channels_last',
            **kwargs)(x))

    #####################################################
    # Embed the discrete image pixels in a vector space #
    #####################################################

    def padding_backend(z):
        return tf.pad(
            z,
            [[0, 0], [0, 0], [0, 0], [0, 1]],
            constant_values=1)

    images_embedding = layers.TimeDistributed(
        layers.Embedding(output_size, filters))(images)
    images_embedding = layers.Lambda(padding_backend)(images_embedding)

    ##############################################
    # Prepare the image for shifted convolutions #
    ##############################################

    top_streams = [down_shift(
        down_shifted_conv2d(
            images_embedding, filters, kernel_size=(2, 3)))]

    initial_top_left_stream_a = down_shift(
        down_shifted_conv2d(
            images_embedding, filters, kernel_size=(1, 3)))
    initial_top_left_stream_b = right_shift(
        down_right_shifted_conv2d(
            images_embedding, filters, kernel_size=(2, 1)))
    top_left_streams = [
        layers.add([
            initial_top_left_stream_a, initial_top_left_stream_b])]

    ######################################################
    # Downsample with Residual Gated Masked Convolutions #
    ######################################################

    for block in range(num_modules):

        for layer in range(num_layers_per_module):

            top_streams.append(gated_resnet(
                top_streams[-1],
                h=conditional_embedding[-(block + 1)],
                conv2d=down_shifted_conv2d,
                nonlinearity=concat_elu,
                kernel_size=(2, 3),
                dropout_rate=dropout_rate,
                **kwargs))

            top_left_streams.append(gated_resnet(
                top_left_streams[-1],
                a=top_streams[-1],
                h=conditional_embedding[-(block + 1)],
                conv2d=down_right_shifted_conv2d,
                nonlinearity=concat_elu,
                kernel_size=(2, 2),
                dropout_rate=dropout_rate,
                **kwargs))

        if block < num_modules - 1:

            top_streams[-1] = down_shifted_conv2d(
                top_streams[-1],
                filters,
                (2, 3),
                strides=(2, 2),
                **kwargs)

            top_left_streams[-1] = down_right_shifted_conv2d(
                top_left_streams[-1],
                filters,
                (2, 2),
                strides=(2, 2),
                **kwargs)

    ####################################################
    # Upsample with Residual Gated Masked Convolutions #
    ####################################################

    top = top_streams.pop()
    top_left = top_left_streams.pop()

    for block in reversed(range(num_modules)):

        if block < num_modules - 1:

            top = down_shifted_conv2d_transpose(
                top,
                filters,
                (2, 3),
                strides=(2, 2),
                **kwargs)

            top_left = down_right_shifted_conv2d_transpose(
                top_left,
                filters,
                (2, 2),
                strides=(2, 2),
                **kwargs)

        for layer in range(num_layers_per_module):

            top = gated_resnet(
                top,
                a=top_streams.pop(),
                h=conditional_embedding[-(block + 1)],
                conv2d=down_shifted_conv2d,
                nonlinearity=concat_elu,
                kernel_size=(2, 3),
                dropout_rate=dropout_rate,
                **kwargs)

            top_left = gated_resnet(
                top_left,
                a=layers.concatenate([top, top_left_streams.pop()]),
                h=conditional_embedding[-(block + 1)],
                conv2d=down_right_shifted_conv2d,
                nonlinearity=concat_elu,
                kernel_size=(2, 2),
                dropout_rate=dropout_rate,
                **kwargs)

    #################################################
    # Compute logits for every image pixel location #
    #################################################

    top_left = concat_elu(top_left)
    logits = layers.Conv2D(
        output_size,
        (1, 1),
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        **kwargs)(top_left)

    return models.Model(inputs=[images, inputs], outputs=logits)
