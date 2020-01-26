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


def pixelcnn_plus_plus(
        output_size,
        image_height=32,
        image_width=32,
        num_modules=3,
        num_layers_per_module=6,
        filters=256,
        dropout_rate=0.1,
        **kwargs
):
    """Build a Pixel CNN model in Keras."""
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


def conditional_pixelcnn_plus_plus(
        conditional_vector_size,
        output_size,
        conditional_height=1,
        conditional_width=1,
        image_height=32,
        image_width=32,
        num_preprocess_layers=3,
        num_modules=3,
        num_layers_per_module=6,
        filters=256,
        dropout_rate=0.1,
        **kwargs
):
    """Build a Conditional Pixel CNN model in Keras."""
    inputs = layers.Input(shape=[
        conditional_height, conditional_width, conditional_vector_size])
    images = layers.Input(shape=[image_height, image_width])

    #####################################################
    # Upsample the conditional inputs to the image size #
    #####################################################

    conditional_embedding = [inputs]
    for i in range(num_preprocess_layers):
        conditional_embedding.append(layers.Conv2DTranspose(
            filters,
            (5, 5),
            strides=(2, 2),
            padding='same',
            data_format='channels_last',
            **kwargs)(conditional_embedding[-1]))

    #####################################################
    # Embed the discrete image pixels in a vector space #
    #####################################################

    images_embedding = layers.TimeDistributed(
        layers.Embedding(output_size, filters))(images)
    images_embedding = layers.Lambda(lambda z: tf.pad(
        z,
        [[0, 0], [0, 0], [0, 0], [0, 1]],
        constant_values=1))(images_embedding)

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
