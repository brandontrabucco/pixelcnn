# Conditional Gated Pixel CNN

This repository implements Conditional Gated Pixel CNN in TensorFlow 2. Have Fun! -Brandon

## Setup

Install this package using pip.

```
pip install git+git://github.com/brandontrabucco/pixelcnn.git
```

## Usage

Create a Gated Masked Pixel CNN Keras Model.

```
model = pixelcnn(
    32,     # input_size
    5,      # num_upconv_layers
    5,      # num_gated_masked_conv_layers
    1024,   # filters
    5)      # kernel_size
```

Fetch the next batch of conditional vectors, and ground truth image pixels.

```
inputs = tf.random.normal([1, 32])
image = tf.random.uniform(
    [1, 32, 32],
    minval=0,
    maxval=256,
    dtype=tf.dtypes.int32)
```

Run the model to predict image logits.

```
logits = model(inputs)
```
