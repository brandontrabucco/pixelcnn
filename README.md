# Conditional Gated Pixel CNN

This repository implements Conditional Gated Pixel CNN in TensorFlow 2. Have Fun! -Brandon

## Setup

Install this package using pip.

```bash
pip install git+git://github.com/brandontrabucco/pixelcnn.git
```

## Usage

Create a Conditional Gated Pixel CNN Keras Model.

```python
model = pixelcnn.pixelcnn_plus_plus(
    32,     # input_size
    256,    # output_size
    5)      # num_upconv_layers
```

Fetch the next batch of conditional vectors to seed the image generation process.

```python
inputs = tf.random.normal([4, 1, 1, 32])
images = tf.random.uniform([4, 32, 32, 256], minval=0, maxval=256, dtype=tf.int32)
```

Run the model to predict image logits.

```python
logits = model([inputs, images])
```
