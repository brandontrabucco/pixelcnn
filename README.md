# Pixel CNN

This repository implements Pixel CNN in TensorFlow 2. Have Fun! -Brandon

## Setup

Install this package using pip.

```bash
pip install git+git://github.com/brandontrabucco/pixelcnn.git
```

## Usage

Create a Pixel CNN Keras Model.

```python
model = pixelcnn.ConditionalPixelCNNPlusPlus(
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
```

Fetch the next batch of conditional vectors to seed the image generation process.

```python
inputs = tf.random.normal([12, 1, 1, 32])
images = tf.zeros([12, 32, 32], dtype=tf.int32)
```

Run the model to predict image logits and select the indices which maximize log probability.

```python
logits = model([images, inputs])
images = tf.argmax(logits, axis=3, output_type=tf.int32)
```
