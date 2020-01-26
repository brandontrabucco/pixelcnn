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
    256,
    32,
    image_height=32,
    image_width=32,
    conditional_height=1,
    conditional_width=1,
    num_preprocess_layers=5,
    num_modules=3,
    num_layers_per_module=6,
    filters=64,
    dropout_rate=0.1,
    class_conditional=True,
    num_classes=10)
```

Fetch the next batch of conditional vectors to seed the image generation process.

```python
images = tf.zeros([12, 32, 32], dtype=tf.int32)
inputs = tf.random.uniform([12, 1, 1], maxval=10, dtype=tf.int32)
```

Run the model to predict image logits and use with your favorite loss function.

```python
logits = model([images, inputs])
```
