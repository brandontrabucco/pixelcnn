"""Author: Brandon Trabucco, Copyright 2020, MIT License"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.0.3',
    'numpy',
    'tensorflow-datasets',
    'matplotlib']


setup(
    name='pixelcnn',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('pixelcnn')],
    description='Conditional Gated Pixel CNN in TensorFlow 2')
