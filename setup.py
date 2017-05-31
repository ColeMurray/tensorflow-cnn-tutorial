from setuptools import setup, find_packages

setup(name='mnist_conv2d_medium_tutorial',
      packages=find_packages(exclude=['tests']),
      zip_safe=False,
      keywords=['convolutional neural network', 'neural network', 'tensorflow', 'mnist'])
