# Capsnet - Traffic sign classifier [Tensorflow]

A Tensorflow implementation of CapsNet(Capsules Net) apply on german traffic sign dataset

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
![completion](https://img.shields.io/badge/completion%20state-80%25-blue.svg?style=plastic)

This implementation is based on this paper: <b>Dynamic Routing Between Capsules</b> (https://arxiv.org/abs/1710.09829) from Sara Sabour, Nicholas Frosst and Geoffrey E. Hinton.

This repository is a work in progress implementation of the Cpasules Net. Since I am using a different dataset (Not MNIST) some details in the architecture are differents. The code for the CapsNet is located in the following file: <b>caps_net.py</b> while the whole model is create inside the <b>model.py</b> file. The two main method used to create the CapsNet from the convolutional layer are  <b>conv_caps_layer</b> and <b>fully_connected_caps_layer</b>

## Requirements
- Python 3
- NumPy 1.13.1
- Tensorflow 1.3.0
- docopt 0.6.2

