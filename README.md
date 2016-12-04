# Rorschach

[![Build Status](https://travis-ci.com/OptimusCrime/master-thesis-code.svg?token=JmzjtQYirFw9etqSW57N&branch=master)](https://travis-ci.com/OptimusCrime/master-thesis-code)

Code behind master thesis for Thomas Gautvedt.

## Information

This project attempts to do letter classificaion in a small search space by using technologies most commonly found in Natural Language Processing.

We do this by building a network that uses a lot of recurrent components, which are able to learn sequence to sequence prediction.

The network looks like this:

![Model network](docs/model_network.png)

## Install

This project requires Python3 with pip. We also advice installing virtualenv and virtualenwrapper.

### Unix / macOS X

Make sure to have python3-dev installed before continuing. Run the setup script with `sh setup_unix.sh`. On Linux we attempt to install graphviz using apt-get, this requires root. On macOS X we attempt tho install graphviz with brew.

Note: If you'd like GPU support in Tensorflow, you need to uninstall the version installed with the shell script. Instead you need to install the GPU version. This requires CUDA toolkit 8.0 and CuDNN v5.

### Windows

Not supported yet.
