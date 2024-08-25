# Deep Learning From Scratch

## Overview

This repository contains some popular deep learning architectures which are implemented using NumPy, without popular libraries like PyTorch or TensorFlow.

These are generally not meant to be optimized and efficient implementations, rather used for understanding how they operate "under the hood".

Architectures currently implemented:

- Multilayer Perceptron (MLP)

- Convolutional Neural Network (CNN)

- Recurrent Neural Network (RNN)

- Long Short Term Memory (LSTM)

## Setup

1. **Set up a virtual environment** (optional):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Side notes

TensorFlow is included in the requirements.txt because of its image datasets.

It is needed only to run the CNN notebook, so you can delete it from requirements.txt if you don't need it.