import numpy as np
from .base import Optimizer, Layer
from .layers import DenseLayer, ConvolutionalLayer

class Optimizer_SGD(Optimizer):

    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0, decay: float = 0) -> None:
        """
        Stochastic gradient descent optimizing algorithm with momentum and decay.

        Parameters
        ----------
        learning_rate : float, default=0.001
            Step size used in gradient descent.

        momentum : float, default=0
            Factor used to scale past gradients.

        decay : float, default=0
            Factor used to reduce learning rate over time.

        Attributes
        ----------
        iterations : int, default=0
            Number of training iterations used to calculate new learning rate with decay.
        """
        self.learning_rate = learning_rate
        self._current_learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self._iterations = 0


    def update_parameters(self) -> None:
        """
        Method for updating current learning rate.

        Returns
        -------
        None
        """
        if self.decay:
            # Inverse decay method
            self._current_learning_rate = self.learning_rate  / (1 + self._iterations * self.decay)

    def update_layer_parameters(self, layer: Layer) -> None:
        """
        Method for updating layer parameters.

        Parameters
        ----------
        layer : Layer
            Layer that is being updated.

        Returns
        -------
        None
        """

        if self.momentum:

            # If the layer doesn't have momentum attribute initialize it
            if not hasattr(layer, 'weight_momentum') and isinstance(layer, DenseLayer):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.biases)

            layer.weight_momentum = self.momentum * layer.weight_momentum - self._current_learning_rate * layer.dweights
            layer.bias_momentum = self.momentum * layer.bias_momentum - self._current_learning_rate * layer.dbiases

            weight_updates = layer.weight_momentum
            bias_updates = layer.bias_momentum

        else:

            # If there is no momentum update weights using vanilla gradient descent
            if isinstance(layer, DenseLayer):
                weight_updates = -self._current_learning_rate * layer.dweights
                layer.weights += weight_updates
            elif isinstance(layer, ConvolutionalLayer):
                kernel_updates = -self._current_learning_rate * layer.dkernels
                layer.kernels += kernel_updates

            bias_updates = -self._current_learning_rate * layer.dbiases
            layer.biases += bias_updates