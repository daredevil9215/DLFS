import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class Layer:

    def forward(self):
        pass

    def backward(self):
        pass

class DenseLayer(Layer):
    
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        """
        Layer of neurons consisting of a weight matrix and bias vector.

        Parameters
        ----------
        n_inputs : int
            Number of inputs that connect to the layer.

        n_neurons : int
            Number of neurons the layer consists of.

        Attributes
        ----------
        weights : numpy.ndarray
            Matrix of weight coefficients.

        biases : numpy.ndaray
            Vector of bias coefficients.
        """

        # Weights are randomly initialized, small random numbers seem to work well
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Bias vector is initialized to a zero vector
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass using the layer. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Store inputs for later use (backpropagation)
        self.inputs = inputs
        # Output is the dot product of the input matrix and weights plus biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the layer. Creates gradient attributes with respect to layer weights, biases and inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        self.dweights = np.dot(self.inputs.T, delta)
        self.dbiases = np.sum(delta, axis=0)
        self.dinputs = np.dot(delta, self.weights.T)

class ConvolutionalLayer(Layer):

    def __init__(self, input_shape, kernel_size, depth) -> None:
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, inputs):
        n_samples = inputs.shape[0]
        self.inputs = inputs
        self.output = np.zeros((n_samples, *self.output_shape))
        for i in range(self.depth):
            for j in range(self.input_depth):
                for k in range(n_samples):
                    self.output[k, i] += signal.correlate2d(self.inputs[k, j], self.kernels[i, j], mode="valid")
                self.output[i] += self.biases[i]

    def backward(self, delta):
        self.dkernels = np.zeros(self.kernels.shape)
        self.dbiases = np.zeros(self.biases.shape)
        self.dinputs = np.zeros(self.inputs.shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                for k in range(len(self.inputs)):
                    self.dkernels[i, j] += signal.correlate2d(self.inputs[k, j], delta[k, j], "valid")
                    self.dbiases[i] += delta[k, j]
                    self.dinputs[k, j] += signal.convolve2d(delta[k, j], self.kernels[i, j], "full")
             

class Activation_Function:

    def forward(self):
        pass

    def backward(self):
        pass

class Linear(Activation_Function):

    def __init__(self) -> None:
        """
        Linear activation function.
        """
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass using Linear activation. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        self.inputs = inputs
        self.output = inputs

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using Linear activation. Creates gradient attribute with respect to inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Derivative of Linear activation
        self.dinputs = delta

class ReLU(Activation_Function):

    def __init__(self) -> None:
        """
        Rectified Linear Unit activation function.
        """
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass using ReLU. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Store inputs for later use (backpropagation)
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using ReLU. Creates gradient attribute with respect to inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        self.dinputs = delta.copy()
        # Derivative of ReLU
        self.dinputs[self.inputs < 0] = 0

class Sigmoid(Activation_Function):

    def __init__(self) -> None:
        """
        Sigmoid activation function.
        """
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass using Sigmoid. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using Sigmoid. Creates gradient attribute with respect to inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Derivative of Sigmoid
        self.dinputs = delta * (1 - self.output) * self.output

class Loss:

    def calculate(self):
        pass
    
    def backward(self):
        pass

class BCE_Loss(Loss):

    def __init__(self) -> None:
        """
        Binary Cross Entropy loss function.
        """
        pass

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Binary Cross Entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        Returns
        -------
        loss : float
        """
        # Clip y_pred so logarithm doesn't become unstable
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(loss)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backward pass using Binary Cross Entropy loss. Creates gradient attribute with respect to predicted values.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        Returns
        -------
        None
        """
        # Clip y_pred so the denominator doesn't become unstable
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # Derivative of BCE with respect to y_pred
        self.dinputs = (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped))

class MSE_Loss(Loss):

    def __init__(self) -> None:
        """
        Mean Squared Error loss function.
        """
        pass

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Mean Squared Error loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        Returns
        -------
        loss : float
        """
        loss = 0.5 * (y_pred - y_true)**2
        return np.mean(loss)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backward pass using Mean Squared Error loss. Creates gradient attribute with respect to predicted values.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        Returns
        -------
        None
        """
        # Derivative of MSE with respect to y_pred
        self.dinputs = (y_pred - y_true)

class Optimizer:

    def update_parameters(self):
        pass

    def update_layer_parameters(self):
        pass

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

class Model:

    def __init__(self, layers: list = None, loss_function: Loss = None, optimizer: Optimizer = None) -> None:
        """
        Neural network model.

        Parameters
        ----------
        layers : list, default=None
            List of layers and activation functions.

        loss_function : Loss, default=None
            Loss function.
        
        optimizer : Optimizer, default=None
            Optimizer algorithm.
        """
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

    def _forward(self, X: np.ndarray) -> None:
        """
        Forward pass.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        Returns
        -------
        None
        """

        # Pass data to the input layer
        self.layers[0].forward(X)

        # Forward data through all the layers
        for idx, layer in enumerate(self.layers[1:], start=1):
            layer.forward(self.layers[idx - 1].output)

        # Output of the model is the output of the last layer
        self.output = self.layers[-1].output

    def _backward(self, y: np.ndarray) -> None:
        """
        Backward pass.
        
        Parameters
        ----------
        y : np.ndarray
            Output values.

        Returns
        -------
        None
        """

        # Backward pass starts with loss function gradient calculation
        self.loss_function.backward(self.output, y)
        self.layers[-1].backward(self.loss_function.dinputs)

        # Pass gradients backwards to all layers
        for idx, layer in reversed(list(enumerate(self.layers[:-1]))):
            layer.backward(self.layers[idx + 1].dinputs)

    def _update_model_parameters(self) -> None:
        """
        Method for updating model parameters (weights and biases).

        Returns
        -------
        None
        """

        # Loop through all layers
        for layer in self.layers:
            # If the layer has weights attribute we can update it
            if hasattr(layer, 'weights') or hasattr(layer, 'kernels'):
                self.optimizer.update_layer_parameters(layer)

        self.optimizer.update_parameters()

    def train(self, X: np.ndarray, y: np.ndarray, iterations: int = 1000, print_every: int = None) -> None:
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        y : np.ndarray
            Output values.

        iterations : int, default=1000
            Number of training iterations.

        print_every : int, default=None
            If given an integer, prints loss value.

        Returns
        -------
        None
        """

        for i in range(iterations):

            # Forward pass
            self._forward(X)

            if print_every is not None:
                if not i % print_every:
                    print(f'Loss: {self.loss_function.calculate(self.output, y)}')

            # Backward pass
            self._backward(y)

            # Update parameters
            self._update_model_parameters()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        Returns
        -------
        prediction : np.ndarray
        """
        self._forward(X)
        return self.output
    
    def add(self, layer : DenseLayer | Activation_Function) -> None:
        """
        Add layer to the network.

        Parameters
        ----------
        layer : Layer | Activation_Function
            Layer to add.

        Returns
        -------
        None
        """
        self.layers.append(layer)