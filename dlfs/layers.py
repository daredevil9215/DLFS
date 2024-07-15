from math import floor
import numpy as np
from scipy import signal
from .base import Layer
from .helpers import dilate, pad_to_shape

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
        Forward pass using the dense layer. Creates output attribute.

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
        Backward pass using the dense layer. Creates gradient attributes with respect to layer weights, biases and inputs.

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

    def __init__(self, input_shape: tuple, output_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        """
        Convolutional layer.

        Parameters
        ----------
        input_shape : tuple
            Dimension of a single sample processed by the layer. For images it's (channels, width, height).

        output_channels : int
            Number of channels of the output array.

        kernel_size : int
            Dimension of a single kernel, square array of shape (kernel_size, kernel_size).

        stride : int, default=1
            Step size at which the kernel moves across the input.

        padding : int, default=0
            Amount of padding added to input.
        """
        # Unpack input_shape tuple
        input_channels, input_width, input_height = input_shape

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Calculate output width and height
        output_width = int(floor((input_width - kernel_size + 2 * padding) / stride) + 1)
        output_height = int(floor((input_height - kernel_size + + 2 * padding) / stride) + 1) 

        # Create output and kernel shapes
        self.output_shape = (output_channels, output_width, output_height)
        self.kernels_shape = (output_channels, input_channels, kernel_size, kernel_size)

        # Initialize layer parameters
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass using the convolutional layer. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Number of samples, first dimension
        n_samples = inputs.shape[0]

        # Store inputs for later use
        self.inputs = inputs

        # Output is 4D tensor of shape (n_samples, output_channels, height, width)
        self.output = np.zeros((n_samples, *self.output_shape))

        # Add bias to output
        self.output += self.biases

        # Loop through each sample, output channel and input channel
        for i in range(n_samples):
            for j in range(self.output_channels):
                for k in range(self.input_channels):
                    # Output is the cross correlation in valid mode between the input and kernel
                    if self.padding:
                        inputs = np.pad(self.inputs[i, k], pad_width=self.padding, mode='constant')
                    else:
                        inputs = self.inputs[i, k].copy()
                    self.output[i, j] += signal.correlate2d(inputs, self.kernels[j, k], mode="valid")[::self.stride, ::self.stride]
            
    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the convolutional layer. Creates gradient attributes with respect to kernels, biases and inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Initialize gradient attributes
        self.dkernels = np.zeros(self.kernels.shape)
        self.dbiases = np.zeros(self.biases.shape)
        """if self.padding:
            input_shape = list(self.inputs.shape)
            input_shape[2] += 2 * self.padding
            input_shape[3] += 2 * self.padding"""
        self.dinputs = np.zeros(self.inputs.shape)

        # Number of samples, first dimension
        n_samples = self.inputs.shape[0]

        # Loop through each sample, output channel and input channel
        for i in range(n_samples):

            # Gradient with respect to biases is the sum of deltas
            self.dbiases += delta[i]

            for j in range(self.output_channels):
                for k in range(self.input_channels):

                    if self.padding:
                        
                        input_padded = np.pad(self.inputs[i, k], pad_width=self.padding)

                        if self.stride == 1:
                            dkernels = self._calculate_kernel_gradient(input_padded, delta[i, j], stride=False)
                            dinputs = self._calculate_input_gradient(delta[i, j], self.kernels[j, k], stride=False)
                            dinputs = dinputs[self.padding:-self.padding, self.padding:-self.padding]
                        else:
                            dkernels = self._calculate_kernel_gradient(input_padded, delta[i, j], stride=True)
                            dinputs = self._calculate_input_gradient(delta[i, j], self.kernels[j, k], stride=True)
                            dinputs = dinputs[self.padding:-self.padding, self.padding:-self.padding]

                    else:
                        if self.stride == 1:
                            dkernels = self._calculate_kernel_gradient(self.inputs[i, k], delta[i, j], stride=False)
                            # Gradient with respect to inputs is the full convolution between delta and kernel
                            dinputs = self._calculate_input_gradient(delta[i, j], self.kernels[j, k], stride=False)
                        else:
                            dkernels = self._calculate_kernel_gradient(self.inputs[i, k], delta[i, j], stride=True)
                            dinputs = self._calculate_input_gradient(delta[i, j], self.kernels[j, k], stride=True)

                            
                    self.dkernels[j, k] += dkernels
                    self.dinputs[i, k] += dinputs

    def _calculate_kernel_gradient(self, inputs: np.ndarray, delta: np.ndarray, stride=False):

        if stride:
            delta_dilated = dilate(delta, stride=self.stride)
            delta_dilated_shape = delta_dilated.shape

            input_shape = self.inputs.shape[-1]
            kernel_shape = self.kernels_shape[-1]
            padding = self.padding

            if delta_dilated_shape == input_shape - kernel_shape + 2 * padding + 1:
                # If dilated delta shape matches the needed correlation shape gradient is computed
                dkernels = signal.correlate2d(inputs, delta_dilated, "valid")
            else:
                # If dilated delta shape doesn't match the needed correlation shape padding is needed
                new_delta_shape = (input_shape - self.kernel_size + 2 * self.padding + 1, input_shape - kernel_shape + 2*self.padding + 1)
                delta_dilated = pad_to_shape(delta_dilated, new_delta_shape)
                dkernels = signal.correlate2d(inputs, delta_dilated, "valid")
        else:
            dkernels = signal.correlate2d(inputs, delta, "valid")
        return dkernels

    def _calculate_input_gradient(self, delta: np.ndarray, kernel: np.ndarray, stride=False):
        if stride:
            delta_dilated = dilate(delta, stride=self.stride)
            dinputs = signal.convolve2d(delta_dilated, kernel, "full")
        else:
            dinputs = signal.convolve2d(delta, kernel, "full")
        return dinputs

class ReshapeLayer(Layer):

    def __init__(self, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs):
        # converts [batch_size, depth, height, width] to [batch_size, depth * height * width]
        batch_size = inputs.shape[0]
        self.output = np.reshape(inputs, (batch_size, self.output_shape))

    def backward(self, delta):
        # converts [batch_size, depth * height * width] to [batch_size, depth, height, width]
        batch_size = delta.shape[0]
        self.dinputs = np.reshape(delta, (batch_size, *self.input_shape))