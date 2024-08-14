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
            Dimension of a single sample processed by the layer. For images it's (channels, height, width).

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
        input_channels, input_height, input_width = input_shape

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Calculate output width and height
        output_height = int((input_height - kernel_size + 2 * padding) / stride) + 1
        output_width = int((input_width - kernel_size + 2 * padding) / stride) + 1

        # Create output and kernel shapes
        self.output_shape = (output_channels, output_height, output_width)
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
                    if self.padding:
                        inputs = np.pad(self.inputs[i, k], pad_width=self.padding, mode='constant')
                    else:
                        inputs = self.inputs[i, k].copy()
                    # Output is the cross correlation in valid mode between the input and kernel
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

                        dkernels = self._calculate_kernel_gradient(input_padded, delta[i, j], self.kernels[j, k], stride=self.stride)
                        dinputs = self._calculate_input_gradient(input_padded, delta[i, j], self.kernels[j, k], stride=self.stride)

                        # Since padding was used gradient needs to be unpadded to match shape
                        dinputs = dinputs[self.padding:-self.padding, self.padding:-self.padding]

                    else:
                        dkernels = self._calculate_kernel_gradient(self.inputs[i, k], delta[i, j], self.kernels[j, k], stride=self.stride)
                        dinputs = self._calculate_input_gradient(self.inputs[i, k], delta[i, j], self.kernels[j, k], stride=self.stride)

                    self.dkernels[j, k] += dkernels
                    self.dinputs[i, k] += dinputs

    def _calculate_kernel_gradient(self, inputs: np.ndarray, delta: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
        """
        Helper method for calculating kernel gradient.

        Parameters
        ----------
        inputs : np.ndarray
            Current sample the gradient is calculated for.

        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        kernel : np.ndarray
            Kernel used in convolutional layer.

        stride : int, default=1
            Step size at which the kernel moves across the input.

        Returns
        -------
        kernel_grad : np.ndarray
            Kernel gradient.
        """

        if stride > 1:

            # If stride is present delta needs to be dilated
            delta_dilated = dilate(delta, stride)

            delta_dilated_height, delta_dilated_width = delta_dilated.shape[-2:]
            input_height, input_width = inputs.shape[-2:]
            kernel_shape = kernel.shape[-1]

            if delta_dilated_height == input_height - kernel_shape + 1 and delta_dilated_width == input_width - kernel_shape + 1:
                # If dilated delta shape matches the needed correlation shape gradient can be computed
                dkernel = signal.correlate2d(inputs, delta_dilated, "valid")
            else:
                # If dilated delta shape doesn't match the needed correlation shape padding is needed
                new_delta_shape = (input_height - kernel_shape + 1, input_width - kernel_shape + 1)
                delta_dilated_padded = pad_to_shape(delta_dilated, new_delta_shape)
                dkernel = signal.correlate2d(inputs, delta_dilated_padded, "valid")

        else:
            # Gradient with respect to kernel is valid cross correlation between inputs and delta
            dkernel = signal.correlate2d(inputs, delta, "valid")

        return dkernel

    def _calculate_input_gradient(self, inputs: np.ndarray, delta: np.ndarray, kernel: np.ndarray, stride: int = 1):
        """
        Helper method for calculating input gradient.

        Parameters
        ----------
        inputs : np.ndarray
            Current sample the gradient is calculated for.

        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        kernel : np.ndarray
            Kernel used in convolutional layer.

        stride : int, default=1
            Step size at which the kernel moves across the input.

        Returns
        -------
        input_grad : np.ndarray
            Input gradient.
        """

        if stride > 1:

            delta_dilated = dilate(delta, stride)

            delta_dilated_height, delta_dilated_width = delta_dilated.shape[-2:]
            input_height, input_width = inputs.shape[-2:]
            kernel_shape = kernel.shape[-1]

            if delta_dilated_height == input_height - kernel_shape + 1 and delta_dilated_width == input_width - kernel_shape + 1:
                # If dilated delta shape matches the needed coonvolution shape gradient can be computed
                dinput = signal.convolve2d(delta_dilated, kernel, "full")
            else:
                # If dilated delta shape doesn't match the needed convolution shape padding is needed
                new_delta_shape = (input_height - kernel_shape + 1, input_width - kernel_shape + 1)
                delta_dilated_padded = pad_to_shape(delta_dilated, new_delta_shape)
                dinput = signal.convolve2d(delta_dilated_padded, kernel, "full")

        else:
            # Gradient with respect to inputs is full convolution between delta and kernel
            dinput = signal.convolve2d(delta, kernel, "full")

        return dinput
    
class MaxPoolLayer(Layer):

    def __init__(self, input_shape: tuple, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        """
        Maxpooling layer.

        Parameters
        ----------
        input_shape : tuple
            Dimension of a single sample processed by the layer. For images it's (channels, height, width).

        kernel_size : int
            Dimension of a kernel, square array of shape (kernel_size, kernel_size).

        stride : int, default=1
            Step size at which the kernel moves across the input.

        padding : int, default=0
            Amount of padding added to input.
        """

        # Unpack the input_shape tuple
        input_channels, input_height, input_width = input_shape

        # Store input channels, kernel size and stride
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Calculate output width and height
        self.output_height = int((input_height - kernel_size + 2 * padding) / stride) + 1
        self.output_width = int((input_width - kernel_size + 2 * padding) / stride) + 1

        # Create output shape
        self.output_shape = (self.input_channels, self.output_height, self.output_width)

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass using the maxpool layer. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        
        # List for storing indices of max elements (used in backward pass)
        self.max_indices = []
        
        # Store inputs
        self.inputs = inputs

        # Number of samples, first dimenison
        n_samples = inputs.shape[0]

        # Output is 4D tensor of shape (n_samples, input_channels, width, height)
        self.output = np.zeros((n_samples, *self.output_shape))

        # Loop through every sample
        for i in range(n_samples):

            # Add empty list to max indices for the current sample
            self.max_indices.append([])

            # Loop through every channel
            for j in range(self.input_channels):

                # Add empty list to max indices for the current channel of the current sample
                self.max_indices[i].append([])

                if self.padding:
                    arr = np.pad(self.inputs[i, j], pad_width=self.padding, mode='constant')
                else:
                    arr = self.inputs[i, j].copy()

                # Loop through each element of the output
                for k in range(self.output_height):

                    # Initalize axis 0 start and end indices 
                    axis_0_start = k * self.stride
                    axis_0_end = axis_0_start + self.kernel_size

                    for l in range(self.output_width):
                        
                        # Initalize axis 1 start and end indices
                        axis_1_start = l*self.stride
                        axis_1_end = axis_1_start + self.kernel_size
                            
                        # Use axis 0 and 1 indices to obtain max pooling region   
                        region = arr[axis_0_start:axis_0_end, axis_1_start:axis_1_end]

                        # Get the max element from the region, save it to output
                        self.output[i, j, k, l] = np.max(region)
                        
                        # Get the index of the max element within the region (region is flattened array in this case)
                        max_index = np.argmax(region)

                        # Calculate the position of the max element within the sample
                        max_element_position = (axis_0_start + (max_index // self.kernel_size), axis_1_start + (max_index % self.kernel_size))

                        # Store the position of max element
                        self.max_indices[i][j].append(max_element_position)

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the maxpool layer. Creates gradient attribute with respect to inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """

        # Initialize inputs gradient
        input_shape = self.inputs.shape
        self.dinputs = np.zeros(input_shape)

        # Number of samples, first dimenison
        n_samples = self.inputs.shape[0]

        if self.padding:
            dinput_height, dinput_width = input_shape[-2:]
            dinput_shape = (dinput_height + 2 * self.padding, dinput_width + 2 * self.padding)
        else:
            dinput_shape = input_shape[-2:]

        # Loop through samples
        for i in range(n_samples):
            # Loop through channels
            for j in range(self.input_channels):
                
                # Initialize gradient for current sample
                dinput = np.zeros(dinput_shape)

                # Loop through pairs of indices zipped with a delta value
                for (k, l), d in zip(self.max_indices[i][j], delta[i, j].flatten()):
                    dinput[k, l] = d

                if self.padding:
                    self.dinputs[i, j] = dinput[self.padding:-self.padding, self.padding:-self.padding]
                else:
                    self.dinputs[i, j] = dinput.copy()

class ReshapeLayer(Layer):

    def __init__(self, input_shape: tuple[int, int, int], output_shape: int) -> None:
        """
        Layer used to reshape (flatten) an array.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            Input shape of a single sample. For images it's (channels, height, width).

        output_shape : int
            Output shape of a single sample.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs: np.ndarray) -> None:
        """
        Reshapes input array from (batch_size, channels, height, width) to (batch_size, channels * height * width). Creates output attribute.

        Parameters
        ----------
        inputs : np.ndarray
            Array to reshape.

        Returns
        -------
        None
        """
        # Store number of samples, first dimension
        batch_size = inputs.shape[0]
        self.output = np.reshape(inputs, (batch_size, self.output_shape))

    def backward(self, delta: np.ndarray) -> None:
        """
        Reshapes input array from (batch_size, channels * height * width) to (batch_size, channels, height, width). Creates gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient to reshape.

        Returns
        -------
        None
        """
        # Store number of samples, first dimension
        batch_size = delta.shape[0]
        self.dinputs = np.reshape(delta, (batch_size, *self.input_shape))

class RecurrentLayer(Layer):

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int) -> None:
        """
        Recurrent layer.

        Parameters
        ----------
        n_inputs : int
            Number of inputs that connect to the layer.

        n_hidden : int
            Number of hidden neurons.

        n_outputs : int
            Number of output neurons.

        Attributes
        ----------
        input_weights : numpy.ndarray
            Matrix of input weight coefficients.

        hidden_weights : numpy.ndarray
            Matrix of hidden weight coefficients.

        output_weights : numpy.ndarray
            Matrix of output weight coefficients.

        input_bias : numpy.ndaray
            Vector of input bias coefficients.

        output_bias : numpy.ndaray
            Vector of output bias coefficients.
        """
        k = 1/np.sqrt(n_hidden)
        self.n_hidden = n_hidden
        self.input_weights = np.random.rand(n_inputs, n_hidden) * 2 * k - k # np.random.uniform(-k, k, (n_inputs, n_hidden))
        self.hidden_weights = np.random.rand(n_hidden, n_hidden) * 2 * k - k # np.random.uniform(-k, k, (n_hidden, n_hidden))
        self.output_weights = np.random.rand(n_hidden, n_outputs) * 2 * k - k  # np.random.uniform(-k, k, (n_hidden, n_outputs))
        self.input_bias = np.random.rand(n_hidden) * 2 * k - k # np.random.uniform(-k, k, (n_hidden))
        self.output_bias = np.random.rand(n_outputs) * 2 * k - k # np.random.uniform(-k, k, (n_outputs))
        
    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass using the recurrent layer. Creates hidden states and output attributes.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Store inputs for later use
        self.inputs = inputs

        # Store number of samples
        self.n_samples = inputs.shape[0]

        self.sequence_length = inputs.shape[1]

        # Initialize output
        self.output = np.zeros((self.n_samples, self.sequence_length, self.output_weights.shape[1]))

        # Initialize hidden states
        self.hidden_states = np.zeros((self.n_samples, self.sequence_length, self.n_hidden))

        for i, sequence in enumerate(inputs):

            for j, x in enumerate(sequence):

                # Reshape to match dimensions
                x = x.reshape(1, -1)

                input_x = np.dot(x, self.input_weights)

                hidden_x = input_x + np.dot(self.hidden_states[i, max(j-1, 0)], self.hidden_weights) + self.input_bias

                # Activation function
                hidden_x = np.tanh(hidden_x)

                # Store current hidden state
                self.hidden_states[i, j] = hidden_x.copy()

                output_x = np.dot(hidden_x, self.output_weights) + self.output_bias

                # Store current output
                self.output[i, j] = output_x.copy()

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the recurrent layer. 
        Creates gradient attributes with respect to input weights, hidden weights, output weights, input bias, output bias and inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Initialize gradient attributes
        self.dinput_weights = np.zeros_like(self.input_weights)
        self.dhidden_weights = np.zeros_like(self.hidden_weights)
        self.dinput_bias = np.zeros_like(self.input_bias)
        self.doutput_weights = np.zeros_like(self.output_weights)
        self.doutput_bias = np.zeros_like(self.output_bias)
        self.dinputs = np.zeros_like(self.inputs, dtype=np.float64)

        for i in range(self.n_samples - 1, -1, -1):

            # Initialize next hidden gradient
            next_hidden_gradient = None

            for j in range(self.sequence_length - 1, -1, -1):

                loss_gradient = delta[i, j].reshape(1, -1)
                hidden_state = self.hidden_states[i, j].reshape(-1, 1)

                self.doutput_weights += np.dot(hidden_state, loss_gradient)
                self.doutput_bias += loss_gradient.reshape(-1)

                hidden_gradient = np.dot(loss_gradient, self.output_weights.T)
                if next_hidden_gradient is not None:
                    hidden_gradient += np.dot(next_hidden_gradient, self.hidden_weights.T)

                dtanh = 1 - self.hidden_states[i, j]**2
                hidden_gradient *= dtanh

                next_hidden_gradient = hidden_gradient.copy()

                if j > 0:
                    self.dhidden_weights += np.dot(self.hidden_states[i, j-1].reshape(-1, 1), hidden_gradient)

                self.dinput_weights += np.dot(self.inputs[i, j].reshape(-1, 1), hidden_gradient)
                self.dinput_bias += hidden_gradient.reshape(-1)
                
                self.dinputs[i, j] += np.dot(self.input_weights, hidden_gradient.T).reshape(-1)

class RecurrentLayerHidden(Layer):

    def __init__(self, n_inputs: int, n_hidden: int) -> None:
        """
        Recurrent layer.

        Parameters
        ----------
        n_inputs : int
            Number of input features.

        n_hidden : int
            Number of hidden features.

        n_outputs : int
            Number of output features.

        Attributes
        ----------
        input_weights : numpy.ndarray
            Matrix of input weight coefficients.

        hidden_weights : numpy.ndarray
            Matrix of hidden weight coefficients.

        output_weights : numpy.ndarray
            Matrix of output weight coefficients.

        input_bias : numpy.ndaray
            Vector of input bias coefficients.

        output_bias : numpy.ndaray
            Vector of output bias coefficients.
        """
        k = 1 / np.sqrt(n_hidden)
        self.n_hidden = n_hidden
        self.input_weights = np.random.uniform(-k, k, (n_inputs, n_hidden))
        self.hidden_weights = np.random.uniform(-k, k, (n_hidden, n_hidden))
        self.input_bias = np.random.uniform(-k, k, (n_hidden))
        
    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass using the recurrent layer. Creates hidden states and output attributes.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Store inputs for later use
        self.inputs = inputs

        # Store number of samples
        self.n_samples = inputs.shape[0]

        self.sequence_length = inputs.shape[1]

        # Initialize output
        self.output = np.zeros((self.n_samples, self.n_hidden))

        # Initialize hidden states
        self.hidden_states = np.zeros((self.n_samples, self.sequence_length, self.n_hidden))

        for i, sequence in enumerate(inputs):

            for j, x in enumerate(sequence):

                # Reshape to match dimensions
                x = x.reshape(1, -1)

                input_x = np.dot(x, self.input_weights)

                hidden_x = input_x + np.dot(self.hidden_states[i, max(j-1, 0)], self.hidden_weights) + self.input_bias

                # Activation function
                hidden_x = np.tanh(hidden_x)

                # Store current hidden state
                self.hidden_states[i, j] = hidden_x.copy()

            # Store current output
            self.output[i] = self.hidden_states[i, -1].copy()

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the recurrent layer. 
        Creates gradient attributes with respect to input weights, hidden weights, output weights, input bias, output bias and inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Initialize gradient attributes
        self.dinput_weights = np.zeros_like(self.input_weights)
        self.dhidden_weights = np.zeros_like(self.hidden_weights)
        self.dinput_bias = np.zeros_like(self.input_bias)
        self.dinputs = np.zeros_like(self.inputs, dtype=np.float64)

        for i in range(self.n_samples - 1, -1, -1):

            # Initialize next hidden gradient
            next_hidden_gradient = None

            for j in range(self.sequence_length - 1, -1, -1):

                if len(delta.shape) == 2:

                    loss_gradient = delta[i].reshape(1, -1)

                    hidden_gradient = loss_gradient.copy()
                    if next_hidden_gradient is not None:
                        hidden_gradient += np.dot(next_hidden_gradient, self.hidden_weights)

                    dtanh = 1 - self.hidden_states[i, j]**2
                    hidden_gradient *= dtanh

                    next_hidden_gradient = hidden_gradient.copy()

                    if j > 0:
                        self.dhidden_weights += np.dot(self.hidden_states[i, j-1].reshape(-1, 1), hidden_gradient)

                    self.dinput_weights += np.dot(self.inputs[i, j].reshape(-1, 1), hidden_gradient)
                    self.dinput_bias += hidden_gradient.reshape(-1)
                    
                    self.dinputs[i, j] += np.dot(self.input_weights, hidden_gradient.T).reshape(-1)

                else:

                    loss_gradient = delta[i, j].reshape(1, -1)

                    hidden_gradient = loss_gradient.copy()
                    if next_hidden_gradient is not None:
                        hidden_gradient += np.dot(next_hidden_gradient, self.hidden_weights)

                    dtanh = 1 - self.hidden_states[i, j]**2
                    hidden_gradient *= dtanh

                    next_hidden_gradient = hidden_gradient.copy()

                    if j > 0:
                        self.dhidden_weights += np.dot(self.hidden_states[i, j-1].reshape(-1, 1), hidden_gradient)

                    self.dinput_weights += np.dot(self.inputs[i, j].reshape(-1, 1), hidden_gradient)
                    self.dinput_bias += hidden_gradient.reshape(-1)
                    
                    self.dinputs[i, j] += np.dot(self.input_weights, hidden_gradient.T).reshape(-1)

class RNN:

    def __init__(self, n_inputs: int, n_hidden: int, n_layers: int = 1) -> None:
        """
        Recurrent neural network.

        Parameters
        ----------
        n_inputs : int
            Number of input features.

        n_hidden : int
            Number of hidden features.
        """
        self.recurrent_layers = [RecurrentLayerHidden(n_inputs, n_hidden)]
        if n_layers > 1:
            for _ in range(n_layers - 1):
                self.recurrent_layers.append(RecurrentLayerHidden(n_hidden, n_hidden))

    def forward(self, inputs: np.ndarray) -> None:

        self.recurrent_layers[0].forward(inputs)

        for idx, layer in enumerate(self.recurrent_layers[1:], start=1):
            layer.forward(self.recurrent_layers[idx - 1].hidden_states)

        self.output = self.recurrent_layers[-1].output.copy()

    def backward(self, delta: np.ndarray) -> None:

        self.recurrent_layers[-1].backward(delta)

        for idx, layer in reversed(list(enumerate(self.recurrent_layers[:-1]))):
            layer.backward(self.recurrent_layers[idx + 1].dinputs)