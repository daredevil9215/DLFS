import numpy as np
from .base import Optimizer, Layer
from .layers import DenseLayer, ConvolutionalLayer, RecurrentLayer, LSTMLayer

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
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.iterations = 0

    def pre_update_parameters(self) -> None:
        """
        Method for updating current learning rate.

        Returns
        -------
        None
        """
        if self.decay:
            # Inverse decay method
            self.current_learning_rate = self.learning_rate  / (1 + self.iterations * self.decay)

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

            if isinstance(layer, DenseLayer):
                # If the layer doesn't have momentum attribute, initialize it
                if not hasattr(layer, 'weight_momentum'):
                    layer.weight_momentum = np.zeros_like(layer.weights)
                    layer.bias_momentum = np.zeros_like(layer.biases)

                layer.weight_momentum = self.momentum * layer.weight_momentum - self.current_learning_rate * layer.dweights
                weight_updates = layer.weight_momentum

                layer.bias_momentum = self.momentum * layer.bias_momentum - self.current_learning_rate * layer.dbiases
                bias_updates = layer.bias_momentum

            elif isinstance(layer, ConvolutionalLayer):
                # If the layer doesn't have kernel attribute, initialize it
                if not hasattr(layer, 'kernel_momentum'):
                    layer.kernel_momentum = np.zeros_like(layer.kernels)
                    layer.bias_momentum = np.zeros_like(layer.biases)

                layer.kernel_momentum = self.momentum * layer.kernel_momentum - self.current_learning_rate * layer.dkernels
                kernel_updates = layer.kernel_momentum

                layer.bias_momentum = self.momentum * layer.bias_momentum - self.current_learning_rate * layer.dbiases
                bias_updates = layer.bias_momentum

            elif isinstance(layer, RecurrentLayer):

                # If the layer doesn't have momentum attributes, initialize them
                if not hasattr(layer, 'input_weights_momentum'):
                    layer.input_weights_momentum = np.zeros_like(layer.input_weights)
                    layer.hidden_weights_momentum = np.zeros_like(layer.hidden_weights)
                    layer.input_bias_momentum = np.zeros_like(layer.input_bias)

                layer.input_weights_momentum = self.momentum * layer.input_weights_momentum - self.current_learning_rate * layer.dinput_weights
                input_weights_updates = layer.input_weights_momentum

                layer.hidden_weights_momentum = self.momentum * layer.hidden_weights_momentum - self.current_learning_rate * layer.dhidden_weights
                hidden_weights_updates = layer.hidden_weights_momentum

                layer.input_bias_momentum = self.momentum * layer.input_bias_momentum - self.current_learning_rate * layer.dinput_bias
                input_bias_updates = layer.input_bias_momentum

            elif isinstance(layer, LSTMLayer):

                if not hasattr(layer, 'input_weights_momentum'):
                    layer.input_weights_momentums = np.zeros_like(layer.input_weights)
                    layer.input_bias_momentums = np.zeros_like(layer.input_bias)
                    layer.forget_weights_momentums = np.zeros_like(layer.forget_weights)
                    layer.forget_bias_momentums = np.zeros_like(layer.forget_bias)
                    layer.candidate_weights_momentums = np.zeros_like(layer.candidate_weights)
                    layer.candidate_bias_momentums = np.zeros_like(layer.candidate_bias)
                    layer.output_weights_momentums = np.zeros_like(layer.output_weights)
                    layer.output_bias_momentums = np.zeros_like(layer.output_bias)

                layer.input_weights_momentum = self.momentum * layer.input_weights_momentum - self.current_learning_rate * layer.dinput_weights
                input_weights_updates = layer.input_weights_momentum
                layer.input_bias_momentum = self.momentum * layer.input_bias_momentum - self.current_learning_rate * layer.dinput_bias
                input_bias_updates = layer.input_bias_momentum

                layer.forget_weights_momentum = self.momentum * layer.forget_weights_momentum - self.current_learning_rate * layer.dforget_weights
                forget_weights_updates = layer.forget_weights_momentum
                layer.forget_bias_momentum = self.momentum * layer.forget_bias_momentum - self.current_learning_rate * layer.dforget_bias
                forget_bias_updates = layer.forget_bias_momentum

                layer.candidate_weights_momentum = self.momentum * layer.candidate_weights_momentum - self.current_learning_rate * layer.dcandidate_weights
                candidate_weights_updates = layer.candidate_weights_momentum
                layer.candidate_bias_momentum = self.momentum * layer.candidate_bias_momentum - self.current_learning_rate * layer.dcandidate_bias
                candidate_bias_updates = layer.candidate_bias_momentum

                layer.output_weights_momentum = self.momentum * layer.output_weights_momentum - self.current_learning_rate * layer.doutput_weights
                output_weights_updates = layer.output_weights_momentum
                layer.output_bias_momentum = self.momentum * layer.output_bias_momentum - self.current_learning_rate * layer.doutput_bias
                output_bias_updates = layer.output_bias_momentum


        else:

            # If there is no momentum update weights using vanilla gradient descent
            if isinstance(layer, DenseLayer):
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases

            elif isinstance(layer, ConvolutionalLayer):
                kernel_updates = -self.current_learning_rate * layer.dkernels
                bias_updates = -self.current_learning_rate * layer.dbiases

            elif isinstance(layer, RecurrentLayer):
                input_weights_updates = -self.current_learning_rate * layer.dinput_weights
                hidden_weights_updates =  -self.current_learning_rate * layer.dhidden_weights
                input_bias_updates = -self.current_learning_rate * layer.dinput_bias

            elif isinstance(layer, LSTMLayer):
                input_weights_updates = -self.current_learning_rate * layer.dinput_weights
                input_bias_updates = -self.current_learning_rate * layer.dinput_bias

                forget_weights_updates = -self.current_learning_rate * layer.dforget_weights
                forget_bias_updates = -self.current_learning_rate * layer.dforget_bias

                candidate_weights_updates = -self.current_learning_rate * layer.dcandidate_weights
                candidate_bias_updates = -self.current_learning_rate * layer.dcandidate_bias

                output_weights_updates = -self.current_learning_rate * layer.doutput_weights
                output_bias_updates = -self.current_learning_rate * layer.doutput_bias
            

        if isinstance(layer, DenseLayer):
            layer.weights += weight_updates
            layer.biases += bias_updates

        elif isinstance(layer, ConvolutionalLayer):
            layer.kernels += kernel_updates
            layer.biases += bias_updates

        elif isinstance(layer, RecurrentLayer):
            layer.input_weights += input_weights_updates
            layer.hidden_weights += hidden_weights_updates
            layer.input_bias += input_bias_updates

        elif isinstance(layer, LSTMLayer):
            layer.input_weights += input_weights_updates
            layer.input_bias += input_bias_updates

            layer.forget_weights += forget_weights_updates
            layer.forget_bias += forget_bias_updates

            layer.candidate_weights += candidate_weights_updates
            layer.candidate_bias += candidate_bias_updates

            layer.output_weights += output_weights_updates
            layer.output_bias += output_bias_updates

    def post_update_parameters(self) -> None:
        self.iterations += 1

class Optimizer_Adam(Optimizer):

    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def pre_update_parameters(self) -> None:
        if self.decay:
            # Inverse decay method
            self.current_learning_rate = self.learning_rate  / (1 + self.iterations * self.decay) 

    def update_layer_parameters(self, layer: Layer):

        if isinstance(layer, LSTMLayer):

            if not hasattr(layer, 'input_weight_cache'):
                layer.input_weight_cache = np.zeros_like(layer.input_weights)
                layer.input_bias_cache = np.zeros_like(layer.input_bias)
                layer.forget_weight_cache = np.zeros_like(layer.forget_weights)
                layer.forget_bias_cache = np.zeros_like(layer.forget_bias)
                layer.candidate_weight_cache = np.zeros_like(layer.candidate_weights)
                layer.candidate_bias_cache = np.zeros_like(layer.candidate_bias)
                layer.output_weight_cache = np.zeros_like(layer.output_weights)
                layer.output_bias_cache = np.zeros_like(layer.output_bias)

                layer.input_weight_momentums = np.zeros_like(layer.input_weights)
                layer.input_bias_momentums = np.zeros_like(layer.input_bias)
                layer.forget_weight_momentums = np.zeros_like(layer.forget_weights)
                layer.forget_bias_momentums = np.zeros_like(layer.forget_bias)
                layer.candidate_weight_momentums = np.zeros_like(layer.candidate_weights)
                layer.candidate_bias_momentums = np.zeros_like(layer.candidate_bias)
                layer.output_weight_momentums = np.zeros_like(layer.output_weights)
                layer.output_bias_momentums = np.zeros_like(layer.output_bias)

            # Input weights
            layer.input_weight_momentums = self.beta_1 * layer.input_weight_momentums + (1-self.beta_1) * layer.dinput_weights
            layer.input_bias_momentums = self.beta_1 * layer.input_bias_momentums + (1-self.beta_1) * layer.dinput_bias
            input_weight_momentums_corrected = layer.input_weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            input_bias_momentums_corrected = layer.input_bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

            layer.input_weight_cache = self.beta_2 * layer.input_weight_cache + (1 - self.beta_2) * layer.dinput_weights**2
            layer.input_bias_cache = self.beta_2 * layer.input_bias_cache + (1 - self.beta_2) * layer.dinput_bias**2
            input_weight_cache_corrected = layer.input_weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
            input_bias_cache_corrected = layer.input_bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.input_weights += -self.current_learning_rate * input_weight_momentums_corrected / (np.sqrt(input_weight_cache_corrected) + self.epsilon)
            layer.input_bias += -self.current_learning_rate * input_bias_momentums_corrected / (np.sqrt(input_bias_cache_corrected) + self.epsilon)

            # Forget weights
            layer.forget_weight_momentums = self.beta_1 * layer.forget_weight_momentums + (1-self.beta_1) * layer.dforget_weights
            layer.forget_bias_momentums = self.beta_1 * layer.forget_bias_momentums + (1-self.beta_1) * layer.dforget_bias
            forget_weight_momentums_corrected = layer.forget_weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            forget_bias_momentums_corrected = layer.forget_bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

            layer.forget_weight_cache = self.beta_2 * layer.forget_weight_cache + (1 - self.beta_2) * layer.dforget_weights**2
            layer.forget_bias_cache = self.beta_2 * layer.forget_bias_cache + (1 - self.beta_2) * layer.dforget_bias**2
            forget_weight_cache_corrected = layer.forget_weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
            forget_bias_cache_corrected = layer.forget_bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.forget_weights += -self.current_learning_rate * forget_weight_momentums_corrected / (np.sqrt(forget_weight_cache_corrected) + self.epsilon)
            layer.forget_bias += -self.current_learning_rate * forget_bias_momentums_corrected / (np.sqrt(forget_bias_cache_corrected) + self.epsilon)

            # Candidate weights
            layer.candidate_weight_momentums = self.beta_1 * layer.candidate_weight_momentums + (1-self.beta_1) * layer.dcandidate_weights
            layer.candidate_bias_momentums = self.beta_1 * layer.candidate_bias_momentums + (1-self.beta_1) * layer.dcandidate_bias
            candidate_weight_momentums_corrected = layer.candidate_weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            candidate_bias_momentums_corrected = layer.candidate_bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

            layer.candidate_weight_cache = self.beta_2 * layer.candidate_weight_cache + (1 - self.beta_2) * layer.dcandidate_weights**2
            layer.candidate_bias_cache = self.beta_2 * layer.candidate_bias_cache + (1 - self.beta_2) * layer.dcandidate_bias**2
            candidate_weight_cache_corrected = layer.candidate_weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
            candidate_bias_cache_corrected = layer.candidate_bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.candidate_weights += -self.current_learning_rate * candidate_weight_momentums_corrected / (np.sqrt(candidate_weight_cache_corrected) + self.epsilon)
            layer.candidate_bias += -self.current_learning_rate * candidate_bias_momentums_corrected / (np.sqrt(candidate_bias_cache_corrected) + self.epsilon)

            # Output weights
            layer.output_weight_momentums = self.beta_1 * layer.output_weight_momentums + (1-self.beta_1) * layer.doutput_weights
            layer.output_bias_momentums = self.beta_1 * layer.output_bias_momentums + (1-self.beta_1) * layer.doutput_bias
            output_weight_momentums_corrected = layer.output_weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            output_bias_momentums_corrected = layer.output_bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

            layer.output_weight_cache = self.beta_2 * layer.output_weight_cache + (1 - self.beta_2) * layer.doutput_weights**2
            layer.output_bias_cache = self.beta_2 * layer.output_bias_cache + (1 - self.beta_2) * layer.doutput_bias**2
            output_weight_cache_corrected = layer.output_weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
            output_bias_cache_corrected = layer.output_bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.output_weights += -self.current_learning_rate * output_weight_momentums_corrected / (np.sqrt(output_weight_cache_corrected) + self.epsilon)
            layer.output_bias += -self.current_learning_rate * output_bias_momentums_corrected / (np.sqrt(output_bias_cache_corrected) + self.epsilon)
    
        elif isinstance(layer, DenseLayer):

            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
            layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases
            weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

            layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
            layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
            weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
            bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
            layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

        elif isinstance(layer, ConvolutionalLayer):

            if not hasattr(layer, 'kernel_cache'):
                layer.kernel_cache = np.zeros_like(layer.kernels)
                layer.bias_cache = np.zeros_like(layer.biases)

                layer.kernel_momentums = np.zeros_like(layer.kernels)
                layer.bias_momentums = np.zeros_like(layer.biases)

            layer.kernel_momentums = self.beta_1 * layer.kernel_momentums + (1-self.beta_1) * layer.dkernels
            layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases
            kernel_momentums_corrected = layer.kernel_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

            layer.kernel_cache = self.beta_2 * layer.kernel_cache + (1 - self.beta_2) * layer.dkernels**2
            layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
            kernel_cache_corrected = layer.kernel_cache / (1 - self.beta_2 ** (self.iterations + 1))
            bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.kernels += -self.current_learning_rate * kernel_momentums_corrected / (np.sqrt(kernel_cache_corrected) + self.epsilon)
            layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

        elif isinstance(layer, RecurrentLayer):
            
            if not hasattr(layer, 'input_weight_cache'):
                layer.input_weight_cache = np.zeros_like(layer.input_weights)
                layer.input_bias_cache = np.zeros_like(layer.input_bias)
                layer.hidden_weight_cache = np.zeros_like(layer.hidden_weights)

                layer.input_weight_momentums = np.zeros_like(layer.input_weights)
                layer.input_bias_momentums = np.zeros_like(layer.input_bias)
                layer.hidden_weight_momentums = np.zeros_like(layer.hidden_weights)

            # Input weights and bias
            layer.input_weights_momentums = self.beta_1 * layer.input_weights_momentums + (1-self.beta_1) * layer.dinput_weights
            layer.input_bias_momentums = self.beta_1 * layer.input_bias_momentums + (1-self.beta_1) * layer.dinput_biases
            input_weights_momentums_corrected = layer.input_weights_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            input_bias_momentums_corrected = layer.input_bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

            layer.input_weights_cache = self.beta_2 * layer.input_weights_cache + (1 - self.beta_2) * layer.dinput_weightss**2
            layer.input_bias_cache = self.beta_2 * layer.input_bias_cache + (1 - self.beta_2) * layer.dinput_biases**2
            input_weights_cache_corrected = layer.input_weights_cache / (1 - self.beta_2 ** (self.iterations + 1))
            input_bias_cache_corrected = layer.input_bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.input_weights += -self.current_learning_rate * input_weights_momentums_corrected / (np.sqrt(input_weights_cache_corrected) + self.epsilon)
            layer.input_biases += -self.current_learning_rate * input_bias_momentums_corrected / (np.sqrt(input_bias_cache_corrected) + self.epsilon)

            # Hidden weights
            layer.hidden_weights_momentums = self.beta_1 * layer.hidden_weights_momentums + (1-self.beta_1) * layer.dhidden_weights
            hidden_weights_momentums_corrected = layer.hidden_weights_momentums / (1 - self.beta_1 ** (self.iterations + 1))

            layer.hidden_weights_cache = self.beta_2 * layer.hidden_weights_cache + (1 - self.beta_2) * layer.dhidden_weightss**2
            hidden_weights_cache_corrected = layer.hidden_weights_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.hidden_weights += -self.current_learning_rate * hidden_weights_momentums_corrected / (np.sqrt(hidden_weights_cache_corrected) + self.epsilon)

    def post_update_parameters(self):
        self.iterations += 1