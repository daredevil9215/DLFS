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

    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999) -> None:
        """
        Adam optimizing algorithm.

        Parameters
        ----------
        learning_rate : float, default=0.001
            Step size used in gradient descent.

        decay : float, default=0
            Factor used to reduce learning rate over time.

        epsilon : float, default=1e-7
            Factor used to avoid divison by zero while updating parameters.

        beta_1 : float, default=0.9
            Exponential decay rate for momentum.
        
        beta_2 : float, default=0.999
            Exponential decay rate for cache.

        Attributes
        ----------
        iterations : int, default=0
            Number of training iterations used to calculate new learning rate with decay.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0 

    def _init_parameters(self, params: tuple) -> list:
        """
        Helper method for initializing Adam parameters of a layer (momentums or cache).

        Parameters
        ----------
        params : tuple
            Tuple of layer parameters.

        Returns
        -------
        adam_params : list
            List of zero-valued momentums or cache arrays.
        """
        return [np.zeros_like(param) for param in params]
    
    def _update_parameters(self, params: np.ndarray, gradients: np.ndarray, momentums: np.ndarray, cache: np.ndarray) -> tuple:
        """
        Helper method for updating Adam parameters of a layer (momentums and caches).

        Parameters
        ----------
        params : np.ndarray
            Layer parameter to be updated.

        gradient : np.ndarray
            Layer gradient used for update.

        momentums : np.ndarray
            Layer momentums used for update.

        cache : np.ndarray
            Layer cache used for update.

        Returns
        -------
        updated_param, new_momentums, new_cache : tuple
            Updated parameter, momentums and cache.
        """
        new_momentums = self.beta_1 * momentums + (1 - self.beta_1) * gradients
        momentums_corrected = new_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        new_cache = self.beta_2 * cache + (1 - self.beta_2) * gradients**2
        cache_corrected = new_cache / (1 - self.beta_2 ** (self.iterations + 1))

        parameter_update = -self.current_learning_rate * momentums_corrected / (np.sqrt(cache_corrected) + self.epsilon)
        return params + parameter_update, new_momentums, new_cache
        
    def _init_lstm(self, layer: LSTMLayer) -> None:
        """
        Helper method for initializing LSTMLayer momentums and caches.

        Parameters
        ----------
        layer : LSTMLayer
            LSTM layer to initialize.

        Returns
        -------
        None
        """

        lstm_parms = (layer.input_weights, layer.input_bias, 
                      layer.forget_weights, layer.forget_bias, 
                      layer.candidate_weights, layer.candidate_bias, 
                      layer.output_weights, layer.output_bias)

        (layer.input_weights_cache, layer.input_bias_cache,
        layer.forget_weights_cache, layer.forget_bias_cache,
        layer.candidate_weights_cache, layer.candidate_bias_cache,
        layer.output_weights_cache, layer.output_bias_cache) = self._init_parameters(lstm_parms)
         
        (layer.input_weights_momentums, layer.input_bias_momentums, 
        layer.forget_weights_momentums, layer.forget_bias_momentums, 
        layer.candidate_weights_momentums, layer.candidate_bias_momentums, 
        layer.output_weights_momentums, layer.output_bias_momentums) = self._init_parameters(lstm_parms)

    def _update_lstm(self, layer: LSTMLayer) -> None:
        """
        Helper method for updating LSTMLayer parameters, momentums and caches.

        Parameters
        ----------
        layer : LSTMLayer
            LSTM layer to update.

        Returns
        -------
        None
        """

        # Input weights
        layer.input_weights, layer.input_weights_momentums, layer.input_weights_cache = self._update_parameters(layer.input_weights, layer.dinput_weights, layer.input_weights_momentums, layer.input_weights_cache)
        layer.input_bias, layer.input_bias_momentums, layer.input_bias_cache = self._update_parameters(layer.input_bias, layer.dinput_bias, layer.input_bias_momentums, layer.input_bias_cache)

        # Forget weights
        layer.forget_weights, layer.forget_weights_momentums, layer.forget_weights_cache = self._update_parameters(layer.forget_weights, layer.dforget_weights, layer.forget_weights_momentums, layer.forget_weights_cache)
        layer.forget_bias, layer.forget_bias_momentums, layer.forget_bias_cache = self._update_parameters(layer.forget_bias, layer.dforget_bias, layer.forget_bias_momentums, layer.forget_bias_cache)

        # Candidate weights
        layer.candidate_weights, layer.candidate_weights_momentums, layer.candidate_weights_cache = self._update_parameters(layer.candidate_weights, layer.dcandidate_weights, layer.candidate_weights_momentums, layer.candidate_weights_cache)
        layer.candidate_bias, layer.candidate_bias_momentums, layer.candidate_bias_cache = self._update_parameters(layer.candidate_bias, layer.dcandidate_bias, layer.candidate_bias_momentums, layer.candidate_bias_cache)

        # Output weights
        layer.output_weights, layer.output_weights_momentums, layer.output_weights_cache = self._update_parameters(layer.output_weights, layer.doutput_weights, layer.output_weights_momentums, layer.output_weights_cache)
        layer.output_bias, layer.output_bias_momentums, layer.output_bias_cache = self._update_parameters(layer.output_bias, layer.doutput_bias, layer.output_bias_momentums, layer.output_bias_cache)

    def _init_dense(self, layer: DenseLayer) -> None:
        """
        Helper method for initializing DenseLayer momentums and caches.

        Parameters
        ----------
        layer : DenseLayer
            Dense layer to initialize.

        Returns
        -------
        None
        """
        dense_parms = (layer.weights, layer.biases)
        layer.weights_cache, layer.bias_cache = self._init_parameters(dense_parms)
        layer.weights_momentums, layer.bias_momentums = self._init_parameters(dense_parms)

    def _update_dense(self, layer: DenseLayer) -> None:
        """
        Helper method for updating DenseLayer parameters, momentums and caches.

        Parameters
        ----------
        layer : DenseLayer
            Dense layer to update.

        Returns
        -------
        None
        """
        layer.weights, layer.weights_momentums, layer.weights_cache = self._update_parameters(layer.weights, layer.dweights, layer.weights_momentums, layer.weights_cache)
        layer.biases, layer.bias_momentums, layer.bias_cache = self._update_parameters(layer.biases, layer.dbiases, layer.bias_momentums, layer.bias_cache)

    def _init_conv(self, layer: ConvolutionalLayer) -> None:
        """
        Helper method for initializing ConvolutionalLayer momentums and caches.

        Parameters
        ----------
        layer : ConvolutionalLayer
            Convolutional layer to initialize.

        Returns
        -------
        None
        """
        conv_params = (layer.kernels, layer.biases)
        layer.kernel_cache, layer.bias_cache = self._init_parameters(conv_params)
        layer.kernel_momentums, layer.bias_momentums = self._init_parameters(conv_params)

    def _update_conv(self, layer: ConvolutionalLayer) -> None:
        """
        Helper method for updating ConvolutionalLayer parameters, momentums and caches.

        Parameters
        ----------
        layer : ConvolutionalLayer
            Convolutional layer to update.

        Returns
        -------
        None
        """
        layer.kernels, layer.kernel_momentums, layer.kernel_cache = self._update_parameters(layer.kernels, layer.dkernels, layer.kernel_momentums, layer.kernel_cache)
        layer.biases, layer.bias_momentums, layer.bias_cache = self._update_parameters(layer.biases, layer.dbiases, layer.bias_momentums, layer.bias_cache)

    def _init_recurrent(self, layer: RecurrentLayer) -> None:
        """
        Helper method for initializing RecurrentLayer momentums and caches.

        Parameters
        ----------
        layer : RecurrentLayer
            Recurrent layer to initialize.

        Returns
        -------
        None
        """
        recurrent_params = (layer.input_weights, layer.input_bias, layer.hidden_weights)
        layer.input_weights_cache, layer.input_bias_cache, layer.hidden_weights_cache = self._init_parameters(recurrent_params)
        layer.input_weights_momentums, layer.input_bias_momentums, layer.hidden_weights_momentums = self._init_parameters(recurrent_params)

    def _update_recurrent(self, layer: RecurrentLayer) -> None:
        """
        Helper method for updating ConvolutionalLayer parameters, momentums and caches.

        Parameters
        ----------
        layer : ConvolutionalLayer
            Convolutional layer to update.

        Returns
        -------
        None
        """
        layer.input_weights, layer.input_weights_momentums, layer.input_weights_cache = self._update_parameters(layer.input_weights, layer.dinput_weights, layer.input_weights_momentums, layer.input_weights_cache)
        layer.input_bias, layer.input_bias_momentums, layer.input_bias_cache = self._update_parameters(layer.input_bias, layer.dinput_bias, layer.input_bias_momentums, layer.input_bias_cache)
        layer.hidden_weights, layer.hidden_weights_momentums, layer.hidden_weights_cache = self._update_parameters(layer.hidden_weights, layer.dhidden_weights, layer.hidden_weights_momentums, layer.hidden_weights_cache)

    def pre_update_parameters(self) -> None:
        """
        Method for updating learning rate based on current number of iterations and decay.

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
            Layer to update.

        Returns
        -------
        None
        """
        
        if isinstance(layer, LSTMLayer):

            if not hasattr(layer, 'input_weights_cache'):
                self._init_lstm(layer)

            self._update_lstm(layer)
    
        elif isinstance(layer, DenseLayer):

            if not hasattr(layer, 'weights_cache'):
                self._init_dense(layer)

            self._update_dense(layer)

        elif isinstance(layer, ConvolutionalLayer):

            if not hasattr(layer, 'kernel_cache'):
                self._init_conv(layer)

            self._update_conv(layer)

        elif isinstance(layer, RecurrentLayer):
            
            if not hasattr(layer, 'input_weights_cache'):
                self._init_recurrent(layer)

            self._update_recurrent(layer)

    def post_update_parameters(self) -> None:
        """
        Method for updating current number of iterations.

        Returns
        -------
        None
        """
        self.iterations += 1