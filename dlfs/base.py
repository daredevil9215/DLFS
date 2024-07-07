import numpy as np

class Layer:

    def __init__(self) -> None:
        """
        Layer abstract base class.

        Attributes
        ----------
        output : np.ndarray
            Output of the layer.

        dinputs : np.ndarray
            Gradient with respect to inputs used in backpropagation.
        """
        self.output: np.ndarray = None
        self.dinputs: np.ndarray = None

    def forward(self, inputs: np.ndarray) -> None:
        pass

    def backward(self, delta: np.ndarray) -> None:
        pass

class Activation:

    def __init__(self) -> None:
        """
        Activation function abstract base class.

        Attributes
        ----------
        output : np.ndarray
            Output of the activation function.

        dinputs : np.ndarray
            Gradient with respect to inputs used in backpropagation.
        """
        self.output: np.ndarray = None
        self.dinputs: np.ndarray = None

    def forward(self, inputs: np.ndarray) -> None:
        pass

    def backward(self, delta: np.ndarray) -> None:
        pass

class Loss:

    def __init__(self) -> None:
        """
        Loss function abstract base class.

        Attributes
        ----------
        dinputs : np.ndarray
            Gradient with respect to inputs used in backpropagation.
        """
        self.dinputs: np.ndarray = None

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        pass

class Optimizer:

    def __init__(self) -> None:
        """
        Optimizer abstract base class.
        """

    def update_parameters(self) -> None:
        pass

    def update_layer_parameters(self, layer: Layer) -> None:
        pass