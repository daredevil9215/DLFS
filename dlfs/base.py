import numpy as np

class Layer:
    """
    Layer abstract base class.
    """
    def forward(self, inputs: np.ndarray) -> None:
        pass

    def backward(self, delta: np.ndarray) -> None:
        pass

class Activation:
    """
    Activation function abstract base class.
    """
    def forward(self, inputs: np.ndarray) -> None:
        pass

    def backward(self, delta: np.ndarray) -> None:
        pass

class Loss:
    """
    Loss function abstract base class.
    """
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        pass

class Optimizer:
    """
    Optimizer abstract base class.
    """
    def pre_update_parameters(self) -> None:
        pass

    def update_layer_parameters(self, layer: Layer) -> None:
        pass

    def post_update_parameters(self) -> None:
        pass