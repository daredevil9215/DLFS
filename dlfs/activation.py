import numpy as np

class Activation:

    def forward(self):
        pass

    def backward(self):
        pass

class Linear(Activation):

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

class ReLU(Activation):

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

class Sigmoid(Activation):

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
