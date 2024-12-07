import numpy as np


# ALL EQUATIONS USED AND THEIR DERIVATIVES


class Activations:
    def __init__(self):
        raise ReferenceError('An object can not exist for this class')
   
    # ========================
    # ACTIVATION FUNCTIONS
    # ========================
   
    def linear(x, dx=False):
        """
        No activation function (linear)


        Args:
            x: The data to apply the activation to.
            dx (bool): Defines if the differenciated function should be applied.
        """
        return x if not dx else x ** 0
       
    def sigmoid(x, dx=False):
        """
        No activation function (linear)


        Args:
            x: The data to apply the activation to.
            dx (bool): Defines if the differenciated function should be applied.
        """
        return 1 / (1 + np.exp(-x)) if not dx else (np.exp(-x)) / ((1 + (np.exp(-x))) ** 2)
   
    def relu(x, dx=False):
        """
        relu activation function


        Args:
            x: The data to apply the activation to.
            dx (bool): Defines if the differenciated function should be applied.
        """
        return np.maximum(0, x) if not dx else np.where(x > 0, 1, 0)
   
    def leaky_relu(x, dx=False):
        """
        leaky relu activation function.


        Args:
            x: The data to apply the activation to.
            dx (bool): Defines if the differenciated function should be applied.
        """
        alpha = 0.01
        return np.maximum(alpha * x, x) if not dx else np.where(x > 0, 1, alpha)
       
    def tanh(x, dx=False):
        """
        tanh activation function


        Args:
            x: The data to apply the activation to.
            dx (bool): Defines if the differenciated function should be applied.
        """
        return np.tanh(x) if not dx else 1 - (np.tanh(x) ** 2)
   
    def elu(x, dx=False):
        """
        elu activation function


        Args:
            x: The data to apply the activation to.
            dx (bool): Defines if the differenciated function should be applied.
        """
        return np.where(x > 0, x, np.exp(x) - 1) if not dx else np.where(x > 0, 1, np.exp(x))
   
    def softmax(x):
        """
        softmax activation function
        derivative should never be necessary


        Args:
            x: The data to apply the activation to.
            dx (bool): Defines if the differenciated function should be applied.
        """
        return np.exp(x - np.max(x)) / sum(np.exp(x - np.max(x)))
           
    def step(x, dx=False):
        """
        step activation function


        Args:
            x: The data to apply the activation to.
            dx (bool): Defines if the differenciated function should be applied.
        """
        return np.where(x > 0, 1, 0) if not dx else 0






class Loss:
    def __init__(self):
        raise ReferenceError('An object can not exist for this class')


    # ========================
    # LOSS FUNCTIONS
    # ========================
   
    def mse(y_pred, y_actu, dx=False):
        """
        Mean squared error cost function (for regression models)


        Args:
            y_pred: The network's predicted values for the given set of inputs
            y_actu: The actually assigned values for the given set of inputs
            dx (bool): Defines if the differenciated function should be applied
        """
        # MEAN SQUARED ERROR
        return np.square(y_pred - y_actu) / 2 if not dx else y_pred - y_actu
   
    def bce(y_pred, y_actu, dx=False):
        """
        Binary-cross entropy cost function (used for binary classification problems)


        Args:
            y_pred: The network's predicted values for the given set of inputs
            y_actu: The actually assigned values for the given set of inputs
            dx (bool): Defines if the differenciated function should be applied
        """
        # BINARY CROSS-ENTROPY LOSS
        epsilon = 1e-10
        return -(y_actu * np.log(y_pred + epsilon)) - ((1 - y_actu) * np.log(1 - y_pred + epsilon)) if not dx else -(y_actu / (y_pred + epsilon)) + ((1 - y_actu) / (1 - y_pred + epsilon))
       
    def cce(y_pred, y_actu, dx=False):
        """
        Categorial-cross entropy cost function (used for multi-class classification problems)


        Args:
            y_pred: The network's predicted values for the given set of inputs
            y_actu: The actually assigned values for the given set of inputs
            dx (bool): Defines if the differenciated function should be applied
        """
        # CATEGORIAL CROSS-ENTROPY LOSS
        epsilon = 1e-10
        return -y_actu * np.log(y_pred + epsilon) if not dx else -y_actu / (y_pred + epsilon)
   
   
   
class Normalization:
    def __init__(self):
        raise ReferenceError('An object can not exist for this class')
       
    # ========================
    # NORMALIZATION EQUATIONS
    # ========================
   
    def n_minmax(dataset: np.ndarray, min: float, max: float, inverse: bool = False):
        """
        Minmax normalization function


        Args:
            dataset (array): The data to be normalized
            min (float): the minimum value of the full dataset
            max (float): the maximum value of the full dataset
            inverse (bool): defines whether or not the dataset should be normalized or de-normalized
        """
        if dataset is not None:
            return (dataset - min) / (max - min) if not inverse else ((dataset * (max - min)) + min)
        return None
   
    def n_zscore(dataset: np.ndarray, mean: float, stddev: float, inverse=False):
        """
        Z-score normalization function


        Args:
            dataset (array): The data to be normalized
            min (float): the minimum value of the full dataset
            max (float): the maximum value of the full dataset
            inverse (bool): defines whether or not the dataset should be normalized or de-normalized
        """
        if dataset is not None:
            return (dataset - mean) / stddev if not inverse else (dataset * stddev) + mean
        return None
