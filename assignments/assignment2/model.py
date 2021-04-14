import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        params = self.params()
        for param_key, param_value in params.items():
            param_value.grad = np.zeros_like(param_value.value)

        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        X = self.fc1.forward(X)
        X = self.relu.forward(X)
        X = self.fc2.forward(X)
        loss, grad = softmax_with_cross_entropy(X, y)

        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)

        for param_key, param_value in params.items():
            reg_loss, reg_grad = l2_regularization(param_value.value, self.reg)
            loss += reg_loss
            param_value.grad += reg_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        X = self.fc1.forward(X)
        X = self.relu.forward(X)
        X = self.fc2.forward(X)
        X = softmax(X)
        return np.argmax(X, axis=1)

    def params(self):
        result = {}
        layers_with_params = [self.fc1, self.fc2]
        for i in range(len(layers_with_params)):
            layer = layers_with_params[i]
            layer_number = str(i)
            for param_key, param_value in layer.params().items():
                result[param_key + str(layer_number)] = param_value

        return result
