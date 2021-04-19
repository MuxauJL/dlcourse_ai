import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax, softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        filter_size = 3
        pool_size = 4
        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, filter_size, padding=1)
        self.relu1 = ReLULayer()
        self.max_pool1 = MaxPoolingLayer(pool_size, stride=pool_size)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size, padding=1)
        self.relu2 = ReLULayer()
        self.max_pool2 = MaxPoolingLayer(pool_size, stride=pool_size)
        self.flatten = Flattener()
        self.fc = FullyConnectedLayer(n_input=4 * conv2_channels, n_output=n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        params = self.params()
        for param_key, param_value in params.items():
            param_value.grad = np.zeros_like(param_value.value)

        # Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        X = self.conv1.forward(X)
        X = self.relu1.forward(X)
        X = self.max_pool1.forward(X)
        X = self.conv2.forward(X)
        X = self.relu2.forward(X)
        X = self.max_pool2.forward(X)
        X = self.flatten.forward(X)
        X = self.fc.forward(X)
        loss, grad = softmax_with_cross_entropy(X, y)

        grad = self.fc.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.max_pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.max_pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

        return loss

    def predict(self, X):
        X = self.conv1.forward(X)
        X = self.relu1.forward(X)
        X = self.max_pool1.forward(X)
        X = self.conv2.forward(X)
        X = self.relu2.forward(X)
        X = self.max_pool2.forward(X)
        X = self.flatten.forward(X)
        X = self.fc.forward(X)
        X = softmax(X)
        return np.argmax(X, axis=1)

    def params(self):
        # Aggregate all the params from all the layers
        # which have parameters
        result = {}
        layers_with_params = [self.conv1, self.conv2, self.fc]
        for i in range(len(layers_with_params)):
            layer = layers_with_params[i]
            layer_number = str(i)
            for param_key, param_value in layer.params().items():
                result[param_key + str(layer_number)] = param_value

        return result
