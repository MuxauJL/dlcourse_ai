import numpy as np
from sys import float_info


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W * W)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    if predictions.ndim == 1:
        preds = predictions - np.max(predictions)
        exps = np.exp(preds)
        return exps / np.sum(exps)
    else:
        preds = predictions - np.max(predictions, axis=1)[:, np.newaxis]
        exps = np.exp(preds)
        return exps / np.sum(exps, axis=1)[:, np.newaxis]


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        return -np.log(probs[target_index] + float_info.epsilon)
    else:
        batch_size = probs.shape[0]
        return -np.sum(np.log(probs[np.arange(batch_size), target_index] + float_info.epsilon))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    scores = softmax(predictions)
    loss = cross_entropy_loss(scores, target_index)

    dprediction = scores
    if scores.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = scores.shape[0]
        dprediction[np.arange(batch_size), target_index] -= 1

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        relu = np.zeros_like(X)
        self.not_null_indices = X > 0
        relu[self.not_null_indices] = X[self.not_null_indices]
        return relu

    def backward(self, d_out):
        d_result = np.zeros_like(d_out)
        d_result[self.not_null_indices] = d_out[self.not_null_indices]
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.matmul(X, self.W.value) + self.B.value

    def backward(self, d_out):
        self.W.grad += np.matmul(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)
        return np.matmul(d_out, self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, stride=1):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.stride = stride

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        assert channels == self.in_channels

        padded_h = height + self.padding + self.padding
        padded_w = width + self.padding + self.padding
        out_height = (padded_h - self.filter_size) // self.stride + 1
        out_width = (padded_w - self.filter_size) // self.stride + 1

        if self.padding > 0:
            self.padded_X = np.zeros((batch_size, padded_h, padded_w, channels))
            self.padded_X[:, self.padding: -self.padding, self.padding: -self.padding, :] = X
        else:
            self.padded_X = X

        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        weights = self.W.value.reshape((self.filter_size * self.filter_size * self.in_channels, self.out_channels))
        bias = self.B.value[np.newaxis, :]
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        shift_y = 0
        for y in range(out_height):
            shift_x = 0
            for x in range(out_width):
                region = self.padded_X[:, shift_y: shift_y + self.filter_size,
                         shift_x: shift_x + self.filter_size, :]
                region = region.reshape((batch_size, -1))
                result[:, y, x, :] = np.matmul(region, weights) + bias

                shift_x += self.stride
            shift_y += self.stride

        return result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, out_height, out_width, out_channels = d_out.shape

        assert out_channels == self.out_channels

        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        weights_shape = (self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        weights_grad = self.W.grad.reshape(weights_shape)
        weights_value = self.W.value.reshape(weights_shape)
        padded_input_grad = np.zeros(self.padded_X.shape)
        region_shape = (batch_size, self.filter_size, self.filter_size, self.in_channels)
        # Try to avoid having any other loops here too
        shift_y = 0
        for y in range(out_height):
            shift_x = 0
            for x in range(out_width):
                # Backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                region = self.padded_X[:, shift_y: shift_y + self.filter_size,
                    shift_x: shift_x + self.filter_size, :]
                region = region.reshape((batch_size, -1))
                local_d_out = d_out[:, y, x, :].reshape(batch_size, -1)
                weights_grad += np.matmul(region.T, local_d_out)
                self.B.grad += np.sum(local_d_out, axis=0)

                local_input_grad = np.matmul(local_d_out, weights_value.T)
                padded_input_grad[:, shift_y: shift_y + self.filter_size,
                    shift_x: shift_x + self.filter_size, :] += local_input_grad.reshape(region_shape)

                shift_x += self.stride
            shift_y += self.stride

        self.W.grad = weights_grad.reshape(self.W.grad.shape)
        if self.padding > 0:
            return padded_input_grad[:, self.padding: -self.padding, self.padding: -self.padding, :]
        else:
            return padded_input_grad

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        result = np.zeros((batch_size, out_height, out_width, channels))
        self.max_indices = np.zeros((batch_size, out_height, out_width, channels), dtype=np.int64)
        for n in range(batch_size):
            shift_y = 0
            for y in range(out_height):
                shift_x = 0
                for x in range(out_width):
                    for c in range(channels):
                        region = X[n, shift_y: shift_y + self.pool_size, shift_x: shift_x + self.pool_size, c]
                        self.max_indices[n, y, x, c] = region.argmax()
                        result[n, y, x, c] = region.max()

                    shift_x += self.stride
                shift_y += self.stride

        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        result_grad = np.zeros(self.X.shape)
        pool_shape = (self.pool_size, self.pool_size)
        for n in range(batch_size):
            shift_y = 0
            for y in range(out_height):
                shift_x = 0
                for x in range(out_width):
                    for c in range(channels):
                        h, w = np.unravel_index(self.max_indices[n, y, x, c], pool_shape)
                        result_grad[n, shift_y + h, shift_x + w, c] += d_out[n, y, x, c]

                    shift_x += self.stride
                shift_y += self.stride

        return result_grad

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape((batch_size, - 1))

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
