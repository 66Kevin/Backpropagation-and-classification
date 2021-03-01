# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#
import os
import numpy as np
import time
import fnn_utils


# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return x * (x > 0.0)


def relu_d(x):
    return 1.0 * (x > 0)


class BackPropagation:
    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.
    def __init__(self, network_shape=[784, 20, 20, 10]):

        # Read the training and test data using cthe provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Scaling the data from [0,255] to [0,1]
        self.trainX, self.testX = self.trainX / 255, self.testX / 255

        # Number of layers in the network
        self.L = len(network_shape)  # Input Layer + 3Hidden Layer + Output Layer

        self.crossings = [(1 if i < 1 else network_shape[i - 1], network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a = [np.zeros(m) for m in network_shape]
        self.db = [np.zeros(m) for m in network_shape]
        self.b = [np.random.normal(0, 1 / 10, m) for m in network_shape]
        self.z = [np.zeros(m) for m in network_shape]
        self.delta = [np.zeros(m) for m in network_shape]
        self.w = [np.random.uniform(-1 / np.sqrt(m0), 1 / np.sqrt(m0), (m1, m0)) for (m0, m1) in self.crossings]
        self.dw = [np.zeros((m1, m0)) for (m0, m1) in self.crossings]
        self.nabla_C_out = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi = relu
        self.phi_d = relu_d

        # Store activations over the batch for plotting
        self.batch_a = [np.zeros(m) for m in network_shape]

    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """
        self.a[0] = x - 0.5  # Center the input values between [-0.5,0.5]
        for layer in range(1, self.L):
            self.z[layer] = np.dot(self.w[layer], self.a[layer - 1]) + self.b[layer]
            # print(np.array(self.w[layer]).shape)
            # print(np.array(self.a[layer - 1]).shape)
            # print(np.array(self.b[layer]).shape)
            self.a[layer] = self.phi(self.z[layer])
        self.a[self.L - 1] = self.softmax(self.a[self.L - 1])
        return self.a[self.L - 1]

    def softmax(self, z):
        z = z - np.max(z)
        exps = np.exp(z)
        if z.ndim == 1:
            return exps / np.sum(np.exp(z))
        return exps / np.sum(np.exp(z), keepdims=True, axis=1)

    def loss(self, pred, y):
        return -np.log(pred[np.argmax(y)])

    def backward(self, x, y):
        """ Compute local gradients, then return gradients of network.
        """
        # derivative for softmax and cross entropy loss together
        self.delta[self.L - 1] = self.a[self.L - 1] - y
        for layer in range(self.L - 2, 0, -1):
            self.delta[layer] = np.dot(self.w[layer + 1].T, self.delta[layer + 1]) * self.phi_d(self.z[layer])

        for layer in range(1, self.L):
            # print(self.delta[layer].reshape(-1,1).shape)
            # print(self.a[layer - 1].reshape(1,-1).shape)
            self.dw[layer] = self.dw[layer] + np.dot(self.delta[layer].reshape(-1, 1), self.a[layer - 1].reshape(1, -1))
            self.db[layer] = self.db[layer] + self.delta[layer]

    # Return predicted image class for input x
    def predict(self, x):
        return np.argmax(self.forward(x))

    # Return predicted percentage for class j
    def predict_pct(self, j):
        return self.a[self.L - 1][j]

    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X), len(Y))
        samples = np.random.randint(num_data, size=N)
        results = [(self.predict(x), np.argmax(y)) for (x, y) in zip(X[samples], Y[samples])]
        return sum(int(x == y) for (x, y) in results) / N

    def sgd(self, batch_size=256, epsilon=0.0001, epochs=30):
        """ Mini-batch gradient descent on training data.
            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N / batch_size)

        # Variables to keep track of statistics
        loss_log = []
        test_acc_log = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True

        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):

            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)

            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append(self.evaluate(self.testX, self.testY, 1000))
            train_acc_log.append(self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            for k in range(num_batches):

                # Reset buffer containing updates
                for layer in range(self.L):
                    self.dw[layer].fill(0.0)
                    self.db[layer].fill(0.0)

                # Mini-batch loop
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k * batch_size + i]]
                    y = self.trainY[permutation[k * batch_size + i]]

                    # Feed forward inputs
                    self.forward(x)

                    # Compute gradients
                    self.backward(x, y)

                    # Update loss log
                    batch_loss += self.loss(self.a[self.L - 1], y)
                    # print(self.loss(self.a[self.L - 1], y))

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size

                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1, self.L):
                    self.w[l] = self.w[l] - epsilon * self.dw[l]
                    self.b[l] = self.b[l] - epsilon * self.db[l]

                # Update logs
                loss_log.append(batch_loss / batch_size)
                batch_loss = 0

                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)

                # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self, show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)
            train_acc = self.evaluate(self.trainX, self.trainY, 1000)
            test_acc = self.evaluate(self.testX, self.testY, 1000)
            # print('train accuracy: ' + str(train_acc))
            # print('test accuracy: ' + str(test_acc))
            # print('train_acc_log: '+str(train_acc_log))
            # print('test_acc_log: ' + str(test_acc_log))


# Start training with default parameters.

def main():
    bp = BackPropagation()
    bp.sgd()


if __name__ == "__main__":
    main()
