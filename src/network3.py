"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.

"""

#### Libraries
# Standard library
import pickle as cPickle
import gzip
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return tf.maximum(0.0, z)
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh

#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU. If this is not desired, then modify network3.py to set the GPU flag to False.")
    # You can set GPU configuration in TensorFlow if needed
else:
    print("Running with a CPU. If this is not desired, then modify network3.py to set the GPU flag to True.")

#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')

    def shared(data):
        shared_x = tf.Variable(np.asarray(data[0], dtype=np.float32), trainable=False)
        shared_y = tf.Variable(np.asarray(data[1], dtype=np.int32), trainable=False)
        return shared_x, tf.cast(shared_y, tf.int32)

    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tf.compat.v1.placeholder(tf.float32, [None, 784])
        self.y = tf.compat.v1.placeholder(tf.int32, [None])
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = len(training_data) // mini_batch_size
        num_validation_batches = len(validation_data) // mini_batch_size
        num_test_batches = len(test_data) // mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([tf.reduce_sum(layer.w ** 2) for layer in self.layers])
        cost = self.layers[-1].cost(self) + \
               0.5 * lmbda * l2_norm_squared / num_training_batches
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(eta)
        grads_and_vars = optimizer.compute_gradients(cost, self.params)
        updates = optimizer.apply_gradients(grads_and_vars)

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        with tf.compat.v1.Session() as sess:
            best_validation_accuracy = 0.0
            best_iteration = 0
            test_accuracy = 0.0
            sess.run(tf.compat.v1.global_variables_initializer())
            
            for epoch in range(epochs):
                for minibatch_index in range(num_training_batches):
                    iteration = num_training_batches * epoch + minibatch_index
                    if iteration % 1000 == 0:
                        print("Training mini-batch number {0}".format(iteration))
                    start = minibatch_index * mini_batch_size
                    end = (minibatch_index + 1) * mini_batch_size
                    _, cost_val = sess.run([updates, cost], feed_dict={self.x: training_x[start:end],
                                                                        self.y: training_y[start:end]})
                    if (iteration + 1) % num_training_batches == 0:
                        validation_accuracy = np.mean([self.layers[-1].accuracy.eval(feed_dict={self.x: validation_x[start:end],
                                                                                                 self.y: validation_y[start:end]})
                                                       for start, end in mini_batch_indices(len(validation_data), mini_batch_size)])
                        print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))
                        if validation_accuracy >= best_validation_accuracy:
                            print("This is the best validation accuracy to date.")
                            best_validation_accuracy = validation_accuracy
                            best_iteration = iteration
                            if test_data:
                                test_accuracy = np.mean([self.layers[-1].accuracy.eval(feed_dict={self.x: test_x[start:end],
                                                                                                 self.y: test_y[start:end]})
                                                         for start, end in mini_batch_indices(len(test_data), mini_batch_size)])
                                print('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))
            print("Finished training network.")
            print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
                best_validation_accuracy, best_iteration))
            print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))
            

def mini_batch_indices(n, mini_batch_size):
    """Helper function to generate indices for mini-batches."""
    indices = np.arange(n)
    np.random.shuffle(indices)
    return [(start, start + mini_batch_size) for start in range(0, n, mini_batch_size)]


#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=tf.nn.sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        self.w = tf.Variable(
            np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape).astype(np.float32),
            name='w')
        self.b = tf.Variable(
            np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)).astype(np.float32),
            name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, self.image_shape)
        conv_out = tf.nn.conv2d(input=self.inpt, filters=self.w, strides=[1, 1, 1, 1], padding='VALID')
        pooled_out = tf.nn.max_pool(value=conv_out, ksize=[1, self.poolsize[0], self.poolsize[1], 1],
                                    strides=[1, self.poolsize[0], self.poolsize[1], 1], padding='VALID')
        self.output = self.activation_fn(pooled_out + tf.reshape(self.b, (1, -1, 1, 1)))
        self.output_dropout = self.output  # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=tf.nn.sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = tf.Variable(
            np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)).astype(np.float32),
            name='w')
        self.b = tf.Variable(
            np.random.normal(loc=0.0, scale=1.0, size=(n_out,)).astype(np.float32),
            name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt
        self.output = self.activation_fn((1-self.p_dropout) * tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.argmax(self.output, axis=1)
        self.inpt_dropout = tf.nn.dropout(inpt_dropout, rate=self.p_dropout)
        self.output_dropout = self.activation_fn(tf.matmul(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return tf.reduce_mean(tf.cast(tf.equal(y, self.y_out), tf.float32))


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = tf.Variable(tf.zeros([n_in, n_out], dtype=tf.float32), name='w')
        self.b = tf.Variable(tf.zeros([n_out], dtype=tf.float32), name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt
        self.output = tf.nn.softmax((1 - self.p_dropout) * tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.argmax(self.output, axis=1)
        self.inpt_dropout = tf.nn.dropout(inpt_dropout, rate=self.p_dropout)
        self.output_dropout = tf.nn.softmax(tf.matmul(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        indices = tf.range(tf.shape(net.y)[0])
        indices = tf.stack([indices, net.y], axis=1)
        return -tf.reduce_mean(tf.gather_nd(tf.math.log(self.output_dropout + 1e-10), indices))

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return tf.reduce_mean(tf.cast(tf.equal(y, self.y_out), tf.float32))



#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].shape[0]
