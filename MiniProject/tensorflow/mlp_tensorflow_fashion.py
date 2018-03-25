# Implementation of a simple MLP network with one hidden layer. Tested on the Fashion MNIST dataset.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
import pandas as pd

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.relu(tf.matmul(X, w_1))  # The activation function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def main():
    fash_data = np.array(pd.read_csv('fashion-mnist_train.csv'))
    train_X = np.ones((fash_data.shape))
    train_X[:,1:] = fash_data[:,1:]
    fash_test = np.array(pd.read_csv('fashion-mnist_test.csv'))
    test_X = np.ones((fash_test.shape))
    test_X[:,1:] = fash_test[:,1:]
    t = fash_data[:,:1]
    train_y = np.zeros((60000,10))
    for i in range(len(t)):
        train_y[i,t[i]] = 1
    print ("train_y: ", train_y)
    print ("train_y.shape: ", train_y.shape)
    T = fash_test[:,:1]
    test_y = np.zeros((10000,10))
    for i in range(len(T)):
        test_y[i,T[i]] = 1
    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 784 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (10 classes)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))
    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)
    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.AdagradOptimizer(0.01,initial_accumulator_value=0.1).minimize(cost)
    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    print("running init\n")
    print(sess.run(init))
    print("running w_1\n")
    print(sess.run(w_1).shape)
    print("running w_2\n")
    print(sess.run(w_2).shape)
    
    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()
