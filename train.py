import numpy as np
import scipy.optimize
import sample_images
from neuralNetwork import initialize, networkCost
import gradient
import display_network
import load_MNIST


##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to
#  change the parameters below.

# number of input units
visible_size = 28 * 28
# number of input units
hidden_size = 196
# weight decay parameter
lambda_ = 3e-3
# debug
debug = False


##======================================================================
## STEP 1: load images and initialize

# Loading 10K images from MNIST database
images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
patches = images[:, 0:10000]# EXPERIMENT WITH THIS

#  Obtain random parameters theta
theta = initialize(hidden_size, visible_size)

##======================================================================
## STEP 2: Implement networkCost
#
#  You can implement all of the components (squared error cost, weight decay term,
#  ) in the cost function at once, but it may be easier to do
#  it step-by-step and run gradient checking (see STEP 3) after each step.  We
#  suggest implementing the networkCost function using the following steps:
#
#  (a) Implement forward propagation in your neural network, and implement the
#      squared error term of the cost function.  Implement backpropagation to
#      compute the derivatives. Then (using lambda=0), run Gradient Checking
#      to verify that the calculations corresponding to the squared error cost
#      term are correct.
#
#  (b) Add in the weight decay term (in both the cost function and the derivative
#      calculations), then re-run Gradient Checking to verify correctness.
#      verify correctness.
#
#  Feel free to change the training settings when debugging your
#  code.  (For example, reducing the training set size or
#  number of hidden units may make your code run faster; and setting lambda
#  to zero may be helpful for debugging.)  However, in your
#  final submission of the visualized weights, please use parameters we
#  gave in Step 0 above.

#...................................................
(cost, grad) = networkCost(theta, visible_size,hidden_size, lambda_, patches)
#FILL UP THE ABOVE FUNCTION
#...................................................

print (cost, grad)
##======================================================================
## STEP 3: Gradient Checking
#
# Hint: If you are debugging your code, performing gradient checking on smaller models
# and smaller training sets (e.g., using only 10 training examples and 1-2 hidden
# units) may speed things up.

# First, lets make sure your numerical gradient computation is correct for a
# simple function.  After you have implemented computeNumericalGradient.m,
# run the following:


if debug:
    gradient.check_gradient()

    # Now we can use it to check your cost function and derivative calculations
    # for the sparse autoencoder.
    # J is the cost function
#.......... WHAT IS THE MEANING OF THIS
    J = lambda x: networkCost(x, visible_size, hidden_size,
                                                             lambda_,
                                                             patches)
    num_grad = gradient.compute_gradient(J, theta)

    # Use this to visually compare the gradients side by side
    print (num_grad, grad)

    # Compare numerically computed gradients with the ones obtained from backpropagation
    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print (diff)
    print ("Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n\n")

##======================================================================
## STEP 4: After verifying that your implementation of
#  sparseAutoencoderCost is correct, You can start training your sparse
#  autoencoder with minFunc (L-BFGS).

#  Randomly initialize the parameters
theta = initialize(hidden_size, visible_size)

J = lambda x: networkCost(x, visible_size, hidden_size,
                                                         lambda_, patches)
options_ = {'maxiter': 400, 'disp': True}
result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
opt_theta = result.x

print (result)

##======================================================================
## STEP 5: Visualization

W1 = opt_theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size).transpose()
display_network.display_network(W1)

