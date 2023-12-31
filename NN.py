from sklearn.neural_network import MLPClassifier
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
import numpy as np

def run(X_tr, y_tr, X_te, y_te):

    best_slice = 20000

    # best_hidden_layer_sizes, test_err = hiddenLayerTest(
    #     X_tr, y_tr, X_te, y_te, best_slice, plot=True
    # )
    best_learning_rate = 0.0001
    best_hidden_layer_sizes = (192, 192)
    alphaTest(X_tr, y_tr, X_te, y_te, best_slice, best_hidden_layer_sizes, best_learning_rate, plot=True)
    
    

def sliceTest(X_tr, y_tr, X_te, y_te, plot=True):


    # Define the slice sizes
    slices = [5000, 10000, 15000, 20000, 26032]

    # Initialize lists to store train and test errors
    train_error = []
    test_error = []

    # Iterate over each slice size
    for sl in slices:
        print("trying slice: ", sl)

        # Slice the training data
        X_tr_subset = X_tr[:sl, :]

        # Create and train the MLP classifier
        learner = MLPClassifier(random_state=1234)
        learner.fit(X_tr_subset, y_tr.ravel())

        # Calculate and store the train and test errors
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))

    # Plot the train and test errors if plot=True
    if plot:
        plt.plot(slices, train_error, label="train")
        plt.plot(slices, test_error, label="test")
        plt.legend()
        plt.xticks(slices)
        plt.show()

    # Find the best slice size with the minimum test error
    best_slice = slices[np.argmin(test_error)]
    print(f"Best Slice: {best_slice}")

    return best_slice, min(test_error)


def hiddenLayerTest(X_tr, y_tr, X_te, y_te, best_slice, plot=True):

    # Define different hidden layer sizes to test
    hidden_layer_sizes = [
        (32, 32),
        (64, 64),
        (96, 96),
        (128, 128),
        (160, 160),
        (192, 192),
    ]

    train_error = []
    test_error = []

    for hidden_layer_size in hidden_layer_sizes:
        print("trying hidden_layer_size: ", hidden_layer_size)

        # Subset the training data
        X_tr_subset = X_tr[:best_slice, :]
        y_tr_subset = y_tr[:best_slice]
        
        # Create and train the MLPClassifier
        learner = MLPClassifier(random_state=1234, hidden_layer_sizes=hidden_layer_size, verbose=False, max_iter=10000)
        learner.fit(X_tr_subset, y_tr_subset.ravel())

        # Calculate and store the train and test errors
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))

    if plot:
        # Plot the train and test errors
        plt.plot(range(len(hidden_layer_sizes)), train_error, label="train")
        plt.plot(range(len(hidden_layer_sizes)), test_error, label="test")
        plt.legend()
        plt.xticks(
            range(len(hidden_layer_sizes)),
            [str(hidden_layer_size) for hidden_layer_size in hidden_layer_sizes],
        )
        plt.ylabel("Error")
        plt.xlabel("Hidden Layer Size")
        plt.title("Hidden Layer Size vs. Error")
        plt.show()

    # Find the best hidden layer size with the minimum test error
    best_hidden_layer_size = hidden_layer_sizes[np.argmin(test_error)]
    print(f"Best Hidden Layer Size: {best_hidden_layer_size}")

    return best_hidden_layer_size, min(test_error)


def activationTest(
    X_tr, y_tr, X_te, y_te, plot=True
):
    """
    Test different activation functions for a neural network classifier.

    Parameters:
    X_tr (array-like): Training data features.
    y_tr (array-like): Training data labels.
    X_te (array-like): Test data features.
    y_te (array-like): Test data labels.
    best_slice (int): Number of samples to use for training and testing.
    best_hidden_layer_sizes (tuple): Sizes of hidden layers in the neural network.
    plot (bool, optional): Whether to plot the train and test errors. Defaults to True.

    Returns:
    tuple: A tuple containing the best activation function and the minimum test error.
    """
    # List of activation functions to test
    activations = ["identity", "logistic", "tanh", "relu"]

    # Lists to store train and test errors for each activation function
    train_error = []
    test_error = []

    # Iterate over each activation function
    for activation in activations:
        print("trying activation: ", activation)


        # Create a MLPClassifier with the current activation function
        learner = MLPClassifier(
            random_state=1234,
            activation=activation,
        )

        # Fit the classifier on the subset of training data
        learner.fit(X_tr, y_tr.ravel())

        # Calculate and store the train and test errors
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))

    # Plot the train and test errors if plot=True
    if plot:
        plt.plot(activations, train_error, label="train")
        plt.plot(activations, test_error, label="test")
        plt.legend()
        plt.xticks(range(len(activations)), activations)
        plt.show()

    # Find the best activation function with the minimum test error
    best_activation = activations[np.argmin(test_error)]
    print(f"Best Activation: {best_activation}")

    return best_activation, min(test_error)


def solverTest(
    X_tr,
    y_tr,
    X_te,
    y_te,
    best_hidden_layer_sizes,
    plot=True,
):
    """
    Test different solvers for the MLPClassifier model and return the best solver and its corresponding test error.

    Parameters:
    - X_tr (array-like): Training data features.
    - y_tr (array-like): Training data labels.
    - X_te (array-like): Test data features.
    - y_te (array-like): Test data labels.
    - best_slice (int): Number of samples to use for training and testing.
    - best_hidden_layer_sizes (tuple): Sizes of the hidden layers in the MLPClassifier model.
    - best_activation (str): Activation function to use in the MLPClassifier model.
    - plot (bool, optional): Whether to plot the train and test errors. Defaults to True.

    Returns:
    - best_solver (str): The best solver found.
    - min(test_error) (float): The minimum test error achieved.
    """
    # List of solvers to try
    solvers = ["lbfgs", "sgd", "adam"]

    # Lists to store train and test errors for each solver
    train_error = []
    test_error = []

    # Iterate over each solver
    for solver in solvers:
        print("trying solver: ", solver)

        # Create and train the MLPClassifier model
        learner = MLPClassifier(
            random_state=1234,
            solver=solver,
            hidden_layer_sizes=best_hidden_layer_sizes,
        )
        learner.fit(X_tr, y_tr.ravel())

        # Calculate and store the train and test errors
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))

    # Plot the train and test errors if plot=True
    if plot:
        plt.plot(solvers, train_error, label="train")
        plt.plot(solvers, test_error, label="test")
        plt.legend()
        plt.xticks(range(len(solvers)), solvers)
        plt.show()

    # Find the best solver based on the minimum test error
    best_solver = solvers[np.argmin(test_error)]
    print(f"Best Solver: {best_solver}")

    # Return the best solver and its corresponding test error
    return best_solver, min(test_error)


def learningRateTest(
    X_tr,
    y_tr,
    X_te,
    y_te,
    best_slice,
    best_hidden_layer_sizes,
    plot=True,
):
    # Define the learning rates to test
    learning_rates = [.00001, .0001, .001, .01, .1, 1]
    train_error = []
    test_error = []
    for lr in learning_rates:
        print("trying learning_rate: ", lr)
        # Subset the training data based on the best slice
        X_tr_subset = X_tr[:best_slice, :]
        y_tr_subset = y_tr[:best_slice]
        # Create an MLPClassifier with the specified parameters
        learner = MLPClassifier(
            random_state=1234,
            learning_rate_init=lr,
            hidden_layer_sizes=best_hidden_layer_sizes,
            max_iter = 10000
        )
        # Fit the learner on the subset of training data
        learner.fit(X_tr_subset, y_tr_subset.ravel())
        # Calculate the training and test error
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))
    if plot:
        # Plot the training and test error for each learning rate
        plt.semilogx(learning_rates, train_error, label="train")
        plt.semilogx(learning_rates, test_error, label="test")
        plt.legend()
        plt.xticks(learning_rates)
        plt.ylabel("Error")
        plt.xlabel("Learning Rate")
        plt.title("Learning Rate vs. Error")
        plt.show()
    # Find the learning rate with the minimum test error
    best_learning_rate = learning_rates[np.argmin(test_error)]
    print(f"Best Learning Rate: {best_learning_rate}")
    return best_learning_rate, min(test_error)


def batchTest(
    X_tr,
    y_tr,
    X_te,
    y_te,
    best_slice,
    best_hidden_layer_sizes,
    best_activation,
    best_learning_rate,
    best_solver,
    plot=True,
):
    """
    Perform batch testing on a neural network model.

    Args:
        X_tr (array-like): Training data features.
        y_tr (array-like): Training data labels.
        X_te (array-like): Testing data features.
        y_te (array-like): Testing data labels.
        best_slice (int): Number of samples to use for training and testing.
        best_hidden_layer_sizes (tuple): Sizes of the hidden layers in the neural network.
        best_activation (str): Activation function for the neural network.
        best_learning_rate (str): Learning rate for the neural network.
        best_solver (str): Solver algorithm for the neural network.
        plot (bool, optional): Whether to plot the training and testing errors. Defaults to True.

    Returns:
        tuple: A tuple containing the best batch size and the minimum testing error.
    """
    # Define the batch sizes to test
    batch_sizes = [100, 200, 300, 400, 500]

    # Initialize lists to store the training and testing errors
    train_error = []
    test_error = []

    # Iterate over each batch size
    for batch_size in batch_sizes:
        print("trying batch_size: ", batch_size)

        # Subset the training data
        X_tr_subset = X_tr[:best_slice, :]
        y_tr_subset = y_tr[:best_slice]

        # Create a MLPClassifier with the specified parameters
        learner = MLPClassifier(
            random_state=1234,
            activation=best_activation,
            batch_size=batch_size,
            learning_rate=best_learning_rate,
            solver=best_solver,
            hidden_layer_sizes=best_hidden_layer_sizes,
        )

        # Fit the model on the subset of training data
        learner.fit(X_tr_subset, y_tr_subset.ravel())

        # Calculate and store the training and testing errors
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))

    # Plot the training and testing errors if plot is True
    if plot:
        plt.plot(batch_sizes, train_error, label="train")
        plt.plot(batch_sizes, test_error, label="test")
        plt.legend()
        plt.xticks(batch_sizes)
        plt.show()

    # Find the best batch size with the minimum testing error
    best_batch_size = batch_sizes[np.argmin(test_error)]

    # Print and return the best batch size and the minimum testing error
    print(f"Best Batch Size: {best_batch_size}")
    return best_batch_size, min(test_error)


def alphaTest(
    X_tr,
    y_tr,
    X_te,
    y_te,
    best_slice,
    best_hidden_layer_sizes,
    best_learning_rate,
    plot=True,
):
    alphas = [0.0001, 0.001, 0.01, 0.1, 1]
    train_error = []
    test_error = []

    # Iterate over different alpha values
    for alpha in alphas:
        print("trying alpha: ", alpha)

        # Subset the training data
        X_tr_subset = X_tr[:best_slice, :]
        y_tr_subset = y_tr[:best_slice]

        # Create and train the MLPClassifier
        learner = MLPClassifier(
            random_state=1234,
            alpha=alpha,
            learning_rate_init=best_learning_rate,
            hidden_layer_sizes=best_hidden_layer_sizes,
        )
        learner.fit(X_tr_subset, y_tr_subset.ravel())

        # Calculate and store the train and test errors
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))

    # Plot the train and test errors
    if plot:
        plt.semilogx(alphas, train_error, label="train")
        plt.semilogx(alphas, test_error, label="test")
        plt.legend()
        plt.xticks(alphas)
        plt.ylabel("Error")
        plt.xlabel("Alpha")
        plt.title("Alpha vs. Error")
        plt.show()

    # Find the best alpha value with minimum test error
    best_alpha = alphas[np.argmin(test_error)]
    print(f"Best Alpha: {best_alpha}")

    return best_alpha, min(test_error)
