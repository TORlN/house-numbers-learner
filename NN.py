from sklearn.neural_network import MLPClassifier
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
import numpy as np

def run(X_tr, y_tr, X_te, y_te):
    # neural networks via sklearn
    # best_slice = sliceTest(X_tr, y_tr, X_te, y_te)
    best_slice = 15000
    best_activation = 'relu'

def sliceTest(X_tr, y_tr, X_te, y_te):
    slices = [5000, 10000, 15000, 20000, 26032]
    train_error = []
    test_error = []
    for sl in slices:
        print("trying slice: ", sl)
        X_tr_subset = X_tr[:sl, :]
        y_tr_subset = y_tr[:sl, :]
        learner = MLPClassifier(random_state=1234)
        learner.fit(X_tr_subset, y_tr_subset.ravel())
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))
    plt.semilogx(slices, train_error, label='train')
    plt.semilogx(slices, test_error, label='test')
    plt.legend()
    plt.show()
    plt.xticks(slices)
    best_slice = slices[np.argmin(test_error)]
    print(f"Best Slice: {best_slice}")
    return best_slice

def activationTest(X_tr, y_tr, X_te, y_te, best_slice):
    activations = ['identity', 'logistic', 'tanh', 'relu']
    train_error = []
    test_error = []
    for activation in activations:
        print("trying activation: ", activation)
        learner = MLPClassifier(random_state=1234, activation=activation)
        learner.fit(X_tr, y_tr.ravel())
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))
    plt.semilogx(activations, train_error, label='train')
    plt.semilogx(activations, test_error, label='test')
    plt.legend()
    plt.show()
    best_activation = activations[np.argmin(test_error)]
    print(f"Best Activation: {best_activation}")
    return best_activation