from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
import numpy as np


def run(X_tr, y_tr, X_te, y_te):
    # Logistic Regression first
    best_c = cValue_Test(X_tr, y_tr, X_te, y_te)
    # best_c = 0.01
    # best_solver = solverTest(X_tr, y_tr, X_te, y_te, best_c)
    best_solver = "lbfgs"
    # best_iter = iterTest(X_tr, y_tr, X_te, y_te, best_c, best_solver)
    # print(best_iter)


def cValue_Test(X_tr, y_tr, X_te, y_te):
    Cvals = [0.000001, 0.00001, 0.0001, 0.001, .01, .1, 1, 10, 100]
    train_error = []
    test_error = []
    for C in Cvals:
        print("trying C value: ", C)
        learner = LogisticRegression(random_state=1234, C=C, max_iter=1000)
        learner.fit(X_tr, y_tr.ravel())
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))
    plt.semilogx(Cvals, train_error, label="train")
    plt.semilogx(Cvals, test_error, label="test")
    plt.xticks(Cvals)
    plt.ylabel("Error")
    plt.xlabel("C Value")
    plt.title("Logistic Regression C Value Test")
    plt.legend()
    plt.show()
    best_c = Cvals[np.argmin(test_error)]
    print(f"Best C Value: {best_c}")
    return best_c


def solverTest(X_tr, y_tr, X_te, y_te, best_c):
    solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    train_error = []
    test_error = []
    for solver in solvers:
        print("trying solver: ", solver)
        learner = LogisticRegression(random_state=1234, C=best_c, solver=solver)
        learner.fit(X_tr, y_tr.ravel())
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))
    plt.semilogx(solvers, train_error, label="train")
    plt.semilogx(solvers, test_error, label="test")
    plt.legend()
    plt.show()
    best_solver = solvers[np.argmin(test_error)]
    print(f"Best Solver: {best_solver}")
    return best_solver


def iterTest(X_tr, y_tr, X_te, y_te, best_c, best_solver):
    iters = [50, 100, 200, 300, 400, 500]
    train_error = []
    test_error = []
    for iter in iters:
        print("trying iter: ", iter)
        learner = LogisticRegression(
            random_state=1234, C=best_c, solver=best_solver, max_iter=iter
        )
        learner.fit(X_tr, y_tr.ravel())
        train_error.append(zero_one_loss(y_tr, learner.predict(X_tr)))
        test_error.append(zero_one_loss(y_te, learner.predict(X_te)))
    plt.semilogx(iters, train_error, label="train")
    plt.semilogx(iters, test_error, label="test")
    plt.legend()
    plt.show()
    best_iter = iters[np.argmin(test_error)]
    print(f"Best Iter: {best_iter}")
    return best_iter
