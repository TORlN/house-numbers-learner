import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# dont use this
def run(X_tr, y_tr, X_te, y_te):
    k_values = list(range(1,6))
    mse_train = []
    mse_eval = []

    for i,k in enumerate(k_values):

        ### YOUR CODE STARTS HERE ###
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_tr, y_tr)

        # Predicting on training data
        yt_pred = knn.predict(X_tr)
        
        # Predicting on testing data
        ye_pred = knn.predict(X_te)

        # Calculating training MSE
        training_mse = mse(y_tr, yt_pred)

        # Calculating testing MSE
        testing_mse = mse(y_te, ye_pred)

        mse_train.append(training_mse)
        mse_eval.append(testing_mse)

    # Finding the best K
    min_tmse = mse_eval[0]
    min_k = 1
    for i, m in enumerate(mse_eval):
        if m < min_tmse:
            min_tmse = m
            min_k = i + 1
    print(f"The best K value for the model is k={min_k}")
    plt.plot(k_values,mse_train,'b-', k_values,mse_eval,'g-', lw=3)
    plt.xlabel("Degree")
    plt.ylabel("MSE")