from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score


from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.linear_model import LinearRegression    # Basic Linear Regression
from sklearn.linear_model import Ridge               # Linear Regression with L2 regularization

from sklearn.model_selection import KFold            # Cross-validation tools
from sklearn.preprocessing import PolynomialFeatures # Feature transformations

from io import StringIO

import plotly.io as pio
pio.renderers.default = 'notebook'

import pandas as pd

import logreg
import scale
import load
import knn
import NN

X_tr, y_tr, X_te, y_te = load.load()
X_tr, y_tr, X_te, y_te = scale.run(X_tr, y_tr, X_te, y_te)
# logreg.run(X_tr, y_tr, X_te, y_te)
# knn.run(X_tr, y_tr, X_te, y_te)
# print("X_tr.shape: ", X_tr.shape)
# print("y_tr.shape: ", y_tr.shape)
# print("X_te.shape: ", X_te.shape)
# print("y_te.shape: ", y_te.shape)
NN.run(X_tr, y_tr, X_te, y_te) 



# throw out logreg because of very high error and limited parameters to test
# throw out knn because of how much time it takes to run
# throw out NN because it doesnt understand depth of image (multiple colors in image)