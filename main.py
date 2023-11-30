import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.linear_model import LinearRegression    # Basic Linear Regression
from sklearn.linear_model import Ridge               # Linear Regression with L2 regularization

from sklearn.model_selection import KFold            # Cross-validation tools

from sklearn.preprocessing import PolynomialFeatures # Feature transformations
from sklearn.preprocessing import StandardScaler

import requests                                      # reading data
from io import StringIO

seed = 1234

import plotly.io as pio
pio.renderers.default = 'notebook'  # You might want to change this for local environment

import pandas as pd
import os

# Adjust file paths as needed for your local file system
file_path = './train_32x32.mat'

print(os.getcwd())

# Check if file is in your specified local directory
print("train_32x32.mat" in os.listdir('./'))


