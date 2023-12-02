import requests 
import os
import scipy.io
def load():
    # Adjust file paths as needed for your local file system
    file_path = 'train_32x32.mat'

    print(os.getcwd())

    # Check if file is in your specified local directory
    print(file_path in os.listdir('./'))

    # Load the training data
    train_mat = scipy.io.loadmat(file_path)

    # categorize the data
    X_tr = train_mat['X']
    y_tr = train_mat['y']
    y_tr = y_tr.astype(int)

    file_path = 'test_32x32.mat'
    print(file_path in os.listdir('./'))\

    # Load the test data
    test_mat = scipy.io.loadmat(file_path)

    X_te = test_mat['X']
    y_te = test_mat['y']
    y_te = y_te.astype(int)
    return X_tr, y_tr, X_te, y_te