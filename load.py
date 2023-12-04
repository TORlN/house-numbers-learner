import os
import scipy.io
import sys
def load():
    # Adjust file paths as needed for your local file system
    file_path = './train_32x32.mat'

    # Check if file is in your specified local directory
    if not os.path.exists(file_path):
        print(file_path," not found, exiting")
        sys.exit()
    else:
        print(file_path, "found")

    # Load the training data
    train_mat = scipy.io.loadmat(file_path)

    # categorize the data
    X_tr = train_mat['X']
    y_tr = train_mat['y']
    y_tr = y_tr.astype(int)

    file_path = "./test_32x32.mat"
    if not os.path.exists(file_path):
        print(file_path," not found, exiting")
        sys.exit()
    else:
        print(file_path, "found")

    # Load the test data
    test_mat = scipy.io.loadmat(file_path)

    X_te = test_mat['X']
    y_te = test_mat['y']
    y_te = y_te.astype(int)
    return X_tr, y_tr, X_te, y_te