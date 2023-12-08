import cv2
import numpy as np
def run(X_tr, y_tr, X_te, y_te):
    X_tr, X_te = greyscale(X_tr, X_te)
    return X_tr, y_tr, X_te, y_te

def greyscale(X_tr, X_te):
    X_tr_greyscale = np.zeros((X_tr.shape[3], 32, 32, 1))
    X_te_greyscale = np.zeros((X_te.shape[3], 32, 32, 1))
    for i in range(X_tr.shape[3]):
        grey_img = cv2.cvtColor(X_tr[:,:,:,i],cv2.COLOR_BGR2GRAY)
        X_tr_greyscale[i] = np.expand_dims(grey_img, axis=-1)
    for i in range(X_te.shape[3]):
        grey_img = cv2.cvtColor(X_te[:,:,:,i],cv2.COLOR_BGR2GRAY)
        X_te_greyscale[i] = np.expand_dims(grey_img, axis=-1)
    return X_tr_greyscale, X_te_greyscale