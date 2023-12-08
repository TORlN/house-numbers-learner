from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def scikitScale(X_tr, y_tr, X_te, y_te):
    # reshape the data
    X_tr = X_tr.reshape(X_tr.shape[0], -1)
    X_te = X_te.reshape(X_te.shape[0], -1)

    # Scale the data
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, y_tr, X_te, y_te
def cnnScale(X_tr, y_tr, X_te, y_te):
    print("Scaling data...")
    X_tr_reshaped = X_tr.reshape(X_tr.shape[3], -1)
    X_te_reshaped = X_te.reshape(X_te.shape[3], -1)

    scaler = MinMaxScaler()

    # Fitting on training, transform on both training and testing
    X_tr_scaled = scaler.fit_transform(X_tr_reshaped)
    X_te_scaled = scaler.transform(X_te_reshaped)

    X_tr = X_tr_scaled.reshape(X_tr.shape)
    X_te = X_te_scaled.reshape(X_te.shape)
    return X_tr, y_tr, X_te, y_te