from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def run(X_tr, y_tr, X_te, y_te):
    # reshape the data
    X_tr = X_tr.reshape(X_tr.shape[0], -1)
    X_te = X_te.reshape(X_te.shape[0], -1)

    # Scale the data
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, y_tr, X_te, y_te