import logreg
import scale
import load
import knn
import NN
import matplotlib.pyplot as plt
import time

X_tr, y_tr, X_te, y_te = load.load()
plt.imshow(X_tr[0,:,:,0], cmap='gray')
# plt.imshow(X_tr[:, :, :, 0])
plt.show()
X_tr, y_tr, X_te, y_te = scale.run(X_tr, y_tr, X_te, y_te)

print("Shape of X_tr:", X_tr.shape)
print("Shape of X_te:", X_te.shape)
print("Shape of y_tr:", y_tr.shape)
print("Shape of y_te:", y_te.shape)
# logreg.run(X_tr, y_tr, X_te, y_te)
# knn.run(X_tr, y_tr, X_te, y_te)

# start = time.time()
NN.run(X_tr, y_tr, X_te, y_te)
# end = time.time()
# print(f"This took {end-start} seconds")

# throw out logreg because of very high error and limited parameters to test
# throw out knn because of how much time it takes to run
# throw out basic NN because it doesnt understand depth of image (multiple colors in image)
# ask revonix for the decision tree code
