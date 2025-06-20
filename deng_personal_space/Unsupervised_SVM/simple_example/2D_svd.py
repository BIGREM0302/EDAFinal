import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Generate a clearly separable 2D dataset
X, y = make_blobs(
    n_samples=200, centers=[(-2, -2), (2, 2)], cluster_std=0.8, random_state=42
)

# Train a linear SVM classifier
clf = SVC(kernel="linear")
clf.fit(X, y)

# Create a mesh grid to plot the decision boundary
xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot the decision boundary and data points
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Linear SVM Classification on 2D Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
