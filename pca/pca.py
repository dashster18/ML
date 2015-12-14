import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing as pp

def draw_line(ax, p1, p2):
    ax.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]))

################################################################
# Simple visualization example to better understand the
# PCA algorithm

viz = pd.read_csv('data/viz_pca.csv', header=None, names=['x1', 'x2'])

# Feature normalization
scaler = pp.StandardScaler().fit(viz.values)
X_norm = scaler.transform(viz.values)

# Comptue SVD
U, s, Vh = la.svd(X_norm)
V = Vh.T

# Visualize the principal eigenvectors
fig = plt.figure(111)
ax = fig.gca()

ax.scatter(X_norm[:, 0], X_norm[:, 1])

# Plots a line between (0,0) and the eigenvector
center = np.array([0, 0])
draw_line(ax, center, V[:, 0])
draw_line(ax, center, V[:, 1])


################################################################
# Reduce data to 1D and then plot
k = 1
basis = V[:, :k]
X_approx = X_norm @ basis
X_rec = X_approx @ basis.T

ax.scatter(X_rec[:, 0], X_rec[:, 1], c='r')

plt.show()
