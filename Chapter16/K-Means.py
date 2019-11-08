# import package
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set paramter k for K-means
k = 3

# randomize the center point. and save the result into C
X = np.random.random((200, 2)) * 10
C_x = np.random.choice(range(0, int(np.max(X[:, 0]))), size = k, replace = False)
C_y = np.random.choice(range(0, int(np.max(X[:, 1]))), size = k, replace = False)
C = np.array(list(zip(C_x, C_y)), dtype = np.float32)

print("The init center point is :")
print(C)

# plot the center point
plt.scatter(X[:, 0], X[:, 1], c = '#050505', s = 7)
plt.scatter(C[:, 0], C[:, 1], marker = '*', s = 300, c = 'g')
plt.show()

# store the previous center point
C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))

# calculate the distance
def dist(a, b, ax = 1):
    return np.linalg.norm(a - b, axis = ax)

error = dist(C, C_old, None)
# iteration for K-mean clustering until converge(that is the error = 0)
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        category = np.argmin(distances)
        clusters[i] = category
    
    # We save the old center points
    C_old = deepcopy(C)
    # and calculate the new center points
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis = 0)
    error = dist(C, C_old, None)

# plot the clusters
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s = 7, c = colors[i])
ax.scatter(C[:, 0], C[:, 1], marker = '*', s = 200, c = '#050505')
plt.show()
