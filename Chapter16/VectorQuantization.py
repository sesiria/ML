#import packages
from pylab import imread, imshow, figure, show, subplot
import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy

# read the image data
img = imread('Tulips.jpg')
imshow(img)
show()
# convert three dimension tensor into two dimension matrix
pixel = img.reshape(img.shape[0] * img.shape[1], 3)
pixel_new = deepcopy(pixel)

print (img.shape)

# construct K-means model
model = KMeans(n_clusters = 16)
labels = model.fit_predict(pixel)
palette = model.cluster_centers_

for i in range(len(pixel)):
    pixel_new[i,:] = palette[labels[i]]

# reshow the compressed image
imshow(pixel_new.reshape(img.shape[0], img.shape[1], 3))
show()