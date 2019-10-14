# generate the random data for two classify problem. the size =5000
import numpy as np

np.random.seed(12)
num_observations = 100 # generate 100 of positive and negative samples

# using the gaussian distribution to generate the sample, build the covariance matrix
# we generate the 20-dimension of the training data, so the covariance matrix is 20 x 20
rand_m = np.random.rand(20, 20)
# make sure that the covariance matrix is semi-defined positive 
cov = np.matmul(rand_m.T, rand_m)

# generate the training samples by gaussian distribution
x1 = np.random.multivariate_normal(np.random.rand(20), cov, num_observations)
x2 = np.random.multivariate_normal(np.random.rand(20) + 5, cov, num_observations)

X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

from sklearn.linear_model import LogisticRegression
# using the L1 regulization, C to control the regulization, when C is large, the regulization is small.
clf = LogisticRegression(fit_intercept = True, C = 0.1, penalty = 'l1', solver = 'liblinear')
clf.fit(X, y)

print("(L1) Logistic Regression with parameters:\n", clf.coef_)

# using the L2 regulization, C to control the regulization, when C is large, the regulization is small.
clf = LogisticRegression(fit_intercept = True, C = 0.1, penalty = 'l2', solver = 'liblinear')
clf.fit(X, y)

print("(L2) Logistic Regression with parameters:\n", clf.coef_)

