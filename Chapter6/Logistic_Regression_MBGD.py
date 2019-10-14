# logistic Regression by mini-batch gradient descent
# import library
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# generate the data set for classify problem the size is 5000
np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations),
                np.ones(num_observations)))

print(X.shape, y.shape)

# Data Visualization
plt.figure(figsize = (12, 8))
plt.scatter(X[:, 0], X[:, 1],
            c = y, alpha = .4)
plt.show()

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# calculate log likelihood
def log_likelihood(X, y, w, b):
    """
    calculate the negative log likelihood, named cross-entropy loss
    the value less, and the performance is better 
    X: training data, N * D
    y: training label,  N * 1
    w: parameter D * 1
    b: bais, which is a scalar
    """

    # get the label related to positive, negative
    pos, neg = np.where(y == 1), np.where(y == 0)

    # calculate the loss for the positive samples.
    pos_sum = np.sum(np.log(sigmoid(X[pos] @ w + b)))

    # calculate the loss for the negative samples.
    neg_sum = np.sum(np.log(1 - sigmoid(X[neg] @ w + b)))

    # return the cross entropy loss
    return -(pos_sum + neg_sum)

# mini-batch gradient descent for logistic regression
def logistic_regression_minibatch(X, y, num_steps, learning_rate):
    """
    base on the gradient descent
    X: training data, N * D
    y: training label,  N * 1
    num_steps: iteration times
    learning_rate: size of the step
    """
    N, D = X.shape
    w, b = np.zeros(X.shape[1]), 0
    for step in range(num_steps):
        # get a random index for minibatch with batch size 100
        index = np.random.randint(0, N, 100)

        # calculate the error between the predict and the actual
        error = sigmoid(X[index] @ w + b) - y[index]

        # calculate the gradient for w, b
        grad_w = np.matmul(X[index].T, error)
        grad_b = np.sum(error)

        # update w, b
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        # calculate the likelihood
        if step % 1000 == 0:
            print(log_likelihood(X, y, w, b))
    
    return w, b

w, b = logistic_regression_minibatch(X, y, num_steps = 500000, learning_rate = 5e-4)
print("(my) logistic regression mini-batch with parameter w, b is :", w, b)

# import logisticRegression model from sklearn
from sklearn.linear_model import LogisticRegression

# Set C very large that is it will not add the regular term.
clf = LogisticRegression(fit_intercept = True, C = 1e15)
clf.fit(X, y)

print("(sklearn) logistic regression with parater w, b is :", clf.coef_, clf.intercept_)