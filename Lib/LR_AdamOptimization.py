# logistic Regression by mini-batch Stochastic Gradient Descent via Adam Algorithm
# import library
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# generate the data set for classify problem the size is 5000
np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal(
    [0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal(
    [1, 4], [[1, .75], [.75, 1]], num_observations)

X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations),
               np.ones(num_observations)))

print(X.shape, y.shape)

# Data Visualization
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1],
            c=y, alpha=.4)
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


def logistic_regression_minibatch(X, y, num_steps, learning_rate, batchsize = 128):
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
        index = np.random.choice(range(0, N), batchsize, replace=False)

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
            print('epoch\t%d\t------\tlost: '%step, log_likelihood(X, y, w, b))

    return w, b

# stochastic gradient descent via Adam for logistic regression


def logistic_regression_adam(X, y, num_steps, learning_rate=0.001, batchsize = 128, beta1=0.9, beta2=0.999, epsilon=10e-8):
    """
    base on the gradient descent of Adam Algorithm
    X: training data, N * D
    y: training label,  N * 1
    num_steps: iteration times
    learning_rate: size of the step
    """
    N, D = X.shape
    w, b = np.zeros(X.shape[1]), 0
    beta1_exp = beta1
    beta2_exp = beta2
    # init the w, b.
    m_w = np.zeros(w.shape)
    m_b = 0
    v_w = np.zeros(w.shape)
    v_b = 0
    for step in range(num_steps):
        # get a random index for minibatch with batch size 100
        index = np.random.choice(range(0, N), batchsize, replace=False)

        # calculate the error between the predict and the actual
        error = sigmoid(X[index] @ w + b) - y[index]

        # calculate the gradient for w, b
        grad_w = np.matmul(X[index].T, error)
        grad_b = np.sum(error)

        # calculate first moment
        m_w = beta1 * m_w + (1 - beta1) * grad_w
        m_b = beta1 * m_b + (1 - beta1) * grad_b

        # calculate the second moment
        v_w = beta2 * v_w + (1 - beta2) * grad_w ** 2
        v_b = beta2 * v_b + (1 - beta2) * grad_b ** 2

        # calculate the bias-corrected first and second moment estimate
        beta1_exp *= beta1
        beta2_exp *= beta2

        bar_m_w = m_w / (1 - beta1_exp)
        bar_m_b = m_b / (1 - beta1_exp)

        bar_v_w = v_w / (1 - beta2_exp)
        bar_v_b = v_b / (1 - beta2_exp)

        # update w, b
        rate = learning_rate * np.sqrt(1 - beta2_exp) / (1 - beta1_exp)
        w = w - rate * bar_m_w / (np.sqrt(bar_v_w) + epsilon)
        b = b - rate * bar_m_b / (np.sqrt(bar_v_b) + epsilon)

        # calculate the likelihood
        if step % 1000 == 0:
            print('epoch\t%d\t------\tlost: '%step, log_likelihood(X, y, w, b))

    return w, b


w, b = logistic_regression_adam(
    X, y, num_steps=100000)
print("(my) logistic regression adam SGD with parameter w, b is :", w, b)

w, b = logistic_regression_minibatch(
    X, y, num_steps=100000, learning_rate = 0.001)
print("(my) logistic regression minibatch with parameter w, b is :", w, b)

# import logisticRegression model from sklearn

# Set C very large that is it will not add the regular term.
clf = LogisticRegression(fit_intercept=True, C=1e15, solver='lbfgs')
clf.fit(X, y)

print("(sklearn) logistic regression with parater w, b is :",
      clf.coef_, clf.intercept_)
