# author sesiria 2019
# a simple XGBoost framework implementation.
# we didn't implementation the pre-sort, block prefetch, approximation approach, out-of-core optimization
# which is the system design by the original paper.
from RegressionTree import regressionTreeRegressor
import numpy as np

# simple XGBoostRegressor Implementation
# we use the mean square error(MSE) for the lost function.
# then gi = y2(yi_hat - yi), hi = 2
class MyXGBoostRegressor:
    def __init__(self, maxDepth, maxLeaves, max_iteration = 50,  C = 0, eta = 1.0):
        self.maxDepth = maxDepth
        self.maxLeaves = maxLeaves
        self.maxTreeSize = max_iteration
        self.C = C
        self.eta = eta
        self.trees = []

    # tain method for XGBoost
    # gi = y2(yi_hat - yi), hi = 2
    def fit(self, data, pred):
        g = -2 * pred
        h = np.ones(len(pred)) * 2
        residual = pred
        for i in  range(self.maxTreeSize):
            tree = regressionTreeRegressor(self.maxDepth, self.maxLeaves, self.C, self.eta, 'xgboost')
            tree.train(data, residual, g, h)
            predictions = tree.predict(data)
            g = 2 * (predictions - residual)
            # update the residual for the current tree.
            residual = residual - predictions
            # we keep the h as before
            self.trees.append(tree)

    # predict method for XGBoost
    def predict(self, test):
        if len(test.shape) == 1:
            return self.calculeValue(test)

        predictions = np.zeros(test.shape[0])
        for n in range(len(predictions)):
            predictions[n] = self.calculeValue(test[n, :])
        return predictions

    # only support for one test instance.
    def calculeValue(self, test):
        result = 0
        for tree in self.trees:
            result += tree.predict(test)
        return result

# **************************unit test function.************************************
def test_xgboost():
    X = np.array([
        [0.23, 61.5, 55.0, 3.95, 3.98, 2.43],
        [0.21, 59.8, 61.0, 3.89, 3.84, 2.31],
        [0.23, 56.9, 65.0, 4.05, 4.07, 2.31],
        [0.29, 62.4, 58.0, 4.20, 4.23, 2.63],
        [0.31, 63.3, 58.0, 4.34, 4.35, 2.75],
        [0.24, 61.9, 57.0, 3.94, 3.96, 2.48],
        [0.24, 62.3, 57.0, 3.95, 3.98, 2.47],
        [0.26, 61.9, 55.0, 4.07, 4.11, 2.53],
        [0.22, 65.1, 61.0, 3.87, 3.78, 2.49],
        [0.23, 59.4, 61.0, 4.00, 4.05, 2.39],
        [0.30, 64.0, 55.0, 4.25, 4.28, 2.73]
    ])
    Y = np.array([326, 326, 327, 334, 335, 336, 336, 337, 337, 338, 339])
    model = MyXGBoostRegressor(5, 3, max_iteration = 100, C=0.1, eta = 0.1)
    model.fit(X, Y)
    test = np.array([0.29, 62.1, 61.0, 3.87, 4.78, 3.49])
    result = model.predict(test)
    temp = 0

def sanity_check():
    test_xgboost()

if __name__ == '__main__':
    sanity_check()