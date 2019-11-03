# author sesiria 2019
# a simple Regression Tree Regressor implementation
import numpy as np
import sys
from collections import Counter

# **************************helper function.************************************
def approximateEqual(X, Y):
    return np.abs(X - Y) < 10e-4

# **********************definition of the Regression Tree***************************
# Node for the Regression tree
class Node:
    # feature_index = -1 for leaf nodes
    def __init__(self, feature_index, splitPoint, predictValue=None):
        # check for valid leaf nodes
        if feature_index == -1:
            assert predictValue != None
        self.feature_index = feature_index
        self.predictValue = predictValue  # it is only valid for leaf node
        self.split_point = splitPoint

# simple regression tree regressor,support for continus input data.
# split method by stand  derivation or xgboost method(first-order and second-order partial derivative of the loss function)
class regressionTreeRegressor:
    # constructor
    # by default we use the standard derivation for split method 'std'
    # if we split node by the XGBoost framework, we use the g, h parameters. 'xgboost'
    def __init__(self, maxDepth, maxLeaves, C=0, eta=1.0, splitMethod='std'):
        self.root = None
        self.maxDepth = maxDepth
        self.maxLeaves = maxLeaves
        # by default we use the standard derivation for split method
        # if we split node by the XGBoost framework, we use the g, h parameters.
        self.splitMethod = splitMethod
        self.C = C          # penality parameter for regulization
        self.eta = eta      # eta is used by the xgboost framework.
        self.left = None    # for left children
        self.right = None   # for the right children

    # train the regression tree model by the normal.
    def fit(self, data, pred):
        self.root = self.split(data, pred, pred, pred, 0)
    
    # train the regression tree model by xgboost method
    def train(self, data, pred, g, h):
        self.root = self.split(data, pred, g, h, 0)

    # predict the regressor via regression tree model
    def predict(self, test):
        if len(test.shape) == 1:
            return self.search(self.root, test)

        predictions = np.zeros(test.shape[0])
        for n in range(len(predictions)):
            predictions[n] = self.search(self.root, test[n, :])
        return predictions

    # search for the node
    def search(self, node, data):
        assert node != None
        # this is a leaf node
        if node.feature_index == -1:
            return node.predictValue

        if node.left != None and node.right != None:
            if (data[node.feature_index] <= node.split_point):
                return self.search(node.left, data)
            else:
                return self.search(node.right, data)
        else:
            if (node.left != None):
                return self.search(node.left, data)
            else:
                return self.search(node.right, data)


    # split nodes
    # the g, h parameter is stand for the first-order derivative and second order derivate used by XGBoost
    def split(self, data, pred, g, h, depth):
        # we stop the split for these condition
        # 1) we meet the max depth
        # 2) we meet the max size of the nodes
        # 3) we have only one features with the same value
        # 4) we have the same labels
        # print('split depth', depth)
        # print('data size', len(data))
        if (depth == self.maxDepth or
            len(data) == self.maxLeaves or
            (data.shape[1] == 1 and len(np.unique(data[:, 0])) == 1) or
            len(np.unique(pred)) == 1
            ):
            targetPred = self.eta * np.mean(pred)
            # for leaf node we don't need to pass the feature index and split point.
            node = Node(feature_index = -1, splitPoint = None, predictValue = targetPred)
            return node

        assert len(data) != 0

        # we need to build the node.
        bestFeature = None
        splitPoint = None
        if self.splitMethod == 'xgboost':
            bestFeature, splitPoint = self.featureSelectionByXGBoost(
                data, g, h)
        elif self.splitMethod == 'std':
            bestFeature, splitPoint = self.featureSelectionByStd(
                data, pred)
        leftIndex = data[:, bestFeature] <= splitPoint
        rightIndex = data[:, bestFeature] > splitPoint

        # we can't split the current node due to one of partitions has the size zero.
        nLeft = len(data[leftIndex])
        nRight = len(data[leftIndex])
        if nLeft == data.shape[0] or nRight == data.shape[0]:
            targetPred = self.eta * np.mean(pred)
            node = Node(feature_index = -1, splitPoint = None, predictValue = targetPred)
            return node

        node = Node(feature_index = bestFeature, splitPoint = splitPoint)
        # create the children for the current node
        node.left = self.split(
            data[leftIndex, :], pred[leftIndex], g[leftIndex], h[leftIndex], depth + 1)
        node.right = self.split(
            data[rightIndex, :], pred[rightIndex], g[rightIndex], h[rightIndex], depth + 1)

        return node

    # iterate for each features ,split point and choose the best ones for current split
    # return the feature index and the split point.
    # we ignore the pre-sorted optimization which is different from the original xgboost.
    def featureSelectionByXGBoost(self, data, g, h):
        assert len(data.shape) == 2
        # choose the best features
        n_features = data.shape[1]
        bestFeatures = 0
        bestSplitPoint = None
        bestScore = 0
        G = np.sum(g)
        H = np.sum(h)
        # iterate for each features
        for i in range(n_features):
            GL = 0
            HL = 0
            Index = np.argsort(data[:, i])
            # iterate for each split points from the current feature.
            for j in Index:
                GL += g[j]
                HL += h[j]
                GR = G - GL
                HR = H - HL
                # score = GL ** 2 / (HL + self.C) + GR ** 2 / \
                #    (HR + self.C) - G ** 2 / (H + self.C)
                # because the G ** 2 / (H + self.C) is constant we ignore the constant.
                score = GL ** 2 / (HL + self.C) + GR ** 2 / (HR + self.C)
                # choose the maximum score
                if bestScore < score:
                    bestScore = score
                    bestFeatures = i
                    bestSplitPoint = data[j, i]
        return bestFeatures, bestSplitPoint

    # by default we enumerate each split point,the features by the standard deviation.
    # we use the variation by implementation cause it is equal to the standard deviation.
    # and we use the raw-score form of the veriation: var = E(X^2) - E(X)^2
    def featureSelectionByStd(self, data, pred):
        assert len(data.shape) == 2
        # choose the best features
        n_size, n_features = data.shape
        bestFeatures = 0
        bestSplitPoint = None
        bestScore = float(sys.maxsize)

        E2 = np.sum(pred ** 2, dtype = np.float64)      # E(X^2)
        E = np.sum(pred, dtype = np.float64)            # E(X)^2
        nLeft = 0

        # iterate for each features
        for i in range(n_features):
            EL2 = 0
            EL = 0
            Index = np.argsort(data[:,i])
            nLeft = 0
            # iterate for each split points from the current feature.
            for j in Index:
                nLeft += 1
                EL2 += pred[j]**2
                EL += pred[j] 
                ER2 = E2 - EL2
                ER = E - EL
                # we choose the min variance split point we have an optimization form
                score = (EL2 - EL**2 / nLeft + ER2 - ER**2 / (n_size - nLeft)) / n_size
                # choose the minscore
                if bestScore > score:
                    bestScore = score
                    bestFeatures = i
                    bestSplitPoint = data[j, i]
        return bestFeatures, bestSplitPoint


# **************************unit test function.************************************
def test_regressionTree():
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
    model = regressionTreeRegressor(10, 3, C=0.1)
    model.fit(X, Y)
    test = np.array([0.25, 62.1, 61.0, 2.87, 4.78, 3.49])
    result = model.predict(test)
    temp = 0

def sanity_check():
    test_regressionTree()


if __name__ == '__main__':
    sanity_check()
