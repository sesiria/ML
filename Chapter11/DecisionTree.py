# author sesiria 2019
# a simple Decision Tree classifier implementation
import numpy as np
import sys
from collections import Counter

# **************************helper function.************************************
# function for calculate the entropy of the data.
# input shape should be (N,)
def calcEntropy(data):
    nSize = len(data)
    classes = np.unique(data)
    p = np.zeros(len(classes))
    for i in range(len(classes)):
        p[i] = len(data[data == classes[i]]) / nSize
    return -np.sum(p * np.log(p))

# func for compare two float numbers
def approximateEqual(X, Y):
    return np.abs(X - Y) < 10e-4

# choose the most frequency label
def chooseMaxFrequency(X):
    count = Counter(X)
    result = sorted(count.items(), key = lambda x : x[1])
    return result[-1][0]

# **********************definition of the Decision Tree***************************
# Node for the decision tree
class Node:
    # feature_index = -1 for leaf nodes
    def __init__(self, feature_index, label = None):
        # check for valid leaf nodes
        if feature_index == -1:
            assert label != None
        self.feature_index = feature_index
        self.label = label  # it is only valid for leaf node
        self.children = {}  # hashtable to store the nodes

# simple decision tree classifier, only support for discrete input.
class DecisionTreeClassifier:
    # constructor
    def __init__(self, maxDepth, maxLeaves):
        self.root = None
        self.maxDepth = maxDepth
        self.maxLeaves = maxLeaves

    # train the decision tree model
    def fit(self, data, label):
        self.root = self.split(data, label, 0)

    # predict the classification via the decision tree model
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
            return node.label
        child = node.children[data[node.feature_index]]
        assert child != None
        return self.search(child, data[[x for x in range(data.shape[0]) if x != node.feature_index]])

    # split nodes
    def split(self, data, label, depth):
        # we stop the split for these condition
        # 1) we meet the max depth
        # 2) we meet the max size of the nodes
        # 3) we have only one features with the same value
        # 4) we have the same labels
        if (depth == self.maxDepth or
           len(data) == self.maxLeaves or 
           (data.shape[1] == 1 and  len(np.unique(data[:, 0]) == 1)) or
           len(np.unique(label)) == 1
           ):
           targetLabel = chooseMaxFrequency(label)
           # for leaf node we don't need to pass the feature index
           node = Node(-1, targetLabel)
           return node

        # we need to build the node.
        bestFeature = self.featureSelection(data, label)
        node = Node(bestFeature)
        partitions = self.splitByfeature(data, bestFeature)
        # create the children for the current node
        for key in partitions:
            index = partitions[key]
            child = self.split(data[index, :][:, [x for x in range(data.shape[1]) if x != bestFeature]], 
                               label[index], 
                               depth + 1)
            node.children[key] = child
        return node
        
 
    # split the data set by the target feature_index
    # return the splitted index
    def splitByfeature(self, data, feature_idx):
        nClass = np.unique(data[:, feature_idx])
        result = {}
        for i in range(len(nClass)):
            index = data[:, feature_idx] == nClass[i]
            result[i] = index
        return result

    # iterate for each features and choose the best one for current split
    def featureSelection(self, data, label):
        # choose the best features
        nSize, n_features = data.shape
        # iterate for each features
        # originEntropy = calcEntropy(label)
        bestFeatures = 0
        minEntropy = float(sys.maxsize)
        for i in range(n_features):
            # split by the current features
            subIndexes = self.splitByfeature(data, i)
            # calculate the current entropys
            currentEntropy = 0
            for key in subIndexes:
                idx = subIndexes[key]
                currentEntropy += len(label[idx]) * calcEntropy(label[idx]) / nSize
            if currentEntropy < minEntropy:
                minEntropy = currentEntropy
                bestFeatures = i
        return bestFeatures

# **************************unit test function.************************************
def test_chooseMaxFrequency():
    A=['a','b','b','c','d','b','a']
    assert chooseMaxFrequency(A) == 'b'
    print("testing...chooseMaxFrequency()")
    print("pass!")

def test_calcEntropy():
    X = np.zeros(5)
    Y = np.ones(5)
    Z = np.array([0, 0, 0, 1, 1, 1])
    assert approximateEqual(calcEntropy(X), 0)
    assert approximateEqual(calcEntropy(Y), 0)
    assert approximateEqual(calcEntropy(Z), 0.69314718)
    print("testing...calcEntropy()")
    print("pass!")

def test_decisionTree():
    clf = DecisionTreeClassifier(100, 100)
    X = np.array([[0, 0, 0], 
                  [1, 0, 0], 
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [1, 0, 0]
                ])
    Y = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    # test splitByfeatures
    result = clf.splitByfeature(X, 0)
    # test featureSelection
    idx = clf.featureSelection(X, Y)
    clf.fit(X, Y)

    X_test = np.array([[1, 0, 0], 
                       [0, 1, 0],
                       [0, 0, 0]])
    predictions = clf.predict(X_test)

def sanity_check():
    # test_chooseMaxFrequency()
    # test_calcEntropy()
    test_decisionTree()

if __name__ == '__main__':
    sanity_check()
