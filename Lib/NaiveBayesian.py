# author sesiria 2019
# a simple Naive Bayesian classifier implementation
import numpy as np

# **********************definition of the Naive Bayesian***************************
class NaiveBayesianClassifier:
    def __init__(self):
        pass
    
    # currently we only support the digital number for labels.
    def fit(self, data, label):
        classes = np.unique(label)
        nWords = data.shape[1]
        # matrix to store the probability for each word in each category.
        self.paramMatrix = np.zeros([nWords, len(classes)], dtype = np.float64)
        self.priorVector = np.zeros(len(classes), dtype = np.float64)
        self.labels = [] # class label

        for i in range(len(classes)):
            c = classes[i]
            nCurrentSize = len(label[label == c])
            # build category hashtable
            self.labels.append(c)
            # we calculate the priorVector
            self.priorVector[i] = nCurrentSize / len(label)
            # calculate the paramMatrix with smoothing
            count = np.sum(data[label == c, :], axis = 0) + 1
            count = count / (nCurrentSize + nWords)
            self.paramMatrix[:, i] = count            

    def predict(self, test):
        if (len(test.shape) == 1):
            return self.getCategory(test)

        predictions = np.zeros(test.shape[0])
        for i in range(test.shape[0]):
            predictions[i] = self.getCategory(test[i, :])
        return predictions

    def getCategory(self, test):
        assert test.shape[0] == self.paramMatrix.shape[0]
        p = np.zeros(len(self.labels))
        for idx in range(len(self.labels)):
            # we use the log trick to avoid the underflow
            p[idx] = np.sum(np.log(self.paramMatrix[:, idx]) * test)

        return self.labels[np.argmax(p)]
        
# **************************unit test function.************************************
def sanity_check():
    X = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 0, 0, 0, 0, 0]
                ])
    Y = np.array([1, 1, 1, 0, 0, 0])
    X_test = np.array([[1, 0, 0, 1, 2, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 1, 0]
                       ])
    clf = NaiveBayesianClassifier()
    clf.fit(X, Y)
    result = clf.predict(X_test)

if __name__ == '__main__':
    sanity_check()
