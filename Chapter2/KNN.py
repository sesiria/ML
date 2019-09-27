from sklearn import datasets
from collections import Counter  # 为了做投票
from sklearn.model_selection import train_test_split
import numpy as np

# 导入iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)


def euc_dis(instance1, instance2):
	"""
	计算两个样本instance1和instance2之间的欧式距离
	instance1: 第一个样本， array型
	instance2: 第二个样本， array型
	"""
	if len(instance1.shape) == 1:
		instance1 = instance1.reshape(1, -1)
	diff = instance1 - instance2
	dist = np.sqrt(np.sum(diff ** 2, axis = 1, keepdims = True))
	return dist
    
    
def knn_classify(X, y, testInstance, k):
	"""
	给定一个测试数据testInstance, 通过KNN算法来预测它的标签。 
	X: 训练数据的特征
	y: 训练数据的标签
	testInstance: 测试数据，这里假定一个测试数据 array型
	k: 选择多少个neighbors? 
	"""
	# TODO  返回testInstance的预测标签 = {0,1,2}
	num_classes = len(np.unique(y))
	distances = euc_dis(X, testInstance)
	classes = y[np.argsort(distances, axis = 0)][:k] # find the labels of the k nearest neighbors
	ypred = np.zeros(num_classes)

	for c in np.unique(classes):
		ypred[c] = len(classes[classes == c])

	return np.argmax(ypred)



# test euc_dis
def test_euc_dis():
	X = np.array([1, 2]).reshape(1, -1)
	Y = np.array([3, 5]).reshape(1, -1)
	L = euc_dis(X, Y)
	print(L)


# 预测结果 
# test_euc_dis()
predictions = [knn_classify(X_train, y_train, data, 3) for data in X_test]
correct = np.count_nonzero((predictions==y_test)==True)
print ("Accuracy is: %.3f" %(correct/len(X_test)))