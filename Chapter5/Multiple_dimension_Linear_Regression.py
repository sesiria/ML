import numpy as np
from sklearn.linear_model import LinearRegression

# 生成样本数据， 特征维度为2
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

# 先使用sklearn自带的库来解决
model = LinearRegression().fit(X, y)

# 打印参数以及偏移量（bias）
print ("基于sklearn的线性回归模型参数为 coef: ", model.coef_, " intercept: %.5f" %model.intercept_)

# TODO: 动手实现多元线性回归的参数估计, 把最后的结果放在res变量里。 res[0]存储的是偏移量，res[1:]存储的是模型参数
N = X.shape[0] # the number of the training examples
X_merge = np.concatenate((np.ones(N).reshape(-1, 1), X), axis = 1)
res = np.linalg.inv(X_merge.T @ X_merge) @ X_merge.T @ y

# 打印参数以偏移量（bias）
print ("通过手动实现的线性回归模型参数为 coef: ", res[1:], " intercept: %.5f"%res[0])