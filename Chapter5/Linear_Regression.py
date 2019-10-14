# 创建数据集，把数据写入到numpy数组
import numpy as np  # 引用numpy库，主要用来做科学计算
import matplotlib.pyplot as plt   # 引用matplotlib库，主要用来画图
data = np.array([[152,51],[156,53],[160,54],[164,55],
                 [168,57],[172,60],[176,62],[180,65],
                 [184,69],[188,72]])

# 打印大小
x, y = data[:,0], data[:,1]
print (x.shape, y.shape)

# 1. 手动实现一个线性回归算法，具体推导细节参考4.1课程视频
# TODO: 实现w和b参数， 这里w是斜率， b是偏移量
xy_mean = np.mean(x * y)
x_mean = np.mean(x)
y_mean = np.mean(y)
x_squared_mean = np.mean(x * x)
w = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
b = y_mean - w * x_mean

print ("通过手动实现的线性回归模型参数: %.5f %.5f"%(w,b))

# 2. 使用sklearn来实现线性回归模型, 可以用来比较一下跟手动实现的结果
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x.reshape(-1,1),y)
print ("基于sklearn的线性回归模型参数：%.5f %.5f"%(model.coef_, model.intercept_))