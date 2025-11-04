"""欠拟合过拟合概念:
        欠拟合：模型在训练集上表现不好，在测试集上也表现不好。模型过于简单
        过拟合：模型在训练集上表现好，在测试集上表现不好。模型过于复杂
        欠拟合在训练集和测试集上的误差都较大
        过拟合在训练集上误差较小，而测试集上误差较大
"""
# 欠拟合过拟合代码实现
# 导包
import numpy as np
import matplotlib.pyplot as plt # 可视化
from sklearn.linear_model import LinearRegression # 线性回归方程
from sklearn.metrics import mean_squared_error # 计算均方误差
from sklearn.model_selection import train_test_split # 数据集切分
def DM_欠拟合():
    # 创建数据集
    np.random.seed(666)
    x = np.random.uniform(-3,3,size=100)
    y = 0.5 * x**2 + x + 2 + np.random.normal(0,1,size=100)
    # 创建模型
    model = LinearRegression()
    # 训练模型
    X1 = x.reshape(-1,1) # 矩阵转置
    model.fit(X1,y)
    # 预测
    y_predict = model.predict(X1)
    # 计算均方误差
    print('均方误差为: ',mean_squared_error(y,y_predict))
    # 可视化
    plt.scatter(x,y) # 散点图
    plt.plot(x,y_predict,color='r') # 参1:x轴数据,参2: y轴数据,参3: 颜色
    plt.show()
def DM_过拟合():
    # 创建数据集
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 创建模型
    model = LinearRegression()
    # 训练模型
    X = x.reshape(-1, 1)  # 矩阵转置
    X2 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10]) # 增加数据次项
    model.fit(X2, y)
    # 预测
    y_predict = model.predict(X2)
    # 计算均方误差
    print('均方误差为: ', mean_squared_error(y, y_predict))
    # 可视化
    plt.scatter(x, y)  # 散点图
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 参1:x轴数据,参2: y轴数据,参3: 颜色
    plt.show()
def DM_正好拟合():
    # 创建数据集
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 创建模型
    model = LinearRegression()
    # 训练模型
    X = x.reshape(-1, 1)  # 矩阵转置
    X3 = np.hstack([X, X ** 2])  # 增加数据二次项
    model.fit(X3, y)
    # 预测
    y_predict = model.predict(X3)
    # 计算均方误差
    print('均方误差为: ', mean_squared_error(y, y_predict))
    # 可视化
    plt.scatter(x, y)  # 散点图
    # np.sort(x)为对x进行排序,y_predict[np.argsort(x)]为x的排序后的索引
    plt.plot(np.sort(x),y_predict[np.argsort(x)], color='r')  # 参1:x轴数据,参2: y轴数据,参3: 颜色
    plt.show()

if __name__ == '__main__':
    DM_欠拟合()
    DM_过拟合()
    DM_正好拟合()