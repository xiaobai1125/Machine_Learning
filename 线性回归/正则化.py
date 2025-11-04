'''
正则化: 解决模型过拟合的方法,即在模型训练中,数据中有些特征影响模型复杂度,
       或者某个特征的异常值较多,所以要尽量减少这个特征的影响(甚至删除),
       这就是正则化.
    L1正则化: 会使权重趋向于0,甚至等于0,使得某些特征失效,达到特征筛选目的.
	API: sklearn.linear_model.Lasso

    L2正则化: 会使权重趋向于0,但一般不会等于0,
	API: sklearn.linear_model.Ridge
'''
# 导包
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import numpy as np


def DM_L1():
    # 创建数据集
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 数据预处理
    X = x.reshape(-1, 1)  # 矩阵转置
    X = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])  # 增加数据次项
    # 创建模型
    model = Lasso(alpha=0.1)
    # 训练模型
    model.fit(X, y)
    # 预测
    y_predict = model.predict(X)
    # 可视化
    plt.scatter(x, y)  # 散点图
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 参1:x轴数据,参2: y轴数据,参3: 颜色
    plt.show()


def DM_L2():
    # 创建数据集
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 数据预处理
    X = x.reshape(-1, 1)  # 矩阵转置
    X = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])  # 增加数据次项
    # 创建模型
    model = Ridge(alpha=0.1)
    # 训练模型
    model.fit(X, y)
    # 预测
    y_predict = model.predict(X)
    # 可视化
    plt.scatter(x, y)  # 散点图
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 参1:x轴数据,参2: y轴数据,参3: 颜色
    plt.show()


if __name__ == '__main__':
    DM_L1()
    DM_L2()
