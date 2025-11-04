# 采用均方误差评估机制
# 导包
from sipbuild.generator.specification import Transfer
# from sklearn.datasets import load_boston                # 数据
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error          # 均方误差评估

from sklearn.linear_model import Ridge, RidgeCV
import pandas as pd
import numpy as np


def Linear_model1():
    # todo 1. 数据获取
    data_boston = pd.read_csv('data/波士顿房价xy.csv')
    data = data_boston.iloc[:, :-1]
    target = data_boston.iloc[:, -1]
    # print(data)  # 查看数据
    # print(target)
    # todo 2. 数据集划分
    x_train,x_test,y_train,y_test = train_test_split(data,target,random_state=22)
    # todo 3. 特征处理
    transfer = StandardScaler() # 标准化
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # todo 4. 模型训练,线性回归(正规方程)
    model = LinearRegression()
    model.fit(x_train,y_train) # 训练
    # todo 5. 模型评估,获取相应系数
    y_predict = model.predict(x_test)
    print('预测结果为: ', y_predict)
    print('模型的权重系数为: ',model.coef_)
    print('模型的偏置为',model.intercept_)
    # 均方误差评估
    print('均方误差为: ',mean_squared_error(y_test,y_predict))


def Linear_model2():
    # todo 1. 数据获取
    data_boston = pd.read_csv('data/波士顿房价xy.csv')
    data = data_boston.iloc[:, :-1]  # :表示所有行。:-1：表示从第一列到 倒数第2列（不包含最后一列）。
    target = data_boston.iloc[:, -1] # :表示所有行。-1：表示最后一列。
    # print(data)
    # print(target)
    # todo 2. 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22)
    # todo 3. 特征处理
    transfer = StandardScaler()  # 标准化
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # todo 4. 模型训练,线性回归(正规方程)
    model = SGDRegressor()
    model.fit(x_train, y_train)  # 训练
    # todo 5. 模型评估,获取相应系数
    y_predict = model.predict(x_test)
    print('预测结果为: ', y_predict)
    print('模型的权重系数为: ', model.coef_)
    print('模型的偏置为', model.intercept_)
    # 均方误差评估
    print('均方误差为: ', mean_squared_error(y_test, y_predict)) #


if __name__ == '__main__':
    print('---------------正规方程---------------')
    Linear_model1()
    print('---------------梯度下降---------------')
    Linear_model2()