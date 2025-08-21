"""
机器学习步骤:
0. 导包
1. 数据获取
2. 数据预处理
3. 特征工程
4. 模型预测
5. 模型评估
"""
# todo 1. 导包
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier



def 分类_KNN():
    # todo 2. 数据获取
    x1 = [[0, 2, 3], [1, 3, 4], [3, 5, 6], [4, 7, 8], [2, 3, 4]]
    y1 = [0, 0, 1, 1, 0]
    x_text = [[4,4,5]]
# 由于是简单的数据不需要进行预处理
# todo 3. 模型预测
# 创建模型对象
    Classfier_modle = KNeighborsClassifier()
    Classfier_modle.fit(x1, y1)
    print('分类模型预测结果: ',Classfier_modle.predict(x_text))

def 回归_KNN():
    # todo 2. 数据获取
    x2 = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    y2 = [0.1, 0.2, 0.3, 0.4]
    x_text = [[2,1,1]]
    # 由于是简单的数据不需要进行预处理
    # todo 3. 模型预测
    Regressor_modle = KNeighborsRegressor(n_neighbors=3)
    Regressor_modle.fit(x2, y2) # 训练模型
    print('回归模型预测结果: ',Regressor_modle.predict(x_text))


if __name__ == '__main__':
    分类_KNN()
    回归_KNN()