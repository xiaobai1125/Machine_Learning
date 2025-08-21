# 导 包
from sklearn.model_selection import GridSearchCV  # 网格搜索交叉验证
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.neighbors import KNeighborsClassifier  # KNN分类
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.metrics import accuracy_score  # 模型评估
from sklearn.datasets import load_iris  # 鸢尾花数据


def GridCV():
    # todo 1. 获取数据集
    iris_data = load_iris()
    # todo 2. 数据处理
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=22)
    # 数据标准化,创建标准化模型
    process = StandardScaler()
    # 对训练数据进行标准化
    x_train = process.fit_transform(x_train)  # 对训练集数据先进行计算 在进行转换
    x_test = process.transform(x_test)  # 对测试集进行转换
    # todo 3. 创建模型
    # 创建模型
    model = KNeighborsClassifier()
    # 使用交叉验证网格搜索,先指定范围
    param_grid = {'n_neighbors': range(1, 10)}
    # 具体 网格搜索+ 交叉验证
    # 参1为模型对象,参2为参数范围,参数3为交叉验证次数
    model_grid = GridSearchCV(model, param_grid, cv=5)
    # 训练模型
    model_grid.fit(x_train, y_train)
    # 预测结果
    print('最高得分', model_grid.best_score_)
    print('最佳参数模型', model_grid.best_estimator_)
    print('交叉验证的结果', model_grid.cv_results_)  # 输出完整的交叉验证结果详情
    print('交叉验证的结果', model_grid.best_params_) # 输出最佳参数
    # 得到最优参数模型后,预测
    model_new = model_grid.best_estimator_
    model_new.fit(x_train, y_train)

    print('准确率: ', model_new.score(x_test, y_test))
    print('准确率: ', accuracy_score(y_test, model_new.predict(x_test)))
    # 创建新数据,对模型进行预测
    x_train_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
    print('预测结果: ', model_new.predict(x_train_new))


if __name__ == '__main__':
    GridCV()
