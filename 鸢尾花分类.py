# 导包
from sklearn.datasets import load_iris  # 鸢尾花数据集
from sklearn.metrics import accuracy_score  # 模型评估,计算准确率
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.preprocessing import StandardScaler  # 标准化
import seaborn as sns  # 数据可视化
import matplotlib.pyplot as plt  # 数据可视化
import pandas as pd


# todo 1. 获取数据集
def dm_loadiris():
    # 获取数据集
    iris_data = load_iris()
    data = iris_data.data  # 数据的特征
    target = iris_data.target  # 数据的标签
    print(data, target)  # 显示数据


# todo 2. 数据展示,处理
def dm_showiris():
    # 获取数据集
    iris_data = load_iris()
    # 数据展示
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)  # 创建数据集 列名为特征名
    iris_df['label'] = iris_data.target  # 添加标签列
    print(iris_df)  # 显示数据集
    # 数据可视化
    # 参1：x轴，参2：y轴，参3：根据不同类别显示不同颜色，参4：数据集，参5：是否画出回归线
    sns.lmplot(x='sepal length (cm)', y='sepal width (cm)', hue='label', data=iris_df, fit_reg=False)
    plt.show()


# todo 3. 数据集划分
def dm_train_test_split():
    # 加载数据集
    iris_data = load_iris()
    # 对数据集进行划分
    # 参1为数据集的特征,参2为数据集的标签,参3为测试集所占比例,参4为随机数种子
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=22)


# todo 4. 模型训练和预测
def dm_modle_train_and_predict():
    # 加载数据集
    iris_data = load_iris()
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=22)
    # 数据标准化,创建标准化模型
    process = StandardScaler()
    # 对训练数据进行标准化
    x_train = process.fit_transform(x_train)  # 对训练集数据先进行计算 在进行转换
    x_test = process.transform(x_test)  # 对测试集进行转换
    # 创建模型
    model = KNeighborsClassifier()
    # 模型训练
    model.fit(x_train, y_train)
    # 模型预测
    y_predict = model.predict(x_test)
    print('预测结果: ', y_predict)
    print('准确率: ', accuracy_score(y_test, y_predict))
    # 创建新数据,对模型进行预测
    x_train_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
    print('新数据预测结果: ', model.predict(x_train_new))
    # 预测概率
    print('预测概率: ', model.predict_proba(x_train_new))
    # 模型评估,两种方式
    # 方式一,计算准确率
    print('准确率: ', model.score(x_test, y_test))
    # 方式二,使用模型评估模块
    print('准确率: ', accuracy_score(y_test, y_predict))


if __name__ == '__main__':
    dm_loadiris()
    dm_showiris()
    dm_train_test_split()
    dm_modle_train_and_predict()
