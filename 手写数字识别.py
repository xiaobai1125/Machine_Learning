# 导包
import matplotlib.pyplot as plt  # 可视化
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.neighbors import KNeighborsClassifier  # KNN分类
import joblib  # 保存
from collections import Counter  # 统计
from PIL import Image  # 图像处理
import numpy as np
import pandas as pd


# 显示图片
def show_digit(idx):
    # 获取数据集
    data = pd.read_csv('data/手写数字识别.csv')
    # 检验非法值
    if idx < 0 or idx > len(data) - 1:
        return
    # 查看数据基本信息
    x = data.iloc[:, 1:]  # 查看所有行,除了第一列所有列
    y = data.iloc[:, 0]  # 查看所有行,第一列
    print(f'数据基本信息: {x.shape})')  # 显示数据集大小
    print(f'类别数据比例: {Counter(y)}')  # counter用于统计数据中各个数字出现的次数
    # 显示图片
    # 将数据形状更改为: 28 * 28
    digit = x.iloc[idx].values.reshape(28, 28)
    # 关闭坐标轴
    plt.axis('off')
    plt.imshow(digit, cmap='gray')  # cmap为颜色映射
    plt.axis('off')  # 关闭坐标.
    # 参1:保存图片的路径,参2:图片的边界,参3:图片的间距,参4:图片的分辨率
    plt.savefig('data/demo2.png', bbox_inches='tight', pad_inches=0, dpi=28)  # 保存图片
    plt.show()


# 训练模型
def train_model():
    # 获取数据
    data = pd.read_csv('data/手写数字识别.csv')
    x = data.iloc[:, 1:].values / 255.0  # 归一化,把数据转化成 0-1 之间,同时values把df转化成 numpy 数组
    y = data.iloc[:, 0].values
    # 划分数据集
    # stratify=y 确保划分比例和数据集一致
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)
    # 模型训练
    model = KNeighborsClassifier(n_neighbors=3)  # 这里可以用网格搜索检查验证选择最优的参数
    model.fit(x_train, y_train)
    # 模型评估
    my_score = model.score(x_test, y_test)
    print('测试机准确率为: ', my_score)
    # 保存模型
    joblib.dump(model, "model/knn.pkl")
    print('模型已保存')


# 测试模型
def use_model():
    img_path = 'data/demo2.png'
    # 读取图片
    img = Image.open(img_path).convert('L')  # 读取图片并转为灰度图
    # 缩放到 28×28
    img = img.resize((28, 28))
    # 转 numpy 并归一化
    img = np.array(img) / 255.0
    img = img.reshape(1, -1)  # 把
    # 模型加载
    model = joblib.load('model/knn.pkl')
    # 预测图片
    y_test = model.predict(img)
    # 显示图片和预测结果
    plt.imshow(img.reshape(28, 28), cmap='gray')  # 显示图片
    plt.axis('off')
    plt.show()
    print('您绘制的数字为: ', y_test)


if __name__ == '__main__':
    show_digit(9)
    train_model()
    use_model()
