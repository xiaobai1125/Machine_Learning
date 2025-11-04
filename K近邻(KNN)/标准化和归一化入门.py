"""为什么进行归一化、标准化
      特征的单位或大小相差较大,或者某些特征的方差相比其他的特征要大出几个数量级,容易影响目标结果,使一些模型无法学习到其他特征.
      归一化 : 通过对原始数据进行变换把数据映射到[min, max](默认为[0, 1])之间
      标准化 : 通过对原始数据进行标准化,转换为均值为0标准差为1的标准正态分布的数据"""

#  导包
from sklearn.preprocessing import MinMaxScaler,StandardScaler # 前者为归一化,后者为标准化
def 归一化():
    # todo 1.获取数据
    x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]
    # todo 2.创建归一化模型
    process = MinMaxScaler()
    # todo 3.进行归一化
    x_process = process.fit_transform(x) # 训练模型并返回归一化结果
    print('归一化结果: ',x_process)

def 标准化():
    # todo 1.获取数据
    x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]
    # todo 2.创建标准化模型
    process = StandardScaler()
    # todo 3.进行标准化
    x_process = process.fit_transform(x)
    print('标准化结果: ',x_process)

if __name__ == '__main__':
    归一化()
    标准化()