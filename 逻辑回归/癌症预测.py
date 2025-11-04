import pandas as pd
import numpy as np
from scipy.signal import TransferFunction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def dm_LoRgisticRegression():
    # 获取数据
    data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
    # print(data.head())
    # 数据处理
    data = data.replace(to_replace='?',value=np.nan)
    data = data.dropna()
    # 数据划分
    x = data.iloc[:,1:-1]
    y = data['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    # 特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    print(f'x-->{x_train.shape}')
    x_test = transfer.transform(x_test)
    print(f'x-->{x_test.shape}')
    # 逻辑回归
    estimator = LogisticRegression()
    estimator.fit(x_train,y_train)

    # 模型评估
    y_pred = estimator.predict(x_test)
    print(f'y_pred-->{y_pred}')
    accuracy = estimator.score(x_test,y_test)
    print(f'accuracy-->{accuracy}')


if __name__ == '__main__':
    dm_LoRgisticRegression()