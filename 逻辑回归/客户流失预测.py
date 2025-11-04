import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def md_ject():
    # 数据处理
    data = pd.read_csv('./data/churn.csv')
    # print(data.head())
    data = pd.get_dummies(data)
    data.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    data.rename(columns={'Churn_Yes': 'flag'}, inplace=True)

    # 特征处理
    x = data[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = data['flag']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=100)
    # 实例化模型
    estimator = LogisticRegression()
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    # 模型评估
    my_accuracy = accuracy_score(y_test,y_pred)
    print(f'my_accuracy-->{my_accuracy}')
    my_score = estimator.score(x_test,y_test)
    print(f'my_score-->{my_score}')
    # 计算AUC
    my_auc = roc_auc_score(y_test,y_pred)
    print(f'my_auc-->{my_auc}')
    result = classification_report(y_test,y_pred,target_names=['flag0', 'flag1'])
    print(f'result-->{result}')
if __name__ == '__main__':
    md_ject()