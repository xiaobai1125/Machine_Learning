from sklearn.metrics import confusion_matrix
import pandas as pd 
def dm01_confusion_matrix():
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']
    labels = ['恶性', '良性']
    dataframe_labels = ['恶性(正例)', '良性(反例)']
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    print('模型A:')
    print('-' * 13)
    y_pred1= ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']
    result = confusion_matrix(y_true, y_pred1,labels=labels)
    print(pd.DataFrame(result, columns=dataframe_labels,  index=dataframe_labels))
    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    print('模型B:')
    print('-' * 13)
    y_pred2= ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性']
    result = confusion_matrix(y_true, y_pred2, labels=labels)
    print(pd.DataFrame(result,  columns=dataframe_labels,index=dataframe_labels))

if __name__ == '__main__':
    dm01_confusion_matrix()