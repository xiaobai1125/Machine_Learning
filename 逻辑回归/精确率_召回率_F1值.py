from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

def dm_accuracy():
    print('准确')
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    y_pred1 = ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]
    result = accuracy_score(y_true,y_pred1)
    print(f'模型A精度{result}')

    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    y_pred2 = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性",  "恶性", "恶性", "恶性", "良性"]
    result = accuracy_score(y_true, y_pred2)
    print(f'模型B精度{result}')

def dm_precision():
    print('精度')
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    y_pred1 = ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]
    result = precision_score(y_true,y_pred1,pos_label='恶性')
    print(f'模型A精度{result}')

    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    y_pred2 = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性",  "恶性", "恶性", "恶性", "良性"]
    result = precision_score(y_true, y_pred2, pos_label='恶性')
    print(f'模型B精度{result}')
def dm_recall():
    print('召回率')
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    y_pred1 = ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]
    result = recall_score(y_true,y_pred1,pos_label='恶性')
    print(f'模型A精度{result}')

    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    y_pred2 = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性",  "恶性", "恶性", "恶性", "良性"]
    result = recall_score(y_true, y_pred2, pos_label='恶性')
    print(f'模型B精度{result}')

def dm_f1():
    print('f1值')
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    y_pred1 = ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]
    result = f1_score(y_true,y_pred1,pos_label='恶性')
    print(f'模型A精度{result}')

    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    y_pred2 = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性",  "恶性", "恶性", "恶性", "良性"]
    result = f1_score(y_true, y_pred2, pos_label='恶性')
    print(f'模型B精度{result}')

if __name__ == '__main__':
    dm_precision()
    dm_accuracy()
    dm_recall()
    dm_f1()