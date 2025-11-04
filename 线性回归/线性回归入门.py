"""线性回归: 就是利用回归方程(函数)对一个或多个自变量(特征值)
            和因变量(目标值)之间关系进行建模的一种分析方式.
   线性回归分类:
            1. 一元线性  y = k * x + b
            2.多元线性   y = w的转置 * x + b
   线性回归API: sklearn.linear_model.LinearRegression

"""

# 身高预测案例
# todo 1.导包
from sklearn.linear_model import LinearRegression  # 线性回归
# todo 2.获取数据
x = [[160],[166],[172],[174],[180]]
y = [56.3,60.6,65.1,68.5,75]
# todo 3.模型训练
# 创建模型对象
Linear_model = LinearRegression()
# 训练模型
Linear_model.fit(x,y)
# 查看模型参数
print('截距',Linear_model.intercept_)
print('斜率',Linear_model.coef_)
# todo 4.模型预测
print('预测结果',Linear_model.predict([[170]]))
