import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_excel('data.xlsx')

# 假设x有四个特征，你需要根据实际情况调整这里的特征列名
feature_columns = ['feature1', 'feature2', 'feature3', 'feature4']

# 提取特征和目标变量
X = data[feature_columns]
y = data['target']

# 使用多项式特征
degree = 2  # 多项式的阶数，可以根据需要调整
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# 创建多项式回归模型
model = LinearRegression()

# 训练模型
model.fit(X_poly, y)

# 在整个数据集上进行预测
y_pred = model.predict(X_poly)

# 打印预测结果和真实值
result_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
print(result_df)

# 评估模型性能
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse}')

# 绘制预测值与实际值的散点图
plt.scatter(y, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values in Polynomial Regression')
plt.show()
