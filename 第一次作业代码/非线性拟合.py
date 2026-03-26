import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==========================================
# 0. 数据加载与预处理
# ==========================================
# 读取训练数据和测试数据
# 请将下面单引号里的路径，替换为你电脑上实际的文件路径
# 使用 read_excel 读取 xlsx 文件，并指定对应的 sheet_name
import pandas as pd
from pathlib import Path

# __file__ 代表当前脚本的路径，.parent 获取它所在的文件夹
current_dir = Path(__file__).parent

# 自动将当前文件夹路径与文件名拼接
file_path = current_dir / '工作簿1.xlsx'

train_df = pd.read_excel(file_path, sheet_name='Sheet1')
test_df = pd.read_excel(file_path, sheet_name='Sheet2')

# 提取特征 x 和目标值 y，并转换为列向量
X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y_complex'].values.reshape(-1, 1)

X_test = test_df['x_new'].values.reshape(-1, 1)
y_test = test_df['y_new_complex'].values.reshape(-1, 1)

# 为线性模型添加偏置项 (x_0 = 1)
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# 定义计算均方误差 (MSE) 的辅助函数
def calc_mse(theta, X, y):
    return np.mean((X.dot(theta) - y) ** 2)

print("================ 线性拟合 ================")

# ==========================================
# 1. 最小二乘法 (Least Squares)
# ==========================================
# 公式: theta = (X^T * X)^(-1) * X^T * y
theta_ls = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

mse_train_ls = calc_mse(theta_ls, X_train_b, y_train)
mse_test_ls = calc_mse(theta_ls, X_test_b, y_test)
print(f"1. 最小二乘法 (LS)   | Train MSE: {mse_train_ls:.4f}, Test MSE: {mse_test_ls:.4f}")


# ==========================================
# 2. 梯度下降法 (Gradient Descent)
# ==========================================
learning_rate = 0.01  # 学习率
n_iterations = 2000   # 迭代次数
m = len(X_train_b)    # 样本数量

# 随机初始化参数
np.random.seed(42)
theta_gd = np.random.randn(2, 1)

for iteration in range(n_iterations):
    # 计算梯度: 2/m * X^T * (X * theta - y)
    gradients = 2/m * X_train_b.T.dot(X_train_b.dot(theta_gd) - y_train)
    # 更新参数
    theta_gd = theta_gd - learning_rate * gradients

mse_train_gd = calc_mse(theta_gd, X_train_b, y_train)
mse_test_gd = calc_mse(theta_gd, X_test_b, y_test)
print(f"2. 梯度下降法 (GD)   | Train MSE: {mse_train_gd:.4f}, Test MSE: {mse_test_gd:.4f}")


# ==========================================
# 3. 牛顿法 (Newton's Method)
# ==========================================
# 随机初始化参数
theta_nt = np.random.randn(2, 1)

# 计算海森矩阵 (Hessian): H = 2/m * X^T * X
H = 2/m * X_train_b.T.dot(X_train_b)
# 计算初始梯度
grad = 2/m * X_train_b.T.dot(X_train_b.dot(theta_nt) - y_train)

# 牛顿法迭代: theta = theta - H^(-1) * grad
# 注意：对于线性回归，牛顿法只需 1 步即可收敛到最优解
theta_nt = theta_nt - np.linalg.inv(H).dot(grad)

mse_train_nt = calc_mse(theta_nt, X_train_b, y_train)
mse_test_nt = calc_mse(theta_nt, X_test_b, y_test)
print(f"3. 牛顿法 (Newton)   | Train MSE: {mse_train_nt:.4f}, Test MSE: {mse_test_nt:.4f}")


print("\n================ 非线性拟合 ================")

# ==========================================
# 4. 多项式回归 (Polynomial Regression) - 9阶
# ==========================================
degree = 9
# 构建多项式特征生成器
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# 使用 scikit-learn 的线性回归求解多项式模型
poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)

# 预测并计算误差
y_train_poly_pred = poly_reg.predict(X_poly_train)
y_test_poly_pred = poly_reg.predict(X_poly_test)

mse_train_poly = mean_squared_error(y_train, y_train_poly_pred)
mse_test_poly = mean_squared_error(y_test, y_test_poly_pred)
print(f"4. 多项式回归 ({degree}阶) | Train MSE: {mse_train_poly:.4f}, Test MSE: {mse_test_poly:.4f}")


# ==========================================
# 5. 可视化对比结果
# ==========================================
plt.figure(figsize=(12, 7))

# 绘制原始数据点
plt.scatter(X_train, y_train, label='Train Data', alpha=0.5, color='gray')
plt.scatter(X_test, y_test, label='Test Data', alpha=0.8, marker='x', color='black')

# 生成用于绘制平滑曲线的 X 值序列
X_plot = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
X_plot_b = np.c_[np.ones((len(X_plot), 1)), X_plot]

# 绘制线性拟合直线 (以最小二乘法结果为例，GD和NT结果与之重合)
y_plot_linear = X_plot_b.dot(theta_ls)
plt.plot(X_plot, y_plot_linear, color='red', linestyle='--', linewidth=2, label='Linear Fit (LS/GD/Newton)')

# 绘制多项式拟合曲线
X_plot_poly = poly_features.transform(X_plot)
y_plot_poly = poly_reg.predict(X_plot_poly)
plt.plot(X_plot, y_plot_poly, color='blue', linewidth=2, label=f'Polynomial Fit (Degree {degree})')

plt.title('Linear vs Polynomial Regression Fit', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)

# 显示图表
plt.tight_layout()
plt.show()