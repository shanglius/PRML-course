import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# __file__ 代表当前脚本的路径，.parent 获取它所在的文件夹
current_dir = Path(__file__).parent

# 自动将当前文件夹路径与文件名拼接
file_path = current_dir / '工作簿1.xlsx'

train_df = pd.read_excel(file_path, sheet_name='Sheet1')
test_df = pd.read_excel(file_path, sheet_name='Sheet2')

# 提取特征 x 和目标变量 y
X_train = train_df['x'].values
y_train = train_df['y_complex'].values
X_test = test_df['x_new'].values
y_test = test_df['y_new_complex'].values

# 为 x 增加一列全为1的常数项 (为了求解截距 theta_0)
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]


# 2. 优化的梯度下降算法实现 (加入早停机制 tolerance)
def gradient_descent_optimized(X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6):
    m = len(y)
    # 随机初始化参数
    theta = np.random.randn(2, 1)
    y = y.reshape(-1, 1)

    cost_history = []

    for iteration in range(n_iterations):
        # 计算预测值和误差
        predictions = X.dot(theta)
        errors = predictions - y

        # 计算梯度并更新参数
        gradients = (1 / m) * X.T.dot(errors)
        theta = theta - learning_rate * gradients

        # 记录当前轮次的均方误差(MSE)
        cost = (1 / m) * np.sum(errors ** 2)
        cost_history.append(cost)

        # 优化点：早停机制 (Early Stopping)
        # 如果损失下降非常缓慢（小于设定的容忍度），则判定为收敛，提前结束训练
        if iteration > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            print(f"在第 {iteration} 次迭代时触发早停 (模型已收敛)。")
            break

    return theta, cost_history


# 3. 设置超参数并运行
alpha = 0.05  # 学习率
iterations = 2000  # 最大迭代次数设大一些，靠早停来控制
tolerance = 1e-6  # 容忍度（用于早停）

theta_best, cost_history = gradient_descent_optimized(
    X_train_b, y_train,
    learning_rate=alpha,
    n_iterations=iterations,
    tol=tolerance
)

# 4. 计算训练误差和测试误差
y_train_predict = X_train_b.dot(theta_best)
mse_train = np.mean((y_train.reshape(-1, 1) - y_train_predict) ** 2)

y_test_predict = X_test_b.dot(theta_best)
mse_test = np.mean((y_test.reshape(-1, 1) - y_test_predict) ** 2)

print("-" * 30)
print(f"截距 (Theta 0): {theta_best[0][0]:.4f}")
print(f"斜率 (Theta 1): {theta_best[1][0]:.4f}")
print(f"训练误差 (Training MSE): {mse_train:.4f}")
print(f"测试误差 (Testing MSE): {mse_test:.4f}")
print("-" * 30)

# 5. 绘图部分 (可视化结果)
# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 5))

# ----- 子图1：回归拟合效果图 -----
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='royalblue', label='训练数据 (Train)', alpha=0.6)
plt.scatter(X_test, y_test, color='darkorange', label='测试数据 (Test)', alpha=0.6)
plt.plot(X_train, y_train_predict, color='crimson', linewidth=2, label='拟合直线')
plt.xlabel('特征 x')
plt.ylabel('目标变量 y')
plt.title('线性回归拟合效果图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# ----- 子图2：损失函数下降曲线 -----
plt.subplot(1, 2, 2)
# 只绘制实际运行的迭代次数部分
plt.plot(range(len(cost_history)), cost_history, color='purple', linewidth=2)
plt.xlabel('迭代次数 (Iterations)')
plt.ylabel('均方误差 (MSE)')
plt.title('训练误差收敛曲线')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()  # 弹出窗口展示图表