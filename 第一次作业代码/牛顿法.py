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
X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y_complex'].values.reshape(-1, 1)
X_test = test_df['x_new'].values.reshape(-1, 1)
y_test = test_df['y_new_complex'].values.reshape(-1, 1)

# 为 X 增加偏置项(截距项) x0 = 1
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]


# 2. 梯度下降算法 (记录迭代历史)
def gradient_descent(X, y, learning_rate=0.05, n_iterations=800):
    m = len(y)
    theta = np.zeros((2, 1))  # 参数初始化为 0
    cost_history = []

    for i in range(n_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (1 / m) * X.T.dot(errors)
        theta = theta - learning_rate * gradients
        cost = (1 / m) * np.sum(errors ** 2)
        cost_history.append(cost)

    return theta, cost_history


theta_best, cost_history = gradient_descent(X_train_b, y_train, learning_rate=0.05, n_iterations=800)
y_train_predict = X_train_b.dot(theta_best)

# 3. 绘图代码
# 设置 Matplotlib 中文字体支持 (不同系统可能需要修改为其他字体如 'Microsoft YaHei')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 5))

# --- 图1：拟合图 ---
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='royalblue', label='训练数据', alpha=0.6)
plt.scatter(X_test, y_test, color='darkorange', label='测试数据', alpha=0.6)
plt.plot(X_train, y_train_predict, color='crimson', linewidth=2.5, label='拟合直线')
plt.xlabel('特征 x')
plt.ylabel('目标变量 y')
plt.title('图1：线性回归拟合效果图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- 图2：迭代次数 vs 误差下降曲线 ---
plt.subplot(1, 2, 2)
plt.plot(range(len(cost_history)), cost_history, color='purple', linewidth=2)
plt.xlabel('迭代次数 (Iterations)')
plt.ylabel('均方误差 (MSE)')
plt.title('图2：迭代次数与训练误差关系图')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()