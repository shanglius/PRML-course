import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 3D 数据生成与划分 (保持绝对一致)
# ==========================================
def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y_labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y_labels


X_train, y_train = make_moons_3d(n_samples=1000, noise=0.2, random_state=42)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2, random_state=99)

# 数据标准化 (SVM 必备)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. 定义三种不同的核函数模型
# ==========================================
# 统一设置 C=10，确保比较的是"核函数"本身的特性，而不是被惩罚系数干扰
models = {
    "Linear Kernel": SVC(kernel='linear', C=10, random_state=42),
    "Polynomial Kernel (d=3)": SVC(kernel='poly', degree=5, C=10, random_state=42),
    "RBF Kernel": SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
}

# ==========================================
# 3. 训练、评估与绘制图表
# ==========================================
# 创建一个 1行3列 的宽幅 3D 图表
fig = plt.figure(figsize=(18, 6))

print("-" * 50)
print(f"{'核函数类型 (Kernel)':<25} | {'训练集准确率':<10} | {'测试集准确率':<10}")
print("-" * 50)

# 遍历训练每一种模型
for i, (name, clf) in enumerate(models.items()):
    # 训练
    clf.fit(X_train_scaled, y_train)

    # 预测
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)

    # 计算准确率
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # 打印表格输出
    print(f"{name:<25} | {train_acc * 100:>8.2f}% | {test_acc * 100:>8.2f}%")

    # ==========================
    # 绘制当前核函数的 3D 结果
    # ==========================
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')

    mis_idx = np.where(y_test != y_test_pred)[0]
    c0_idx = np.where(y_test == 0)[0]
    c1_idx = np.where(y_test == 1)[0]

    # 画出原始点
    ax.scatter(X_test[c0_idx, 0], X_test[c0_idx, 1], X_test[c0_idx, 2], c='skyblue', alpha=0.6, label='Class 0')
    ax.scatter(X_test[c1_idx, 0], X_test[c1_idx, 1], X_test[c1_idx, 2], c='salmon', alpha=0.6, label='Class 1')

    # 标记错误点
    ax.scatter(X_test[mis_idx, 0], X_test[mis_idx, 1], X_test[mis_idx, 2], c='black', marker='x', s=60,
               depthshade=False)

    ax.set_title(f'{name}\nTest Acc: {test_acc * 100:.1f}%')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

print("-" * 50)
plt.tight_layout()
plt.show()