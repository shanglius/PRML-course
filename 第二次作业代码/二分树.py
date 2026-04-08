import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 3D 双月数据生成函数
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

# 生成 2000 个数据点
X, labels = make_moons_3d(n_samples=1000, noise=0.2, random_state=42)

# ==========================================
# 2. 划分数据集与模型训练
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# 注意：这里我们故意不设置 max_depth，让树完全生长以达到 100% 的训练集准确率
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# ==========================================
# 3. 计算并打印性能量化结果 (对应你的表1)
# ==========================================
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("-" * 30)
print(f"表 1 决策树分类性能量化结果")
print("-" * 30)
print(f"数据集类型\t\t准确率 (Accuracy)")
print(f"训练集 (Training Set)\t{train_accuracy * 100:.2f}%")
print(f"测试集 (Testing Set)\t{test_accuracy * 100:.2f}%")
print("-" * 30)

# ==========================================
# 4. 绘制包含错误标记的 3D 散点图
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 提取预测错误的点的索引
misclassified_idx = np.where(y_test != y_test_pred)[0]

# 提取真实的类别数据
class0_idx = np.where(y_test == 0)[0]
class1_idx = np.where(y_test == 1)[0]

# 绘制真实的 Class 0 (浅蓝色)
ax.scatter(X_test[class0_idx, 0], X_test[class0_idx, 1], X_test[class0_idx, 2],
           c='skyblue', marker='o', label='Class 0 (True)', alpha=0.7)

# 绘制真实的 Class 1 (浅红色)
ax.scatter(X_test[class1_idx, 0], X_test[class1_idx, 1], X_test[class1_idx, 2],
           c='salmon', marker='o', label='Class 1 (True)', alpha=0.7)

# 叠加绘制分类错误的点 (黑色 X)
# s=60 控制 X 的大小，depthshade=False 保证 X 标记不会因为 3D 深度而变淡
ax.scatter(X_test[misclassified_idx, 0], X_test[misclassified_idx, 1], X_test[misclassified_idx, 2],
           c='black', marker='x', s=60, label='Misclassified', depthshade=False)

# 设置图表元素
ax.set_title('Decision Tree Classification')
ax.legend(loc='upper right')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()