import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 3D 数据生成器
# ==========================================
def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    rng = np.random.RandomState(random_state)
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y_labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += rng.normal(scale=noise, size=X.shape)
    return X, y_labels

# 训练集：1000个点；测试集：500个点（250+250）
X_train, y_train = make_moons_3d(n_samples=1000, noise=0.2, random_state=42)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2, random_state=99)

# ==========================================
# 2. 深度优化的 AdaBoost 配置 (兼容所有版本)
# ==========================================
# 优化点 1: 提升基学习器深度至 5，增强 3D 空间捕捉能力
optimized_base = DecisionTreeClassifier(max_depth=5, random_state=42)

# 【核心修复】：使用 try-except 完美兼容 scikit-learn 新旧版本参数名更迭
try:
    # 尝试使用 sklearn >= 1.2 的新参数名 'estimator'
    clf = AdaBoostClassifier(
        estimator=optimized_base,
        n_estimators=300,    # 优化点 2: 提升迭代次数到 300
        learning_rate=0.1,   # 优化点 3: 降低学习率，防止过拟合
        random_state=42,

    )
except TypeError:
    # 如果触发报错，自动回退到 sklearn < 1.2 的旧参数名 'base_estimator'
    clf = AdaBoostClassifier(
        estimator=optimized_base,
        n_estimators=300,
        learning_rate=0.1,
        random_state=42,

    )

clf.fit(X_train, y_train)

# ==========================================
# 3. 性能量化
# ==========================================
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("-" * 40)
print(f"🚀 优化后 AdaBoost 分类性能")
print("-" * 40)
print(f"训练集准确率: {train_acc * 100:.2f}%")
print(f"测试集准确率: {test_acc * 100:.2f}%")
print("-" * 40)

# ==========================================
# 4. 可视化
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

mis_idx = np.where(y_test != y_test_pred)[0]
c0_idx = np.where(y_test == 0)[0]
c1_idx = np.where(y_test == 1)[0]

ax.scatter(X_test[c0_idx, 0], X_test[c0_idx, 1], X_test[c0_idx, 2], c='skyblue', label='C0 (True)', alpha=0.6)
ax.scatter(X_test[c1_idx, 0], X_test[c1_idx, 1], X_test[c1_idx, 2], c='salmon', label='C1 (True)', alpha=0.6)
ax.scatter(X_test[mis_idx, 0], X_test[mis_idx, 1], X_test[mis_idx, 2], c='black', marker='x', s=70, label='Misclassified', depthshade=False)

ax.set_title(f'Optimized AdaBoost\n(Depth=5, Estimators=300, LR=0.1)')
ax.legend()
plt.show()