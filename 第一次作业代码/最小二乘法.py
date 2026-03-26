import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
def main():
    # __file__ 代表当前脚本的路径，.parent 获取它所在的文件夹
    current_dir = Path(__file__).parent

    # 自动将当前文件夹路径与文件名拼接
    file_path = current_dir / '工作簿1.xlsx'

    train_df = pd.read_excel(file_path, sheet_name='Sheet1')
    test_df = pd.read_excel(file_path, sheet_name='Sheet2')
    # 2. 提取特征矩阵 (X) 和 目标向量 (y)
    # pandas 提取单列作为特征时，使用双括号 [['x']] 确保其为二维数组 (n_samples, n_features)
    X_train = train_df[['x']].values
    y_train = train_df['y_complex'].values

    X_test = test_df[['x_new']].values
    y_test = test_df['y_new_complex'].values

    # 3. 初始化并训练线性回归模型（本质即普通最小二乘法 OLS）
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. 使用模型进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 5. 计算并打印均方误差 (MSE)
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    # 打印模型参数：斜率 (coef_) 和 截距 (intercept_)
    print("-" * 30)
    print(f"拟合直线方程: y = {model.coef_[0]:.4f} * x + ({model.intercept_:.4f})")
    print(f"训练误差 (MSE): {train_error:.4f}")
    print(f"测试误差 (MSE): {test_error:.4f}")
    print("-" * 30)

    # 6. 可视化拟合结果
    plt.figure(figsize=(10, 6))

    # 绘制训练集（蓝色）和测试集（绿色）的散点图
    plt.scatter(X_train, y_train, color='blue', label='Train Data (Sheet1)', alpha=0.6)
    plt.scatter(X_test, y_test, color='green', label='Test Data (Sheet2)', alpha=0.6)

    # 为了画出平滑且贯穿全图的拟合直线，生成一个覆盖所有 x 范围的连续点集
    x_min = min(X_train.min(), X_test.min())
    x_max = max(X_train.max(), X_test.max())
    X_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_range_pred = model.predict(X_range)

    # 绘制拟合直线（红色）
    plt.plot(X_range, y_range_pred, color='red', label='Linear Fit (Least Squares)', linewidth=2)

    # 设置图表格式
    plt.xlabel('X (Feature)')
    plt.ylabel('Y (Target)')
    plt.title('Linear Regression Model vs Data')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 展示图表
    plt.show()

if __name__ == "__main__":
    main()