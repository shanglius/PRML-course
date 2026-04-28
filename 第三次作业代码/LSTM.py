import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import math

# 1. 将时间序列转换为监督学习问题的辅助函数
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 拼接起来
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 删除包含NaN的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# =======================
# 2. 数据加载与预处理
# =======================
# 加载数据 (设置date为索引)
dataset = pd.read_csv('LSTM-Multivariate_pollution.csv', header=0, index_col=0)
values = dataset.values

# 类别特征编码：风向（wnd_dir）是字符串类别，需要转换为数字
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# 确保所有数据是float32类型
values = values.astype('float32')

# 归一化特征：LSTM对输入数值的范围比较敏感，推荐将其缩放到 [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# 将时间序列问题转换为监督学习问题
# 设定用前1个小时(t-1)的数据，预测当前时刻(t)的污染值
reframed = series_to_supervised(scaled, 1, 1)

# 删除我们不想预测的列（保留 var1(t) 即污染预测，删除 t 时刻的其他气象变量）
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print("预处理后的数据前5行：")
print(reframed.head())

# =======================
# 3. 划分训练集和测试集
# =======================
values = reframed.values
# 假设用前一年的数据（约365*24=8760小时）作为训练集
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# 拆分输入(X)和输出(y)
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# 将输入重塑为 3D 格式，即 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print('训练集维度:', train_X.shape, train_y.shape)
print('测试集维度:', test_X.shape, test_y.shape)

# =======================
# 4. 构建并训练 LSTM 模型
# =======================
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

print("\n开始训练模型...")
history = model.fit(train_X, train_y,
                    epochs=50,
                    batch_size=72,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False)

# =======================
# 5. 可视化训练过程损失
# =======================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test (Validation) Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.show()

# =======================
# 6. 预测与逆缩放还原真实数值
# =======================
yhat = model.predict(test_X)
test_X_reshape = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# 逆缩放预测值
inv_yhat = np.concatenate((yhat, test_X_reshape[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# 逆缩放真实值
test_y_reshape = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y_reshape, test_X_reshape[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# 计算预测的 RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print(f'\n测试集预测 RMSE: {rmse:.3f}')

# =======================
# 7. 预测值与实际观测值对比图 (新增部分)
# =======================
# 7.1 全局对比图（展示测试集全貌）
plt.figure(figsize=(15, 6))
plt.plot(inv_y, label='Actual Pollution (Ground Truth)', color='blue', alpha=0.6, linewidth=1)
plt.plot(inv_yhat, label='Predicted Pollution', color='red', alpha=0.6, linewidth=1)
plt.title('Global Comparison: Actual vs Predicted Pollution over Entire Test Set')
plt.xlabel('Time Steps (Hours)')
plt.ylabel('Pollution Concentration')
plt.legend()
plt.show()

# 7.2 局部对比图（提取前500小时，以便细致观察曲线拟合度）
zoom_range = 500
plt.figure(figsize=(15, 6))
plt.plot(inv_y[:zoom_range], label='Actual Pollution (Ground Truth)', color='blue', marker='.', markersize=4, linewidth=1.5)
plt.plot(inv_yhat[:zoom_range], label='Predicted Pollution', color='red', marker='.', markersize=4, linewidth=1.5)
plt.title(f'Local Zoom: Actual vs Predicted Pollution (First {zoom_range} Hours)')
plt.xlabel('Time Steps (Hours)')
plt.ylabel('Pollution Concentration')
plt.legend()
plt.show()