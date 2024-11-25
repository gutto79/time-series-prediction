import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dropout

# 1. データの取得
data_path = "resourses/stock_price.csv"  
df = pd.read_csv(data_path, encoding="utf-8").iloc[::-1].reset_index(drop=True)

df['日付け'] = pd.to_datetime(df['日付け'])
df.set_index('日付け', inplace=True)

# 終値を予測対象として使用
data = df['終値']

# 2. 移動平均の計算と特徴量追加
df['MA_5'] = df['終値'].rolling(window=5).mean()  # 5日移動平均
df['MA_20'] = df['終値'].rolling(window=20).mean()  # 20日移動平均
df['MA_diff'] = df['MA_5'] - df['MA_20']  # 5日移動平均と20日移動平均の差

# 移動平均計算によるNaNを削除
df = df.dropna()


# 特徴量として「終値」と「MA_diff」を使用
features = df[['終値', 'MA_diff']]


# 正規化
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# 3. 訓練データとテストデータの分割
training_size = int(len(features_scaled) * 0.8)
train_data, test_data = features_scaled[:training_size], features_scaled[training_size:]

# 4. データをLSTM用に変換
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])  
        Y.append(dataset[i + time_step, 0])  
    return np.array(X), np.array(Y)

time_step = 60  # 60日分のデータを使って1日を予測
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 5. LSTMモデルの入力形状を調整
# (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# 6. LSTMモデルの構築
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))  
model.add(Dropout(0.3))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))  

# 7. モデルのコンパイルと訓練
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=5)

# 8. 予測
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 9. 予測結果を逆変換
train_predict = scaler.inverse_transform(np.hstack((train_predict, np.zeros((train_predict.shape[0], 1)))))
test_predict = scaler.inverse_transform(np.hstack((test_predict, np.zeros((test_predict.shape[0], 1)))))
y_test = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1)))))

train_predict = train_predict[:, 0]  # 終値のみ取得
test_predict = test_predict[:, 0]
y_test = y_test[:, 0]

# 10. モデルの評価
mse = mean_squared_error(y_test, test_predict)
r2 = r2_score(y_test, test_predict)

print('MSE: {:.2f}'.format(mse))

# 11. プロット
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['終値'], label='Historical')

# 訓練データの予測プロット
train_predict_plot = np.empty_like(features_scaled[:, 0])
train_predict_plot[:] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step] = train_predict

# テストデータの予測プロット
test_predict_plot = np.empty_like(features_scaled[:, 0])
test_predict_plot[:] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2)+1 :len(features_scaled) - 1] = test_predict

# プロット
plt.plot(df.index, train_predict_plot, label='Train Predict')
plt.plot(df.index, test_predict_plot, label='Test Predict')
plt.legend()
plt.show()
