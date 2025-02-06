# train_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
import os

# 从欧易API获取历史K线数据
def fetch_okx_historical_data(symbol='SOL-USDT', timeframe='1H', limit=1000):
    url = "https://www.okx.com/api/v5/market/history-candles"
    headers = {
        'OK-ACCESS-KEY': os.getenv('OKX_API_KEY'),
        'OK-ACCESS-SECRET': os.getenv('OKX_API_SECRET')
    }
    params = {
        'instId': symbol,
        'bar': timeframe,
        'limit': limit
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()['data']
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df.sort_values('timestamp')
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

# 数据预处理和训练模型
def train_lstm_model():
    # 1. 获取数据
    data = fetch_okx_historical_data()
    if data is None:
        return

    # 2. 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # 3. 创建时间序列数据集
    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # 调整形状为 [样本数, 时间步, 特征数]

    # 4. 构建模型
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(time_step, 1)))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 5. 训练模型
    model.fit(X, y, epochs=20, batch_size=64, verbose=1)

    # 6. 保存模型和Scaler
    model.save('lstm_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved.")

if __name__ == '__main__':
    train_lstm_model()
