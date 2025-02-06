# main.py
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# 加载预训练模型和Scaler
model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

# 3Commas API 客户端
class ThreeCommasClient:
    def __init__(self, api_key, api_secret):
        self.base_url = "https://api.3commas.io/public/api"
        self.headers = {'APIKEY': api_key, 'SECRET': api_secret}

    def get_bots(self):
        response = requests.get(f"{self.base_url}/ver1/bots", headers=self.headers)
        return response.json() if response.status_code == 200 else None

    def update_take_profit_and_stop_loss(self, bot_id, take_profit, stop_loss):
        url = f"{self.base_url}/ver1/bots/{bot_id}/update"
        data = {
            'take_profit': take_profit,
            'stop_loss': stop_loss
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json() if response.status_code == 200 else None

# 欧易 API 客户端
class OKXClient:
    def __init__(self, api_key, api_secret):
        self.base_url = "https://www.okx.com/api/v5"
        self.headers = {'OK-ACCESS-KEY': api_key, 'OK-ACCESS-SECRET': api_secret}

    def get_historical_data(self, symbol='SOL-USDT', timeframe='1H', limit=1000):
        params = {'instId': symbol, 'bar': timeframe, 'limit': limit}
        response = requests.get(f"{self.base_url}/market/history-candles", headers=self.headers, params=params)
        return response.json()['data'] if response.status_code == 200 else None

# 使用 LSTM 模型预测价格
def predict_with_lstm(data):
    last_60 = data['close'].values[-60:].reshape(-1, 1)
    scaled_data = scaler.transform(last_60)
    predicted_price = model.predict(scaled_data.reshape(1, 60, 1))
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]
    return predicted_price

# 主逻辑
def main():
    # 初始化客户端
    three_commas_client = ThreeCommasClient(
        os.getenv('3COMMAS_API_KEY'),
        os.getenv('3COMMAS_API_SECRET')
    )
    okx_client = OKXClient(
        os.getenv('OKX_API_KEY'),
        os.getenv('OKX_API_SECRET')
    )

    # 获取欧易历史数据
    data = okx_client.get_historical_data()
    if data is None:
        print("Failed to fetch historical data.")
        return

    # 预测价格
    predicted_price = predict_with_lstm(data)
    print(f"Predicted Price: {predicted_price}")

    # 获取 3Commas 的机器人并更新止盈止损
    bots = three_commas_client.get_bots()
    if bots:
        for bot in bots:
            bot_id = bot['id']
            # 根据预测价格调整止盈止损
            take_profit = predicted_price * 1.02  # 2% 止盈
            stop_loss = predicted_price * 0.98   # 2% 止损
            three_commas_client.update_take_profit_and_stop_loss(bot_id, take_profit, stop_loss)
            print(f"Updated bot {bot_id} with TP: {take_profit}, SL: {stop_loss}")

if __name__ == '__main__':
    main()
