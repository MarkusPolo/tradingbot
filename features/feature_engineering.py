import pandas as pd
import ta
import os


def load_data(path='../data/BTC_USDT.csv'):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df


def compute_features(df):
    # Preisbewegung
    df['return'] = df['close'].pct_change()

    # Gleitende Durchschnitte (SMA & EMA)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    for span in [5, 8, 20, 21, 50, 200]:
        df[f'ema_{span}'] = ta.trend.ema_indicator(df['close'], window=span)

    # RSI
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

    # MACD
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])

    # Bollinger-Bänder
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    # Average True Range (ATR)
    df['atr_14'] = ta.volatility.average_true_range(
        high=df['high'], low=df['low'], close=df['close'], window=14)

    # On-Balance Volume
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    # Fibonacci Retracement Levels (20er Fenster)
    window = 20
    max_h = df['high'].rolling(window).max()
    min_l = df['low'].rolling(window).min()
    diff = max_h - min_l
    fib_levels = [0.236, 0.382, 0.5, 0.618]
    for level in fib_levels:
        df[f'fib_{int(level*1000) if level!=0.5 else "50"}'] = max_h - diff * level

    # Zeitbasierte Features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek

    # NaN-Werte entfernen
    df = df.dropna()
    return df


if __name__ == '__main__':
    df = load_data()
    df_feat = compute_features(df)
    os.makedirs('../data/features', exist_ok=True)
    df_feat.to_csv('../data/features/BTC_features.csv')
    print('Feature-Dataset gespeichert unter ../data/features/BTC_features.csv')
