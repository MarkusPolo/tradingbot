import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os


def load_features(path='../data/features/BTC_features.csv'):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df


def detect_regimes(df, n_clusters=3):
    # Wähle Features für Clustering: Rendite und Volatilität
    df['volatility'] = df['close'].rolling(window=20).std()
    df_reg = df[['return', 'volatility']].dropna()

    # Standardisierung
    scaler = StandardScaler()
    X = scaler.fit_transform(df_reg)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    df_reg['regime'] = labels
    return df_reg, kmeans


if __name__ == '__main__':
    df = load_features()
    df_reg, model = detect_regimes(df)
    # Index als Spalte zurücksetzen, damit CSV eine 'timestamp'-Spalte enthält
    df_reg = df_reg.reset_index()
    os.makedirs('../data/regime', exist_ok=True)
    # Speichern der Regime-Daten mit Timestamp-Spalte
    df_reg.to_csv('../data/regime/BTC_regimes.csv', index=False)
    # Modell speichern
    import joblib
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/kmeans_regime.pkl')
    print('Regime-Daten gespeichert unter ../data/regime/BTC_regimes.csv')
    print('KMeans-Modell gespeichert unter ../models/kmeans_regime.pkl')
