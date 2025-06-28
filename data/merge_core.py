# data/merge_core.py
"""
Schreibt BTC_features_core.csv (Features + Regime + Anomaly)
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
df_feat = pd.read_csv(BASE/"data/features/BTC_features.csv", parse_dates=["timestamp"])
df_reg  = pd.read_csv(BASE/"data/regime/BTC_regimes.csv", parse_dates=["timestamp"])
df_anom = pd.read_csv(BASE/"data/anomaly/BTC_anomalies.csv", parse_dates=["timestamp"])

df = (df_feat
      .merge(df_reg[["timestamp","regime"]], on="timestamp", how="left")
      .merge(df_anom[["timestamp","reconstruction_error","anomaly"]], on="timestamp", how="left")
      .fillna(0)
      .sort_values("timestamp")
      .reset_index(drop=True))

out = BASE/"data/features/BTC_features_core.csv"
df.to_csv(out, index=False)
print("✅ Core-Features →", out)
