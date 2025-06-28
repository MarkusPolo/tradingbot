# data/merge_features.py

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))          #  <─  wichtig

"""
Merged CSV mit allen Ebenen (Features, Regime, Anomaly, Edge-Score, Embedding)
schreibt ../data/features/BTC_features_merged.csv
"""
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sequence.sequence_encoder import SequenceEncoder, WINDOW_SIZE, HIDDEN_DIM, NUM_LAYERS

BASE_DIR   = Path(__file__).resolve().parents[1]
FP_FEAT    = BASE_DIR / "data/features/BTC_features.csv"
FP_REGIME  = BASE_DIR / "data/regime/BTC_regimes.csv"
FP_ANOM    = BASE_DIR / "data/anomaly/BTC_anomalies.csv"
FP_EDGE    = BASE_DIR / "data/edge/BTC_edge.csv"        # optional, wenn vorhanden
OUT_CSV    = BASE_DIR / "data/features/BTC_features_merged.csv"
ENC_PATH   = BASE_DIR / "models/seq_encoder.pth"

def merge():
    df = pd.read_csv(FP_FEAT, parse_dates=["timestamp"])
    df_reg = pd.read_csv(FP_REGIME, parse_dates=["timestamp"])
    df_anom = pd.read_csv(FP_ANOM,  parse_dates=["timestamp"])

    df = (df
          .merge(df_reg[["timestamp","regime"]], on="timestamp", how="left")
          .merge(df_anom[["timestamp","reconstruction_error","anomaly"]], on="timestamp", how="left"))

    if FP_EDGE.exists():
        df_edge = pd.read_csv(FP_EDGE, parse_dates=["timestamp"])
        df = df.merge(df_edge[["timestamp","edge_score"]], on="timestamp", how="left")

    # --- Sequenz-Embeddings --------------------------------------------------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols].values)

    enc = SequenceEncoder(len(num_cols), HIDDEN_DIM, NUM_LAYERS)
    enc.load_state_dict(torch.load(ENC_PATH, map_location="cpu"))
    enc.eval()

    embs = []
    with torch.no_grad():
        for i in range(len(df)):
            if i < WINDOW_SIZE:
                embs.append(np.zeros(HIDDEN_DIM*2))
            else:
                win = X_scaled[i-WINDOW_SIZE:i]
                emb = enc(torch.tensor(win[None,...], dtype=torch.float32))
                embs.append(emb.numpy().flatten())
    emb_arr = np.vstack(embs)
    emb_cols = [f"seq_emb_{k}" for k in range(emb_arr.shape[1])]
    df[emb_cols] = emb_arr

    df.fillna(0, inplace=True)
    df.to_csv(OUT_CSV, index=False)
    print("✅ Merged Features →", OUT_CSV)

if __name__ == "__main__":
    merge()
