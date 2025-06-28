# data/merge_features.py

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from sequence.sequence_encoder import SequenceEncoder, WINDOW_SIZE, HIDDEN_DIM, NUM_LAYERS

CORE_PATH = BASE / "data/features/BTC_features_core.csv"
EDGE_PATH = BASE / "data/edge/BTC_edge.csv"
ENC_PATH  = BASE / "models/seq_encoder.pth"
OUT_PATH  = BASE / "data/features/BTC_features_merged.csv"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512  # kannst du erhöhen, wenn genug GPU-Speicher

# Daten laden
df = pd.read_csv(CORE_PATH, parse_dates=["timestamp"])
if EDGE_PATH.exists():
    df_edge = pd.read_csv(EDGE_PATH, parse_dates=["timestamp"])
    df = df.merge(df_edge[["timestamp", "edge_score"]], on="timestamp", how="left")
df = df.sort_values("timestamp").reset_index(drop=True)

# Numerische Features normalisieren
num_cols = df.select_dtypes(include=np.number).columns.tolist()
X = df[num_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sliding Windows erzeugen (startet ab WINDOW_SIZE)
X_seq = np.stack([
    X_scaled[i - WINDOW_SIZE:i]
    for i in range(WINDOW_SIZE, len(X_scaled))
])

# Leerer Ziel-Speicher (alle Zeilen, Embedding-Spalten)
embedding_array = np.zeros((len(X_scaled), HIDDEN_DIM * 2), dtype=np.float32)

# Modell laden
model = SequenceEncoder(X_seq.shape[2], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
model.load_state_dict(torch.load(ENC_PATH, map_location=DEVICE))
model.eval()

# Batches verarbeiten und direkt schreiben
with torch.no_grad():
    for i in tqdm(range(0, len(X_seq), BATCH_SIZE), desc="⚡ Batch-Embedding", unit="batch"):
        end = min(i + BATCH_SIZE, len(X_seq))
        batch = torch.tensor(X_seq[i:end], dtype=torch.float32, device=DEVICE)
        out = model(batch).cpu().numpy()  # [B, 2*H]
        embedding_array[i + WINDOW_SIZE : i + WINDOW_SIZE + out.shape[0]] = out

# In DataFrame einfügen (fragmentfrei)
emb_cols = [f"seq_emb_{k}" for k in range(embedding_array.shape[1])]
emb_df = pd.DataFrame(embedding_array, columns=emb_cols)
df = pd.concat([df.reset_index(drop=True), emb_df], axis=1, copy=False)
df = df.copy()  # Speicher defragmentieren

df.fillna(0, inplace=True)
df.to_csv(OUT_PATH, index=False)
print("✅ Features + Embeddings gespeichert →", OUT_PATH)

