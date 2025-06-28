# sequence/train_encoder.py
"""
Trainiert SequenceEncoder auf BTC_features_core.csv
"""
from pathlib import Path
import sys, numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))
from sequence.sequence_encoder import SequenceEncoder, WINDOW_SIZE, HIDDEN_DIM, NUM_LAYERS

DATA_PATH = BASE/"data/features/BTC_features_core.csv"
MODEL_OUT = BASE/"models/seq_encoder.pth"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH     = 32
EPOCHS    = 20
LR        = 1e-3

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
num_cols = df.select_dtypes("number").columns
X = df[num_cols].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

seqs = [X[i:i+WINDOW_SIZE] for i in range(0, len(X)-WINDOW_SIZE, 10)]
X = np.stack(seqs)
loader = torch.utils.data.DataLoader(
    torch.tensor(X, dtype=torch.float32),
    batch_size=BATCH, shuffle=True, pin_memory=(DEVICE.type=="cuda")
)

model = SequenceEncoder(X.shape[2], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

model.train()
for epoch in range(1, EPOCHS+1):
    tot = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optim.zero_grad()
        out = model(batch)
        loss = loss_fn(out, torch.zeros_like(out))
        loss.backward(); optim.step()
        tot += loss.item()*batch.size(0)
    print(f"Epoch {epoch}/{EPOCHS}  Loss {tot/len(loader.dataset):.6f}")

MODEL_OUT.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
print("✅ Neuer Encoder gespeichert →", MODEL_OUT)

