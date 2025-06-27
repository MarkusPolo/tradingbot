import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

# Konfiguration
DATA_PATH = '../data/features/BTC_features.csv'
MODEL_DIR = '../models'
WINDOW_SIZE = 60       # Input-Sequenzlänge in Kerzen
HIDDEN_DIM = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers):
        super(SequenceEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        out, _ = self.lstm(x)
        # letzter Zeitschritt beider Richtungen
        return out[:, -1, :]


def load_sequences(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Features normalisieren
    features = df.select_dtypes(include=[np.number]).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # Sliding-window erzeugen
    sequences = []
    for i in range(0, len(features) - WINDOW_SIZE, 10):
        sequences.append(features[i : i + WINDOW_SIZE])
    X = np.stack(sequences)

    return X, scaler


def train_encoder():
    X, scaler = load_sequences(DATA_PATH)
    n_samples, seq_len, feat_dim = X.shape

    model = SequenceEncoder(feat_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # DataLoader mit GPU-Optimierungen
    tensor_data = torch.tensor(X, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type=='cuda'),
        num_workers=4
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for (batch,) in loader:
            x_batch = batch.to(DEVICE)
            optimizer.zero_grad()
            encoded = model(x_batch)
            # Dummy-Ziel: Nullen
            loss = criterion(encoded, torch.zeros_like(encoded))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {total_loss/len(dataset):.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'seq_encoder.pth'))
    print("Sequenz-Encoder gespeichert unter", os.path.join(MODEL_DIR, 'seq_encoder.pth'))
    return model, scaler

if __name__ == '__main__':
    train_encoder()
