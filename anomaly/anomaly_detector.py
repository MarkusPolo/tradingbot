import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

# Konfiguration
DATA_PATH = '../data/features/BTC_features.csv'
MODEL_DIR = '../models'
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Autoencoder-Modell
def Autoencoder(input_dim):
    class AE(nn.Module):
        def __init__(self, in_dim):
            super(AE, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, in_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)
    return AE(input_dim)


def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    X = df.select_dtypes(include=[np.number]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, df.index


def train_autoencoder(X):
    input_dim = X.shape[1]
    model = Autoencoder(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.Tensor(X))
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0.0
        for batch in loader:
            x_batch = batch[0].to(DEVICE)
            optimizer.zero_grad()
            x_hat = model(x_batch)
            loss = criterion(x_hat, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {epoch_loss/len(dataset):.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'autoencoder.pth'))
    print('Autoencoder-Modell gespeichert unter', os.path.join(MODEL_DIR, 'autoencoder.pth'))
    return model


def detect_anomalies(model, X, scaler, timestamps):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.Tensor(X).to(DEVICE)
        X_hat = model(X_tensor).cpu().numpy()
    errors = np.mean((X_hat - X)**2, axis=1)
    thresh = np.quantile(errors, 0.95)
    anomalies = errors > thresh
    df_anom = pd.DataFrame({'timestamp': timestamps, 'reconstruction_error': errors, 'anomaly': anomalies})
    os.makedirs('../data/anomaly', exist_ok=True)
    df_anom.to_csv('../data/anomaly/BTC_anomalies.csv', index=False)
    print('Anomalie-Daten gespeichert unter ../data/anomaly/BTC_anomalies.csv (Threshold:', round(thresh,6), ')')
    return df_anom


if __name__ == '__main__':
    X, scaler, timestamps = load_data(DATA_PATH)
    model = train_autoencoder(X)
    detect_anomalies(model, X, scaler, timestamps)
