import sys, os
# Projekt-Root zum Pfad hinzufügen, damit 'sequence' gefunden wird
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(top_dir)

import gym
import numpy as np
import torch
from gym import spaces
import pandas as pd
from sequence.sequence_encoder import SequenceEncoder, load_sequences
from sequence.sequence_encoder import WINDOW_SIZE, HIDDEN_DIM, NUM_LAYERS

class TradingEnv(gym.Env):
    def __init__(self, data_path='../data/features/BTC_features.csv'):
        super().__init__()
        # Sequenzdaten laden
        X, _ = load_sequences(data_path)
        self.data = X
        self.n_steps = len(X)
        self.current_step = 0
        # Aktionen: 0=Hold,1=Buy,2=Sell
        self.action_space = spaces.Discrete(3)
        # Beobachtung: Encoder-Ausgabe dim=hidden_dim*2
        obs_dim = HIDDEN_DIM * 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        # Encoder laden
        self.encoder = SequenceEncoder(X.shape[2], HIDDEN_DIM, NUM_LAYERS)
        self.encoder.load_state_dict(torch.load('../models/seq_encoder.pth'))
        self.encoder.eval()

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        seq = torch.tensor(self.data[self.current_step], dtype=torch.float32)
        with torch.no_grad():
            feat = self.encoder(seq.unsqueeze(0)).squeeze(0).numpy()
        return feat

    def step(self, action):
        # Dummy-Reward
        reward = 0.0
        self.current_step += 1
        done = self.current_step >= self.n_steps
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}
