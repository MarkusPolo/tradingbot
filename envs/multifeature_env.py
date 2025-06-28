# envs/multifeature_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

LOOKBACK = 100
COST_BPS = 2      # 0.02 %

class MultiFeatureTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        assert len(df) > LOOKBACK + 1, "DataFrame zu kurz"
        self.df = df.reset_index(drop=True)
        self.n_feat = df.shape[1] - 1        # timestamp raus
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(LOOKBACK * self.n_feat,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx         = self.np_random.integers(LOOKBACK, len(self.df)-1)
        self.pos         = 0
        self.entry_price = 0.0
        self.balance     = 0.0
        self.equity      = []
        return self._obs(), {}

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        price      = float(self.df["close"].iat[self.idx])
        price_prev = float(self.df["close"].iat[self.idx - 1])

        closing = (action == 0 and self.pos != 0)
        pnl     = ((price - self.entry_price) / self.entry_price * self.pos) if self.entry_price else 0.0

        # ---------- Reward (Basis-BPS) --------------------------------------
        reward = (price - price_prev) / price_prev * self.pos
        reward -= COST_BPS / 10000     # Handelskosten
        if closing and self.entry_price:
            reward += pnl

        # ---------- Aktionen ausführen --------------------------------------
        if action == 1 and self.pos == 0:             # Long
            self.pos, self.entry_price = 1, price
        elif action == 2 and self.pos == 0:           # Short
            self.pos, self.entry_price = -1, price
        elif closing:
            self.balance += pnl * self.entry_price
            self.pos, self.entry_price = 0, 0.0

        unreal = (price - self.entry_price) * self.pos
        self.equity.append(self.balance + unreal)

        self.idx += 1
        done = self.idx >= len(self.df) - 1
        return self._obs(), float(reward * 100), done, False, {}

    # ------------------------------------------------------------------ obs
    def _obs(self):
        win = self.df.iloc[self.idx - LOOKBACK: self.idx]
        return win.drop(columns=["timestamp"]).to_numpy(np.float32).flatten()
