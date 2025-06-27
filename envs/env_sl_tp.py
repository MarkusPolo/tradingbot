# env_sl_tp.py
# -------------------------------------------------------------
# Stop-Loss/Take-Profit-Environment mit Reward-Shaping
# – vollständig auf dein neues Feature-Set angepasst –
# -------------------------------------------------------------
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# ────────────────  Hyper-Parameter  ────────────────
LOOKBACK       = 100         # Bars je Observation-Window
SL_PCT         = 0.005       # 0 .5 % Stop-Loss
TP_PCT         = 0.01        # 1 % Basis-Take-Profit
PENALTY_DD     = 0.6         # Drawdown-Gewicht
PENALTY_VOL    = 0.25        # Equity-Volatilitäts-Penalty
HOLD_PEN       = 1/500       # Strafe pro Bar ohne Position
FLIP_PEN       = 0.0004      # Strafe für jeden Flip-Trade
ALPHA_LOSS     = 1.5         # Verluste höher bestrafen
BETA_GAIN      = 1           # Gewinne 1 : 1 belohnen
BONUS_FACTOR   = 0.0005      # End-Bonus in % des Final-Equity
DTP_LOOKBACK   = 20          # Fenster für dyn. TP-Vol-Check
DTP_MAX_FACTOR = 2.0         # Zusätzl. TP-Faktor bei Ruhe

# ────────────────  Feature-Definition  ────────────────
FEATURE_COLS = [
    "open","high","low","close","volume","return",
    "sma_20","ema_5","ema_8","ema_20","ema_21","ema_50","ema_200",
    "rsi_14","macd","macd_signal",
    "bb_high","bb_low",
    "atr_14","obv",
    "fib_236","fib_382","fib_50","fib_618",
    "hour","dayofweek",
]
# 27 Features  →  obs-shape = (27 * LOOKBACK,)

# ────────────────  Hilfs-Funktionen  ────────────────
def dynamic_tp(equity_hist: list[float], base_tp: float = TP_PCT) -> float:
    """Passt TP an Equity-Volatilität der letzten DTP_LOOKBACK Bars an."""
    if len(equity_hist) < DTP_LOOKBACK + 1:
        return base_tp
    window = equity_hist[-(DTP_LOOKBACK + 1):]
    returns = np.diff(window)
    vol = np.std(returns) / (np.mean(np.abs(window)) + 1e-8)
    factor = 1 + DTP_MAX_FACTOR * max(0.0, 1 - vol)
    return base_tp * factor


def compute_reward(
    position: int,
    price: float,
    prev_price: float,
    entry_price: float,
    closing_action: bool,
    hold_time: int,
    equity_hist: list[float],
    action_changed: bool,
) -> float:
    """Reward in Prozentpunkten (×100) unter Berücksichtigung aller Penalties."""
    ret = (price - prev_price) / prev_price
    base = BETA_GAIN * ret if position * ret >= 0 else ALPHA_LOSS * ret
    reward = base * position - 0.0002  # fixe Transaktionskosten

    # Realisierte PnL beim Schließen additiv
    if closing_action and entry_price > 0:
        tot = (price - entry_price) / entry_price * position
        reward += tot

    # Flip-Penalty
    if action_changed:
        reward -= FLIP_PEN

    # Drawdown-Penalty
    if equity_hist:
        peak = max(equity_hist)
        dd = min(0.0, (equity_hist[-1] - peak) / (abs(peak) + 1e-8))
        reward += PENALTY_DD * dd

    # Equity-Vol-Penalty
    if len(equity_hist) >= DTP_LOOKBACK + 1:
        window = equity_hist[-(DTP_LOOKBACK + 1):]
        vol = np.std(np.diff(window)) / (np.mean(np.abs(window)) + 1e-8)
        reward -= PENALTY_VOL * vol

    # Halte-Penalty
    reward -= hold_time * HOLD_PEN
    return float(reward * 100)  # Basispunkte → Prozentpunkte


# ────────────────  Environment  ────────────────
class AlpacaSLTPEnv(gym.Env):
    """Discrete-Action Env<br>0 = flat/close · 1 = long · 2 = short"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        assert set(["timestamp"] + FEATURE_COLS).issubset(df.columns), "Spalten fehlen"
        self.df = (
            df.sort_values("timestamp")
              .reset_index(drop=True)
              .loc[:, ["timestamp"] + FEATURE_COLS]
              .astype({"hour": "int8", "dayofweek": "int8"})
        )
        assert len(self.df) > LOOKBACK + 1, "DataFrame zu kurz"

        n_feat = len(FEATURE_COLS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_feat * LOOKBACK,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.reset()

    # ────── Gym-Setup ──────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx    = self.np_random.integers(LOOKBACK, len(self.df) - 1)
        self.position    = 0             # 0 flat, 1 long, −1 short
        self.entry_price = 0.0
        self.hold_time   = 0
        self.balance     = 0.0
        self.equity      = []
        obs = self._obs()
        return obs, {}

    def step(self, action: int):
        price      = float(self.df["close"].iat[self.step_idx])
        prev_price = float(self.df["close"].iat[self.step_idx - 1])
        closing    = (action == 0 and self.position != 0)

        # Flip-Erkennung
        action_changed = (
            (action == 0 and self.position != 0) or
            (action == 1 and self.position != 1) or
            (action == 2 and self.position != -1)
        )

        # dynamischer TP
        pnl = (price - self.entry_price) / self.entry_price * self.position if self.position else 0
        if self.position and pnl >= dynamic_tp(self.equity):
            action, closing, action_changed = 0, True, True
        # fester SL/TP
        if self.position and (pnl <= -SL_PCT or pnl >= TP_PCT):
            action, closing, action_changed = 0, True, True

        reward = compute_reward(
            self.position, price, prev_price,
            self.entry_price, closing,
            self.hold_time, self.equity,
            action_changed
        )

        # Idle-Penalty
        if self.position == 0 and action == 0:
            reward -= 0.001

        # Order-Ausführung
        if action == 1 and self.position == 0:
            self.position, self.entry_price, self.hold_time = 1, price, 0
        elif action == 2 and self.position == 0:
            self.position, self.entry_price, self.hold_time = -1, price, 0
        elif closing:
            self.balance += (price - self.entry_price) * self.position
            self.position, self.hold_time = 0, 0

        # Mark-to-Market
        unreal = (price - self.entry_price) * self.position
        self.equity.append(self.balance + unreal)

        # Zeitschritt vor
        self.step_idx += 1
        terminated = self.step_idx >= len(self.df) - 1

        if self.position:
            self.hold_time += 1

        # End-Bonus
        if terminated:
            reward += BONUS_FACTOR * (self.equity[-1] if self.equity else 0)

        obs = self._obs()
        return obs, float(reward), terminated, False, {}

    # ────── Hilfsmethode ──────
    def _obs(self) -> np.ndarray:
        window = self.df.iloc[self.step_idx - LOOKBACK:self.step_idx]
        return (
            window[FEATURE_COLS]
            .to_numpy(dtype=np.float32)
            .flatten()
        )

