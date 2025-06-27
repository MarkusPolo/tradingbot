# Stelle sicher, dass `scripts`-Ordner im PYTHONPATH ist oder importiere env relativ:
from env import TradingEnv  # statt 'from scripts.env'
from stable_baselines3 import PPO
import os

# RL-Agent Training

def train_rl():
    env = TradingEnv()
    model = PPO('MlpPolicy', env, verbose=1, device='cpu')
    model.learn(total_timesteps=100_000)
    os.makedirs('../models', exist_ok=True)
    model.save('../models/rl_agent')
    print('RL-Agent gespeichert unter ../models/rl_agent.zip')

if __name__ == '__main__':
    # Führe das Script aus aus dem `scripts`-Ordner mit: `python3 rl_agent.py`
    train_rl()
