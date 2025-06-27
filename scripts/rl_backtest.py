import sys, os
# Projekt-Root zum Pfad hinzufügen, damit 'sequence' gefunden wird
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(top_dir)
import pandas as pd, numpy as np, pickle
from sb3_contrib.ppo_recurrent import RecurrentPPO
from envs.env_sl_tp import AlpacaSLTPEnv

CSV       = "../data/features/BTC_features.csv"
MODEL     = "../models/ppo_rl.zip"
VECNORM   = "../models/vecnorm.pkl"

df = pd.read_csv(CSV, parse_dates=["timestamp"])
env = AlpacaSLTPEnv(df)

# VecNormalize laden → rms, eps, clip
with open(VECNORM, "rb") as f:
    vec = pickle.load(f)
rms, eps, clip = vec.obs_rms, vec.epsilon, vec.clip_obs

def norm(x): return np.clip((x - rms.mean)/np.sqrt(rms.var+eps), -clip, clip)

model: RecurrentPPO = RecurrentPPO.load(MODEL)
obs, _ = env.reset()
obs = norm(obs)
state, done = None, False
equity = []

while not done:
    action, state = model.predict(obs, state=state, deterministic=True)
    obs_raw, reward, done, _, _ = env.step(int(action))
    obs = norm(obs_raw)
    equity.append(env.equity[-1] if env.equity else 0)

print("Final equity:", equity[-1])
print("Trades total:", len(env.equity))

