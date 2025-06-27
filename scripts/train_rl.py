import sys, os
# Projekt-Root zum Pfad hinzufügen, damit 'sequence' gefunden wird
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(top_dir)
# train_rl.py
import multiprocessing as mp
from pathlib import Path
import os, pickle
import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import torch
from envs.env_sl_tp import AlpacaSLTPEnv, LOOKBACK
# ---------- Konfiguration ----------
CSV         = "../data/features/BTC_features.csv"   # <– deine Datei
MODEL_DIR   = "../models"
MODEL_PATH  = Path(MODEL_DIR) / "ppo_rl.zip"
VEC_PATH    = Path(MODEL_DIR) / "vecnorm.pkl"
TIMESTEPS   = 400_000
NUM_ENVS    = 8             # → kannst du auch 1 setzen
# ------------------------------------

def make_env(df):
    return lambda: AlpacaSLTPEnv(df)

def main():
    mp.set_start_method("spawn", force=True)  # robust unter WSL/Windows

    df = pd.read_csv(CSV, parse_dates=["timestamp"])

    # -------- VecEnv ----------
    if NUM_ENVS == 1:
        venv = DummyVecEnv([make_env(df)])
    else:
        venv = SubprocVecEnv([make_env(df) for _ in range(NUM_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    # -------- Modell ----------
    policy_kwargs = dict(
        net_arch=dict(          # ←  nicht mehr [dict(...)]
            pi=[128, 128, 64],
            vf=[128, 128, 64],
        ),
        lstm_hidden_size=256,
        shared_lstm=True,       # ein gemeinsames LSTM
        enable_critic_lstm=False,  # Critic teilt dasselbe LSTM
    )



    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Device:", device)

    model = RecurrentPPO(
        RecurrentActorCriticPolicy,
        env=venv,
        device=device,
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.02,
        n_steps=1024,
        batch_size=1024,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
    )

    print("⏳ Training …")
    model.learn(total_timesteps=TIMESTEPS)

    # -------- Speichern --------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(str(MODEL_PATH))
    with open(VEC_PATH, "wb") as f:
        pickle.dump(venv, f)
    print("✅ Modell & VecNormalize gespeichert:", MODEL_PATH, VEC_PATH)

if __name__ == "__main__":
    main()
