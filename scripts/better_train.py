#!/usr/bin/env python3
# train_rl_cpu64.py
#
# Vollständiges Beispiel für reines-CPU-Training mit 64 parallelen Envs.
# Tested: Python 3.10, PyTorch ≥ 2.1, SB3 ≥ 2.2


import sys, os
# Projekt-Root zum Pfad hinzufügen, damit 'sequence' gefunden wird
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(top_dir)

# ───────────── 1. BEFORE torch import: Thread-Budget Hauptprozess ─────────────
import os, sys
os.environ["OMP_NUM_THREADS"] = "32"     # Hauptprozess: 32 Threads fürs NN
os.environ["MKL_NUM_THREADS"] = "32"

# ────────────────────────── Standard-Imports ──────────────────────────
import pickle, multiprocessing as mp
from pathlib import Path
import pandas as pd
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from envs.env_sl_tp import AlpacaSLTPEnv

# ───────────── 2. Allgemeine Konfiguration ─────────────
CSV         = "../data/features/BTC_features.csv"
MODEL_DIR   = "../models"
MODEL_PATH  = Path(MODEL_DIR) / "ppo_rl.zip"
VEC_PATH    = Path(MODEL_DIR) / "vecnorm.pkl"

TOTAL_STEPS = 400_000
N_ENVS      = min(64, mp.cpu_count())   # 64 parallele Prozesse
N_STEPS     = 256                       # → 16 384 Samples je Policy-Update

# ───────────── 3. Multiprocessing-Methode ─────────────
try:
    mp.set_start_method("fork", force=True)   # Linux/WSL2: Copy-on-Write
except RuntimeError:
    mp.set_start_method("spawn", force=True)  # Windows-Fallback

# ───────────── 4. Thread-Budget *nur* im Hauptprozess setzen ─────────────
torch.set_num_threads(32)
try:
    torch.set_num_interop_threads(4)
except RuntimeError:
    pass                            # Falls schon gesetzt, ignorieren

# ───────────── 5. Hilfs-Closure für jede Env ─────────────
def make_env(df, rank: int):
    """Erzeugt ein Alpaca-Env; Kinderprozesse laufen single-threaded."""
    def _init():
        # Keine Torch-Aufrufe hier – nur Env-Threads begrenzen!
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        return AlpacaSLTPEnv(df)
    _init.__name__ = f"AlpacaSLTPEnv_{rank}"
    return _init

# ───────────── 6. Hauptfunktion ─────────────
def main() -> None:
    # ----- Daten einmal laden -----
    df = pd.read_csv(CSV, parse_dates=["timestamp"], low_memory=False)

    # ----- VecEnv aufbauen -----
    if N_ENVS == 1:
        venv = DummyVecEnv([make_env(df, 0)])
    else:
        venv = SubprocVecEnv(
            [make_env(df, i) for i in range(N_ENVS)],
            start_method=mp.get_start_method(),
        )
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    # ----- Policy-Architektur -----
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        lstm_hidden_size=256,
        shared_lstm=True,
        enable_critic_lstm=False,
    )

    model = RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=venv,
        device="cpu",
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.02,
        n_steps=N_STEPS,
        batch_size=(N_STEPS * N_ENVS) // 2,   # 8 192
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
    )

    # PyTorch 2.x – Graph-Kompilierung (macht auch CPU schneller)
    if hasattr(torch, "compile"):
        model.policy = torch.compile(model.policy, mode="max-autotune")

    print(f"🚀 Training startet …  Envs: {N_ENVS}  |  Haupt-Threads: 32")
    model.learn(total_timesteps=TOTAL_STEPS)
    print("✅ Training abgeschlossen")

    # ----- Model speichern -----
    Path(MODEL_DIR).mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    with open(VEC_PATH, "wb") as f:
        pickle.dump(venv, f)
    print(f"💾 Gespeichert unter: {MODEL_PATH}  und  {VEC_PATH}")

# ───────────── 7. Entry-Point ─────────────
if __name__ == "__main__":
    main()
