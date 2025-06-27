#!/usr/bin/env python3
"""
Parallel-Backtest von 50 Episoden für den PPO-Recurrent-Agent
------------------------------------------------------------
• nutzt SubprocVecEnv mit allen verfügbaren CPU-Kernen (max 8)
• zeigt Live-Fortschritt via tqdm
• schreibt ausführliche Episoden-Metriken in CSV
"""

import sys, os
# Projekt-Root zum Pfad hinzufügen, damit 'sequence' gefunden wird
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(top_dir)

import os, pickle, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd, torch, tqdm
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from envs.env_sl_tp import AlpacaSLTPEnv

# ----- Pfade & Parameter -----
CSV         = "../data/features/BTC_features.csv"
MODEL_PATH  = "../models/ppo_rl.zip"
VEC_PATH    = "../models/vecnorm.pkl"
OUT_CSV     = "../results/backtest_50_runs.csv"

N_EPISODES  = 50
MAX_CPU     = 8
N_ENVS      = 8  # 1..8
SEED_BASE   = 42

# suppress TensorFlow spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def make_env(df, rank: int):
    def _init():
        env = AlpacaSLTPEnv(df)
        env.reset(seed=SEED_BASE + rank)
        return env
    return _init


def run_backtests():
    mp.set_start_method("spawn", force=True)     # notwendig unter WSL/Win

    df = pd.read_csv(CSV, parse_dates=["timestamp"])

    # VecNorm laden
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)
    rms, eps, clip = vec.obs_rms, vec.epsilon, vec.clip_obs
    norm = lambda x: np.clip((x - rms.mean)/np.sqrt(rms.var+eps), -clip, clip)

    # Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: RecurrentPPO = RecurrentPPO.load(MODEL_PATH, device=device)

    # VecEnv
    if N_ENVS == 1:
        venv = DummyVecEnv([make_env(df, 0)])
    else:
        venv = SubprocVecEnv([make_env(df, i) for i in range(N_ENVS)])

    # Initialobs
    obs = norm(venv.reset())
    state = None
    episode_results, episodes_done = [], 0

    pbar = tqdm.tqdm(total=N_EPISODES, desc="Backtesting", ncols=80)

    while episodes_done < N_EPISODES:
        actions, state = model.predict(obs, state=state, deterministic=True)
        new_obs, _, dones, infos = venv.step(actions)
        obs = np.vstack([norm(o) for o in new_obs])

        for idx, done in enumerate(dones):
            if not done:
                continue
            equity_hist = venv.get_attr("equity", indices=[idx])[0]
            equity = equity_hist[-1] if equity_hist else 0.0
            trades = len(env.equity)
            episode_results.append({
                "episode": episodes_done + 1,
                "env_idx": idx,
                "final_equity": equity,
                "trades": trades
            })
            pbar.set_postfix(eq=f"{equity:.1f}", trades=trades)
            pbar.update(1)
            episodes_done += 1

            # reset_done() gibt neue Beobachtungen nur für DONE-Envs zurück
            new_obs_done = venv.reset_done()
            # ersetze die Zeilen in obs an den DONE-Indizes
            obs[dones] = norm(new_obs_done)
            if state is not None:
                for s in state:              # hidden & cell nullen
                    s[:, dones, :] = 0.0

    pbar.close()

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_results).to_csv(OUT_CSV, index=False)
    print("\n✅ 50-Run-Backtest fertig →", OUT_CSV)
    print(pd.DataFrame(episode_results)["final_equity"].describe())


if __name__ == "__main__":
    run_backtests()
