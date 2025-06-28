#!/usr/bin/env python3
# multi_backtest_fixed.py
# – funktional mit allen SB3-Versionen –
# – CPU-only (CUDA-Spam unterdrückt) –

import os, sys, multiprocessing as mp
from pathlib import Path

# ── 0. Umgebungsvariablen ──────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # keine CUDA-Warnungen

# ── 1. Projekt-Root für lokale Imports ─────────────────────────────────
proj_root = Path(__file__).resolve().parent.parent
sys.path.append(str(proj_root))

# ── 2. Standard-Imports ────────────────────────────────────────────────
import numpy as np, pandas as pd, torch, tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from sb3_contrib.ppo_recurrent import RecurrentPPO
from envs.env_sl_tp import AlpacaSLTPEnv

# ── 3. Pfade & Parameter ───────────────────────────────────────────────
CSV            = "../data/features/BTC_features.csv"
MODEL_REPAIRED = "../models/ppo_rl_repaired.zip"
VEC_PATH       = "../models/vecnorm.pkl"
OUT_CSV        = "../results/backtest_50_runs.csv"

N_EPISODES = 50
N_ENVS     = min(64, mp.cpu_count())
SEED_BASE  = 42

# ── 4. Env-Factory ─────────────────────────────────────────────────────
def make_env(df, rank):
    def _init():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        env = AlpacaSLTPEnv(df)
        env.reset(seed=SEED_BASE + rank)
        return env
    _init.__name__ = f"AlpacaSLTPEnv_{rank}"
    return _init

# ── 5. Teil-Reset-Funktion (SB3 < 2.3) ─────────────────────────────────

# ── Teil-Reset (robust für Objekt-Arrays & Dict-Obs) ────────────────
def partial_reset(vec_env, idx_array, obs_holder):
    """Setzt genau die Envs in idx_array zurück und schreibt in-place auf obs_holder."""
    for i in idx_array:
        # env_method gibt eine Liste zurück → erstes Element nehmen
        new_obs = vec_env.env_method("reset", indices=[int(i)])[0]

        # Box-Observation (objekt-Array) …
        if isinstance(obs_holder, np.ndarray):
            obs_holder[int(i)] = new_obs               # direkte Zuweisung reicht

        # Dict-Observation …
        elif isinstance(obs_holder, dict):
            for k in obs_holder.keys():
                obs_holder[k][int(i)] = new_obs[k]

        else:
            raise TypeError(f"Unsupported obs type: {type(obs_holder)}")


# ── 6. Backtest-Loop ───────────────────────────────────────────────────
def run_backtests() -> None:
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        mp.set_start_method("spawn", force=True)

    df = pd.read_csv(CSV, parse_dates=["timestamp"], low_memory=False)

    base_env = (DummyVecEnv([make_env(df, 0)]) if N_ENVS == 1
                else SubprocVecEnv([make_env(df, i) for i in range(N_ENVS)],
                                   start_method=mp.get_start_method()))

    venv = VecNormalize.load(VEC_PATH, base_env)
    venv.training = False
    venv.norm_reward = False

    torch.set_grad_enabled(False)
    model: RecurrentPPO = RecurrentPPO.load(MODEL_REPAIRED, device="cpu")
    model.policy.eval()

    obs         = venv.reset()
    lstm_state  = None
    done_total  = 0
    results     = []

    pbar = tqdm.tqdm(total=N_EPISODES, ncols=90, desc="Backtesting")

    has_reset_done = hasattr(venv, "reset_done")

    while done_total < N_EPISODES:
        act, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)
        obs, _, dones, infos = venv.step(act)

        finished = np.where(dones)[0]
        for idx in finished:
            info   = infos[idx]
            equity = info.get("final_equity", 0.0)
            trades = info.get("num_trades",   0)
            if equity == 0.0:                       # Fallback falls Env nix liefert
                hist = venv.get_attr("equity", indices=[idx])[0]
                equity, trades = (hist[-1], len(hist)) if hist else (0.0, 0)

            results.append((done_total + 1, int(idx), equity, trades))
            pbar.set_postfix(eq=f"{equity:.1f}", trades=trades)
            pbar.update(1)
            done_total += 1

        if finished.size:
            if has_reset_done:                      # SB3 ≥ 2.3
                obs_reset = venv.reset_done()
                obs[dones] = obs_reset
            else:                                   # SB3 < 2.3
                partial_reset(venv, finished, obs)

            if lstm_state is not None:              # LSTM-State nullen
                for s in lstm_state:
                    s[:, dones, :] = 0.0

    pbar.close()
    venv.close()

    df_res = pd.DataFrame(results,
                          columns=["episode", "env_idx", "final_equity", "trades"])
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUT_CSV, index=False)

    print(f"\n✅ {N_EPISODES} Backtests fertig → {OUT_CSV}")
    print(df_res["final_equity"].describe())

# ── 7. Entry-Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    run_backtests()
