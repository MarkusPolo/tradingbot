

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))          #  <─  wichtig


# train/train_rl_agent.py
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback

from pathlib import Path
import pandas as pd
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from envs.multifeature_env import MultiFeatureTradingEnv
import torch, os

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data/features/BTC_features_merged.csv"
MODEL_PATH = BASE / "models/ppo_multifeat.zip"
VEC_PATH   = BASE / "models/vecnorm.pkl"
TIMESTEPS  = 300_000
LOOKBACK   = 100

policy_kwargs = dict(
    net_arch=[dict(pi=[256,128], vf=[256,128])],
    lstm_hidden_size=256,
    shared_lstm=True,
    enable_critic_lstm=False,
)

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="🏋️ RL Training", unit="steps")

    def _on_step(self) -> bool:
        self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


def make_env(df_slice):
    return lambda: MultiFeatureTradingEnv(df_slice)

def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
    split = int(0.8 * len(df))
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    # ---------- VecEnv -----------------------------------------------------
    n_envs = os.cpu_count() // 2
    env = SubprocVecEnv([make_env(train_df) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentPPO(
        RecurrentActorCriticPolicy,
        env=env,
        verbose=1,
        n_steps=1024,
        batch_size=1024,
        learning_rate=1e-4,
        ent_coef=0.01,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        device=device,
    )

    callback = TqdmCallback(total_timesteps=TIMESTEPS)
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    model.save(MODEL_PATH)
    env.save(VEC_PATH)
    print("✅ Modell & Normalizer gespeichert")

if __name__ == "__main__":
    main()
