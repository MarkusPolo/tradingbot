import pandas as pd
import numpy as np
from env import TradingEnv
from stable_baselines3 import PPO
from tqdm import tqdm
import os

def run_backtest(model_path, features_csv, cash=100_000, debug=True, log_steps=False):
    # 1. Environment & Modell laden
    env = TradingEnv(data_path=features_csv)
    env.initial_cash = cash  # falls unterstützt
    model = PPO.load(model_path)
    obs = env.reset()

    trades = []
    step_logs = []
    step = 0
    DEBUG_INTERVAL = 500

    total_steps = getattr(env, "n_steps", 10_000)  # ✅ fallback falls nicht verfügbar


    for _ in tqdm(range(total_steps), desc="Backtesting"):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if info.get("trade_executed"):
            trades.append({
                "timestamp": info.get("timestamp"),
                "action": action,
                "price": info.get("price"),
                "qty": info.get("qty"),
                "cash": info.get("cash"),
                "position_value": info.get("position_value"),
                "portfolio_value": info.get("portfolio_value"),
                "reward": reward
            })

        if log_steps:
            step_logs.append({
                "timestamp": info.get("timestamp"),
                "action": action,
                "reward": reward,
                "portfolio_value": info.get("portfolio_value")
            })

        if debug and step % DEBUG_INTERVAL == 0:
            print(f"[{info.get('timestamp')}] Step {step} | Action: {action} | "
                  f"Reward: {reward:.6f} | Portfolio: {info.get('portfolio_value', 0):.2f}")

        step += 1
        if done:
            break

    # Ergebnisse
    df_trades = pd.DataFrame(trades)
    pnl = env.portfolio_value - cash
    win_rate = (df_trades.query("reward > 0").shape[0] / len(df_trades)) if len(df_trades) else 0.0

    os.makedirs("../results", exist_ok=True)
    df_trades.to_csv("../results/backtest_trades.csv", index=False)

    print("\n===== Backtest abgeschlossen =====")
    print(f"P&L: {pnl:.2f}")
    print(f"Anzahl Trades: {len(df_trades)}")
    print(f"Win-Rate: {win_rate:.2%}")
    print("Trade-Log gespeichert unter ../results/backtest_trades.csv")

    if log_steps:
        pd.DataFrame(step_logs).to_csv("../results/step_logs.csv", index=False)
        print("Step-Log gespeichert unter ../results/step_logs.csv")

if __name__ == "__main__":
    run_backtest(
        model_path="../models/rl_agent.zip",
        features_csv="../data/features/BTC_features.csv",
        cash=100_000,
        debug=True,
        log_steps=True
    )

