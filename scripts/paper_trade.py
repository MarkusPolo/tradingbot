from env import TradingEnv
import os
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
from stable_baselines3 import PPO

# API-Keys als ENV-Variablen
API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

# Parametrisierung
desired_qty = 0.001  # Basisgröße für Orders
max_steps = 1000     # maximale Schritte

# Umgebung und Agent
env = TradingEnv()
model = PPO.load('../models/rl_agent', env=env)
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

obs = env.reset()
done = False
step = 0
in_position = False  # Track, ob wir BTC halten

while not done and step < max_steps:
    action, _ = model.predict(obs)

    # Aktion: BUY (öffnen)
    if action == 1 and not in_position:
        try:
            api.submit_order('BTCUSD', desired_qty, 'buy', 'market', 'gtc')
            in_position = True
            print(f"Buy-Order ausgeführt: {desired_qty} BTC.")
        except APIError as e:
            print(f"Buy-Order fehlgeschlagen: {e}")

    # Aktion: SELL (close) nur wenn Position offen
    elif action == 2 and in_position:
        try:
            # Bei Crypto ist Sell ein Close der Position
            api.submit_order('BTCUSD', desired_qty, 'sell', 'market', 'gtc')
            in_position = False
            print(f"Position geschlossen: {desired_qty} BTC verkauft.")
        except APIError as e:
            print(f"Sell-Order fehlgeschlagen: {e}")

    # Halte- oder ungültige Aktion: nichts tun

    step += 1
    obs, _, done, _ = env.step(action)

print(f'Paper Trading abgeschlossen nach {step} Schritten.')
