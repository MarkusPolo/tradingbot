import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import sys

# Exchange und Symbol
echange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})
symbol = 'BTC/USDT'
timeframe = '1m'

# Hilfsfunktion: In Millisekunden
def to_millis(dt):
    return int(dt.timestamp() * 1000)

# Daten in Blöcken holen bis zum aktuellen Zeitpunkt mit Debugging
def fetch_full_ohlcv(symbol, since, timeframe='1m', limit=1000):
    all_data = []
    iteration = 0
    start_time = time.time()
    print(f"Starte Datensammlung ab {datetime.utcfromtimestamp(since/1000)} UTC...", flush=True)
    while True:
        iteration += 1
        try:
            data = echange.fetch_ohlcv(symbol, since=since, limit=limit, timeframe=timeframe)
        except Exception as e:
            print(f"Fehler beim Abruf: {e}. Warte 5 Sekunden und versuche erneut...", flush=True)
            time.sleep(5)
            continue
        if not data:
            print("Keine weiteren Daten erhalten, Abbruch der Schleife.", flush=True)
            break
        batch_count = len(data)
        last_ts = data[-1][0]
        all_data.extend(data)
        since = last_ts + 1
        elapsed = time.time() - start_time
        print(f"Iteration {iteration}: {batch_count} Kerzen von bis {datetime.utcfromtimestamp(data[0][0]/1000)} bis {datetime.utcfromtimestamp(last_ts/1000)} (insgesamt {len(all_data)}). Laufzeit: {elapsed:.1f}s", flush=True)
        time.sleep(0.2)
        # Abbruch, wenn aktueller Stand erreicht
        if datetime.utcfromtimestamp(last_ts/1000) >= datetime.utcnow() - timedelta(minutes=1):
            print("Aktuelles Zeitfenster erreicht, Beende Datensammlung.", flush=True)
            break
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == '__main__':
    end = datetime.utcnow()
    start = end - timedelta(days=730)
    since = to_millis(start)
    
    df = fetch_full_ohlcv(symbol, since, timeframe=timeframe)
    path = '../data/BTC_USDT.csv'
    df.to_csv(path, index=False)
    print(f'Daten gespeichert: {path}')
