import time
import pandas as pd
from config import Config
import zipfile
import io
from trading_system import TradingSystem
import asyncio
import decimal as Decimal

def load_historical_data(filename='historical_data.csv.zip'):
    with zipfile.ZipFile(filename) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(io.BytesIO(f.read()))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df
async def main():
    config = Config()
    historical_data = load_historical_data()
    trading_system = TradingSystem(config, historical_data)
    await trading_system.start()
    await trading_system.main_loop()

if __name__ == "__main__":
    asyncio.run(main())
    import tracemalloc
    tracemalloc.start()
