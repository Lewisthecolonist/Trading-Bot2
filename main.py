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
    config = Config()  # Assume you have a Config class
    historical_data = load_historical_data()  # Implement this function to load your historical data
    trading_system = TradingSystem(config, historical_data)
    
    await trading_system.start()

    # Example of using wallet functions
    eth_balance, usdt_balance = await trading_system.check_wallet_balance()
    
    to_address = "RECIPIENT_ADDRESS"  # Replace with actual recipient address
    amount = Decimal("0.1")  # 0.1 ETH
    
    gas_estimate, gas_price = await trading_system.estimate_transaction_cost(to_address, amount)
    if gas_estimate and gas_price:
        tx_hash = await trading_system.send_transaction(to_address, amount)
        if tx_hash:
            print(f"Transaction sent successfully: {tx_hash}")

    # Run for a specific duration or until a condition is met
    await asyncio.sleep(config.TRADING_DURATION)
    trading_system.stop()

if __name__ == "__main__":
    asyncio.run(main())
