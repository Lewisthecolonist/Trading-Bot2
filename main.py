import time
import pandas as pd
from config import Config
import zipfile
import io
from trading_system import TradingSystem
import asyncio
import decimal as Decimal
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def load_historical_data(filename='historical_data.csv.zip'):
    with zipfile.ZipFile(filename) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(io.BytesIO(f.read()))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

async def main():
    logger.debug("Starting application...")
    try:
        config = Config()
        logger.debug("Config loaded")
        
        logger.debug("Loading historical data...")
        historical_data = load_historical_data()
        logger.debug(f"Historical data loaded: {len(historical_data)} rows")
        
        logger.debug("Initializing trading system...")
        trading_system = TradingSystem(config, historical_data)
        
        logger.debug("Starting trading system...")
        await trading_system.start()
        
        logger.debug("Starting main loop...")
        await trading_system.main_loop()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

# Add this to main.py temporarily
def verify_data():
    try:
        data = load_historical_data()
        print(f"Data columns: {data.columns}")
        print(f"First row: {data.iloc[0]}")
    except Exception as e:
        print(f"Data verification failed: {e}")

# Call before main()
verify_data()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down by user request...")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)        logger.error(f"Startup error: {str(e)}", exc_info=True)    