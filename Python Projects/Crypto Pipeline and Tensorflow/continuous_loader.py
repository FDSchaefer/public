import os
import time
import logging
from datetime import datetime, timedelta
from BinanceTimescaleDB import BinanceTimescaleDB

os.makedirs('/logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/loader.log'),
        logging.StreamHandler()
    ]
)

def main():
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'crypto_data'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password')
    }
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    update_interval = int(os.getenv('UPDATE_INTERVAL', 60))
    
    db = BinanceTimescaleDB(db_params)
    
    logging.info("Starting continuous loader")
    logging.info(f"Symbols: {symbols}")
    logging.info(f"Update interval: {update_interval}s")
    
    # Initial setup
    while True:
        try:
            db.connect()
            db.setup_database()
            break
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            time.sleep(5)
    
    # Continuous loop
    while True:
        try:
            for symbol in symbols:
                end_date = datetime.now()
                start_date = start_date = '2025-01-01'
                
                logging.info(f"Updating {symbol}")
                
                kline_data = db.download_klines(
                    symbol=symbol,
                    interval='1m',
                    start_date=start_date,
                    end_date=end_date,
                    force_override=False
                )
                
                if kline_data:
                    db.insert_data(symbol, kline_data)
                
                time.sleep(1)
            
            logging.info(f"Sleeping for {update_interval}s")
            time.sleep(update_interval)
            
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(10)
            try:
                db.close()
                db.connect()
            except:
                pass

if __name__ == "__main__":
    main()