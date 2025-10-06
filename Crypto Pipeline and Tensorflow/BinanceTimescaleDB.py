"""
TimescaleDB Crypto Data Downloader and Storage System

Prerequisites:
1. Docker installed and running
2. Python packages: pip install requests pandas psycopg2-binary

Setup:
1. Start TimescaleDB: docker-compose up -d
2. Run this script: python script.py
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import psycopg2
from psycopg2.extras import execute_values

class BinanceTimescaleDB:
    def __init__(self, db_params):
        """
        Initialize connection to TimescaleDB.
        
        db_params should be a dict with: host, port, database, user, password
        """
        self.db_params = db_params
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.conn = None
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            print("✓ Connected to TimescaleDB")
        except Exception as e:
            print(f"✗ Error connecting to database: {e}")
            raise
    
    def setup_database(self):
        """Create tables and hypertables for crypto data."""
        with self.conn.cursor() as cur:
            # Create the main table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crypto_ohlcv (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    quote_volume DOUBLE PRECISION,
                    trades INTEGER,
                    taker_buy_base DOUBLE PRECISION,
                    taker_buy_quote DOUBLE PRECISION,
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            # Convert to hypertable (time-series optimized)
            cur.execute("""
                SELECT create_hypertable('crypto_ohlcv', 'time', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Create indexes for better query performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_time 
                ON crypto_ohlcv (symbol, time DESC);
            """)
            
            # Enable compression (saves storage for old data)
            cur.execute("""
                ALTER TABLE crypto_ohlcv SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol'
                );
            """)
            
            # Add compression policy (compress data older than 7 days)
            cur.execute("""
                SELECT add_compression_policy('crypto_ohlcv', 
                    compress_after => INTERVAL '7 days',
                    if_not_exists => TRUE
                );
            """)
            
            # Create continuous aggregate for 5-minute candles
            cur.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS crypto_ohlcv_5m
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('5 minutes', time) AS bucket,
                    symbol,
                    FIRST(open, time) as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    LAST(close, time) as close,
                    SUM(volume) as volume,
                    SUM(quote_volume) as quote_volume,
                    SUM(trades) as trades
                FROM crypto_ohlcv
                GROUP BY bucket, symbol
                WITH NO DATA;
            """)
            
            # Add refresh policy for continuous aggregate
            cur.execute("""
                SELECT add_continuous_aggregate_policy('crypto_ohlcv_5m',
                    start_offset => INTERVAL '1 day',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)
            
            self.conn.commit()
            print("✓ Database schema created successfully")
            print("  - Hypertable: crypto_ohlcv")
            print("  - Compression enabled (7 days)")
            print("  - Continuous aggregate: crypto_ohlcv_5m")
    
    def get_existing_data_range(self, symbol):
        """Get the date range of existing data for a symbol."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    MIN(time) as earliest,
                    MAX(time) as latest,
                    COUNT(*) as count
                FROM crypto_ohlcv
                WHERE symbol = %s
            """, (symbol,))
            
            result = cur.fetchone()
            if result and result[0]:
                return {
                    'earliest': result[0],
                    'latest': result[1],
                    'count': result[2]
                }
            return None
    
    def find_missing_ranges(self, symbol, start_date, end_date):
        """
        Find date ranges that are missing in the database.
        Returns list of (start, end) tuples for missing ranges.
        """
        existing = self.get_existing_data_range(symbol)
        
        if not existing:
            # No data exists, return full range
            return [(start_date, end_date)]
        
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date
        
        # Make timezone-naive for comparison (database returns timezone-aware)
        existing_start = existing['earliest'].replace(tzinfo=None)
        existing_end = existing['latest'].replace(tzinfo=None)
        
        missing_ranges = []
        
        # Check if requested range is before existing data
        if start_dt < existing_start:
            missing_ranges.append((start_dt, min(existing_start, end_dt)))
        
        # Check if requested range is after existing data
        if end_dt > existing_end:
            missing_ranges.append((max(existing_end, start_dt), end_dt))
        
        return missing_ranges
    
    def download_klines(self, symbol, interval='1m', start_date=None, end_date=None, force_override=False):
        """
        Download historical kline data from Binance.
        
        Parameters:
        - force_override: If True, download all data ignoring what exists in DB
        """
        
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        elif isinstance(start_date, datetime):
            start_dt = start_date
        else:
            start_dt = datetime.now() - timedelta(days=7)
            
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        elif isinstance(end_date, datetime):
            end_dt = end_date
        else:
            end_dt = datetime.now()
        
        # Check for existing data unless override flag is set
        if not force_override:
            existing = self.get_existing_data_range(symbol)
            
            if existing:
                print(f"ℹ Existing data found for {symbol}:")
                print(f"  Range: {existing['earliest']} to {existing['latest']}")
                print(f"  Records: {existing['count']:,}")
                
                missing_ranges = self.find_missing_ranges(symbol, start_dt, end_dt)
                
                if not missing_ranges:
                    print(f"✓ All requested data already exists. Skipping download.")
                    print(f"  Use force_override=True to re-download.")
                    return []
                else:
                    print(f"ℹ Will download {len(missing_ranges)} missing range(s):")
                    for start, end in missing_ranges:
                        print(f"  - {start} to {end}")
            else:
                print(f"ℹ No existing data for {symbol}. Downloading full range.")
                missing_ranges = [(start_dt, end_dt)]
        else:
            print(f"⚠ OVERRIDE mode: Re-downloading all data for {symbol}")
            missing_ranges = [(start_dt, end_dt)]
        
        all_data = []
        
        for range_start, range_end in missing_ranges:
            start_ts = int(range_start.timestamp() * 1000)
            end_ts = int(range_end.timestamp() * 1000)
            current_ts = start_ts
            
            print(f"\nDownloading {symbol} from {range_start} to {range_end}")
            
            while current_ts < end_ts:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_ts,
                    'endTime': end_ts,
                    'limit': 1000
                }
                
                try:
                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        break
                        
                    all_data.extend(data)
                    current_ts = data[-1][6] + 1
                    
                    print(f"  Downloaded {len(all_data)} candles...", end='\r')
                    time.sleep(0.5)
                    
                except requests.exceptions.RequestException as e:
                    print(f"\n✗ Error downloading data: {e}")
                    break
        
        if all_data:
            print(f"\n✓ Downloaded {len(all_data)} total candles")
        
        return all_data
    
    def insert_data(self, symbol, kline_data):
        """Insert kline data into TimescaleDB."""
        if not kline_data:
            print("No data to insert")
            return
        
        # Prepare data for insertion and remove duplicates within the batch
        records = []
        seen = set()
        
        for candle in kline_data:
            timestamp = datetime.fromtimestamp(candle[0] / 1000)
            key = (timestamp, symbol)
            
            # Skip if we've already seen this timestamp for this symbol
            if key in seen:
                continue
            
            seen.add(key)
            records.append((
                timestamp,                                   # time
                symbol,                                      # symbol
                float(candle[1]),                           # open
                float(candle[2]),                           # high
                float(candle[3]),                           # low
                float(candle[4]),                           # close
                float(candle[5]),                           # volume
                float(candle[7]),                           # quote_volume
                int(candle[8]),                             # trades
                float(candle[9]),                           # taker_buy_base
                float(candle[10])                           # taker_buy_quote
            ))
        
        if len(records) < len(kline_data):
            print(f"ℹ Removed {len(kline_data) - len(records)} duplicate records from batch")
        
        # Bulk insert using execute_values (much faster than individual inserts)
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO crypto_ohlcv 
                (time, symbol, open, high, low, close, volume, quote_volume, 
                 trades, taker_buy_base, taker_buy_quote)
                VALUES %s
                ON CONFLICT (time, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    quote_volume = EXCLUDED.quote_volume,
                    trades = EXCLUDED.trades,
                    taker_buy_base = EXCLUDED.taker_buy_base,
                    taker_buy_quote = EXCLUDED.taker_buy_quote
                """,
                records
            )
            self.conn.commit()
        
        print(f"✓ Inserted {len(records)} records for {symbol}")
    
    def get_stats(self):
        """Display database statistics."""
        with self.conn.cursor() as cur:
            # Count records per symbol
            cur.execute("""
                SELECT symbol, 
                       COUNT(*) as records,
                       MIN(time) as earliest,
                       MAX(time) as latest,
                       pg_size_pretty(pg_total_relation_size('crypto_ohlcv')) as total_size
                FROM crypto_ohlcv
                GROUP BY symbol
                ORDER BY symbol;
            """)
            
            results = cur.fetchall()
            
            print("\n" + "="*70)
            print("DATABASE STATISTICS")
            print("="*70)
            
            for row in results:
                symbol, records, earliest, latest, size = row
                print(f"\n{symbol}:")
                print(f"  Records: {records:,}")
                print(f"  Date range: {earliest} to {latest}")
                if size:
                    print(f"  Size: {size}")
            
            # Show compression stats (only if compression has occurred)
            try:
                cur.execute("""
                    SELECT 
                        pg_size_pretty(before_compression_total_bytes) as before,
                        pg_size_pretty(after_compression_total_bytes) as after,
                        ROUND(100 - (after_compression_total_bytes::numeric / 
                              before_compression_total_bytes::numeric * 100), 2) as savings_pct
                    FROM timescaledb_information.compressed_chunk_stats
                    WHERE hypertable_name = 'crypto_ohlcv';
                """)
                
                compression_stats = cur.fetchall()
                if compression_stats and len(compression_stats) > 0:
                    print("\n" + "-"*70)
                    print("COMPRESSION STATISTICS")
                    print("-"*70)
                    for before, after, savings in compression_stats:
                        print(f"Before: {before} | After: {after} | Savings: {savings}%")
                else:
                    print("\n" + "-"*70)
                    print("COMPRESSION: No compressed chunks yet (data < 7 days old)")
                    print("-"*70)
            except Exception as e:
                print("\n" + "-"*70)
                print("COMPRESSION: Not available yet (data needs to be > 7 days old)")
                print("-"*70)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("\n✓ Database connection closed")


def main():
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'crypto_data',
        'user': 'postgres',
        'password': 'password'
    }
    
    # Symbols to download
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    
    # Date range (adjust as needed)
    start_date = '2025-01-01'
    end_date = datetime.now()
    
    # Set to True to re-download all data even if it exists
    force_override = False
    
    # Initialize
    db = BinanceTimescaleDB(db_params)
    
    try:
        # Connect and setup
        db.connect()
        db.setup_database()
        
        # Download and insert data for each symbol
        for symbol in symbols:
            print(f"\n{'='*70}")
            print(f"Processing {symbol}")
            print(f"{'='*70}")
            
            kline_data = db.download_klines(
                symbol=symbol,
                interval='1m',
                start_date=start_date,
                end_date=end_date,
                force_override=force_override
            )
            
            if kline_data:
                db.insert_data(symbol, kline_data)
            else:
                print(f"⊘ No new data to insert for {symbol}")
            
            time.sleep(1)  # Be nice to Binance API
        
        # Show statistics
        db.get_stats()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()