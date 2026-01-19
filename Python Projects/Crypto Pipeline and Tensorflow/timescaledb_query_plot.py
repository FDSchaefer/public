"""
TimescaleDB Crypto Data Query and Visualization

Prerequisites:
pip install psycopg2-binary pandas matplotlib sqlalchemy

Usage:
python query_plot.py
"""

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sqlalchemy import create_engine

class CryptoDataVisualizer:
    def __init__(self, db_params):
        """Initialize database connection."""
        self.db_params = db_params
        self.conn = None
        self.engine = None
        
    def connect(self):
        """Connect to TimescaleDB."""
        try:
            # Create SQLAlchemy engine for pandas
            connection_string = (
                f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
                f"@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"
            )
            self.engine = create_engine(connection_string)
            
            # Also keep psycopg2 connection for non-pandas operations if needed
            self.conn = psycopg2.connect(**self.db_params)
            
            print("✓ Connected to TimescaleDB")
        except Exception as e:
            print(f"✗ Error connecting to database: {e}")
            raise
    
    def query_data(self, symbol, start_date=None, end_date=None, interval='1m'):
        """
        Query OHLCV data for a symbol.
        
        Parameters:
        - symbol: Trading pair (e.g., 'BTCUSDT')
        - start_date: Start date as string 'YYYY-MM-DD' or datetime
        - end_date: End date as string 'YYYY-MM-DD' or datetime
        - interval: '1m', '5m', '15m', '1h', '4h', '1d' (for aggregated data)
        
        Returns: pandas DataFrame
        """
        
        # Build the query based on interval
        if interval == '1m':
            # Query raw 1-minute data
            query = """
                SELECT time, open, high, low, close, volume
                FROM crypto_ohlcv
                WHERE symbol = %(symbol)s
            """
            params = {'symbol': symbol}
            
        elif interval == '5m':
            # Use continuous aggregate if available
            query = """
                SELECT bucket as time, open, high, low, close, volume
                FROM crypto_ohlcv_5m
                WHERE symbol = %(symbol)s
            """
            params = {'symbol': symbol}
            
        else:
            # Aggregate on the fly for other intervals
            interval_map = {
                '15m': '15 minutes',
                '1h': '1 hour',
                '4h': '4 hours',
                '1d': '1 day'
            }
            
            time_bucket = interval_map.get(interval, '1 hour')
            
            query = f"""
                SELECT 
                    time_bucket('{time_bucket}', time) AS time,
                    FIRST(open, time) as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    LAST(close, time) as close,
                    SUM(volume) as volume
                FROM crypto_ohlcv
                WHERE symbol = %(symbol)s
            """
            params = {'symbol': symbol}
        
        # Add date filters if provided
        if start_date:
            query += " AND time >= %(start_date)s"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND time <= %(end_date)s"
            params['end_date'] = end_date
        
        # Add grouping for aggregated queries
        if interval not in ['1m', '5m']:
            query += " GROUP BY time"
        
        query += " ORDER BY time ASC"
        
        # Execute query using SQLAlchemy engine with named parameters
        df = pd.read_sql_query(query, self.engine, params=params)
        
        print(f"✓ Queried {len(df)} candles for {symbol}")
        return df
    
    def get_available_symbols(self):
        """Get list of available symbols in database."""
        query = """
            SELECT DISTINCT symbol, 
                   COUNT(*) as records,
                   MIN(time) as start_date,
                   MAX(time) as end_date
            FROM crypto_ohlcv
            GROUP BY symbol
            ORDER BY symbol
        """
        
        df = pd.read_sql_query(query, self.engine)
        return df
    
    def plot_price(self, df, symbol, title=None, save_path=None):
        """
        Plot close price over time.
        
        Parameters:
        - df: DataFrame with 'time' and 'close' columns
        - symbol: Symbol name for labels
        - title: Optional custom title
        - save_path: Optional path to save figure
        """
        
        if df.empty:
            print("✗ No data to plot")
            return
        
        plt.figure(figsize=(14, 7))
        
        # Main price plot
        plt.plot(df['time'], df['close'], linewidth=1.5, color='#2962FF', label='Close Price')
        
        # Formatting
        plt.title(title or f'{symbol} Price Chart', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (USDT)', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to {save_path}")
        
        plt.show()
    
    def plot_ohlc(self, df, symbol, title=None, save_path=None):
        """
        Plot OHLC (candlestick-style) chart.
        
        Parameters:
        - df: DataFrame with OHLC data
        - symbol: Symbol name
        - title: Optional custom title
        - save_path: Optional path to save figure
        """
        
        if df.empty:
            print("✗ No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(df['time'], df['close'], linewidth=1.5, color='#2962FF', label='Close')
        ax1.fill_between(df['time'], df['low'], df['high'], alpha=0.2, color='#2962FF', label='High-Low Range')
        
        ax1.set_title(title or f'{symbol} OHLC Chart', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        
        # Volume chart
        colors = ['#26A69A' if df.loc[i, 'close'] >= df.loc[i, 'open'] else '#EF5350' 
                  for i in range(len(df))]
        ax2.bar(df['time'], df['volume'], color=colors, alpha=0.7, width=0.8)
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Rotate x-axis labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to {save_path}")
        
        plt.show()
    
    def plot_multiple_symbols(self, symbols, start_date=None, end_date=None, 
                             interval='1h', normalize=True, save_path=None):
        """
        Plot multiple symbols on the same chart for comparison.
        
        Parameters:
        - symbols: List of symbols to plot
        - start_date, end_date: Date range
        - interval: Time interval for aggregation
        - normalize: If True, normalize prices to start at 100
        - save_path: Optional path to save figure
        """
        
        plt.figure(figsize=(14, 7))
        
        for symbol in symbols:
            df = self.query_data(symbol, start_date, end_date, interval)
            
            if not df.empty:
                if normalize:
                    # Normalize to percentage change from first value
                    df['normalized'] = (df['close'] / df['close'].iloc[0]) * 100
                    plt.plot(df['time'], df['normalized'], linewidth=2, label=symbol, alpha=0.8)
                else:
                    plt.plot(df['time'], df['close'], linewidth=2, label=symbol, alpha=0.8)
        
        plt.title('Crypto Price Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Normalized Price (%)' if normalize else 'Price (USDT)', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to {save_path}")
        
        plt.show()
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


def main():
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'crypto_data',
        'user': 'postgres',
        'password': 'password'
    }
    
    # Initialize visualizer
    viz = CryptoDataVisualizer(db_params)
    
    try:
        viz.connect()
        
        # Show available data
        print("\n" + "="*70)
        print("AVAILABLE DATA")
        print("="*70)
        available = viz.get_available_symbols()
        print(available.to_string(index=False))
        print()
        
        # Example 1: Plot single symbol with close price
        print("\n" + "="*70)
        print("EXAMPLE 1: BTC Close Price (Last 7 days)")
        print("="*70)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        btc_data = viz.query_data('BTCUSDT', start_date, end_date, interval='1h')
        viz.plot_price(btc_data, 'BTCUSDT', 
                      title='Bitcoin (BTC) - Last 7 Days',
                      save_path='btc_7days.png')
        
        # Example 2: Plot OHLC chart with volume
        print("\n" + "="*70)
        print("EXAMPLE 2: ETH OHLC Chart (Last 3 days)")
        print("="*70)
        
        start_date = end_date - timedelta(days=3)
        eth_data = viz.query_data('ETHUSDT', start_date, end_date, interval='15m')
        viz.plot_ohlc(eth_data, 'ETHUSDT',
                     title='Ethereum (ETH) - Last 3 Days',
                     save_path='eth_ohlc_3days.png')
        
        # Example 3: Compare multiple symbols
        print("\n" + "="*70)
        print("EXAMPLE 3: Compare All Symbols (Normalized)")
        print("="*70)
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
        start_date = end_date - timedelta(days=30)
        
        viz.plot_multiple_symbols(symbols, start_date, end_date, 
                                 interval='4h', normalize=True,
                                 save_path='comparison_30days.png')
        
        print("\n" + "="*70)
        print("DONE! Check your directory for saved charts.")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        viz.close()


if __name__ == "__main__":
    main()