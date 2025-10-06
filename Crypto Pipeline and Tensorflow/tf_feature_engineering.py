"""
TensorFlow Feature Engineering Pipeline for Crypto OHLCV Data

This script creates features from TimescaleDB crypto data for TensorFlow model training.
Data is downsampled to 5-minute intervals.
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class CryptoFeatureEngineer:
    """
    Feature engineering pipeline for crypto OHLCV data.
    Prepares data for TensorFlow model training.
    """
    
    def __init__(self, db_params, scaling_method='standard'):
        """
        Initialize feature engineer.
        
        Args:
            db_params: Database connection parameters
            scaling_method: 'standard', 'minmax', or 'robust'
        """
        self.db_params = db_params
        self.conn = None
        self.engine = None
        
        # Initialize scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        
        self.feature_names = []
        
    def connect(self):
        """Connect to TimescaleDB."""
        try:
            connection_string = (
                f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
                f"@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"
            )
            self.engine = create_engine(connection_string)
            self.conn = psycopg2.connect(**self.db_params)
            print("Connected to TimescaleDB")
        except Exception as e:
            print(f"Error connecting: {e}")
            raise
    
    def load_data(self, symbol, days=30):
        """Load historical OHLCV data at 5-minute intervals."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    
        query = """
            SELECT 
                time_bucket('5 minutes', time) AS time,
                FIRST(open, time) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close, time) as close,
                SUM(volume) as volume
            FROM crypto_ohlcv
            WHERE symbol = %s
                AND time >= %s
                AND time <= %s
            GROUP BY time_bucket('5 minutes', time)
            ORDER BY time ASC
        """
        df = pd.read_sql_query(query, self.conn, params=(symbol, start_date, end_date))
        print(f"Loaded {len(df)} 5-minute candles for {symbol} (manually aggregated)")
    
        return df
    
    def create_technical_features(self, df):
        """Create technical analysis features."""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Typical Price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Moving Averages (adjusted for 5-min intervals)
        for window in [5, 10, 20, 50, 100]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Price position relative to MAs
        df['price_sma5_ratio'] = df['close'] / df['sma_5']
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['price_sma50_ratio'] = df['close'] / df['sma_50']
        
        # MA crossovers
        df['sma5_sma20_ratio'] = df['sma_5'] / df['sma_20']
        df['sma20_sma50_ratio'] = df['sma_20'] / df['sma_50']
        
        # Volatility Features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'volatility_high_low_{window}'] = df['high_low_pct'].rolling(window=window).std()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume Features
        for window in [5, 10, 20]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
        
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        # Volume-Price Trend
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        # Momentum Indicators
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        df['rsi_7'] = self._calculate_rsi(df['close'], period=7)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                   df['close'].shift(period) * 100)
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # Lag Features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Rolling Statistics
        for window in [5, 10, 20]:
            df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'close_skew_{window}'] = df['close'].rolling(window=window).skew()
            df[f'close_kurt_{window}'] = df['close'].rolling(window=window).kurt()
        
        # Time-based Features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_target_variables(self, df, prediction_horizon=1, classification_threshold=0.0):
        """
        Create target variables for training.
        
        Args:
            df: DataFrame with features
            prediction_horizon: Number of periods ahead to predict (1 = next 5-min period)
            classification_threshold: Threshold for binary classification (% change)
        
        Returns:
            DataFrame with target variables added
        """
        df = df.copy()
        
        # Regression targets
        df['target_price'] = df['close'].shift(-prediction_horizon)
        df['target_return'] = (df['target_price'] - df['close']) / df['close']
        df['target_log_return'] = np.log(df['target_price'] / df['close'])
        
        # Classification targets
        df['target_direction'] = (df['target_return'] > classification_threshold).astype(int)
        
        # Multi-class classification (down, neutral, up)
        df['target_movement'] = pd.cut(
            df['target_return'],
            bins=[-np.inf, -0.001, 0.001, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        # Future volatility (for risk prediction)
        df['target_volatility'] = df['close'].pct_change().shift(-prediction_horizon).rolling(
            window=prediction_horizon).mean()
        
        return df
    
    def prepare_for_tensorflow(self, df, test_size=0.2, sequence_length=None):
        """
        Prepare data for TensorFlow training.
        
        Args:
            df: DataFrame with features and targets
            test_size: Fraction of data to use for testing
            sequence_length: If provided, create sequences for LSTM/RNN
        
        Returns:
            Dictionary containing train/test splits and metadata
        """
        # Drop rows with NaN values
        df_clean = df.dropna()
        print(f"Cleaned data: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows with NaN)")
        
        # Separate features and targets
        exclude_cols = ['time', 'target_price', 'target_return', 'target_log_return', 
                       'target_direction', 'target_movement', 'target_volatility']
        
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df_clean[feature_cols].values
        
        # Multiple target options
        y_regression = df_clean['target_return'].values
        y_classification = df_clean['target_direction'].values
        y_multiclass = df_clean['target_movement'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split (time-series aware - no shuffling)
        split_idx = int(len(X_scaled) * (1 - test_size))
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        
        y_reg_train = y_regression[:split_idx]
        y_reg_test = y_regression[split_idx:]
        
        y_clf_train = y_classification[:split_idx]
        y_clf_test = y_classification[split_idx:]
        
        y_multi_train = y_multiclass[:split_idx]
        y_multi_test = y_multiclass[split_idx:]
        
        # Create sequences for LSTM if requested
        if sequence_length:
            X_train_seq, y_reg_train_seq, y_clf_train_seq = self._create_sequences(
                X_train, y_reg_train, y_clf_train, sequence_length)
            X_test_seq, y_reg_test_seq, y_clf_test_seq = self._create_sequences(
                X_test, y_reg_test, y_clf_test, sequence_length)
            
            return {
                'X_train': X_train_seq,
                'X_test': X_test_seq,
                'y_regression_train': y_reg_train_seq,
                'y_regression_test': y_reg_test_seq,
                'y_classification_train': y_clf_train_seq,
                'y_classification_test': y_clf_test_seq,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'sequence_length': sequence_length,
                'n_features': len(feature_cols)
            }
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of features: {X_train.shape[1]}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_regression_train': y_reg_train,
            'y_regression_test': y_reg_test,
            'y_classification_train': y_clf_train,
            'y_classification_test': y_clf_test,
            'y_multiclass_train': y_multi_train,
            'y_multiclass_test': y_multi_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'n_features': len(feature_cols)
        }
    
    def _create_sequences(self, X, y_reg, y_clf, sequence_length):
        """Create sequences for LSTM/RNN models."""
        X_seq, y_reg_seq, y_clf_seq = [], [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_reg_seq.append(y_reg[i])
            y_clf_seq.append(y_clf[i])
        
        return np.array(X_seq), np.array(y_reg_seq), np.array(y_clf_seq)
    
    def save_features(self, df, filepath='features.csv'):
        """Save engineered features to CSV."""
        df.to_csv(filepath, index=False)
        print(f"Features saved to {filepath}")
    
    def get_feature_importance_data(self, data_dict):
        """
        Get feature names and data for feature importance analysis.
        
        Returns:
            feature_names, X_train, y_train for sklearn models
        """
        return (
            self.feature_names,
            data_dict['X_train'],
            data_dict['y_classification_train']
        )
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


def main():
    # Database configuration
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'crypto_data',
        'user': 'postgres',
        'password': 'password'
    }
    
    # Initialize feature engineer
    engineer = CryptoFeatureEngineer(db_params, scaling_method='standard')
    
    try:
        engineer.connect()
        
        # Load data (5-minute intervals)
        symbol = 'BTCUSDT'
        df = engineer.load_data(symbol, days=90)
        
        # Create technical features
        print("\nCreating technical features...")
        df_features = engineer.create_technical_features(df)
        
        # Create target variables
        print("Creating target variables...")
        df_with_targets = engineer.create_target_variables(
            df_features,
            prediction_horizon=1,
            classification_threshold=0.0
        )
        
        # Save intermediate results
        engineer.save_features(df_with_targets, 'crypto_features_5min.csv')
        
        # Prepare for TensorFlow (standard format)
        print("\nPreparing data for TensorFlow...")
        data = engineer.prepare_for_tensorflow(
            df_with_targets,
            test_size=0.2
        )
        
        print(f"\n{'='*50}")
        print("DATA PREPARATION COMPLETE")
        print(f"{'='*50}")
        print(f"Features shape: {data['X_train'].shape}")
        print(f"Number of features: {data['n_features']}")
        print(f"\nSample feature names:")
        for i, name in enumerate(data['feature_names'][:10]):
            print(f"  {i+1}. {name}")
        print(f"  ... and {len(data['feature_names']) - 10} more features")
        
        # Prepare for LSTM (sequence format)
        # 72 periods = 6 hours at 5-min intervals
        print(f"\n{'='*50}")
        print("Preparing sequences for LSTM...")
        sequence_data = engineer.prepare_for_tensorflow(
            df_with_targets,
            test_size=0.2,
            sequence_length=72
        )
        
        print(f"LSTM input shape: {sequence_data['X_train'].shape}")
        print(f"  (samples, timesteps, features)")
        
        # Save to numpy files
        np.save('X_train.npy', data['X_train'])
        np.save('X_test.npy', data['X_test'])
        np.save('y_train.npy', data['y_classification_train'])
        np.save('y_test.npy', data['y_classification_test'])
        print("\nData saved to .npy files")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engineer.close()


if __name__ == "__main__":
    main()