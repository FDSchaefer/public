"""
Crypto Price Movement Prediction using ML

Prerequisites:
pip install scikit-learn psycopg2-binary pandas numpy matplotlib

Usage:
python ml_predictor.py
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CryptoPricePredictor:
    def __init__(self, db_params):
        self.db_params = db_params
        self.conn = None
        self.scaler = StandardScaler()
        self.direction_model = None  # Predict up/down
        self.swing_model = None  # Predict price change magnitude
        
    def connect(self):
        """Connect to TimescaleDB."""
        self.conn = psycopg2.connect(**self.db_params)
        print("✓ Connected to TimescaleDB")
    
    def load_data(self, symbol, days=30):
        """Load historical data and prepare features."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query = """
            SELECT time, open, high, low, close, volume
            FROM crypto_ohlcv
            WHERE symbol = %s
              AND time >= %s
              AND time <= %s
            ORDER BY time ASC
        """
        
        df = pd.read_sql_query(query, self.conn, params=(symbol, start_date, end_date))
        print(f"✓ Loaded {len(df)} records for {symbol}")
        return df
    
    def create_features(self, df):
        """Engineer features for ML model."""
        df = df.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['close_open_spread'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Price position relative to MAs
        df['price_to_sma5'] = df['close'] / df['sma_5'] - 1
        df['price_to_sma20'] = df['close'] / df['sma_20'] - 1
        
        # Volatility
        df['volatility_5'] = df['close'].rolling(window=5).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()
        
        # Volume features
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        df['momentum'] = df['close'] - df['close'].shift(4)
        
        # Lag features (past prices)
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Target variables
        df['future_close'] = df['close'].shift(-1)  # Next period close
        df['future_change'] = (df['future_close'] - df['close']) / df['close']
        df['direction'] = (df['future_change'] > 0).astype(int)  # 1=up, 0=down
        
        # Potential swing (high-low range in next period)
        df['future_high'] = df['high'].shift(-1)
        df['future_low'] = df['low'].shift(-1)
        df['future_swing'] = (df['future_high'] - df['future_low']) / df['close']
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_training_data(self, df):
        """Prepare X and y for training."""
        feature_cols = [
            'price_change', 'high_low_spread', 'close_open_spread',
            'price_to_sma5', 'price_to_sma20',
            'volatility_5', 'volatility_20',
            'volume_ratio', 'rsi', 'momentum',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5'
        ]
        
        # Drop rows with NaN values
        df_clean = df.dropna(subset=feature_cols + ['direction', 'future_swing'])
        
        X = df_clean[feature_cols]
        y_direction = df_clean['direction']
        y_swing = df_clean['future_swing']
        
        print(f"✓ Prepared {len(X)} training samples with {len(feature_cols)} features")
        return X, y_direction, y_swing, df_clean.reset_index(drop=True)
    
    def train_models(self, X, y_direction, y_swing, df_clean):
        """Train both direction and swing prediction models."""
        # Split data
        split_idx = int(len(X) * 0.8)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_dir_train = y_direction.iloc[:split_idx]
        y_dir_test = y_direction.iloc[split_idx:]
        y_swing_train = y_swing.iloc[:split_idx]
        y_swing_test = y_swing.iloc[split_idx:]
        
        # Get test period info for daily predictions
        test_df = df_clean.iloc[split_idx:].copy()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train direction classifier (up/down)
        print("\nTraining direction classifier...")
        self.direction_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        self.direction_model.fit(X_train_scaled, y_dir_train)
        
        # Evaluate direction model
        dir_pred = self.direction_model.predict(X_test_scaled)
        dir_proba = self.direction_model.predict_proba(X_test_scaled)
        dir_accuracy = (dir_pred == y_dir_test).mean()
        print(f"✓ Direction accuracy: {dir_accuracy:.2%}")
        
        # Train swing magnitude regressor
        print("\nTraining swing magnitude regressor...")
        self.swing_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=20,
            learning_rate=0.1,
            random_state=42
        )
        self.swing_model.fit(X_train_scaled, y_swing_train)
        
        # Evaluate swing model
        swing_pred = self.swing_model.predict(X_test_scaled)
        swing_r2 = r2_score(y_swing_test, swing_pred)
        swing_rmse = np.sqrt(mean_squared_error(y_swing_test, swing_pred))
        print(f"✓ Swing prediction R²: {swing_r2:.3f}")
        print(f"✓ Swing prediction RMSE: {swing_rmse:.4f}")
        
        return X_test_scaled, y_dir_test, y_swing_test, dir_pred, swing_pred, dir_proba, test_df
    
    def analyze_daily_predictions(self, test_df, dir_pred, swing_pred, dir_proba):
        """Analyze predictions for each day in test period."""
        results = []
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            
            # Actual values
            actual_direction = 1 if row['future_change'] > 0 else 0
            actual_price_change = row['future_change']
            actual_swing = row['future_swing']
            
            # Predicted values
            pred_direction = dir_pred[i]
            pred_swing = swing_pred[i]
            confidence = dir_proba[i][pred_direction]
            
            # Calculate predicted price change
            pred_price_change = pred_swing if pred_direction == 1 else -pred_swing
            
            # Calculate profit/loss if trading on prediction
            if pred_direction == actual_direction:
                direction_correct = True
                # Simplified P&L: if direction correct, gain is proportional to actual move
                trade_pnl = abs(actual_price_change)
            else:
                direction_correct = False
                # If wrong direction, lose proportional to move
                trade_pnl = -abs(actual_price_change)
            
            results.append({
                'time': row['time'],
                'actual_close': row['close'],
                'actual_direction': 'up' if actual_direction == 1 else 'down',
                'pred_direction': 'up' if pred_direction == 1 else 'down',
                'confidence': confidence,
                'direction_correct': direction_correct,
                'actual_change_pct': actual_price_change * 100,
                'pred_change_pct': pred_price_change * 100,
                'actual_swing_pct': actual_swing * 100,
                'pred_swing_pct': pred_swing * 100,
                'trade_pnl_pct': trade_pnl * 100
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate cumulative performance
        results_df['cumulative_pnl'] = results_df['trade_pnl_pct'].cumsum()
        
        return results_df
    
    def plot_daily_performance(self, results_df):
        """Visualize daily prediction performance."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3)
        
        # 1. Cumulative P&L
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(results_df['time'], results_df['cumulative_pnl'], 
                linewidth=2, color='#2962FF', label='Cumulative P&L')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.fill_between(results_df['time'], results_df['cumulative_pnl'], 0, 
                         where=results_df['cumulative_pnl'] >= 0, alpha=0.3, color='green')
        ax1.fill_between(results_df['time'], results_df['cumulative_pnl'], 0, 
                         where=results_df['cumulative_pnl'] < 0, alpha=0.3, color='red')
        ax1.set_title('Cumulative Trading P&L (Test Period)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Daily P&L
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['green' if x > 0 else 'red' for x in results_df['trade_pnl_pct']]
        ax2.bar(range(len(results_df)), results_df['trade_pnl_pct'], 
               color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Daily P&L', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Daily Return (%)')
        ax2.set_xlabel('Day')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Prediction confidence distribution
        ax3 = fig.add_subplot(gs[1, 1])
        correct = results_df[results_df['direction_correct'] == True]['confidence']
        incorrect = results_df[results_df['direction_correct'] == False]['confidence']
        ax3.hist([correct, incorrect], bins=20, label=['Correct', 'Incorrect'], 
                alpha=0.7, color=['green', 'red'])
        ax3.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Actual vs Predicted Price Change
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.scatter(results_df['actual_change_pct'], results_df['pred_change_pct'], 
                   c=results_df['direction_correct'].map({True: 'green', False: 'red'}),
                   alpha=0.6, s=50)
        lims = [min(results_df['actual_change_pct'].min(), results_df['pred_change_pct'].min()),
                max(results_df['actual_change_pct'].max(), results_df['pred_change_pct'].max())]
        ax4.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Prediction')
        ax4.set_title('Actual vs Predicted Change', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Actual Change (%)')
        ax4.set_ylabel('Predicted Change (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Actual vs Predicted Swing
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.scatter(results_df['actual_swing_pct'], results_df['pred_swing_pct'], 
                   alpha=0.6, s=50, color='purple')
        lims = [0, max(results_df['actual_swing_pct'].max(), results_df['pred_swing_pct'].max())]
        ax5.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Prediction')
        ax5.set_title('Actual vs Predicted Swing', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Actual Swing (%)')
        ax5.set_ylabel('Predicted Swing (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Daily accuracy
        ax6 = fig.add_subplot(gs[3, :])
        window = 10  # Rolling window for accuracy
        rolling_accuracy = results_df['direction_correct'].rolling(window=window).mean() * 100
        ax6.plot(range(len(rolling_accuracy)), rolling_accuracy, 
                linewidth=2, color='#FF6D00', label=f'{window}-Day Rolling Accuracy')
        ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax6.fill_between(range(len(rolling_accuracy)), 50, rolling_accuracy, 
                        where=rolling_accuracy >= 50, alpha=0.3, color='green')
        ax6.fill_between(range(len(rolling_accuracy)), 50, rolling_accuracy, 
                        where=rolling_accuracy < 50, alpha=0.3, color='red')
        ax6.set_title('Rolling Accuracy Over Time', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Day')
        ax6.set_ylabel('Accuracy (%)')
        ax6.set_ylim([0, 100])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.savefig('daily_performance.png', dpi=300, bbox_inches='tight')
        print("✓ Daily performance saved to daily_performance.png")
        plt.show()
    
    def print_performance_summary(self, results_df):
        """Print detailed performance statistics."""
        print("\n" + "="*70)
        print("HISTORICAL PERFORMANCE SUMMARY (TEST PERIOD)")
        print("="*70)
        
        total_days = len(results_df)
        correct_predictions = results_df['direction_correct'].sum()
        accuracy = (correct_predictions / total_days) * 100
        
        total_pnl = results_df['cumulative_pnl'].iloc[-1]
        winning_days = (results_df['trade_pnl_pct'] > 0).sum()
        losing_days = (results_df['trade_pnl_pct'] < 0).sum()
        
        avg_win = results_df[results_df['trade_pnl_pct'] > 0]['trade_pnl_pct'].mean()
        avg_loss = results_df[results_df['trade_pnl_pct'] < 0]['trade_pnl_pct'].mean()
        
        max_win = results_df['trade_pnl_pct'].max()
        max_loss = results_df['trade_pnl_pct'].min()
        
        avg_confidence = results_df['confidence'].mean()
        
        print(f"\nPeriod: {results_df['time'].iloc[0]} to {results_df['time'].iloc[-1]}")
        print(f"Total Trading Days: {total_days}")
        
        print(f"\n📊 DIRECTION ACCURACY:")
        print(f"  Correct Predictions: {correct_predictions}/{total_days}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Average Confidence: {avg_confidence:.2%}")
        
        print(f"\n💰 P&L PERFORMANCE:")
        print(f"  Total Return: {total_pnl:+.2f}%")
        print(f"  Winning Days: {winning_days} ({winning_days/total_days*100:.1f}%)")
        print(f"  Losing Days: {losing_days} ({losing_days/total_days*100:.1f}%)")
        print(f"  Average Win: +{avg_win:.2f}%")
        print(f"  Average Loss: {avg_loss:.2f}%")
        print(f"  Best Day: +{max_win:.2f}%")
        print(f"  Worst Day: {max_loss:.2f}%")
        
        if avg_loss != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
            print(f"  Win/Loss Ratio: {win_loss_ratio:.2f}")
        
        print(f"\n📈 PRICE CHANGE PREDICTIONS:")
        change_mae = np.abs(results_df['actual_change_pct'] - results_df['pred_change_pct']).mean()
        print(f"  Mean Absolute Error: {change_mae:.2f}%")
        
        print(f"\n📊 SWING PREDICTIONS:")
        swing_mae = np.abs(results_df['actual_swing_pct'] - results_df['pred_swing_pct']).mean()
        print(f"  Mean Absolute Error: {swing_mae:.2f}%")
        
        print("\n" + "="*70)
    
    def predict_next(self, df, current_price):
        """Make prediction for the next period."""
        # Use last row for prediction
        feature_cols = [
            'price_change', 'high_low_spread', 'close_open_spread',
            'price_to_sma5', 'price_to_sma20',
            'volatility_5', 'volatility_20',
            'volume_ratio', 'rsi', 'momentum',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5'
        ]
        
        X_latest = df[feature_cols].iloc[[-1]]
        X_scaled = self.scaler.transform(X_latest)
        
        # Predict direction
        direction = self.direction_model.predict(X_scaled)[0]
        direction_proba = self.direction_model.predict_proba(X_scaled)[0]
        
        # Predict swing
        swing = self.swing_model.predict(X_scaled)[0]
        
        # Calculate price targets
        predicted_change = swing if direction == 1 else -swing
        predicted_price = current_price * (1 + predicted_change)
        potential_high = current_price * (1 + swing)
        potential_low = current_price * (1 - swing)
        
        print("\n" + "="*70)
        print("NEXT PERIOD PREDICTION")
        print("="*70)
        print(f"Current Price: ${current_price:.2f}")
        print(f"\nDirection: {'UP ↑' if direction == 1 else 'DOWN ↓'}")
        print(f"Confidence: {direction_proba[direction]:.1%}")
        print(f"\nPredicted Price: ${predicted_price:.2f} ({predicted_change:+.2%})")
        print(f"Potential Range: ${potential_low:.2f} - ${potential_high:.2f}")
        print(f"Expected Swing: ±{swing:.2%}")
        print("="*70)
        
        return {
            'direction': 'up' if direction == 1 else 'down',
            'confidence': direction_proba[direction],
            'predicted_price': predicted_price,
            'potential_high': potential_high,
            'potential_low': potential_low,
            'expected_swing': swing
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("\n✓ Database connection closed")


def main():
    # Database connection
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'crypto_data',
        'user': 'postgres',
        'password': 'password'
    }
    
    # Initialize predictor
    predictor = CryptoPricePredictor(db_params)
    
    try:
        predictor.connect()
        
        # Load and prepare data
        symbol = 'BTCUSDT'
        print(f"\nProcessing {symbol}...")
        df = predictor.load_data(symbol, days=90)
        
        # Create features
        df_features = predictor.create_features(df)
        
        # Prepare training data
        X, y_direction, y_swing, df_clean = predictor.prepare_training_data(df_features)
        
        # Train models
        X_test, y_dir_test, y_swing_test, dir_pred, swing_pred, dir_proba, test_df = predictor.train_models(
            X, y_direction, y_swing, df_clean
        )
        
        # Analyze daily predictions
        print("\nAnalyzing daily predictions...")
        results_df = predictor.analyze_daily_predictions(test_df, dir_pred, swing_pred, dir_proba)
        
        # Print performance summary
        predictor.print_performance_summary(results_df)
        
        # Visualize daily performance
        predictor.plot_daily_performance(results_df)
        
        # Save detailed results to CSV
        results_df.to_csv('daily_predictions.csv', index=False)
        print("\n✓ Daily predictions saved to daily_predictions.csv")
        
        # Make prediction for next period
        current_price = df_clean['close'].iloc[-1]
        prediction = predictor.predict_next(df_clean, current_price)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        predictor.close()


if __name__ == "__main__":
    main()