import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedPredictor:
    def __init__(self, asset_type="crypto"):
        self.asset_type = asset_type
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def clean_data(self, df):
        """Clean data to remove infinity and invalid values"""
        df = df.copy()
        
        # Replace infinity with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill and backward fill to handle NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        return df
    
    def create_advanced_features(self, df):
        """Create advanced features for prediction"""
        df = df.copy()
        
        # Clean data first
        df = self.clean_data(df)
        
        # Basic returns and price features
        df['Return'] = df['Close'].pct_change()
        df['High_Low_Pct'] = np.where(df['Close'] != 0, (df['High'] - df['Low']) / df['Close'], 0)
        df['Price_Change'] = df['Close'] - df['Open']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Handle potential infinity in volume change
        df['Volume_Change'] = np.where(np.isinf(df['Volume_Change']), 0, df['Volume_Change'])
        df['Volume_Change'] = np.where(np.isnan(df['Volume_Change']), 0, df['Volume_Change'])
        
        # Technical indicators with multiple timeframes
        windows = [3, 5, 8, 13, 21, 34, 55]  # Fibonacci sequence
        
        for window in windows:
            # Price-based features
            df['SMA_' + str(window)] = df['Close'].rolling(window=window).mean()
            df['EMA_' + str(window)] = df['Close'].ewm(span=window).mean()
            
            # Weighted moving average with error handling
            try:
                df['WMA_' + str(window)] = df['Close'].rolling(window=window).apply(
                    lambda x: np.average(x, weights=np.arange(1, len(x)+1)) if len(x) > 0 else 0, raw=True)
            except:
                df['WMA_' + str(window)] = df['Close'].rolling(window=window).mean()
            
            # Volatility features
            df['Volatility_' + str(window)] = df['Close'].pct_change().rolling(window=window).std()
            
            # Momentum features with error handling
            shifted_price = df['Close'].shift(window)
            df['Momentum_' + str(window)] = np.where(shifted_price != 0, 
                                                   df['Close'] / shifted_price - 1, 0)
            
            # Rate of Change
            df['ROC_' + str(window)] = df['Close'].pct_change(periods=window)
        
        # Advanced technical indicators
        # Triple Exponential Moving Average (TEMA)
        ema_1 = df['Close'].ewm(span=9).mean()
        ema_2 = ema_1.ewm(span=9).mean()
        ema_3 = ema_2.ewm(span=9).mean()
        df['TEMA'] = 3 * ema_1 - 3 * ema_2 + ema_3
        
        # Hull Moving Average (simplified)
        df['HMA'] = df['Close'].rolling(20).mean()  # Simplified version
        
        # Ichimoku Cloud components (simplified)
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        df['Tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        df['Kijun_sen'] = (high_26 + low_26) / 2
        df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        
        # Advanced volatility measures
        returns = df['Close'].pct_change()
        df['GARCH_vol'] = returns.rolling(20).var()  # Simplified GARCH
        
        # Market microstructure features
        volume_ma = df['Volume'].rolling(20).mean()
        df['Volume_Profile'] = np.where(volume_ma != 0, df['Volume'] / volume_ma, 0)
        df['Price_Volume_Trend'] = (df['Close'].pct_change() * df['Volume']).cumsum()
        
        # Statistical arbitrage features
        close_ma = df['Close'].rolling(20).mean()
        close_std = df['Close'].rolling(20).std()
        df['Z_Score'] = np.where(close_std != 0, (df['Close'] - close_ma) / close_std, 0)
        df['Bollinger_Position'] = np.where(close_std != 0, 
                                          (df['Close'] - close_ma) / (close_std * 2), 0)
        
        # Clean data again after feature creation
        df = self.clean_data(df)
        
        return df
    
    def prepare_advanced_data(self, df):
        """Prepare data with advanced features"""
        # Clean data first
        df = self.clean_data(df)
        
        # Target variables
        df['Target_Return'] = df['Close'].pct_change().shift(-1)
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Select comprehensive feature set
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Close',
            'Return', 'High_Low_Pct', 'Price_Change', 'Volume_Change',
            'SMA_3', 'SMA_5', 'SMA_8', 'SMA_13', 'SMA_21', 'SMA_34', 'SMA_55',
            'EMA_3', 'EMA_5', 'EMA_8', 'EMA_13', 'EMA_21', 'EMA_34', 'EMA_55',
            'WMA_21', 'WMA_55',
            'Volatility_3', 'Volatility_5', 'Volatility_8', 'Volatility_13', 'Volatility_21', 
            'Volatility_34', 'Volatility_55',
            'Momentum_3', 'Momentum_5', 'Momentum_8', 'Momentum_13', 'Momentum_21',
            'Momentum_34', 'Momentum_55',
            'ROC_3', 'ROC_5', 'ROC_8', 'ROC_13', 'ROC_21', 'ROC_34', 'ROC_55',
            'TEMA', 'HMA', 'Tenkan_sen', 'Kijun_sen', 'Senkou_span_a',
            'GARCH_vol', 'Volume_Profile', 'Price_Volume_Trend', 'Z_Score', 'Bollinger_Position'
        ]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in df_clean.columns]
        X = df_clean[available_features]
        y_return = df_clean['Target_Return']
        y_direction = df_clean['Target_Direction']
        
        # Final cleaning to remove any infinity values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_return = y_return.replace([np.inf, -np.inf], 0).fillna(0)
        y_direction = y_direction.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Ensure finite values only
        X = X[np.isfinite(X).all(axis=1)]
        y_return = y_return[np.isfinite(y_return)]
        y_direction = y_direction[np.isfinite(y_direction)]
        
        # Align indices
        common_index = X.index.intersection(y_return.index).intersection(y_direction.index)
        X = X.loc[common_index]
        y_return = y_return.loc[common_index]
        y_direction = y_direction.loc[common_index]
        
        self.feature_columns = available_features
        
        return X, y_return, y_direction, df_clean
    
    def create_diverse_ensemble(self, X_train, y_train):
        """Create diverse ensemble of advanced models"""
        print("Training diverse ensemble of advanced models...")
        
        # Clean training data
        X_train = X_train.replace([np.inf, -np.inf], 0)
        y_train = y_train.replace([np.inf, -np.inf], 0)
        
        # Traditional ML models with advanced parameters
        models = {
            'xgb': XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
            )
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            print("Training " + name.upper() + "...")
            try:
                if name in ['ridge', 'lasso']:
                    # Scale data for linear models
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    # Handle infinity in scaled data
                    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
                    self.scalers[name] = scaler
                    model.fit(X_train_scaled, y_train)
                elif name == 'mlp':
                    # Scale data for neural network
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    # Handle infinity in scaled data
                    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
                    self.scalers[name] = scaler
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                trained_models[name] = model
            except Exception as e:
                print("Error training " + name + ": " + str(e))
        
        self.models = trained_models
        return trained_models
    
    def advanced_ensemble_predict(self, X):
        """Make predictions using advanced ensemble"""
        predictions = {}
        
        # Clean input data
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Get predictions from traditional models
        for name, model in self.models.items():
            try:
                if name in ['ridge', 'lasso', 'mlp']:
                    X_scaled = self.scalers[name].transform(X)
                    # Handle infinity in scaled data
                    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                print("Error predicting with " + name + ": " + str(e))
                predictions[name] = np.zeros(len(X))
        
        # Weighted ensemble
        weights = {
            'xgb': 0.25, 'rf': 0.20, 'gb': 0.20, 'ridge': 0.10, 'lasso': 0.10, 'mlp': 0.15
        }
        
        # Combine predictions
        weighted_pred = np.zeros(len(X))
        total_weight = 0
        for name, pred in predictions.items():
            if name in weights:
                weighted_pred += weights[name] * pred
                total_weight += weights[name]
        
        # Normalize if not all models worked
        if total_weight > 0:
            weighted_pred = weighted_pred / total_weight * sum(weights.values())
        
        return weighted_pred, predictions, weights
    
    def train_and_evaluate(self, symbol="ETH-USD"):
        """Train advanced model and evaluate performance"""
        print("🚀 Advanced Prediction Model")
        print("="*60)
        
        # Download data
        print("Downloading " + symbol + " data...")
        try:
            data = yf.download(symbol, start="2020-01-01", end="2024-01-01")
            if data.empty:
                print("No data downloaded for " + symbol)
                return None
            print("Downloaded " + str(len(data)) + " records")
        except Exception as e:
            print("Error downloading data: " + str(e))
            return None
        
        # Prepare advanced features
        print("Creating advanced features...")
        try:
            X, y_return, y_direction, clean_data = self.prepare_advanced_data(data)
            print("Prepared data with " + str(len(self.feature_columns)) + " features")
            print("Final data shape: " + str(X.shape))
            
            if len(X) < 10:
                print("Insufficient data for training")
                return None
                
        except Exception as e:
            print("Error preparing data: " + str(e))
            return None
        
        # Split data (time series aware)
        if len(X) < 20:
            print("Not enough data for proper train/test split")
            return None
            
        split_index = int(len(X) * 0.8)
        if split_index >= len(X):
            split_index = len(X) - 5
            
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y_return.iloc[:split_index]
        y_test = y_return.iloc[split_index:]
        y_train_direction = y_direction.iloc[:split_index]
        y_test_direction = y_direction.iloc[split_index:]
        
        print("Training set: " + str(len(X_train)) + " samples")
        print("Test set: " + str(len(X_test)) + " samples")
        
        # Train diverse ensemble
        try:
            self.create_diverse_ensemble(X_train, y_train)
        except Exception as e:
            print("Error creating ensemble: " + str(e))
            return None
        
        # Make predictions
        try:
            ensemble_pred, individual_preds, weights = self.advanced_ensemble_predict(X_test)
        except Exception as e:
            print("Error making predictions: " + str(e))
            return None
        
        # Calculate metrics
        try:
            mse = mean_squared_error(y_test, ensemble_pred)
            mae = mean_absolute_error(y_test, ensemble_pred)
            r2 = r2_score(y_test, ensemble_pred)
            
            # Direction accuracy
            pred_direction = (ensemble_pred > 0).astype(int)
            direction_accuracy = accuracy_score(y_test_direction, pred_direction)
            
            print("\n" + "="*70)
            print("ADVANCED MODEL RESULTS")
            print("="*70)
            print("Mean Squared Error: " + "{:.6f}".format(mse))
            print("Mean Absolute Error: " + "{:.6f}".format(mae))
            print("R² Score: " + "{:.4f}".format(r2))
            print("Direction Accuracy: " + "{:.4f}".format(direction_accuracy) + " (" + "{:.2f}".format(direction_accuracy*100) + "%)")
            
            # Model weights and performance
            print("\nModel Weights:")
            print("-" * 30)
            for name, weight in weights.items():
                if name in individual_preds and len(individual_preds[name]) == len(y_test):
                    try:
                        perf = mean_squared_error(y_test, individual_preds[name])
                        print("  " + name.upper() + ": " + "{:.3f}".format(weight) + " (MSE: " + "{:.6f}".format(perf) + ")")
                    except:
                        print("  " + name.upper() + ": " + "{:.3f}".format(weight) + " (MSE: N/A)")
            
            # Generate prediction for next day
            if len(X) > 0:
                latest_features = X.iloc[-1:].copy()
                next_day_pred, _, _ = self.advanced_ensemble_predict(latest_features)
                next_day_pred = float(next_day_pred[0]) if len(next_day_pred) > 0 else 0.0
                
                # Confidence interval estimation (simplified)
                prediction_std = np.std(ensemble_pred)
                confidence_lower = next_day_pred - 1.96 * prediction_std
                confidence_upper = next_day_pred + 1.96 * prediction_std
                
                current_price = float(data['Close'].iloc[-1])
                predicted_price = current_price * (1 + next_day_pred)
                lower_price = current_price * (1 + confidence_lower)
                upper_price = current_price * (1 + confidence_upper)
                
                print("\n🔮 Next Trading Day Prediction:")
                print("  Current Price: $" + "{:.2f}".format(current_price))
                print("  Predicted Return: " + "{:+.2f}".format(next_day_pred*100) + "%")
                print("  Predicted Price: $" + "{:.2f}".format(predicted_price))
                print("  Expected Change: $" + "{:+.2f}".format(predicted_price - current_price))
                
                # Risk-adjusted prediction
                risk_adjusted_return = next_day_pred * 0.7  # Conservative adjustment
                risk_adjusted_price = current_price * (1 + risk_adjusted_return)
                print("\n🛡️  Risk-Adjusted Prediction:")
                print("  Conservative Return: " + "{:+.2f}".format(risk_adjusted_return*100) + "%")
                print("  Conservative Price: $" + "{:.2f}".format(risk_adjusted_price))
                
                return {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'next_day_return': next_day_pred,
                    'next_day_price': predicted_price,
                    'current_price': current_price,
                    'risk_adjusted_return': risk_adjusted_return,
                    'risk_adjusted_price': risk_adjusted_price
                }
            else:
                print("No data available for prediction")
                return None
                
        except Exception as e:
            print("Error calculating metrics: " + str(e))
            return None

# Run the advanced models
print("🚀 Launching Advanced Prediction Models")
print("="*80)

try:
    # S&P 500 Advanced Model
    print("Initializing S&P 500 Advanced Model...")
    sp500_advanced = AdvancedPredictor(asset_type="equity")
    sp500_results = sp500_advanced.train_and_evaluate("^GSPC")

    print("\n" + "="*80)

    # Ethereum Advanced Model  
    print("Initializing Ethereum Advanced Model...")
    eth_advanced = AdvancedPredictor(asset_type="crypto")
    eth_results = eth_advanced.train_and_evaluate("ETH-USD")

    print("\n" + "="*80)
    print("MODELS COMPLETED")
    print("="*80)
    
    if sp500_results:
        print("S&P 500 model completed successfully")
    else:
        print("S&P 500 model failed")
        
    if eth_results:
        print("Ethereum model completed successfully")
    else:
        print("Ethereum model failed")
    
    print("\n🎯 Advanced models with infinity value handling are ready!")
    
except Exception as e:
    print("Major error running advanced models: " + str(e))
    print("Please make sure required packages are installed:")
    print("pip install yfinance pandas numpy scikit-learn xgboost")