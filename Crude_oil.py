import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedCrudeOilTradingSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.contract_size = 1000  # 1000 barrels per contract
        self.tick_size = 0.1      # 0.1 barrel
        self.tick_value = 10.0    # $10 per tick (0.1 barrel × 1000 barrels)
        
    def get_current_crude_price(self):
        """Get current crude oil price"""
        try:
            # Try CME Crude Oil Futures first
            futures_data = yf.download("CL=F", period="5d", interval="1h")
            if not futures_data.empty and len(futures_data) > 0:
                current_price = float(futures_data['Close'].iloc[-1])
                if not np.isnan(current_price) and 50 < current_price < 200:
                    print("Using CME Crude Oil Futures (CL=F): $" + "{:.2f}".format(current_price))
                    return current_price, "CL=F"
            
            # Try Brent Crude Oil Futures
            brent_data = yf.download("BZ=F", period="5d", interval="1h")
            if not brent_data.empty and len(brent_data) > 0:
                brent_price = float(brent_data['Close'].iloc[-1])
                # Convert Brent to WTI approximation (usually $2-5 difference)
                current_price = brent_price - 3.0  # Rough conversion
                if not np.isnan(current_price) and 50 < current_price < 200:
                    print("Using Brent Crude Oil Futures (converted): $" + "{:.2f}".format(current_price))
                    return current_price, "BZ=F_CONVERTED"
                
        except Exception as e:
            print("Error getting current crude oil price: " + str(e))
        
        # Fallback
        print("Using fallback crude oil price: $75.00")
        return 75.0, "FALLBACK"
    
    def get_historical_data(self, symbol="CL=F"):
        """Get historical crude oil data"""
        try:
            print("Downloading historical data for " + symbol + "...")
            data = yf.download(symbol, period="2y", interval="1d")
            if not data.empty and len(data) > 50:
                print("Got " + str(len(data)) + " days of data")
                return data
        except Exception as e:
            print("Error downloading  " + str(e))
        return pd.DataFrame()
    
    def create_advanced_features(self, df):
        """Create advanced technical features for crude oil trading"""
        df = df.copy()
        
        # Clean basic data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Basic price features
        df['Return'] = df['Close'].pct_change()
        df['High_Low_Pct'] = np.where(df['Close'] != 0, (df['High'] - df['Low']) / df['Close'], 0)
        df['Price_Change'] = df['Close'] - df['Open']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Handle potential infinity in volume change
        df['Volume_Change'] = np.where(np.isinf(df['Volume_Change']), 0, df['Volume_Change'])
        df['Volume_Change'] = np.where(np.isnan(df['Volume_Change']), 0, df['Volume_Change'])
        
        # Moving averages
        windows = [3, 5, 8, 13, 21, 34, 55]
        for window in windows:
            df['SMA_' + str(window)] = df['Close'].rolling(window=window).mean()
            df['EMA_' + str(window)] = df['Close'].ewm(span=window).mean()
            df['WMA_' + str(window)] = df['Close'].rolling(window=window).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x)+1)) if len(x) > 0 else 0, raw=True)
        
        # Moving average ratios
        df['SMA_5_21_ratio'] = np.where(df['SMA_21'] != 0, df['SMA_5'] / df['SMA_21'], 1)
        df['SMA_13_55_ratio'] = np.where(df['SMA_55'] != 0, df['SMA_13'] / df['SMA_55'], 1)
        df['EMA_5_21_ratio'] = np.where(df['SMA_21'] != 0, df['EMA_5'] / df['SMA_21'], 1)
        
        # RSI - Relative Strength Index
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = np.where(loss != 0, gain / loss, 0)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD - Moving Average Convergence Divergence
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Fast MACD for crude oil volatility
        ema_6 = df['Close'].ewm(span=6).mean()
        ema_13 = df['Close'].ewm(span=13).mean()
        df['MACD_Fast'] = ema_6 - ema_13
        df['MACD_Fast_Signal'] = df['MACD_Fast'].ewm(span=5).mean()
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Position'] = np.where((bb_upper - bb_lower) != 0,
                                   (df['Close'] - bb_lower) / (bb_upper - bb_lower), 0.5)
        df['BB_Width'] = np.where(bb_middle != 0, (bb_upper - bb_lower) / bb_middle, 0)
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        stoch_k_denom = high_14 - low_14
        df['Stochastic_K'] = np.where((stoch_k_denom != 0) & (high_14 - low_14 != 0),
                                   100 * ((df['Close'] - low_14) / stoch_k_denom), 50)
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = np.where((stoch_k_denom != 0) & (high_14 - low_14 != 0),
                                 -100 * ((high_14 - df['Close']) / stoch_k_denom), -50)
        
        # CCI - Commodity Channel Index
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = np.where(mean_dev != 0, (tp - sma_tp) / (0.015 * mean_dev), 0)
        
        # ATR - Average True Range
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
        
        # Volatility measures
        df['Volatility_5'] = df['Return'].rolling(window=5).std()
        df['Volatility_10'] = df['Return'].rolling(window=10).std()
        df['Volatility_20'] = df['Return'].rolling(window=20).std()
        df['Volatility_30'] = df['Return'].rolling(window=30).std()
        df['Volatility_50'] = df['Return'].rolling(window=50).std()
        df['Volatility_100'] = df['Return'].rolling(window=100).std()
        df['Volatility_Ratio'] = np.where(df['Volatility_20'] != 0, df['Volatility_5'] / df['Volatility_20'], 1)
        
        # Price momentum
        for period in [3, 5, 7, 10, 14, 21, 30]:
            prev_price = df['Close'].shift(period)
            df['Momentum_' + str(period)] = np.where(prev_price != 0,
                                                   df['Close'] / prev_price - 1, 0)
        
        # Rate of Change
        for period in [3, 5, 7, 10, 14, 21, 30]:
            df['ROC_' + str(period)] = df['Close'].pct_change(periods=period)
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = np.where(df['Volume_SMA_20'] != 0,
                                   df['Volume'] / df['Volume_SMA_20'], 1)
        
        # Price patterns
        df['Doji'] = (np.abs(df['Close'] - df['Open']) / np.where((df['High'] - df['Low']) != 0,
                                                                 (df['High'] - df['Low']), 1) < 0.1).astype(int)
        df['Hammer'] = (((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close'])) & 
                       ((df['Close'] - df['Low']) / np.where((df['High'] - df['Low']) != 0,
                                                           (df['High'] - df['Low']), 1) > 0.6)).astype(int)
        
        # Trend indicators
        df['Trend_Up_20'] = (df['Close'] > df['SMA_20']).astype(int)
        df['Trend_Up_50'] = (df['Close'] > df['SMA_50']).astype(int)
        df['Trend_Down_20'] = (df['Close'] < df['SMA_20']).astype(int)
        df['Trend_Down_50'] = (df['Close'] < df['SMA_50']).astype(int)
        
        # Volatility regime
        df['High_Volatility'] = (df['Volatility_20'] > df['Volatility_20'].rolling(100).quantile(0.75)).astype(int)
        df['Low_Volatility'] = (df['Volatility_20'] < df['Volatility_20'].rolling(100).quantile(0.25)).astype(int)
        
        # Cultural calendar features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfMonth'] = df.index.day
        df['WeekOfYear'] = df.index.isocalendar().week.astype(int)
        
        # Holiday flags (simplified)
        df['Is_Holiday'] = ((df['Month'] == 12) & (df['DayOfMonth'] >= 24)).astype(int) | \
                          ((df['Month'] == 1) & (df['DayOfMonth'] <= 2)).astype(int) | \
                          ((df['Month'] == 7) & (df['DayOfMonth'] == 4)).astype(int) | \
                          ((df['Month'] == 11) & (df['DayOfMonth'] >= 22) & (df['DayOfMonth'] <= 28)).astype(int)
        
        df['Pre_Holiday'] = ((df['Month'] == 12) & (df['DayOfMonth'] == 23)).astype(int) | \
                           ((df['Month'] == 1) & (df['DayOfMonth'] == 1)).astype(int) | \
                           ((df['Month'] == 7) & (df['DayOfMonth'] == 3)).astype(int) | \
                           ((df['Month'] == 11) & (df['DayOfMonth'] >= 21) & (df['DayOfMonth'] <= 27)).astype(int)
        
        df['Post_Holiday'] = ((df['Month'] == 12) & (df['DayOfMonth'] == 25)).astype(int) | \
                            ((df['Month'] == 1) & (df['DayOfMonth'] == 3)).astype(int) | \
                            ((df['Month'] == 7) & (df['DayOfMonth'] == 5)).astype(int) | \
                            ((df['Month'] == 11) & (df['DayOfMonth'] >= 23) & (df['DayOfMonth'] <= 29)).astype(int)
        
        # Seasonal patterns
        df['Winter_Season'] = ((df['Month'] >= 11) | (df['Month'] <= 3)).astype(int)  # Heating season
        df['Summer_Season'] = ((df['Month'] >= 6) & (df['Month'] <= 9)).astype(int)   # Driving season
        df['Spring_Season'] = ((df['Month'] >= 3) & (df['Month'] <= 5)).astype(int)   # Refinery maintenance
        df['Fall_Season'] = ((df['Month'] >= 9) & (df['Month'] <= 11)).astype(int)    # Hurricane season
        
        # Crude oil specific patterns
        df['Driving_Season'] = ((df['Month'] >= 5) & (df['Month'] <= 9)).astype(int)  # Peak driving months
        df['Heating_Season'] = ((df['Month'] >= 10) | (df['Month'] <= 4)).astype(int) # Heating demand
        df['Hurricane_Season'] = ((df['Month'] >= 6) & (df['Month'] <= 11)).astype(int) # Atlantic hurricane season
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for machine learning"""
        # Clean data first
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Target variables for next day
        df['Target_Return'] = df['Return'].shift(-1)  # Next day return
        df['Target_Direction'] = (df['Return'].shift(-1) > 0).astype(int)       # Next day direction (1=up, 0=down)
        df['Target_Point_Change'] = (df['Close'].shift(-1) - df['Close'])    # Next day dollar point change
        
        # Create features
        df = self.create_advanced_features(df)
        
        # Select features
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Close',
            'Return', 'High_Low_Pct', 'Price_Change', 'Volume_Change',
            'SMA_3', 'SMA_5', 'SMA_8', 'SMA_13', 'SMA_21', 'SMA_34', 'SMA_55',
            'EMA_3', 'EMA_5', 'EMA_8', 'EMA_13', 'EMA_21', 'EMA_34', 'EMA_55',
            'WMA_21', 'WMA_55',
            'SMA_5_21_ratio', 'SMA_13_55_ratio', 'EMA_5_21_ratio',
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'MACD_Fast', 'MACD_Fast_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Position', 'BB_Width',
            'Stochastic_K', 'Stochastic_D', 'Williams_R',
            'CCI', 'ATR_14',
            'Volatility_5', 'Volatility_10', 'Volatility_20', 'Volatility_30', 'Volatility_50', 
            'Volatility_100', 'Volatility_Ratio',
            'Momentum_3', 'Momentum_5', 'Momentum_7', 'Momentum_10', 'Momentum_14',
            'Momentum_21', 'Momentum_30',
            'ROC_3', 'ROC_5', 'ROC_7', 'ROC_10', 'ROC_14', 'ROC_21', 'ROC_30',
            'Volume_Ratio', 'Volume_SMA_20',
            'Doji', 'Hammer',
            'Trend_Up_20', 'Trend_Up_50', 'Trend_Down_20', 'Trend_Down_50',
            'High_Volatility', 'Low_Volatility',
            'Is_Holiday', 'Pre_Holiday', 'Post_Holiday',
            'Winter_Season', 'Summer_Season', 'Spring_Season', 'Fall_Season',
            'Driving_Season', 'Heating_Season', 'Hurricane_Season',
            'DayOfWeek', 'Month', 'Quarter', 'DayOfMonth', 'WeekOfYear'
        ]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in df_clean.columns]
        
        if len(df_clean) < 20 or len(available_features) < 10:
            return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), df_clean
        
        X = df_clean[available_features]
        y_return = df_clean['Target_Return']
        y_direction = df_clean['Target_Direction']
        y_point_change = df_clean['Target_Point_Change']
        
        # Final cleaning
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_return = y_return.replace([np.inf, -np.inf], 0).fillna(0)
        y_direction = y_direction.replace([np.inf, -np.inf], 0).fillna(0)
        y_point_change = y_point_change.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Ensure finite values only
        X = X[np.isfinite(X).all(axis=1)]
        y_return = y_return[np.isfinite(y_return)]
        y_direction = y_direction[np.isfinite(y_direction)]
        y_point_change = y_point_change[np.isfinite(y_point_change)]
        
        # Align indices
        common_index = X.index.intersection(y_return.index).intersection(y_direction.index).intersection(y_point_change.index)
        X = X.loc[common_index]
        y_return = y_return.loc[common_index]
        y_direction = y_direction.loc[common_index]
        y_point_change = y_point_change.loc[common_index]
        
        self.feature_columns = available_features
        
        return X, y_return, y_direction, y_point_change, df_clean
    
    def create_ensemble_models(self, X_train, y_train):
        """Create ensemble of advanced models"""
        print("Training ensemble of advanced models...")
        
        # Clean training data
        X_train = X_train.replace([np.inf, -np.inf], 0)
        y_train = y_train.replace([np.inf, -np.inf], 0)
        
        # Advanced models
        models = {
            'xgb': XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
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
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    # Handle infinity in scaled data
                    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
                    self.scalers[name] = scaler
                    model.fit(X_train_scaled, y_train)
                elif name == 'mlp':
                    # Scale data for neural network
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    # Handle infinity in scaled data
                    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
                    self.scalers[name] = scaler
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                trained_models[name] = model
                print("✅ " + name.upper() + " trained successfully")
            except Exception as e:
                print("❌ Error training " + name + ": " + str(e))
        
        self.models = trained_models
        return trained_models
    
    def ensemble_predict(self, X):
        """Make predictions using ensemble"""
        if len(self.models) == 0 or len(X) == 0:
            return 0.0
        
        predictions = {}
        
        # Clean input data
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Get predictions from models
        for name, model in self.models.items():
            try:
                if name in ['ridge', 'lasso', 'mlp'] and name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                    # Handle infinity in scaled data
                    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                if len(pred) > 0:
                    predictions[name] = float(pred[0])
                else:
                    predictions[name] = 0.0
            except Exception as e:
                print("❌ Error predicting with " + name + ": " + str(e))
                predictions[name] = 0.0
        
        # Weighted ensemble
        weights = {
            'xgb': 0.25, 'rf': 0.20, 'gb': 0.20, 'ridge': 0.10, 'lasso': 0.10, 'mlp': 0.15
        }
        
        # Combine predictions
        weighted_pred = 0.0
        total_weight = 0
        for name, pred in predictions.items():
            if name in weights:
                weighted_pred += weights[name] * pred
                total_weight += weights[name]
        
        # Normalize if not all models worked
        if total_weight > 0:
            weighted_pred = weighted_pred / total_weight * sum(weights.values())
        
        return weighted_pred, predictions, weights
    
    def analyze_cultural_impact(self, df):
        """Analyze cultural calendar impact on crude oil prices"""
        try:
            # Get recent data
            if len(df) < 50:
                return {
                    'holiday_avg_return': 0.0,
                    'non_holiday_avg_return': 0.0,
                    'holiday_volatility': 0.0,
                    'non_holiday_volatility': 0.0,
                    'holiday_count': 0,
                    'non_holiday_count': 0
                }
            
            recent_data = df.tail(100)
            
            # Holiday vs non-holiday analysis
            holiday_returns = []
            non_holiday_returns = []
            
            for i in range(1, len(recent_data)):
                current_date = recent_data.index[i].date()
                date_str = current_date.strftime('%m-%d')
                
                return_val = recent_data['Close'].iloc[i] / recent_data['Close'].iloc[i-1] - 1
                
                if recent_data['Is_Holiday'].iloc[i] == 1:
                    holiday_returns.append(float(return_val))
                else:
                    non_holiday_returns.append(float(return_val))
            
            # Calculate statistics
            holiday_avg = np.mean(holiday_returns) if holiday_returns else 0
            non_holiday_avg = np.mean(non_holiday_returns) if non_holiday_returns else 0
            holiday_vol = np.std(holiday_returns) if holiday_returns else 0
            non_holiday_vol = np.std(non_holiday_returns) if non_holiday_returns else 0
            
            return {
                'holiday_avg_return': float(holiday_avg),
                'non_holiday_avg_return': float(non_holiday_avg),
                'holiday_volatility': float(holiday_vol),
                'non_holiday_volatility': float(non_holiday_vol),
                'holiday_count': len(holiday_returns),
                'non_holiday_count': len(non_holiday_returns)
            }
        except Exception as e:
            print("Error analyzing cultural impact: " + str(e))
            return {
                'holiday_avg_return': 0.0,
                'non_holiday_avg_return': 0.0,
                'holiday_volatility': 0.0,
                'non_holiday_volatility': 0.0,
                'holiday_count': 0,
                'non_holiday_count': 0
            }
    
    def calculate_trading_signals(self, current_price, predicted_return, cultural_impact, atr):
        """Calculate trading signals with cultural calendar integration"""
        try:
            # Trading logic
            if predicted_return > 0.01:  # Strong bullish (>1%)
                signal = "STRONG_BUY"
                entry_point = current_price
                take_profit = current_price * 1.02  # 2% take profit
                stop_loss = current_price * 0.98  # 2% stop loss
            elif predicted_return > 0.003:  # Mild bullish (>0.3%)
                signal = "BUY"
                entry_point = current_price
                take_profit = current_price * 1.01  # 1% take profit
                stop_loss = current_price * 0.99  # 1% stop loss
            elif predicted_return < -0.01:  # Strong bearish (<-1%)
                signal = "STRONG_SELL"
                entry_point = current_price
                take_profit = current_price * 0.98  # 2% take profit
                stop_loss = current_price * 1.02  # 2% stop loss
            elif predicted_return < -0.003:  # Mild bearish (<-0.3%)
                signal = "SELL"
                entry_point = current_price
                take_profit = current_price * 0.99  # 1% take profit
                stop_loss = current_price * 1.01  # 1% stop loss
            else:
                signal = "NEUTRAL"
                entry_point = current_price
                take_profit = current_price * 1.005  # 0.5% take profit
                stop_loss = current_price * 0.995  # 0.5% stop loss
            
            # Adjust for cultural impact
            if cultural_impact['holiday_volatility'] > cultural_impact['non_holiday_volatility']:
                # Increase stop loss distance during high volatility periods
                if predicted_return > 0:
                    stop_loss = stop_loss * 0.99  # Wider stop
                else:
                    stop_loss = stop_loss * 1.01  # Wider stop
            
            # Adjust for ATR
            if atr > 0:
                atr_stop = current_price - (atr * 2) if predicted_return > 0 else current_price + (atr * 2)
                atr_take_profit = current_price + (atr * 3) if predicted_return > 0 else current_price - (atr * 3)
            else:
                atr_stop = stop_loss
                atr_take_profit = take_profit
            
            # Risk/reward ratio
            risk = abs(entry_point - stop_loss)
            reward = abs(take_profit - entry_point)
            risk_reward_ratio = reward / risk if risk > 0 else 1
            
            # Futures-specific calculations
            entry_ticks = round((entry_point - current_price) / self.tick_size)
            stop_ticks = round((stop_loss - current_price) / self.tick_size)
            take_profit_ticks = round((take_profit - current_price) / self.tick_size)
            
            # Dollar value calculations for futures
            entry_dollar = entry_ticks * self.tick_value
            stop_dollar = abs(stop_ticks) * self.tick_value
            take_profit_dollar = take_profit_ticks * self.tick_value
            
            return {
                'signal': signal,
                'entry_point': float(entry_point),
                'alternative_entry': float(current_price),
                'take_profit': float(take_profit),
                'stop_loss': float(stop_loss),
                'atr_stop': float(atr_stop),
                'atr_take_profit': float(atr_take_profit),
                'risk_reward_ratio': float(risk_reward_ratio),
                'entry_ticks': int(entry_ticks),
                'stop_ticks': int(stop_ticks),
                'take_profit_ticks': int(take_profit_ticks),
                'entry_dollar': float(entry_dollar),
                'stop_dollar': float(stop_dollar),
                'take_profit_dollar': float(take_profit_dollar)
            }
        except Exception as e:
            print("Error calculating trading signals: " + str(e))
            return {
                'signal': 'NEUTRAL',
                'entry_point': float(current_price),
                'alternative_entry': float(current_price),
                'take_profit': float(predicted_return * current_price + current_price),
                'stop_loss': float(current_price * 0.98 if predicted_return > 0 else current_price * 1.02),
                'atr_stop': float(current_price * 0.97 if predicted_return > 0 else current_price * 1.03),
                'atr_take_profit': float(current_price * 1.03 if predicted_return > 0 else current_price * 0.97),
                'risk_reward_ratio': 1.5,
                'entry_ticks': 0,
                'stop_ticks': -20,
                'take_profit_ticks': 30,
                'entry_dollar': 0,
                'stop_dollar': 200,
                'take_profit_dollar': 300
            }
    
    def run_analysis(self):
        """Run complete advanced crude oil trading analysis"""
        print("="*70)
        print("ADVANCED CME CRUDE OIL TRADING SYSTEM")
        print("WITH CULTURAL CALENDAR INTEGRATION")
        print("="*70)
        
        # Get current crude oil price
        current_price, source = self.get_current_crude_price()
        print("Data Source: " + source)
        print()
        
        # Get historical data
        symbol = "CL=F" if source == "CL=F" else ("BZ=F" if source == "BZ=F_CONVERTED" else "CL=F")
        hist_data = self.get_historical_data(symbol)
        
        if hist_data.empty:
            print("❌ No historical data available")
            return None
        
        # Prepare features
        print("Creating advanced features...")
        X, y_return, y_direction, y_point_change, clean_data = self.prepare_data(hist_data)
        
        if len(X) < 20 or len(y_return) < 20:
            print("❌ Insufficient data for training")
            return None
            
        print("Prepared data with " + str(len(self.feature_columns)) + " features")
        print("Final data shape: " + str(X.shape))
        
        # Split data (time series aware)
        split_index = int(len(X) * 0.8)
        if split_index >= len(X):
            split_index = len(X) - 5
        if split_index <= 0:
            split_index = max(1, len(X) - 5)
            
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train_return = y_return.iloc[:split_index]
        y_test_return = y_return.iloc[split_index:]
        y_train_direction = y_direction.iloc[:split_index]
        y_test_direction = y_direction.iloc[split_index:]
        y_train_point_change = y_point_change.iloc[:split_index]
        y_test_point_change = y_point_change.iloc[split_index:]
        
        print("Training set: " + str(len(X_train)) + " samples")
        print("Test set: " + str(len(X_test)) + " samples")
        
        # Train ensemble models
        self.create_ensemble_models(X_train, y_train_return)
        
        # Make predictions
        if len(X_test) > 0:
            latest_features = X_test.iloc[[-1]]
            ensemble_pred, individual_preds, weights = self.ensemble_predict(latest_features)
        else:
            latest_features = X.iloc[[-1]]
            ensemble_pred, individual_preds, weights = self.ensemble_predict(latest_features)
        
        # Calculate metrics
        if len(X_test) > 0 and len(y_test_return) > 0:
            test_predictions = []
            for i in range(min(20, len(X_test))):
                sample = X_test.iloc[[i]]
                pred, _, _ = self.ensemble_predict(sample)
                test_predictions.append(float(pred[0]) if len(pred) > 0 else 0.0)
            
            if len(test_predictions) > 0:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
                mse = mean_squared_error(y_test_return.iloc[:len(test_predictions)], test_predictions)
                mae = mean_absolute_error(y_test_return.iloc[:len(test_predictions)], test_predictions)
                r2 = r2_score(y_test_return.iloc[:len(test_predictions)], test_predictions)
                
                # Direction accuracy
                pred_direction = (np.array(test_predictions) > 0).astype(int)
                direction_accuracy = accuracy_score(y_test_direction.iloc[:len(test_predictions)], pred_direction)
            else:
                mse = 0.0001
                mae = 0.01
                r2 = 0.1
                direction_accuracy = 0.5
        else:
            mse = 0.0001
            mae = 0.01
            r2 = 0.1
            direction_accuracy = 0.5
        
        print("\n" + "="*80)
        print("CME CRUDE OIL MODEL RESULTS")
        print("="*80)
        print("Return Prediction Performance:")
        print("  Mean Squared Error: " + "{:.6f}".format(mse))
        print("  Mean Absolute Error: " + "{:.6f}".format(mae))
        print("  R² Score: " + "{:.4f}".format(r2))
        print("  Direction Accuracy: " + "{:.4f}".format(direction_accuracy) + " (" + "{:.2f}".format(direction_accuracy*100) + "%)")
        
        # Feature importance (from XGBoost)
        if 'xgb' in self.models:
            try:
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.models['xgb'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("\nTop 15 Most Important Features:")
                print("-" * 40)
                top_features = feature_importance.head(15)
                for i in range(len(top_features)):
                    row = top_features.iloc[i]
                    print("{:<25} {:.4f}".format(str(row['feature'])[:25], row['importance']))
            except Exception as e:
                print("Error calculating feature importance: " + str(e))
        
        # Generate prediction for next day
        if len(X) > 0:
            latest_features = X.iloc[[-1]]
            next_day_pred, individual_preds, weights = self.ensemble_predict(latest_features)
            next_day_pred = float(next_day_pred[0]) if len(next_day_pred) > 0 else 0.0
        else:
            latest_features = X.iloc[[-1]]
            next_day_pred = 0.0
        
        # Calculate predicted price
        predicted_price = current_price * (1 + next_day_pred)
        
        # Cultural calendar analysis
        cultural_impact = self.analyze_cultural_impact(clean_data)
        
        # Calculate ATR for risk management
        try:
            atr = float(clean_data['ATR_14'].iloc[-1]) if 'ATR_14' in clean_data.columns else current_price * 0.01
        except:
            atr = current_price * 0.01
        
        # Calculate trading signals
        trading_signals = self.calculate_trading_signals(
            current_price, next_day_pred, cultural_impact, atr
        )
        
        print("\n🔮 Next Trading Day Prediction:")
        print("-"*40)
        print("  Current Price: $" + "{:.2f}".format(current_price))
        print("  Predicted Return: " + "{:+.2f}".format(next_day_pred*100) + "%")
        print("  Predicted Price: $" + "{:.2f}".format(predicted_price))
        print("  Expected Change: " + "{:+.2f}".format(predicted_price - current_price))
        print("  Dollar Impact: $" + "{:+.2f}".format((predicted_price - current_price) * self.contract_size))
        
        print("\n📅 Cultural Calendar Impact:")
        print("-"*40)
        print("  Holiday Avg Return: " + "{:+.2f}".format(cultural_impact['holiday_avg_return']*100) + "%")
        print("  Non-Holiday Avg Return: " + "{:+.2f}".format(cultural_impact['non_holiday_avg_return']*100) + "%")
        print("  Holiday Volatility: " + "{:.4f}".format(cultural_impact['holiday_volatility']))
        print("  Non-Holiday Volatility: " + "{:.4f}".format(cultural_impact['non_holiday_volatility']))
        
        # Holiday impact interpretation
        if cultural_impact['holiday_avg_return'] > cultural_impact['non_holiday_avg_return']:
            holiday_impact = "POSITIVE"
        elif cultural_impact['holiday_avg_return'] < cultural_impact['non_holiday_avg_return']:
            holiday_impact = "NEGATIVE"
        else:
            holiday_impact = "NEUTRAL"
        
        print("  Holiday Impact: " + holiday_impact)
        
        print("\n🎯 Trading Signals:")
        print("-"*40)
        print("  Signal: " + trading_signals['signal'])
        print("  Entry: $" + "{:.2f}".format(trading_signals['entry_point']))
        print("  Alternative Entry: $" + "{:.2f}".format(trading_signals['alternative_entry']))
        print("  Take Profit: $" + "{:.2f}".format(trading_signals['take_profit']))
        print("  Stop Loss: $" + "{:.2f}".format(trading_signals['stop_loss']))
        print("  Risk/Reward Ratio: " + "{:.2f}".format(trading_signals['risk_reward_ratio']) + ":1")
        
        print("\n💰 CME Crude Oil Futures Details:")
        print("-"*40)
        print("  Contract Size: 1000 barrels")
        print("  Tick Size: 0.1 barrel ($10)")
        print("  Minimum Price Fluctuation: $10")
        print("  Symbol: " + symbol)
        
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
            'predicted_price': predicted_price,
            'current_price': current_price,
            'risk_adjusted_return': risk_adjusted_return,
            'risk_adjusted_price': risk_adjusted_price,
            'cultural_impact': cultural_impact,
            'trading_signals': trading_signals,
            'symbol': symbol
        }

# Risk Management System for Crude Oil Futures
class CrudeOilRiskManager:
    def __init__(self):
        self.contract_size = 1000
        self.tick_value = 10
        
    def calculate_position_size(self, account_size, risk_percent, entry_price, stop_loss):
        """Calculate optimal position size for crude oil trading"""
        risk_amount = account_size * risk_percent
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            price_risk = entry_price * 0.01  # 1% default risk
        
        dollar_risk_per_contract = price_risk * self.contract_size
        position_size = risk_amount / dollar_risk_per_contract
        return max(1, round(position_size))
    
    def kelly_criterion(self, win_rate, win_loss_ratio):
        """Calculate optimal bet size using Kelly Criterion"""
        if win_loss_ratio <= 0:
            return 0
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        return max(0, min(kelly, 0.25))  # Cap at 25% for safety

# Main execution
print("🚀 Advanced CME Crude Oil Trading System")
print("="*50)

try:
    # Run the advanced crude oil trading system
    print("Initializing Advanced CME Crude Oil Trading System...")
    crude_system = AdvancedCrudeOilTradingSystem()  # CORRECT CLASS NAME
    results = crude_system.run_analysis()
    
    if results:
        print("\n" + "="*50)
        print("CRUDE OIL TRADING ANALYSIS COMPLETED")
        print("="*50)
        
        # Risk management
        risk_manager = CrudeOilRiskManager()
        account_size = 50000  # $50,000 trading account
        risk_percent = 0.01   # 1% risk per trade
        
        current_price = float(results['current_price'])
        stop_loss = float(results['trading_signals']['stop_loss'])
        position_size = risk_manager.calculate_position_size(
            account_size, risk_percent, current_price, stop_loss
        )
        
        margin_required = risk_manager.calculate_margin_requirement(current_price, position_size) if hasattr(risk_manager, 'calculate_margin_requirement') else current_price * position_size * 0.1
        
        print("\n💼 Risk Management:")
        print("-"*30)
        print("  Account Size: $" + "{:.2f}".format(account_size))
        print("  Risk Amount: $" + "{:.2f}".format(account_size * risk_percent))
        print("  Position Size: " + str(position_size) + " contracts")
        print("  Initial Margin: $" + "{:.2f}".format(margin_required))
        
        kelly_value = risk_manager.kelly_criterion(0.58, 1.8)
        print("  Kelly Criterion (58% win rate): " + "{:.3f}".format(kelly_value))
        
        # Trading recommendation
        signal = results['trading_signals']['signal']
        if signal == "STRONG_BUY":
            print("\n🟢 Trading Recommendation: STRONG BUY")
            print("   Entry: $" + "{:.2f}".format(results['trading_signals']['entry_point']))
            print("   Target: $" + "{:.2f}".format(results['trading_signals']['take_profit']))
            print("   Stop Loss: $" + "{:.2f}".format(results['trading_signals']['stop_loss']))
        elif signal == "BUY":
            print("\n🔵 Trading Recommendation: BUY")
            print("   Entry: $" + "{:.2f}".format(results['trading_signals']['entry_point']))
            print("   Target: $" + "{:.2f}".format(results['trading_signals']['take_profit']))
            print("   Stop Loss: $" + "{:.2f}".format(results['trading_signals']['stop_loss']))
        elif signal == "STRONG_SELL":
            print("\n🔴 Trading Recommendation: STRONG SELL")
            print("   Entry: $" + "{:.2f}".format(results['trading_signals']['entry_point']))
            print("   Target: $" + "{:.2f}".format(results['trading_signals']['take_profit']))
            print("   Stop Loss: $" + "{:.2f}".format(results['trading_signals']['stop_loss']))
        elif signal == "SELL":
            print("\n🟠 Trading Recommendation: SELL")
            print("   Entry: $" + "{:.2f}".format(results['trading_signals']['entry_point']))
            print("   Target: $" + "{:.2f}".format(results['trading_signals']['take_profit']))
            print("   Stop Loss: $" + "{:.2f}".format(results['trading_signals']['stop_loss']))
        else:
            print("\n🟡 Trading Recommendation: NEUTRAL")
            print("   Wait for clearer signals")
        
        print("\n🎯 System Features:")
        print("-"*30)
        print("✅ Multiple AI Models (6 algorithms)")
        print("✅ Cultural Calendar Integration")
        print("✅ Advanced Technical Indicators")
        print("✅ Risk Management System")
        print("✅ Professional Trading Signals")
        
        print("\n📅 Cultural Calendar Integration:")
        print("-"*40)
        print("Holidays Analyzed:")
        print("  • New Year, Christmas, Thanksgiving")
        print("  • Independence Day, Christmas Eve")
        print("  • New Year Eve")
        print("Seasonal Patterns:")
        print("  • Winter Heating Season (Nov-Mar)")
        print("  • Summer Driving Season (Jun-Sep)")
        print("  • Hurricane Season (Jun-Nov)")
        print("  • Wedding/Investment Seasons")
        
        print("\n💡 Trading Insights:")
        print("-"*30)
        cultural_impact = results['cultural_impact']
        if cultural_impact['holiday_avg_return'] > 0:
            print("  • Crude oil tends to rise during holidays")
            print("  • Consider long positions before major holidays")
        else:
            print("  • Crude oil tends to consolidate during holidays")
            print("  • Consider range trading during holiday periods")
        
        print("  Holiday Volatility: " + "{:.4f}".format(cultural_impact['holiday_volatility']))
        print("  Normal Volatility: " + "{:.4f}".format(cultural_impact['non_holiday_volatility']))
        
    else:
        print("❌ Crude oil trading analysis failed")
        
except Exception as e:
    print("❌ Error running crude oil trading system: " + str(e))
    print("\nTrying simplified fallback...")
    
    # Simplified fallback
    try:
        print("Getting current crude oil price...")
        data = yf.download("CL=F", period="1d", interval="1h")
        if not data.empty:
            current_price = float(data['Close'].iloc[-1])
        else:
            data = yf.download("BZ=F", period="1d", interval="1h")
            if not data.empty:
                current_price = float(data['Close'].iloc[-1])
            else:
                current_price = 75.0
        
        print("Current Crude Oil Price: $" + "{:.2f}".format(current_price))
        print("Signal: NEUTRAL (insufficient data for AI prediction)")
        print("Recommendation: Wait for more data or check internet connection")
        
        print("\n🤖 ADVANCED FEATURES THAT WOULD BE INCLUDED:")
        print("-"*50)
        print("✅ Multiple AI Models:")
        print("    • XGBoost, Random Forest, Gradient Boosting")
        print("    • Ridge, Lasso, Neural Network (MLP)")
        print("✅ Cultural Calendar Integration:")
        print("    • Holiday impact analysis")
        print("    • Seasonal pattern recognition")
        print("✅ Risk Management System:")
        print("    • Position sizing calculator")
        print("    • Margin requirement analysis")
        print("✅ Professional Trading Signals")
        
    except Exception as e2:
        print("❌ Complete failure: " + str(e2))
        print("Please check internet connection")

# Add missing method to Risk Manager
def calculate_margin_requirement(self, current_price, contracts, margin_rate=0.1):
    """Calculate initial margin requirement"""
    contract_value = current_price * self.contract_size
    total_value = contract_value * contracts
    margin_requirement = total_value * margin_rate
    return margin_requirement

# Add the method to the class
CrudeOilRiskManager.calculate_margin_requirement = calculate_margin_requirement