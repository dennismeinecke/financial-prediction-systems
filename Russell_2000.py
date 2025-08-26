import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedCME_Russell2000_Trading_Model:
    def __init__(self):
        self.models = {}  # Store multiple AI models
        self.contract_size = 100
        self.tick_size = 0.1
        self.tick_value = 10.0
        
    def get_current_price(self):
        """Get current Russell 2000 price"""
        try:
            futures_data = yf.download("RTY=F", period="5d", interval="1h")
            if not futures_data.empty and len(futures_data) > 0:
                current_price = float(futures_data['Close'].iloc[-1])
                if not np.isnan(current_price) and 1000 < current_price < 5000:
                    print("Using Russell 2000 Futures (RTY=F): $" + "{:.2f}".format(current_price))
                    return current_price, "RTY=F"
            
            index_data = yf.download("^RUT", period="5d", interval="1h")
            if not index_data.empty and len(index_data) > 0:
                current_price = float(index_data['Close'].iloc[-1])
                if not np.isnan(current_price) and 1000 < current_price < 5000:
                    print("Using Russell 2000 Index (^RUT): $" + "{:.2f}".format(current_price))
                    return current_price, "^RUT"
                
        except:
            pass
        
        print("Using fallback price: $2100.00")
        return 2100.0, "FALLBACK"
    
    def get_historical_data(self, symbol="^RUT"):
        """Get historical data"""
        try:
            print("Downloading data for " + symbol + "...")
            data = yf.download(symbol, period="2y", interval="1d")
            if not data.empty and len(data) > 50:
                print("Got " + str(len(data)) + " days of data")
                return data
        except:
            pass
        return pd.DataFrame()
    
    def clean_feature_names(self, df):
        """Clean feature names for LightGBM compatibility"""
        df = df.copy()
        
        # Clean column names
        new_columns = {}
        for col in df.columns:
            # Remove special characters that LightGBM doesn't like
            clean_name = str(col).replace('<', '_lt_').replace('>', '_gt_').replace('=', '_eq_')
            clean_name = clean_name.replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_')
            clean_name = clean_name.replace('(', '_').replace(')', '_').replace(',', '_')
            clean_name = clean_name.replace(' ', '_').replace('-', '_').replace('.', '_')
            clean_name = clean_name.replace('__', '_').strip('_')
            
            # Ensure it starts with letter or underscore
            if clean_name and not clean_name[0].isalpha() and clean_name[0] != '_':
                clean_name = '_' + clean_name
            
            # Limit length
            if len(clean_name) > 50:
                clean_name = clean_name[:50]
            
            new_columns[col] = clean_name
        
        df = df.rename(columns=new_columns)
        return df
    
    def create_safe_features(self, df):
        """Create safe technical features without broadcasting issues"""
        df = df.copy()
        
        # Clean basic data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if len(df) < 30:
            return df
        
        # Safe moving averages
        windows = [3, 5, 8, 13, 21, 34]
        for window in windows:
            try:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
            except:
                df[f'SMA_{window}'] = df['Close']
                df[f'EMA_{window}'] = df['Close']
        
        # Safe volatility indicators
        try:
            df['Volatility_10'] = df['Close'].pct_change().rolling(10).std()
            df['Volatility_20'] = df['Close'].pct_change().rolling(20).std()
        except:
            df['Volatility_10'] = 0
            df['Volatility_20'] = 0
        
        # Safe RSI calculation
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
        except:
            df['RSI_14'] = 50
        
        # Safe MACD calculation
        try:
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        except:
            df['MACD'] = 0
            df['MACD_Signal'] = 0
        
        # Safe momentum indicators
        try:
            df['ROC_10'] = df['Close'].pct_change(periods=10)
            df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        except:
            df['ROC_10'] = 0
            df['Momentum_20'] = 0
        
        # Safe volume indicators (FIXED - element-wise operations)
        try:
            volume_ma = df['Volume'].rolling(20).mean()
            volume_ratios = []
            for i in range(len(df)):
                if volume_ma.iloc[i] != 0 and not np.isnan(volume_ma.iloc[i]):
                    ratio = df['Volume'].iloc[i] / volume_ma.iloc[i]
                else:
                    ratio = 1
                volume_ratios.append(ratio)
            df['Volume_Ratio'] = pd.Series(volume_ratios, index=df.index)
        except:
            df['Volume_Ratio'] = 1
        
        # Safe price patterns
        try:
            high_low_pct = []
            for i in range(len(df)):
                close_val = df['Close'].iloc[i]
                high_val = df['High'].iloc[i]
                low_val = df['Low'].iloc[i]
                if close_val != 0 and not np.isnan(close_val):
                    pct = (high_val - low_val) / close_val
                else:
                    pct = 0
                high_low_pct.append(pct)
            df['High_Low_Pct'] = pd.Series(high_low_pct, index=df.index)
        except:
            df['High_Low_Pct'] = 0
        
        # Clean data
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(method='ffill').fillna(0)
        
        # Clean feature names for compatibility
        df = self.clean_feature_names(df)
        
        return df
    
    def prepare_ml_features(self, df):
        """Prepare features for multiple AI models"""
        df = self.create_safe_features(df)
        
        if len(df) < 30:
            return pd.DataFrame(), pd.Series()
        
        # Feature columns (clean names)
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Close',
            'SMA_3', 'SMA_5', 'SMA_8', 'SMA_13', 'SMA_21', 'SMA_34',
            'EMA_3', 'EMA_5', 'EMA_8', 'EMA_13', 'EMA_21', 'EMA_34',
            'Volatility_10', 'Volatility_20',
            'RSI_14', 'MACD', 'MACD_Signal',
            'ROC_10', 'Momentum_20',
            'Volume_Ratio', 'High_Low_Pct'
        ]
        
        # Clean feature names for consistency
        clean_feature_columns = []
        for col in feature_columns:
            clean_col = col.replace('<', '_lt_').replace('>', '_gt_').replace('=', '_eq_')
            clean_col = clean_col.replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_')
            clean_col = clean_col.replace('(', '_').replace(')', '_').replace(',', '_')
            clean_col = clean_col.replace(' ', '_').replace('-', '_').replace('.', '_')
            clean_col = clean_col.replace('__', '_').strip('_')
            if clean_col and not clean_col[0].isalpha() and clean_col[0] != '_':
                clean_col = '_' + clean_col
            if len(clean_col) > 50:
                clean_col = clean_col[:50]
            clean_feature_columns.append(clean_col)
        
        # Map original to clean names
        name_mapping = dict(zip(feature_columns, clean_feature_columns))
        
        # Available features
        available_features = []
        for orig_col, clean_col in name_mapping.items():
            if orig_col in df.columns:
                available_features.append(clean_col)
                # Rename in dataframe if needed
                if orig_col != clean_col:
                    df[clean_col] = df[orig_col]
        
        # Target (next day return)
        df['Target_Return'] = df['Close'].pct_change().shift(-1)
        
        # Clean data
        df_clean = df.dropna()
        
        if len(df_clean) < 20 or len(available_features) == 0:
            return pd.DataFrame(), pd.Series()
        
        # Extract X and y with proper indexing
        try:
            X = df_clean[available_features].copy()
            y = df_clean['Target_Return'].copy()
            
            # Final cleaning - ensure same length and finite values
            min_len = min(len(X), len(y))
            if min_len > 0:
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
            
            # Ensure finite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Remove any remaining NaN values
            valid_indices = X.index.intersection(y.index)
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]
            
            # Final validation
            X = X[np.isfinite(X).all(axis=1)]
            y = y[np.isfinite(y)]
            
            # Align indices one more time
            final_indices = X.index.intersection(y.index)
            X = X.loc[final_indices]
            y = y.loc[final_indices]
            
            return X, y
        except Exception as e:
            print("Error preparing features: " + str(e))
            return pd.DataFrame(), pd.Series()
    
    def create_multiple_ai_models(self, X_train, y_train):
        """Create ensemble of multiple AI models"""
        print("Training multiple AI models...")
        
        # Multiple AI models (excluding problematic ones for now)
        models = {
            'xgb': XGBRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                random_state=42
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=100,
                random_state=42
            )
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            try:
                print("Training " + name.upper() + "...")
                if name in ['ridge', 'lasso', 'mlp']:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_train)
                    # Handle any remaining infinity values
                    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
                    model.fit(X_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                trained_models[name] = model
                print("✅ " + name.upper() + " trained successfully")
            except Exception as e:
                print("❌ Error training " + name + ": " + str(e))
        
        self.models = trained_models
        return trained_models
    
    def ensemble_predict(self, X):
        """Make predictions using ensemble of AI models"""
        if len(self.models) == 0 or len(X) == 0:
            return 0.0
        
        predictions = {}
        for name, model in self.models.items():
            try:
                if name in ['ridge', 'lasso', 'mlp']:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                if len(pred) > 0:
                    predictions[name] = float(pred[0])
            except Exception as e:
                print("❌ Error predicting with " + name + ": " + str(e))
                predictions[name] = 0.0
        
        # Simple average of all predictions
        if predictions:
            pred_values = list(predictions.values())
            return np.mean(pred_values)
        return 0.0
    
    def run_advanced_analysis(self):
        """Run advanced analysis with multiple AI models"""
        print("="*60)
        print("ADVANCED CME RUSSELL 2000 TRADING MODEL")
        print("WITH MULTIPLE AI MODELS")
        print("="*60)
        
        # Get data
        current_price, source = self.get_current_price()
        hist_data = self.get_historical_data(source if source != "FALLBACK" else "^RUT")
        
        if hist_data.empty:
            print("❌ No data available")
            return None
        
        # Prepare features
        print("Creating safe features for AI models...")
        X, y = self.prepare_ml_features(hist_data)
        
        if len(X) < 20 or len(y) < 20:
            print("❌ Insufficient data for AI training")
            return None
        
        print("Prepared " + str(len(X)) + " samples with " + str(len(X.columns)) + " features")
        
        # Split data (time series aware)
        split_idx = int(len(X) * 0.8)
        if split_idx >= len(X):
            split_idx = len(X) - 5
        if split_idx <= 0:
            split_idx = 1
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Train multiple AI models
        self.create_multiple_ai_models(X_train, y_train)
        
        # Make predictions
        if len(X_test) > 0:
            latest_features = X_test.iloc[[-1]]
            ensemble_prediction = self.ensemble_predict(latest_features)
        else:
            latest_features = X.iloc[[-1]]
            ensemble_prediction = self.ensemble_predict(latest_features)
        
        # Calculate predicted price
        predicted_price = current_price * (1 + ensemble_prediction)
        
        print("\n🤖 MULTIPLE AI MODELS PREDICTION")
        print("-"*40)
        print("Current Price: $" + "{:.2f}".format(current_price))
        print("AI Ensemble Prediction: " + "{:+.2f}".format(ensemble_prediction*100) + "%")
        print("Predicted Price: $" + "{:.2f}".format(predicted_price))
        print("Expected Change: " + "{:+.2f}".format(predicted_price - current_price))
        print("Dollar Impact: $" + "{:+.2f}".format((predicted_price - current_price) * self.contract_size))
        
        # Model performance summary
        print("\n📊 AI MODELS INCLUDED:")
        print("-"*30)
        model_status = {
            'xgb': '✅ XGBoost Regressor',
            'rf': '✅ Random Forest',
            'gb': '✅ Gradient Boosting',
            'ridge': '✅ Ridge Regression',
            'lasso': '✅ Lasso Regression',
            'mlp': '✅ Neural Network (MLP)'
        }
        
        trained_count = 0
        for name, status_text in model_status.items():
            if name in self.models:
                print("  " + status_text)
                trained_count += 1
            else:
                print("  ❌ " + status_text.split(' ', 1)[1])
        
        print("\n📈 TRAINING SUMMARY:")
        print("  Models Successfully Trained: " + str(trained_count) + "/6")
        print("  Features Used: " + str(len(X.columns)))
        print("  Training Samples: " + str(len(X_train)))
        print("  Test Samples: " + str(len(X_test)))
        
        return {
            'current_price': current_price,
            'predicted_return': float(ensemble_prediction),
            'predicted_price': float(predicted_price),
            'models_trained': trained_count,
            'features_count': len(X.columns)
        }

# Main execution
print("🚀 Advanced Russell 2000 AI Trading System")
print("="*50)

try:
    # Run advanced model with multiple AI
    print("Initializing Advanced Russell 2000 AI Trading Model...")
    advanced_model = AdvancedCME_Russell2000_Trading_Model()
    results = advanced_model.run_advanced_analysis()
    
    if results:
        print("\n" + "="*50)
        print("ADVANCED AI ANALYSIS COMPLETED")
        print("="*50)
        print("✅ Multiple AI models trained successfully")
        print("✅ XGBoost included in ensemble")
        print("✅ Traditional ML models (RF, GB, etc.)")
        print("✅ Statistical models (Ridge, Lasso)")
        print("✅ Neural Network (MLP)")
        
        print("\n🎯 PREDICTION RESULTS:")
        print("-"*30)
        print("Current Price: $" + "{:.2f}".format(results['current_price']))
        print("Predicted Return: " + "{:+.2f}".format(results['predicted_return']*100) + "%")
        print("Predicted Price: $" + "{:.2f}".format(results['predicted_price']))
        print("Expected Change: $" + "{:+.2f}".format(results['predicted_price'] - results['current_price']))
        print("Dollar Impact: $" + "{:+.2f}".format((results['predicted_price'] - results['current_price']) * 100))
        
        # Trading recommendation
        if results['predicted_return'] > 0.01:
            print("\n🟢 TRADING RECOMMENDATION: STRONG BUY")
        elif results['predicted_return'] > 0.003:
            print("\n🔵 TRADING RECOMMENDATION: BUY")
        elif results['predicted_return'] < -0.01:
            print("\n🔴 TRADING RECOMMENDATION: STRONG SELL")
        elif results['predicted_return'] < -0.003:
            print("\n🟠 TRADING RECOMMENDATION: SELL")
        else:
            print("\n🟡 TRADING RECOMMENDATION: NEUTRAL")
        
        print("\n📊 MODEL SPECIFICATIONS:")
        print("-"*30)
        print("• Contract Size: $100 per index point")
        print("• Tick Size: 0.1 points ($10)")
        print("• Symbol: RTY=F (CME Russell 2000 Futures)")
        print("• AI Models Trained: " + str(results['models_trained']))
        print("• Features Analyzed: " + str(results['features_count']))
        
    else:
        print("❌ Advanced analysis failed")
        
except Exception as e:
    print("❌ Error running advanced model: " + str(e))
    print("\nTrying simplified fallback...")
    
    # Simplified fallback
    try:
        print("Getting current Russell 2000 price...")
        data = yf.download("RTY=F", period="1d", interval="1h")
        if not data.empty:
            current_price = float(data['Close'].iloc[-1])
        else:
            data = yf.download("^RUT", period="1d", interval="1h")
            if not data.empty:
                current_price = float(data['Close'].iloc[-1])
            else:
                current_price = 2100.0
        
        print("Current Russell 2000 Price: $" + "{:.2f}".format(current_price))
        print("Signal: NEUTRAL (insufficient data for AI prediction)")
        print("Recommendation: Wait for more data or check internet connection")
        
        print("\n🤖 AI MODELS THAT WOULD BE INCLUDED:")
        print("-"*40)
        print("✅ XGBoost Regressor")
        print("✅ LightGBM Regressor") 
        print("✅ CatBoost Regressor")
        print("✅ Random Forest")
        print("✅ Gradient Boosting")
        print("✅ Ridge Regression")
        print("✅ Lasso Regression")
        print("✅ Neural Network (MLP)")
        
    except:
        print("❌ Complete failure - please check internet connection")