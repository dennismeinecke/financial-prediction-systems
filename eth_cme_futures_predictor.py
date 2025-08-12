# eth_cme_futures_predictor.py (Enhanced with XGBoost)
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
import json
import os
from datetime import datetime

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow and protobuf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with 'pip install xgboost'")

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Global settings
tf.random.set_seed(42)
np.random.seed(42)

class EthereumCMEPredictor:
    def __init__(self, model_dir="eth_cme_models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.seq_feature_columns = []
        self.training_history = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def fetch_eth_cme_data(self, period="2y"):
        """Fetch Ethereum CME futures data"""
        print("Fetching Ethereum CME futures data...")
        # Try CME ETH futures ticker first, fallback to spot ETH
        try:
            eth = yf.Ticker("ETH=F")  # CME Ethereum futures
            data = eth.history(period=period)
            if len(data) < 10:  # If no CME data, fallback to spot
                raise Exception("No CME data available")
        except:
            print("CME data not available, using spot ETH data...")
            eth = yf.Ticker("ETH-USD")
            data = eth.history(period=period)
        return data
    
    def fetch_bitcoin_data(self, period="2y"):
        """Fetch Bitcoin data for correlation features"""
        print("Fetching Bitcoin data...")
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period=period)
        return data
    
    def fetch_vix_data(self, period="2y"):
        """Fetch VIX (Volatility Index) data"""
        try:
            print("Fetching VIX data...")
            vix = yf.Ticker("^VIX")
            data = vix.history(period=period)
            return data
        except:
            print("Warning: Could not fetch VIX data")
            return None
    
    def clean_infinity_values(self, df):
        """Clean infinity and NaN values from dataframe"""
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        df = df.fillna(0)
        return df
    
    def compute_atr(self, df, window=14):
        """Compute Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = pd.Series(true_range).rolling(window=window).mean()
        return atr.fillna(method='bfill').fillna(0)
    
    def compute_obv(self, df):
        """Compute On-Balance Volume"""
        obv = pd.Series(index=df.index)
        obv.iloc[0] = 0
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def compute_cci(self, df, window=20):
        """Compute Commodity Channel Index"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=window).mean()
        mean_dev = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma_tp) / (0.015 * mean_dev)
        return cci.fillna(0)
    
    def compute_williams_r(self, df, window=14):
        """Compute Williams %R"""
        highest_high = df['High'].rolling(window=window).max()
        lowest_low = df['Low'].rolling(window=window).min()
        williams_r = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
        return williams_r.fillna(0)
    
    def compute_adx(self, df, window=14):
        """Compute Average Directional Index"""
        # Calculate True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        tr = np.max(ranges, axis=1)
        
        # Calculate +DM and -DM
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        # Smooth TR, +DM, -DM
        tr_smooth = pd.Series(tr).rolling(window=window).mean()
        plus_dm_smooth = plus_dm.rolling(window=window).mean()
        minus_dm_smooth = minus_dm.rolling(window=window).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        
        # Calculate ADX
        adx = dx.rolling(window=window).mean()
        
        return adx.fillna(0)
    
    def create_features(self, eth_data, btc_data=None, vix_data=None):
        """Create technical indicators and features for Ethereum"""
        df = eth_data.copy()
        
        # Daily percentage change with safe division
        df['Daily_Return'] = df['Close'].pct_change()
        df['Daily_Return'] = df['Daily_Return'].replace([np.inf, -np.inf], 0)
        
        # Technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['RSI'] = self.compute_rsi(df['Close'])
        df['Volatility'] = df['Daily_Return'].rolling(window=10).std()
        df['ATR'] = self.compute_atr(df)
        df['CCI'] = self.compute_cci(df)
        df['Williams_R'] = self.compute_williams_r(df)
        df['ADX'] = self.compute_adx(df)
        
        # Safe volume change calculation
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], 0)
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['High_Low_Pct'] = df['High_Low_Pct'].replace([np.inf, -np.inf], 0)
        df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
        df['Price_Change'] = df['Price_Change'].replace([np.inf, -np.inf], 0)
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Position'] = df['BB_Position'].replace([np.inf, -np.inf], 0)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stochastic_K'] = df['Stochastic_K'].replace([np.inf, -np.inf], 50)
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
        
        # Momentum indicators
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
        df['ROC_10'] = df['ROC_10'].replace([np.inf, -np.inf], 0)
        
        # Volume indicators
        df['OBV'] = self.compute_obv(df)
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], 0)
        
        # Cross-market features (if BTC data available)
        if btc_data is not None:
            btc_returns = btc_data['Close'].pct_change().replace([np.inf, -np.inf], 0)
            df['BTC_Return'] = btc_returns.reindex(df.index, method='ffill')
            df['ETH_BTC_Correlation'] = df['Daily_Return'].rolling(window=30).corr(df['BTC_Return'])
            df['BTC_Volume_Change'] = btc_data['Volume'].pct_change().replace([np.inf, -np.inf], 0).reindex(df.index, method='ffill')
        else:
            df['BTC_Return'] = 0
            df['ETH_BTC_Correlation'] = 0
            df['BTC_Volume_Change'] = 0
        
        # Market sentiment features (if VIX data available)
        if vix_data is not None:
            vix_close = vix_data['Close'].reindex(df.index, method='ffill')
            df['VIX'] = vix_close.fillna(method='bfill').fillna(0)
            df['VIX_MA_10'] = df['VIX'].rolling(window=10).mean()
            df['VIX_RSI'] = self.compute_rsi(pd.Series(df['VIX']))
        else:
            df['VIX'] = 0
            df['VIX_MA_10'] = 0
            df['VIX_RSI'] = 50
        
        # Lagged features with safe handling
        for i in range(1, 6):
            df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume_Change'].shift(i)
            df[f'RSI_Lag_{i}'] = df['RSI'].shift(i)
            df[f'Volatility_Lag_{i}'] = df['Volatility'].shift(i)
            df[f'Volume_Ratio_Lag_{i}'] = df['Volume_Ratio'].shift(i)
            if btc_data is not None:
                df[f'BTC_Return_Lag_{i}'] = df['BTC_Return'].shift(i)
        
        # Target variable (next day's return) with safe handling
        df['Target'] = df['Daily_Return'].shift(-1)
        df['Target'] = df['Target'].replace([np.inf, -np.inf], 0)
        
        # Clean the data
        df = self.clean_infinity_values(df)
        
        return df.dropna()
    
    def compute_rsi(self, prices, window=14):
        """Compute Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.replace([np.inf, -np.inf], 50)
    
    def prepare_sequences(self, data, sequence_length=20):
        """Prepare sequences for time series models"""
        feature_columns = [col for col in data.columns if col not in 
                          ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
        
        features_df = data[feature_columns].copy()
        target_df = data['Target'].copy()
        
        features_df = features_df.replace([np.inf, -np.inf], 0)
        target_df = target_df.replace([np.inf, -np.inf], 0)
        
        scaler_features = StandardScaler()
        scaler_target = StandardScaler()
        
        features = features_df.values
        target = target_df.values.reshape(-1, 1)
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        features_scaled = scaler_features.fit_transform(features)
        target_scaled = scaler_target.fit_transform(target).flatten()
        
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target_scaled[i])
        
        return (np.array(X), np.array(y), 
                scaler_features, scaler_target, 
                feature_columns)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def build_gru_model(self, input_shape):
        """Build GRU model"""
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block"""
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)

        y = Dense(ff_dim, activation="relu")(x)
        y = Dropout(dropout)(y)
        y = Dense(inputs.shape[-1])(y)
        y = LayerNormalization(epsilon=1e-6)(x + y)
        return y
    
    def build_transformer_model(self, input_shape):
        """Build Transformer model"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        for _ in range(2):
            x = self.transformer_encoder(x, head_size=32, num_heads=2, ff_dim=64, dropout=0.1)
        
        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model with data cleaning and outlier clipping"""
        X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train_clean = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values in training data
        y_train_clean = np.clip(y_train_clean, -0.2, 0.2)  # Crypto can be more volatile
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train_clean, y_train_clean)
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with data cleaning and outlier clipping"""
        if not XGBOOST_AVAILABLE:
            return None
            
        X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train_clean = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values in training data
        y_train_clean = np.clip(y_train_clean, -0.2, 0.2)
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_clean, y_train_clean)
        return model
    
    def clip_predictions(self, predictions, min_val=-0.2, max_val=0.2):
        """Clip predictions to reasonable range for crypto"""
        if isinstance(predictions, dict):
            return {k: np.clip(v, min_val, max_val) for k, v in predictions.items()}
        else:
            return np.clip(predictions, min_val, max_val)
    
    def calculate_risk_metrics(self, data, predictions):
        """Calculate risk metrics for the prediction"""
        current_volatility = data['Volatility'].iloc[-1]
        current_atr = data['ATR'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Value at Risk (simplified)
        var_95 = current_price * current_volatility * 1.645  # 95% confidence level
        var_99 = current_price * current_volatility * 2.33   # 99% confidence level
        
        # Risk-adjusted prediction
        prediction_std = np.std(list(predictions.values())) if isinstance(predictions, dict) else 0.01
        confidence_interval = 1.96 * prediction_std  # 95% confidence interval
        
        return {
            'current_volatility': float(current_volatility),
            'current_atr': float(current_atr),
            'value_at_risk_95': float(var_95),
            'value_at_risk_99': float(var_99),
            'prediction_std': float(prediction_std),
            'confidence_interval': float(confidence_interval)
        }
    
    def get_market_regime(self, data):
        """Determine current market regime based on indicators"""
        rsi = data['RSI'].iloc[-1]
        volatility = data['Volatility'].iloc[-1]
        vix = data['VIX'].iloc[-1] if 'VIX' in data.columns else 20
        btc_corr = data['ETH_BTC_Correlation'].iloc[-1] if 'ETH_BTC_Correlation' in data.columns else 0
        
        # Market regime classification
        if rsi > 70:
            trend = "Overbought"
        elif rsi < 30:
            trend = "Oversold"
        else:
            trend = "Neutral"
        
        if volatility > data['Volatility'].quantile(0.75):
            vol_regime = "High Volatility"
        elif volatility < data['Volatility'].quantile(0.25):
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Normal Volatility"
        
        if vix > 30:
            fear_greed = "High Fear"
        elif vix < 20:
            fear_greed = "Low Fear (Greed)"
        else:
            fear_greed = "Neutral"
        
        if btc_corr > 0.7:
            correlation_regime = "Strong BTC Correlation"
        elif btc_corr < 0.3:
            correlation_regime = "Low BTC Correlation"
        else:
            correlation_regime = "Moderate BTC Correlation"
        
        return {
            'trend': trend,
            'volatility_regime': vol_regime,
            'fear_greed_index': fear_greed,
            'correlation_regime': correlation_regime,
            'rsi_level': float(rsi),
            'vix_level': float(vix),
            'btc_correlation': float(btc_corr)
        }
    
    def adjust_prediction_for_risk(self, ensemble_pred, risk_metrics, market_regime):
        """Adjust prediction based on risk factors"""
        # Risk adjustment factor based on volatility
        vol_adjustment = min(1.0, 1.0 / (1.0 + risk_metrics['current_volatility'] * 5))
        
        # Market regime adjustment
        regime_multiplier = 1.0
        if market_regime['trend'] == "Overbought":
            regime_multiplier = 0.7  # Reduce bullish predictions
        elif market_regime['trend'] == "Oversold":
            regime_multiplier = 1.3  # Increase bullish predictions
        
        # Fear/Greed adjustment
        if market_regime['fear_greed_index'] == "High Fear":
            regime_multiplier *= 0.8
        elif market_regime['fear_greed_index'] == "Low Fear (Greed)":
            regime_multiplier *= 1.2
        
        # BTC correlation adjustment
        if market_regime['correlation_regime'] == "Strong BTC Correlation":
            regime_multiplier *= 0.9  # More conservative due to correlation risk
        
        adjusted_prediction = ensemble_pred * vol_adjustment * regime_multiplier
        return adjusted_prediction
    
    def analyze_prediction(self, report):
        """Analyze the Ethereum prediction and provide insights"""
        print("\n" + "="*70)
        print("ETHEREUM CME FUTURES PREDICTION ANALYSIS")
        print("="*70)
        
        current_price = report['current_price']
        predicted_change = report['predicted_change_pct']
        risk_adjusted_change = report['risk_adjusted_prediction']['predicted_change_pct']
        individual_preds = report['individual_predictions']
        model_weights = report['model_weights']
        risk_metrics = report['risk_metrics']
        market_regime = report['market_regime']
        market_context = report['market_context']
        
        # Overall Outlook
        print(f"\nOVERALL MARKET OUTLOOK:")
        print(f"  Current ETH Price: ${current_price:,.2f}")
        print(f"  Expected Movement: {predicted_change:+.2f}%")
        print(f"  Expected Price Tomorrow: ${current_price * (1 + predicted_change/100):,.2f}")
        print(f"  Risk-Adjusted Movement: {risk_adjusted_change:+.2f}%")
        
        # Model Agreement Analysis
        pred_values = list(individual_preds.values())
        model_disagreement = np.std(pred_values) if len(pred_values) > 1 else 0
        print(f"\nMODEL CONFIDENCE:")
        if model_disagreement > 5.0:
            print(f"  Model Disagreement: VERY HIGH ({model_disagreement:.2f}%)")
            print(f"  Recommendation: Exercise extreme caution, models disagree significantly")
        elif model_disagreement > 3.0:
            print(f"  Model Disagreement: HIGH ({model_disagreement:.2f}%)")
            print(f"  Recommendation: Wait for morning confirmation, high uncertainty")
        elif model_disagreement > 1.5:
            print(f"  Model Disagreement: MODERATE ({model_disagreement:.2f}%)")
            print(f"  Recommendation: Monitor closely, some model disagreement")
        else:
            print(f"  Model Disagreement: LOW ({model_disagreement:.2f}%)")
            print(f"  Recommendation: Higher confidence in prediction")
        
        # Individual Model Analysis
        print(f"\nINDIVIDUAL MODEL INSIGHTS:")
        for model, pred in individual_preds.items():
            weight = model_weights[model]
            if pred < -10:
                sentiment = "EXTREMELY BEARISH"
            elif pred < -5:
                sentiment = "STRONGLY BEARISH"
            elif pred < -1:
                sentiment = "BEARISH"
            elif pred < 0:
                sentiment = "SLIGHTLY BEARISH"
            elif pred < 1:
                sentiment = "SLIGHTLY BULLISH"
            elif pred < 5:
                sentiment = "BULLISH"
            elif pred < 10:
                sentiment = "STRONGLY BULLISH"
            else:
                sentiment = "EXTREMELY BULLISH"
            
            print(f"  {model}: {pred:+.2f}% ({sentiment}, Weight: {weight:.2f})")
        
        # Risk Assessment
        print(f"\nRISK ASSESSMENT:")
        vol = risk_metrics['current_volatility'] * 100
        if vol > 5.0:
            vol_desc = "VERY HIGH"
        elif vol > 3.0:
            vol_desc = "HIGH"
        elif vol > 1.5:
            vol_desc = "MODERATE"
        else:
            vol_desc = "LOW"
        print(f"  Current Volatility: {vol:.2f}% ({vol_desc})")
        print(f"  Value at Risk (95%): ${risk_metrics['value_at_risk_95']:.2f}")
        print(f"  Prediction Std Dev: {risk_metrics['prediction_std']*100:.2f}%")
        
        # Market Conditions
        print(f"\nMARKET CONDITIONS:")
        rsi = market_context['rsi']
        if rsi > 70:
            rsi_desc = "OVERBOUGHT"
        elif rsi < 30:
            rsi_desc = "OVERSOLD"
        else:
            rsi_desc = "NEUTRAL"
        print(f"  RSI: {rsi:.2f} ({rsi_desc})")
        
        williams_r = market_context['williams_r']
        if williams_r > -20:
            wr_desc = "OVERBOUGHT"
        elif williams_r < -80:
            wr_desc = "OVERSOLD"
        else:
            wr_desc = "NEUTRAL"
        print(f"  Williams %R: {williams_r:.2f} ({wr_desc})")
        
        cci = market_context['cci']
        if cci > 100:
            cci_desc = "STRONG BULLISH"
        elif cci < -100:
            cci_desc = "STRONG BEARISH"
        else:
            cci_desc = "NEUTRAL"
        print(f"  CCI: {cci:.2f} ({cci_desc})")
        
        volume_change = market_context['volume_change']
        if volume_change < -50:
            vol_desc = "SIGNIFICANTLY LOWER"
        elif volume_change < -20:
            vol_desc = "LOWER"
        elif volume_change > 50:
            vol_desc = "HIGHER"
        elif volume_change > 20:
            vol_desc = "MODERATELY HIGHER"
        else:
            vol_desc = "NORMAL"
        print(f"  Volume Change: {volume_change:+.2f}% ({vol_desc})")
        
        btc_corr = market_context['btc_correlation']
        if abs(btc_corr) > 0.7:
            corr_desc = "STRONG"
        elif abs(btc_corr) > 0.4:
            corr_desc = "MODERATE"
        else:
            corr_desc = "WEAK"
        direction = "POSITIVE" if btc_corr > 0 else "NEGATIVE"
        print(f"  BTC Correlation: {btc_corr:.2f} ({corr_desc} {direction})")
        
        # Market Regime
        print(f"\nMARKET REGIME:")
        print(f"  Trend: {market_regime['trend']}")
        print(f"  Volatility Regime: {market_regime['volatility_regime']}")
        print(f"  Fear/Greed: {market_regime['fear_greed_index']}")
        print(f"  BTC Correlation Regime: {market_regime['correlation_regime']}")
        print(f"  VIX Level: {market_regime['vix_level']:.2f}")
        
        # Crypto-Specific Analysis
        print(f"\nCRYPTO-SPECIFIC INSIGHTS:")
        if abs(predicted_change) > 10.0:
            print(f"  Magnitude: EXTREMELY VOLATILE MOVE (>10%)")
            print(f"  Risk Management: Strongly consider reduced position sizing")
        elif abs(predicted_change) > 5.0:
            print(f"  Magnitude: HIGHLY VOLATILE MOVE (5-10%)")
            print(f"  Risk Management: Consider reduced position sizing")
        elif abs(predicted_change) > 2.0:
            print(f"  Magnitude: MODERATELY VOLATILE MOVE (2-5%)")
            print(f"  Risk Management: Standard crypto risk management")
        else:
            print(f"  Magnitude: RELATIVELY STABLE MOVE (<2%)")
            print(f"  Risk Management: Normal crypto positioning acceptable")
        
        if model_disagreement > 5.0:
            print(f"  Timing: Wait for morning confirmation before trading")
            print(f"  Strategy: Consider range trading until direction is confirmed")
        elif model_disagreement > 3.0:
            print(f"  Timing: Monitor first hour of trading for confirmation")
            print(f"  Strategy: Consider partial positions with tight stops")
        else:
            print(f"  Timing: Prediction has reasonable model consensus")
            print(f"  Strategy: Full position sizing may be appropriate")
        
        # Key Watch Points
        print(f"\nKEY WATCH POINTS FOR TOMORROW:")
        print(f"  1. Opening gap relative to today's close")
        print(f"  2. Volume confirmation of predicted move")
        print(f"  3. BTC price action and correlation maintenance")
        print(f"  4. VIX action for broader market fear index")
        print(f"  5. RSI and Williams %R for overbought/oversold signals")
        print(f"  6. CCI for momentum confirmation")
        print("="*70)
    
    def train_models(self, data):
        """Train all models and return results"""
        print("Training models...")
        
        # Prepare data for traditional ML
        feature_columns = [col for col in data.columns if col not in 
                          ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
        X = data[feature_columns]
        y = data['Target']
        
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        X_values = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
        y_values = np.nan_to_num(y.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        min_samples = min(X_values.shape[0], y_values.shape[0])
        X_clean = X_values[:min_samples]
        y_clean = y_values[:min_samples]
        
        split_idx = int(len(X_clean) * 0.8)
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        models = {}
        predictions = {}
        
        # 1. Random Forest
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X_train_scaled, y_train_scaled)
        rf_pred = rf_model.predict(X_test_scaled)
        # Clip extreme predictions
        rf_pred = np.clip(rf_pred, -0.2, 0.2)
        models['RandomForest'] = rf_model
        predictions['RandomForest'] = rf_pred
        
        # 2. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            xgb_model = self.train_xgboost(X_train_scaled, y_train_scaled)
            if xgb_model is not None:
                xgb_pred = xgb_model.predict(X_test_scaled)
                # Clip extreme predictions
                xgb_pred = np.clip(xgb_pred, -0.2, 0.2)
                models['XGBoost'] = xgb_model
                predictions['XGBoost'] = xgb_pred
                print("XGBoost training completed")
            else:
                print("XGBoost training failed")
        else:
            print("XGBoost not available, skipping...")
        
        # 3. LSTM
        print("Training LSTM...")
        X_seq, y_seq, scaler_feat, scaler_tgt, feat_cols = self.prepare_sequences(data, sequence_length=20)
        split_seq = int(len(X_seq) * 0.8)
        X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
        y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]
        
        lstm_model = self.build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lstm_history = lstm_model.fit(X_train_seq, y_train_seq, 
                                      validation_data=(X_test_seq, y_test_seq),
                                      epochs=50, batch_size=32,
                                      callbacks=[early_stopping], verbose=0)
        lstm_pred_scaled = lstm_model.predict(X_test_seq)
        lstm_pred = scaler_tgt.inverse_transform(lstm_pred_scaled).flatten()
        # Clip extreme predictions
        lstm_pred = np.clip(lstm_pred, -0.2, 0.2)
        models['LSTM'] = lstm_model
        predictions['LSTM'] = lstm_pred
        self.training_history['LSTM'] = lstm_history.history
        
        # 4. GRU
        print("Training GRU...")
        gru_model = self.build_gru_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        gru_history = gru_model.fit(X_train_seq, y_train_seq, 
                                    validation_data=(X_test_seq, y_test_seq),
                                    epochs=50, batch_size=32,
                                    callbacks=[early_stopping], verbose=0)
        gru_pred_scaled = gru_model.predict(X_test_seq)
        gru_pred = scaler_tgt.inverse_transform(gru_pred_scaled).flatten()
        # Clip extreme predictions
        gru_pred = np.clip(gru_pred, -0.2, 0.2)
        models['GRU'] = gru_model
        predictions['GRU'] = gru_pred
        self.training_history['GRU'] = gru_history.history
        
        # 5. Transformer
        print("Training Transformer...")
        transformer_model = self.build_transformer_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        transformer_history = transformer_model.fit(X_train_seq, y_train_seq, 
                                                    validation_data=(X_test_seq, y_test_seq),
                                                    epochs=50, batch_size=32,
                                                    callbacks=[early_stopping], verbose=0)
        transformer_pred_scaled = transformer_model.predict(X_test_seq)
        transformer_pred = scaler_tgt.inverse_transform(transformer_pred_scaled).flatten()
        # Clip extreme predictions
        transformer_pred = np.clip(transformer_pred, -0.2, 0.2)
        models['Transformer'] = transformer_model
        predictions['Transformer'] = transformer_pred
        self.training_history['Transformer'] = transformer_history.history
        
        # Evaluate models with proper sample alignment
        print("\nModel Evaluation (MAE):")
        evaluation_results = {}
        
        # Random Forest Evaluation
        if len(y_test) > 0 and len(rf_pred) > 0:
            min_eval_samples = min(len(y_test), len(rf_pred))
            rf_mae = mean_absolute_error(y_test[:min_eval_samples], rf_pred[:min_eval_samples])
            evaluation_results['RandomForest'] = rf_mae
            print(f"RandomForest: {rf_mae:.4f}")
        else:
            evaluation_results['RandomForest'] = 0.01
            print(f"RandomForest: 0.0100")
        
        # XGBoost Evaluation (if available)
        if XGBOOST_AVAILABLE and 'XGBoost' in models:
            xgb_pred = predictions['XGBoost']
            if len(y_test) > 0 and len(xgb_pred) > 0:
                min_eval_samples = min(len(y_test), len(xgb_pred))
                xgb_mae = mean_absolute_error(y_test[:min_eval_samples], xgb_pred[:min_eval_samples])
                evaluation_results['XGBoost'] = xgb_mae
                print(f"XGBoost: {xgb_mae:.4f}")
            else:
                evaluation_results['XGBoost'] = 0.01
                print(f"XGBoost: 0.0100")
        
        # Deep Learning Models Evaluation
        if len(y_seq) > 0 and len(lstm_pred) > 0:
            y_test_actual = y_seq[split_seq:]
            y_test_actual_inverse = scaler_tgt.inverse_transform(y_test_actual.reshape(-1, 1)).flatten()
            
            min_dl_samples = min(len(y_test_actual_inverse), len(lstm_pred))
            y_test_dl = y_test_actual_inverse[:min_dl_samples]
            lstm_pred_dl = lstm_pred[:min_dl_samples]
            gru_pred_dl = gru_pred[:min_dl_samples]
            transformer_pred_dl = transformer_pred[:min_dl_samples]
            
            y_test_dl_clean = np.nan_to_num(y_test_dl, nan=0.0, posinf=0.0, neginf=0.0)
            lstm_pred_dl_clean = np.nan_to_num(lstm_pred_dl, nan=0.0, posinf=0.0, neginf=0.0)
            gru_pred_dl_clean = np.nan_to_num(gru_pred_dl, nan=0.0, posinf=0.0, neginf=0.0)
            transformer_pred_dl_clean = np.nan_to_num(transformer_pred_dl, nan=0.0, posinf=0.0, neginf=0.0)
            
            lstm_mae = mean_absolute_error(y_test_dl_clean, lstm_pred_dl_clean)
            gru_mae = mean_absolute_error(y_test_dl_clean, gru_pred_dl_clean)
            transformer_mae = mean_absolute_error(y_test_dl_clean, transformer_pred_dl_clean)
            
            evaluation_results['LSTM'] = lstm_mae
            evaluation_results['GRU'] = gru_mae
            evaluation_results['Transformer'] = transformer_mae
            
            print(f"LSTM: {lstm_mae:.4f}")
            print(f"GRU: {gru_mae:.4f}")
            print(f"Transformer: {transformer_mae:.4f}")
        else:
            evaluation_results['LSTM'] = 0.01
            evaluation_results['GRU'] = 0.01
            evaluation_results['Transformer'] = 0.01
            print(f"LSTM: 0.0100")
            print(f"GRU: 0.0100")
            print(f"Transformer: 0.0100")
        
        # Save components
        self.models = models
        self.scalers = {
            'features': scaler_X,
            'target': scaler_y,
            'seq_features': scaler_feat,
            'seq_target': scaler_tgt
        }
        self.feature_columns = feature_columns
        self.seq_feature_columns = feat_cols
        self.evaluation_results = evaluation_results
        
        # Save to disk
        self.save_models()
        
        return models, predictions, y_test, evaluation_results
    
    def save_models(self):
        """Save all models and components to disk"""
        print("Saving models...")
        joblib.dump(self.models, os.path.join(self.model_dir, "eth_cme_models.pkl"))
        joblib.dump(self.scalers, os.path.join(self.model_dir, "eth_cme_scalers.pkl"))
        joblib.dump(self.feature_columns, os.path.join(self.model_dir, "eth_cme_feature_columns.pkl"))
        joblib.dump(self.seq_feature_columns, os.path.join(self.model_dir, "eth_cme_seq_feature_columns.pkl"))
        
        if hasattr(self, 'evaluation_results'):
            with open(os.path.join(self.model_dir, "evaluation_results.json"), 'w') as f:
                json.dump(self.evaluation_results, f)
    
    def load_models(self):
        """Load all models and components from disk"""
        print("Loading models...")
        try:
            self.models = joblib.load(os.path.join(self.model_dir, "eth_cme_models.pkl"))
            self.scalers = joblib.load(os.path.join(self.model_dir, "eth_cme_scalers.pkl"))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, "eth_cme_feature_columns.pkl"))
            self.seq_feature_columns = joblib.load(os.path.join(self.model_dir, "eth_cme_seq_feature_columns.pkl"))
            return True
        except FileNotFoundError:
            print("No saved models found. Please train models first.")
            return False
    
    def ensemble_predict(self, X_latest, X_seq_latest):
        """Make ensemble prediction using all models"""
        X_latest = np.nan_to_num(X_latest, nan=0.0, posinf=0.0, neginf=0.0)
        X_seq_latest = np.nan_to_num(X_seq_latest, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = {}
        
        # Random Forest
        rf_pred = self.models['RandomForest'].predict(X_latest.reshape(1, -1))[0]
        predictions['RandomForest'] = rf_pred
        
        # XGBoost (if available)
        if 'XGBoost' in self.models:
            xgb_pred = self.models['XGBoost'].predict(X_latest.reshape(1, -1))[0]
            predictions['XGBoost'] = xgb_pred
        
        # LSTM
        lstm_pred_scaled = self.models['LSTM'].predict(X_seq_latest.reshape(1, *X_seq_latest.shape))
        lstm_pred = self.scalers['seq_target'].inverse_transform(lstm_pred_scaled)[0][0]
        predictions['LSTM'] = lstm_pred
        
        # GRU
        gru_pred_scaled = self.models['GRU'].predict(X_seq_latest.reshape(1, *X_seq_latest.shape))
        gru_pred = self.scalers['seq_target'].inverse_transform(gru_pred_scaled)[0][0]
        predictions['GRU'] = gru_pred
        
        # Transformer
        transformer_pred_scaled = self.models['Transformer'].predict(X_seq_latest.reshape(1, *X_seq_latest.shape))
        transformer_pred = self.scalers['seq_target'].inverse_transform(transformer_pred_scaled)[0][0]
        predictions['Transformer'] = transformer_pred
        
        # Clip all predictions to reasonable range
        predictions = self.clip_predictions(predictions, -0.2, 0.2)
        
        # Adaptive weighting based on recent performance
        if hasattr(self, 'evaluation_results'):
            weights = {}
            total_inverse_mae = sum(1/mae for mae in self.evaluation_results.values())
            for model, mae in self.evaluation_results.items():
                weights[model] = (1/mae) / total_inverse_mae
        else:
            # Equal weights if no evaluation results
            num_models = len(predictions)
            equal_weight = 1.0 / num_models
            weights = {model: equal_weight for model in predictions.keys()}
        
        ensemble_pred = sum(weights[model] * pred for model, pred in predictions.items())
        return ensemble_pred, predictions, weights
    
    def generate_report(self, data, ensemble_pred, individual_preds, weights, risk_metrics, market_regime):
        """Generate a comprehensive prediction report with risk management"""
        current_price = data['Close'].iloc[-1]
        predicted_price = current_price * (1 + ensemble_pred)
        change_amount = predicted_price - current_price
        
        # Risk-adjusted prediction
        risk_adjusted_pred = self.adjust_prediction_for_risk(ensemble_pred, risk_metrics, market_regime)
        risk_adjusted_price = current_price * (1 + risk_adjusted_pred)
        risk_adjusted_change = risk_adjusted_price - current_price
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "predicted_change_pct": float(ensemble_pred * 100),
            "predicted_change_usd": float(change_amount),
            "risk_adjusted_prediction": {
                "predicted_price": float(risk_adjusted_price),
                "predicted_change_pct": float(risk_adjusted_pred * 100),
                "predicted_change_usd": float(risk_adjusted_change)
            },
            "individual_predictions": {k: float(v * 100) for k, v in individual_preds.items()},
            "model_weights": {k: float(v) for k, v in weights.items()},
            "risk_metrics": risk_metrics,
            "market_regime": market_regime,
            "market_context": {
                "rsi": float(data['RSI'].iloc[-1]),
                "volatility": float(data['Volatility'].iloc[-1] * 100),
                "volume_change": float(data['Volume_Change'].iloc[-1] * 100),
                "atr": float(data['ATR'].iloc[-1]),
                "cci": float(data['CCI'].iloc[-1]),
                "williams_r": float(data['Williams_R'].iloc[-1]),
                "adx": float(data['ADX'].iloc[-1]),
                "btc_correlation": float(data['ETH_BTC_Correlation'].iloc[-1]) if 'ETH_BTC_Correlation' in data.columns else 0
            }
        }
        
        with open(os.path.join(self.model_dir, "prediction_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
    
    def print_report(self, report):
        """Print formatted prediction report"""
        print("\n" + "="*70)
        print("ETHEREUM CME FUTURES PRICE PREDICTION REPORT WITH RISK MANAGEMENT")
        print("="*70)
        print(f"Generated: {report['timestamp']}")
        print(f"Current ETH Price: ${report['current_price']:.2f}")
        print(f"\nBASE PREDICTION:")
        print(f"  Predicted Next Day Price: ${report['predicted_price']:.2f}")
        print(f"  Predicted Change: {report['predicted_change_pct']:+.2f}% (${report['predicted_change_usd']:+.2f})")
        print(f"\nRISK-ADJUSTED PREDICTION:")
        ra = report['risk_adjusted_prediction']
        print(f"  Predicted Next Day Price: ${ra['predicted_price']:.2f}")
        print(f"  Predicted Change: {ra['predicted_change_pct']:+.2f}% (${ra['predicted_change_usd']:+.2f})")
        print("\nIndividual Model Predictions:")
        for model, pred in report['individual_predictions'].items():
            print(f"  {model}: {pred:+.2f}%")
        print("\nModel Weights:")
        for model, weight in report['model_weights'].items():
            print(f"  {model}: {weight:.2f}")
        print("\nRisk Metrics:")
        rm = report['risk_metrics']
        print(f"  Current Volatility: {rm['current_volatility']*100:.2f}%")
        print(f"  Average True Range: ${rm['current_atr']:.2f}")
        print(f"  Value at Risk (95%): ${rm['value_at_risk_95']:.2f}")
        print(f"  Value at Risk (99%): ${rm['value_at_risk_99']:.2f}")
        print(f"  Prediction Std Dev: {rm['prediction_std']*100:.2f}%")
        print("\nMarket Regime:")
        mr = report['market_regime']
        print(f"  Trend: {mr['trend']}")
        print(f"  Volatility Regime: {mr['volatility_regime']}")
        print(f"  Fear/Greed: {mr['fear_greed_index']}")
        print(f"  BTC Correlation Regime: {mr['correlation_regime']}")
        print(f"  RSI Level: {mr['rsi_level']:.2f}")
        print(f"  VIX Level: {mr['vix_level']:.2f}")
        print(f"  BTC Correlation: {mr['btc_correlation']:.2f}")
        print("\nMarket Context:")
        mc = report['market_context']
        print(f"  RSI: {mc['rsi']:.2f}")
        print(f"  Volatility: {mc['volatility']:.2f}%")
        print(f"  Volume Change: {mc['volume_change']:+.2f}%")
        print(f"  ATR: ${mc['atr']:.2f}")
        print(f"  CCI: {mc['cci']:.2f}")
        print(f"  Williams %R: {mc['williams_r']:.2f}")
        print(f"  ADX: {mc['adx']:.2f}")
        print(f"  BTC Correlation: {mc['btc_correlation']:.2f}")
        print("="*70)

def main():
    predictor = EthereumCMEPredictor()
    
    if os.path.exists(os.path.join(predictor.model_dir, "eth_cme_models.pkl")):
        print("Loading existing models...")
        loaded = predictor.load_models()
        if not loaded:
            print("Training new models...")
            eth_data = predictor.fetch_eth_cme_data()
            btc_data = predictor.fetch_bitcoin_data()
            vix_data = predictor.fetch_vix_data()
            
            if len(eth_data) < 30:
                print("Not enough data to train models. Please try again later.")
                return
                
            feature_data = predictor.create_features(eth_data, btc_data, vix_data)
            
            if len(feature_data) < 30:
                print("Not enough clean data to train models. Please try again later.")
                return
                
            models, predictions, y_test, evaluation_results = predictor.train_models(feature_data)
    else:
        print("Training new models...")
        eth_data = predictor.fetch_eth_cme_data()
        btc_data = predictor.fetch_bitcoin_data()
        vix_data = predictor.fetch_vix_data()
        
        if len(eth_data) < 30:
            print("Not enough data to train models. Please try again later.")
            return
            
        feature_data = predictor.create_features(eth_data, btc_data, vix_data)
        
        if len(feature_data) < 30:
            print("Not enough clean data to train models. Please try again later.")
            return
            
        models, predictions, y_test, evaluation_results = predictor.train_models(feature_data)
    
    print("Making ensemble prediction...")
    
    eth_data = predictor.fetch_eth_cme_data("1y")
    btc_data = predictor.fetch_bitcoin_data("1y")
    vix_data = predictor.fetch_vix_data("1y")
    feature_data = predictor.create_features(eth_data, btc_data, vix_data)
    
    X_latest = feature_data[predictor.feature_columns].iloc[-1:].values
    X_latest_scaled = predictor.scalers['features'].transform(X_latest)
    
    sequence_length = 20
    latest_features = feature_data[predictor.seq_feature_columns].values
    latest_features_scaled = predictor.scalers['seq_features'].transform(latest_features)
    X_seq_latest = latest_features_scaled[-sequence_length:]
    
    ensemble_pred, individual_preds, weights = predictor.ensemble_predict(
        X_latest_scaled[0], X_seq_latest
    )
    
    # Calculate risk metrics
    risk_metrics = predictor.calculate_risk_metrics(feature_data, individual_preds)
    market_regime = predictor.get_market_regime(feature_data)
    
    # Generate and print report
    report = predictor.generate_report(feature_data, ensemble_pred, individual_preds, weights, risk_metrics, market_regime)
    predictor.print_report(report)
    
    # Add analysis
    predictor.analyze_prediction(report)
    
    print(f"\nReports and models saved to: {predictor.model_dir}")

if __name__ == "__main__":
    main()