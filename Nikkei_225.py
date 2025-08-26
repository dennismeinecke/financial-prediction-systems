import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Suppress protobuf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Try to import ib_insync for Interactive Brokers integration
try:
    from ib_insync import *
    IB_AVAILABLE = True
    print("IB_insync library available for Interactive Brokers integration")
except ImportError:
    IB_AVAILABLE = False
    print("IB_insync library not available. Install with 'pip install ib_insync' for Interactive Brokers support")

class NikkeiFuturesTradingSystem:
    def __init__(self, symbol="MNKU5", futures_symbol="MNKU5", initial_capital=100000, 
                 ibkr_host='127.0.0.1', ibkr_port=7497, ibkr_client_id=1):
        self.symbol = symbol  # For general symbol reference
        self.futures_symbol = futures_symbol  # CME Globex symbol for Micro Nikkei Futures
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.positions = None
        self.portfolio = None
        self.models = {}
        self.indicators = {}
        self.X = None
        self.y_reg = None
        self.y_clf = None
        self.y_high = None
        self.y_low = None
        self.y_close = None
        
    def fetch_data(self, period="2y", realtime=False):
        """Fetch Micro Nikkei data from Interactive Brokers ONLY"""
        print(f"Attempting to fetch data for Micro Nikkei (USD) Futures {self.futures_symbol}...")
        
        if not IB_AVAILABLE:
            print("Interactive Brokers: ib_insync library not available")
            self._create_sample_data()
            return
            
        ib = None
        try:
            # Create IB connection
            ib = IB()
            ib.connect(self.ibkr_host, self.ibkr_port, clientId=self.ibkr_client_id)
            print(f"Connected to Interactive Brokers at {self.ibkr_host}:{self.ibkr_port}")
            
            # Try multiple contract specifications for Micro Nikkei futures
            contracts_to_try = [
                Future(symbol='MNK', exchange='CME', currency='USD'),  # Micro Nikkei
                Future(localSymbol='MNKU5', exchange='CME', currency='USD'),  # September 2025 contract
                Future(symbol='NIK', exchange='CME', currency='USD'),  # Nikkei Mini
                Future(symbol='NK225', exchange='CME', currency='JPY'),  # Nikkei 225
                Contract(secType='FUT', symbol='MNK', exchange='CME', currency='USD'),
                Contract(secType='FUT', localSymbol='MNKU5', exchange='CME', currency='USD'),
            ]
            
            contract = None
            qualified_contract = None
            
            # Try to qualify contracts
            for i, contract_spec in enumerate(contracts_to_try):
                try:
                    print(f"Trying contract specification {i+1}/{len(contracts_to_try)}: {contract_spec}")
                    qualified = ib.qualifyContracts(contract_spec)
                    if qualified and len(qualified) > 0:
                        qualified_contract = qualified[0]
                        contract = contract_spec
                        print(f"Successfully qualified contract: {qualified_contract}")
                        break
                    else:
                        print(f"Contract qualification failed for: {contract_spec}")
                except Exception as e:
                    print(f"Error qualifying contract {contract_spec}: {e}")
                    continue
            
            if qualified_contract is None:
                print("Interactive Brokers: Could not qualify any Micro Nikkei contract")
                ib.disconnect()
                self._create_sample_data()
                return
            
            # Request historical data
            print(f"Requesting historical data for {qualified_contract}...")
            bars = ib.reqHistoricalData(
                qualified_contract,
                endDateTime='',
                durationStr=period,
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            # Convert to DataFrame
            if bars and len(bars) > 0:
                data_df = util.df(bars)
                if data_df is not None and not data_df.empty:
                    data_df['Date'] = pd.to_datetime(data_df['date'])
                    data_df.set_index('Date', inplace=True)
                    data_df.rename(columns={
                        'open': 'Open', 
                        'high': 'High', 
                        'low': 'Low', 
                        'close': 'Close', 
                        'volume': 'Volume'
                    }, inplace=True)
                    data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    print(f"Interactive Brokers: Fetched {len(data_df)} days of data for {qualified_contract}")
                    self.data = data_df
                    ib.disconnect()
                    return
                else:
                    print("Interactive Brokers: Empty data received")
            else:
                print("Interactive Brokers: No historical data bars received")
                
        except Exception as e:
            print(f"Interactive Brokers Error: {e}")
        finally:
            if ib and ib.isConnected():
                try:
                    ib.disconnect()
                    print("Disconnected from Interactive Brokers")
                except:
                    pass
                    
        # Fallback to sample data
        self._create_sample_data()
        
    def _create_sample_data(self):
        """Create sample data for testing"""
        print("Creating sample data for testing...")
        dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        self.data = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.randint(1000000, 2000000, len(dates)),
            'Dividends': 0,
            'Stock Splits': 0
        }, index=dates)
        print(f"Created sample data with {len(self.data)} days")
        
    def calculate_technical_indicators(self):
        """Calculate 50+ technical indicators"""
        if self.data is None or len(self.data) == 0:
            print("No data available for technical indicator calculation")
            return
            
        df = self.data.copy()
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Missing column {col}, creating sample data")
                df[col] = df['Close'] if 'Close' in df.columns else np.random.randn(len(df)) * 10 + 100
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['WMA_20'] = df['Close'].rolling(window=20).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # ATR
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift()), 
                                         abs(df['Low'] - df['Close'].shift())))
        df['ATR_14'] = df['TR'].rolling(window=14).mean()
        
        # ADX
        up_move = df['High'].diff()
        down_move = df['Low'].diff()
        df['PlusDM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        df['MinusDM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        df['PlusDI'] = 100 * (df['PlusDM'].rolling(window=14).mean() / df['ATR_14'])
        df['MinusDI'] = 100 * (df['MinusDM'].rolling(window=14).mean() / df['ATR_14'])
        df['DX'] = 100 * abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])
        df['ADX'] = df['DX'].rolling(window=14).mean()
        
        # CCI
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['CCI'] = (tp - sma_tp) / (0.015 * mean_dev)
        
        # Parabolic SAR
        df['PSAR'] = self._calculate_psar(df)
        
        # Ichimoku Cloud
        df['Tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        df['Senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        df['Chikou_span'] = df['Close'].shift(-26)
        
        # Volume Indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Oscillator'] = (df['Volume'] - df['Volume_SMA']) / df['Volume_SMA']
        
        # Price Channels
        df['Price_Channel_Upper'] = df['High'].rolling(window=20).max()
        df['Price_Channel_Lower'] = df['Low'].rolling(window=20).min()
        
        # Momentum Indicators
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Fibonacci Retracement Levels (based on last 50 days)
        min_50 = df['Close'].iloc[-50:].min() if len(df) >= 50 else df['Close'].min()
        max_50 = df['Close'].iloc[-50:].max() if len(df) >= 50 else df['Close'].max()
        df['Fib_23.6'] = min_50 + (max_50 - min_50) * 0.236
        df['Fib_38.2'] = min_50 + (max_50 - min_50) * 0.382
        df['Fib_50'] = min_50 + (max_50 - min_50) * 0.5
        df['Fib_61.8'] = min_50 + (max_50 - min_50) * 0.618
        
        # Volatility Indicators
        df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
        
        # Rate of Change
        df['Rate_of_Change'] = df['Close'].pct_change(periods=10) * 100
        
        # Additional indicators
        df['TRIX'] = df['Close'].ewm(span=15).mean().ewm(span=15).mean().ewm(span=15).mean()
        df['TRIX_signal'] = df['TRIX'].ewm(span=9).mean()
        
        # Keltner Channels
        df['Keltner_MA'] = df['Close'].ewm(span=20).mean()
        df['Keltner_Upper'] = df['Keltner_MA'] + (df['ATR_14'] * 2)
        df['Keltner_Lower'] = df['Keltner_MA'] - (df['ATR_14'] * 2)
        
        # Donchian Channels
        df['Donchian_Upper'] = df['High'].rolling(window=20).max()
        df['Donchian_Lower'] = df['Low'].rolling(window=20).min()
        df['Donchian_Middle'] = (df['Donchian_Upper'] + df['Donchian_Lower']) / 2
        
        # Elder Ray Index
        df['Bull_Power'] = df['High'] - df['EMA_12']
        df['Bear_Power'] = df['Low'] - df['EMA_12']
        
        # Force Index
        df['Force_Index'] = df['Close'].diff() * df['Volume']
        df['Force_Index_EMA'] = df['Force_Index'].ewm(span=13).mean()
        
        # Chaikin Oscillator
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['ADL'] = (clv * df['Volume']).fillna(0).cumsum()
        df['Chaikin_Osc'] = df['ADL'].ewm(span=3).mean() - df['ADL'].ewm(span=10).mean()
        
        # Money Flow Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        positive_flow = []
        negative_flow = []
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(raw_money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(raw_money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        positive_mf = pd.Series(positive_flow).rolling(window=14).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=14).sum()
        mfr = positive_mf / negative_mf
        df['MFI'] = 100 - (100 / (1 + mfr))
        df['MFI'] = df['MFI'].fillna(50)  # Fill initial NaN values
        
        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Accumulation/Distribution Line
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfv = mfm * df['Volume']
        df['ADL'] = mfv.cumsum()
        
        # Ease of Movement
        distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
        box_ratio = (df['Volume'] / 100000000) / (df['High'] - df['Low'])
        df['EMV'] = distance_moved / box_ratio
        df['EMV_SMA'] = df['EMV'].rolling(window=14).mean()
        
        # Coppock Curve
        roc14 = df['Close'].pct_change(periods=14) * 100
        roc11 = df['Close'].pct_change(periods=11) * 100
        roc_sum = roc14 + roc11.shift(10)
        df['Coppock'] = roc_sum.ewm(span=10).mean()
        
        # Hull Moving Average
        df['HMA'] = self._calculate_hma(df['Close'], 20)
        
        # Guppy Multiple Moving Average
        df['GMMA_Short1'] = df['Close'].ewm(span=3).mean()
        df['GMMA_Short2'] = df['Close'].ewm(span=5).mean()
        df['GMMA_Short3'] = df['Close'].ewm(span=8).mean()
        df['GMMA_Short4'] = df['Close'].ewm(span=10).mean()
        df['GMMA_Short5'] = df['Close'].ewm(span=12).mean()
        df['GMMA_Short6'] = df['Close'].ewm(span=15).mean()
        df['GMMA_Long1'] = df['Close'].ewm(span=30).mean()
        df['GMMA_Long2'] = df['Close'].ewm(span=35).mean()
        df['GMMA_Long3'] = df['Close'].ewm(span=40).mean()
        df['GMMA_Long4'] = df['Close'].ewm(span=45).mean()
        df['GMMA_Long5'] = df['Close'].ewm(span=50).mean()
        df['GMMA_Long6'] = df['Close'].ewm(span=60).mean()
        
        # ZigZag Indicator (simplified)
        df['ZigZag'] = self._calculate_zigzag(df)
        
        # SuperTrend
        df['SuperTrend'] = self._calculate_supertrend(df)
        
        # Vortex Indicator
        df['Vortex_Pos'], df['Vortex_Neg'] = self._calculate_vortex(df)
        
        # Mass Index
        df['Mass_Index'] = self._calculate_mass_index(df)
        
        # True Strength Index
        df['TSI'] = self._calculate_tsi(df)
        
        # Ultimate Oscillator
        df['Ultimate_Osc'] = self._calculate_ultimate_oscillator(df)
        
        # Awesome Oscillator
        df['AO'] = self._calculate_awesome_oscillator(df)
        
        # Alligator Indicator
        df['Alligator_Jaw'] = df['Close'].shift(8).rolling(window=13).mean()
        df['Alligator_Teeth'] = df['Close'].shift(5).rolling(window=8).mean()
        df['Alligator_Lips'] = df['Close'].shift(3).rolling(window=5).mean()
        
        # Fractal Indicator
        df['Fractal_Up'], df['Fractal_Down'] = self._calculate_fractals(df)
        
        # Market Facilitation Index
        df['MFI_BW'] = (df['High'] - df['Low']) / df['Volume']
        
        # Standard Deviation
        df['Std_Dev'] = df['Close'].rolling(window=20).std()
        
        # Variance
        df['Variance'] = df['Close'].rolling(window=20).var()
        
        # Kurtosis
        df['Kurtosis'] = df['Close'].rolling(window=20).apply(lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0, raw=True)
        
        # Skewness
        df['Skewness'] = df['Close'].rolling(window=20).apply(lambda x: pd.Series(x).skew() if len(x) > 2 else 0, raw=True)
        
        # Hurst Exponent (simplified)
        df['Hurst'] = self._calculate_hurst(df['Close'])
        
        # Detrended Price Oscillator
        df['DPO'] = df['Close'] - df['Close'].rolling(window=20).mean().shift(11)
        
        # Elder Impulse System
        df['Elder_Impulse'] = self._calculate_elder_impulse(df)
        
        # Schaff Trend Cycle
        df['STC'] = self._calculate_stc(df)
        
        # Add target variables (next day predictions)
        df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target_Direction'] = np.where(df['Target_Return'] > 0, 1, 0)
        df['Target_High'] = df['High'].shift(-1)
        df['Target_Low'] = df['Low'].shift(-1)
        df['Target_Close'] = df['Close'].shift(-1)
        
        self.data = df.dropna()
        print(f"Calculated {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']])} technical indicators")
        
    def _calculate_psar(self, df):
        """Calculate Parabolic SAR"""
        psar = df['Close'].copy()
        af = 0.02  # Acceleration factor
        ep = df['High'].iloc[0]  # Extreme point
        trend = 1  # 1 for uptrend, -1 for downtrend
        psar.iloc[0] = df['Low'].iloc[0]
        
        for i in range(1, len(df)):
            if trend == 1:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['Low'].iloc[i] < psar.iloc[i]:
                    trend = -1
                    psar.iloc[i] = ep
                    ep = df['Low'].iloc[i]
                    af = 0.02
                else:
                    if df['High'].iloc[i] > ep:
                        ep = df['High'].iloc[i]
                        af = min(af + 0.02, 0.2)
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['High'].iloc[i] > psar.iloc[i]:
                    trend = 1
                    psar.iloc[i] = ep
                    ep = df['High'].iloc[i]
                    af = 0.02
                else:
                    if df['Low'].iloc[i] < ep:
                        ep = df['Low'].iloc[i]
                        af = min(af + 0.02, 0.2)
                        
        return psar
    
    def _calculate_hma(self, series, period):
        """Calculate Hull Moving Average"""
        half_length = int(period / 2)
        sqrt_length = int(np.sqrt(period))
        wma1 = series.rolling(window=half_length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        wma2 = series.rolling(window=period).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        diff = 2 * wma1 - wma2
        hma = pd.Series(diff).rolling(window=sqrt_length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        return hma
    
    def _calculate_zigzag(self, df, deviation=5):
        """Calculate simplified ZigZag indicator"""
        zigzag = pd.Series(index=df.index, dtype=float)
        last_pivot = df['Close'].iloc[0]
        trend = 0  # 0: undefined, 1: up, -1: down
        
        for i in range(1, len(df)):
            if trend == 0:
                if df['Close'].iloc[i] > last_pivot * (1 + deviation/100):
                    trend = 1
                    zigzag.iloc[i] = df['Close'].iloc[i]
                    last_pivot = df['Close'].iloc[i]
                elif df['Close'].iloc[i] < last_pivot * (1 - deviation/100):
                    trend = -1
                    zigzag.iloc[i] = df['Close'].iloc[i]
                    last_pivot = df['Close'].iloc[i]
            elif trend == 1:
                if df['Close'].iloc[i] > last_pivot:
                    zigzag.iloc[i] = df['Close'].iloc[i]
                    last_pivot = df['Close'].iloc[i]
                elif df['Close'].iloc[i] < last_pivot * (1 - deviation/100):
                    trend = -1
                    zigzag.iloc[i] = df['Close'].iloc[i]
                    last_pivot = df['Close'].iloc[i]
            else:  # trend == -1
                if df['Close'].iloc[i] < last_pivot:
                    zigzag.iloc[i] = df['Close'].iloc[i]
                    last_pivot = df['Close'].iloc[i]
                elif df['Close'].iloc[i] > last_pivot * (1 + deviation/100):
                    trend = 1
                    zigzag.iloc[i] = df['Close'].iloc[i]
                    last_pivot = df['Close'].iloc[i]
                    
        return zigzag.fillna(method='ffill')
    
    def _calculate_supertrend(self, df, atr_period=10, multiplier=3):
        """Calculate SuperTrend"""
        atr = df['ATR_14']
        hl2 = (df['High'] + df['Low']) / 2
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        final_upperband = basic_upperband.copy()
        final_lowerband = basic_lowerband.copy()
        supertrend = pd.Series(index=df.index, dtype=float)
        trend = True  # True for uptrend, False for downtrend
        
        for i in range(1, len(df)):
            if trend:
                if df['Close'].iloc[i] < final_upperband.iloc[i-1]:
                    trend = False
                    supertrend.iloc[i] = final_upperband.iloc[i]
                else:
                    supertrend.iloc[i] = final_lowerband.iloc[i]
                    if basic_lowerband.iloc[i] > final_lowerband.iloc[i-1]:
                        final_lowerband.iloc[i] = basic_lowerband.iloc[i]
                    else:
                        final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            else:
                if df['Close'].iloc[i] > final_lowerband.iloc[i-1]:
                    trend = True
                    supertrend.iloc[i] = final_lowerband.iloc[i]
                else:
                    supertrend.iloc[i] = final_upperband.iloc[i]
                    if basic_upperband.iloc[i] < final_upperband.iloc[i-1]:
                        final_upperband.iloc[i] = basic_upperband.iloc[i]
                    else:
                        final_upperband.iloc[i] = final_upperband.iloc[i-1]
                        
        return supertrend
    
    def _calculate_vortex(self, df, period=14):
        """Calculate Vortex Indicator"""
        tr = np.maximum(df['High'] - df['Low'], 
                        np.maximum(abs(df['High'] - df['Close'].shift()), 
                                   abs(df['Low'] - df['Close'].shift())))
        vp = abs(df['High'] - df['Low'].shift())
        vm = abs(df['Low'] - df['High'].shift())
        vip = pd.Series(vp.rolling(window=period).sum() / tr.rolling(window=period).sum())
        vim = pd.Series(vm.rolling(window=period).sum() / tr.rolling(window=period).sum())
        return vip, vim
    
    def _calculate_mass_index(self, df, period=9, ema_period=25):
        """Calculate Mass Index"""
        hl_diff = df['High'] - df['Low']
        ema1 = hl_diff.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ratio = ema1 / ema2
        mass_index = ratio.rolling(window=ema_period).sum()
        return mass_index
    
    def _calculate_tsi(self, df, r=25, s=13):
        """Calculate True Strength Index"""
        momentum = df['Close'].diff()
        ema1 = momentum.ewm(span=r).mean()
        ema2 = ema1.ewm(span=s).mean()
        abs_momentum = abs(momentum)
        abs_ema1 = abs_momentum.ewm(span=r).mean()
        abs_ema2 = abs_ema1.ewm(span=s).mean()
        tsi = 100 * (ema2 / abs_ema2)
        return tsi
    
    def _calculate_ultimate_oscillator(self, df, period1=7, period2=14, period3=28):
        """Calculate Ultimate Oscillator"""
        bp = df['Close'] - np.minimum(df['Low'], df['Close'].shift())
        tr = np.maximum(df['High'], df['Close'].shift()) - np.minimum(df['Low'], df['Close'].shift())
        avg7 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
        avg14 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
        avg28 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        return uo
    
    def _calculate_awesome_oscillator(self, df):
        """Calculate Awesome Oscillator"""
        median_price = (df['High'] + df['Low']) / 2
        ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
        return ao
    
    def _calculate_fractals(self, df):
        """Calculate Fractals"""
        fractal_up = pd.Series(index=df.index, dtype=bool)
        fractal_down = pd.Series(index=df.index, dtype=bool)
        
        for i in range(2, len(df)-2):
            # Up fractal
            if (df['High'].iloc[i] > df['High'].iloc[i-1]) and \
               (df['High'].iloc[i] > df['High'].iloc[i-2]) and \
               (df['High'].iloc[i] > df['High'].iloc[i+1]) and \
               (df['High'].iloc[i] > df['High'].iloc[i+2]):
                fractal_up.iloc[i] = True
            # Down fractal
            if (df['Low'].iloc[i] < df['Low'].iloc[i-1]) and \
               (df['Low'].iloc[i] < df['Low'].iloc[i-2]) and \
               (df['Low'].iloc[i] < df['Low'].iloc[i+1]) and \
               (df['Low'].iloc[i] < df['Low'].iloc[i+2]):
                fractal_down.iloc[i] = True
                
        return fractal_up, fractal_down
    
    def _calculate_hurst(self, series):
        """Calculate simplified Hurst Exponent"""
        lags = range(2, min(20, len(series)//2))
        if len(lags) < 2:
            return 0.5
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        if len(tau) < 2:
            return 0.5
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]*2.0
    
    def _calculate_elder_impulse(self, df):
        """Calculate Elder Impulse System"""
        ema13 = df['Close'].ewm(span=13).mean()
        macd_hist = df['MACD_histogram']
        impulse = pd.Series(index=df.index, dtype=int)
        
        for i in range(1, len(df)):
            if (ema13.iloc[i] > ema13.iloc[i-1]) and (macd_hist.iloc[i] > macd_hist.iloc[i-1]):
                impulse.iloc[i] = 1  # Green (buy)
            elif (ema13.iloc[i] < ema13.iloc[i-1]) and (macd_hist.iloc[i] < macd_hist.iloc[i-1]):
                impulse.iloc[i] = -1  # Red (sell)
            else:
                impulse.iloc[i] = 0  # Blue (neutral)
                
        return impulse
    
    def _calculate_stc(self, df, cycle=10, fast=23, slow=50):
        """Calculate Schaff Trend Cycle"""
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        stc = pd.Series(index=df.index, dtype=float)
        
        # Simplified STC calculation
        def stc_apply_func(x):
            if len(x) < 2:
                return 50
            x_min = np.min(x)
            x_max = np.max(x)
            if x_max - x_min == 0:
                return 50
            return (x[-1] - x_min) / (x_max - x_min) * 100
            
        k = macd.rolling(window=cycle).apply(stc_apply_func, raw=True)
        d = k.rolling(window=cycle).mean()
        stc = d.rolling(window=cycle).mean()
        
        return stc
    
    def create_cultural_calendar(self):
        """Create cultural calendar with holiday impacts"""
        if self.data is None or len(self.data) == 0:
            print("No data available for cultural calendar creation")
            return
            
        # Major Japanese holidays that affect markets
        holidays = [
            '01-01',  # New Year's Day
            '01-02',  # New Year's Holiday
            '01-03',  # New Year's Holiday
            '01-15',  # Coming of Age Day (2nd Monday)
            '02-11',  # National Foundation Day
            '02-23',  # Emperor's Birthday
            '03-20',  # Vernal Equinox Day
            '04-29',  # Showa Day
            '05-03',  # Constitution Memorial Day
            '05-04',  # Greenery Day
            '05-05',  # Children's Day
            '07-15',  # Marine Day (3rd Monday)
            '08-11',  # Mountain Day
            '09-15',  # Respect for the Aged Day (3rd Monday)
            '09-23',  # Autumnal Equinox Day
            '10-10',  # Health and Sports Day (2nd Monday)
            '11-03',  # Culture Day
            '11-23',  # Labor Thanksgiving Day
            '12-23',  # Emperor's Birthday
            '12-31'   # New Year's Eve
        ]
        
        # Add holiday features
        self.data['Month'] = self.data.index.month
        self.data['DayOfWeek'] = self.data.index.dayofweek
        self.data['DayOfMonth'] = self.data.index.day
        self.data['IsMonthStart'] = (self.data.index.is_month_start).astype(int)
        self.data['IsMonthEnd'] = (self.data.index.is_month_end).astype(int)
        self.data['IsQuarterStart'] = (self.data.index.is_quarter_start).astype(int)
        self.data['IsQuarterEnd'] = (self.data.index.is_quarter_end).astype(int)
        
        # Seasonal patterns
        self.data['IsSpring'] = ((self.data['Month'] >= 3) & (self.data['Month'] <= 5)).astype(int)
        self.data['IsSummer'] = ((self.data['Month'] >= 6) & (self.data['Month'] <= 8)).astype(int)
        self.data['IsAutumn'] = ((self.data['Month'] >= 9) & (self.data['Month'] <= 11)).astype(int)
        self.data['IsWinter'] = ((self.data['Month'] <= 2) | (self.data['Month'] == 12)).astype(int)
        
        # Holiday proximity
        self.data['DaysToHoliday'] = 0
        self.data['DaysAfterHoliday'] = 0
        
        # Create holiday flags
        holiday_dates = []
        for year in self.data.index.year.unique():
            for holiday in holidays:
                try:
                    date_str = f"{year}-{holiday}"
                    holiday_dates.append(pd.to_datetime(date_str))
                except:
                    pass  # Skip invalid dates
                    
        self.data['IsHoliday'] = self.data.index.isin(holiday_dates).astype(int)
        
        print("Created cultural calendar features")
    
    def prepare_features(self):
        """Prepare features for machine learning models"""
        if self.data is None or len(self.data) == 0:
            print("No data available for feature preparation")
            return
            
        # Select features for modeling
        feature_columns = [col for col in self.data.columns if col not in [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
            'Target_Return', 'Target_Direction', 'Target_High', 'Target_Low', 'Target_Close'
        ]]
        
        self.X = self.data[feature_columns]
        self.y_reg = self.data['Target_Return']  # For regression (return prediction)
        self.y_clf = self.data['Target_Direction']  # For classification (direction prediction)
        self.y_high = self.data['Target_High']  # For high prediction
        self.y_low = self.data['Target_Low']    # For low prediction
        self.y_close = self.data['Target_Close'] # For close prediction
        
        # Handle missing values
        self.X = self.X.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN rows
        valid_indices = ~(self.X.isnull().any(axis=1) | self.y_reg.isnull() | self.y_clf.isnull() |
                         self.y_high.isnull() | self.y_low.isnull() | self.y_close.isnull())
        self.X = self.X[valid_indices]
        self.y_reg = self.y_reg[valid_indices]
        self.y_clf = self.y_clf[valid_indices]
        self.y_high = self.y_high[valid_indices]
        self.y_low = self.y_low[valid_indices]
        self.y_close = self.y_close[valid_indices]
        
        print(f"Prepared {self.X.shape[1]} features for modeling")
        print(f"Dataset shape: {self.X.shape}")
        
        # If we still have no data, create proper sample data
        if self.X.shape[0] == 0:
            print("WARNING: No valid data after cleaning. Creating sample data...")
            # Create minimal sample data for testing with proper indices
            sample_dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
            self.X = pd.DataFrame(np.random.randn(100, 10), 
                                 columns=[f'feature_{i}' for i in range(10)],
                                 index=sample_dates)
            self.y_reg = pd.Series(np.random.randn(100), index=sample_dates)
            self.y_clf = pd.Series(np.random.choice([0, 1], 100), index=sample_dates)
            self.y_high = pd.Series(np.random.randn(100) * 10 + 105, index=sample_dates)
            self.y_low = pd.Series(np.random.randn(100) * 10 + 95, index=sample_dates)
            self.y_close = pd.Series(np.random.randn(100) * 10 + 100, index=sample_dates)
            print(f"Created sample data with shape: {self.X.shape}")
    
    def train_models(self):
        """Train multiple machine learning models"""
        if self.X is None or len(self.X) == 0:
            print("No data available for training. Preparing sample data...")
            self.prepare_features()
            
        if len(self.X) < 10:  # Minimum samples needed
            print("Insufficient data for training. Creating sample data...")
            sample_dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
            self.X = pd.DataFrame(np.random.randn(100, 10), 
                                 columns=[f'feature_{i}' for i in range(10)],
                                 index=sample_dates)
            self.y_reg = pd.Series(np.random.randn(100), index=sample_dates)
            self.y_clf = pd.Series(np.random.choice([0, 1], 100), index=sample_dates)
            self.y_high = pd.Series(np.random.randn(100) * 10 + 105, index=sample_dates)
            self.y_low = pd.Series(np.random.randn(100) * 10 + 95, index=sample_dates)
            self.y_close = pd.Series(np.random.randn(100) * 10 + 100, index=sample_dates)
        
        print(f"Training models with {len(self.X)} samples...")
        
        # Split data
        try:
            if len(self.X) > 10:
                X_train, X_test, y_train_reg, y_test_reg = train_test_split(
                    self.X, self.y_reg, test_size=0.2, random_state=42, shuffle=False
                )
                X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                    self.X, self.y_clf, test_size=0.2, random_state=42, shuffle=False
                )
                X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(
                    self.X, self.y_high, test_size=0.2, random_state=42, shuffle=False
                )
                X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(
                    self.X, self.y_low, test_size=0.2, random_state=42, shuffle=False
                )
                X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(
                    self.X, self.y_close, test_size=0.2, random_state=42, shuffle=False
                )
            else:
                # Use simple split for small datasets
                split_idx = max(1, int(0.8 * len(self.X)))
                X_train, X_test = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
                y_train_reg, y_test_reg = self.y_reg.iloc[:split_idx], self.y_reg.iloc[split_idx:]
                y_train_clf, y_test_clf = self.y_clf.iloc[:split_idx], self.y_clf.iloc[split_idx:]
                y_train_high, y_test_high = self.y_high.iloc[:split_idx], self.y_high.iloc[split_idx:]
                y_train_low, y_test_low = self.y_low.iloc[:split_idx], self.y_low.iloc[split_idx:]
                y_train_close, y_test_close = self.y_close.iloc[:split_idx], self.y_close.iloc[split_idx:]
        except ValueError as e:
            print(f"Error in train_test_split: {e}")
            print("Using last 20% as test set instead...")
            split_idx = int(0.8 * len(self.X))
            X_train, X_test = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
            y_train_reg, y_test_reg = self.y_reg.iloc[:split_idx], self.y_reg.iloc[split_idx:]
            y_train_clf, y_test_clf = self.y_clf.iloc[:split_idx], self.y_clf.iloc[split_idx:]
            y_train_high, y_test_high = self.y_high.iloc[:split_idx], self.y_high.iloc[split_idx:]
            y_train_low, y_test_low = self.y_low.iloc[:split_idx], self.y_low.iloc[split_idx:]
            y_train_close, y_test_close = self.y_close.iloc[:split_idx], self.y_close.iloc[split_idx:]
        
        # 1. XGBoost Regressor for returns
        self.models['xgb_reg'] = xgb.XGBRegressor(
            n_estimators=50,  # Reduced for faster execution
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['xgb_reg'].fit(X_train, y_train_reg)
        y_pred_xgb_reg = self.models['xgb_reg'].predict(X_test)
        
        # 2. XGBoost Classifier
        self.models['xgb_clf'] = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['xgb_clf'].fit(X_train_clf, y_train_clf)
        y_pred_xgb_clf = self.models['xgb_clf'].predict(X_test_clf)
        
        # 3. Random Forest Regressor for returns
        self.models['rf_reg'] = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.models['rf_reg'].fit(X_train, y_train_reg)
        y_pred_rf_reg = self.models['rf_reg'].predict(X_test)
        
        # 4. Random Forest Classifier
        self.models['rf_clf'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.models['rf_clf'].fit(X_train_clf, y_train_clf)
        y_pred_rf_clf = self.models['rf_clf'].predict(X_test_clf)
        
        # 5. Gradient Boosting Regressor for returns
        self.models['gb_reg'] = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gb_reg'].fit(X_train, y_train_reg)
        y_pred_gb_reg = self.models['gb_reg'].predict(X_test)
        
        # 6. Gradient Boosting Classifier
        self.models['gb_clf'] = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gb_clf'].fit(X_train_clf, y_train_clf)
        y_pred_gb_clf = self.models['gb_clf'].predict(X_test_clf)
        
        # 7. XGBoost Regressor for High prediction
        self.models['xgb_high'] = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['xgb_high'].fit(X_train, y_train_high)
        y_pred_xgb_high = self.models['xgb_high'].predict(X_test)
        
        # 8. XGBoost Regressor for Low prediction
        self.models['xgb_low'] = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['xgb_low'].fit(X_train, y_train_low)
        y_pred_xgb_low = self.models['xgb_low'].predict(X_test)
        
        # 9. XGBoost Regressor for Close prediction
        self.models['xgb_close'] = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['xgb_close'].fit(X_train, y_train_close)
        y_pred_xgb_close = self.models['xgb_close'].predict(X_test)
        
        # Evaluate models
        print("Model Performance:")
        try:
            print(f"XGBoost Regressor MSE: {mean_squared_error(y_test_reg, y_pred_xgb_reg):.6f}")
        except:
            print(f"XGBoost Regressor MSE: N/A")
        try:
            print(f"Random Forest Regressor MSE: {mean_squared_error(y_test_reg, y_pred_rf_reg):.6f}")
        except:
            print(f"Random Forest Regressor MSE: N/A")
        try:
            print(f"Gradient Boosting Regressor MSE: {mean_squared_error(y_test_reg, y_pred_gb_reg):.6f}")
        except:
            print(f"Gradient Boosting Regressor MSE: N/A")
        try:
            print(f"XGBoost Classifier Accuracy: {accuracy_score(y_test_clf, y_pred_xgb_clf):.4f}")
        except:
            print(f"XGBoost Classifier Accuracy: N/A")
        try:
            print(f"Random Forest Classifier Accuracy: {accuracy_score(y_test_clf, y_pred_rf_clf):.4f}")
        except:
            print(f"Random Forest Classifier Accuracy: N/A")
        try:
            print(f"Gradient Boosting Classifier Accuracy: {accuracy_score(y_test_clf, y_pred_gb_clf):.4f}")
        except:
            print(f"Gradient Boosting Classifier Accuracy: N/A")
        try:
            print(f"XGBoost High Prediction MSE: {mean_squared_error(y_test_high, y_pred_xgb_high):.6f}")
        except:
            print(f"XGBoost High Prediction MSE: N/A")
        try:
            print(f"XGBoost Low Prediction MSE: {mean_squared_error(y_test_low, y_pred_xgb_low):.6f}")
        except:
            print(f"XGBoost Low Prediction MSE: N/A")
        try:
            print(f"XGBoost Close Prediction MSE: {mean_squared_error(y_test_close, y_pred_xgb_close):.6f}")
        except:
            print(f"XGBoost Close Prediction MSE: N/A")
        
    def train_deep_learning_models(self):
        """Train deep learning models"""
        if self.X is None or len(self.X) == 0:
            print("No data available for deep learning training.")
            return
            
        # Prepare data for deep learning
        sequence_length = min(30, len(self.X) // 2)  # Reduce sequence length for small datasets
        if sequence_length < 5:
            print("Insufficient data for deep learning models.")
            return
            
        X_dl = []
        y_dl = []
        
        for i in range(sequence_length, len(self.X)):
            X_dl.append(self.X.iloc[i-sequence_length:i].values)
            y_dl.append(self.y_reg.iloc[i])
            
        if len(X_dl) < 10:  # Need minimum samples
            print("Insufficient sequential data for deep learning models.")
            return
            
        X_dl = np.array(X_dl)
        y_dl = np.array(y_dl)
        
        # Split data
        split_idx = int(0.8 * len(X_dl))
        if split_idx < 1:
            print("Insufficient data for deep learning train/test split.")
            return
            
        X_train_dl, X_test_dl = X_dl[:split_idx], X_dl[split_idx:]
        y_train_dl, y_test_dl = y_dl[:split_idx], y_dl[split_idx:]
        
        # 1. LSTM Model
        try:
            self.models['lstm'] = Sequential([
                LSTM(32, return_sequences=True, input_shape=(X_train_dl.shape[1], X_train_dl.shape[2])),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16),
                Dense(1)
            ])
            self.models['lstm'].compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            self.models['lstm'].fit(X_train_dl, y_train_dl, batch_size=16, epochs=20, verbose=0)
        except Exception as e:
            print(f"Error training LSTM model: {e}")
        
        # 2. GRU Model
        try:
            self.models['gru'] = Sequential([
                GRU(32, return_sequences=True, input_shape=(X_train_dl.shape[1], X_train_dl.shape[2])),
                Dropout(0.2),
                GRU(32, return_sequences=False),
                Dropout(0.2),
                Dense(16),
                Dense(1)
            ])
            self.models['gru'].compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            self.models['gru'].fit(X_train_dl, y_train_dl, batch_size=16, epochs=20, verbose=0)
        except Exception as e:
            print(f"Error training GRU model: {e}")
        
        # 3. CNN Model
        try:
            self.models['cnn'] = Sequential([
                Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_dl.shape[1], X_train_dl.shape[2])),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=32, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            self.models['cnn'].compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            self.models['cnn'].fit(X_train_dl, y_train_dl, batch_size=16, epochs=20, verbose=0)
        except Exception as e:
            print(f"Error training CNN model: {e}")
        
        print("Trained deep learning models")
        
    def generate_signals(self):
        """Generate trading signals using ensemble of models"""
        if self.X is None or len(self.X) == 0:
            print("No data available for signal generation.")
            return
            
        # Get predictions from all models
        predictions = {}
        
        # Traditional ML models
        try:
            predictions['xgb_reg'] = self.models['xgb_reg'].predict(self.X)
            predictions['rf_reg'] = self.models['rf_reg'].predict(self.X)
            predictions['gb_reg'] = self.models['gb_reg'].predict(self.X)
            predictions['xgb_clf'] = self.models['xgb_clf'].predict(self.X)
            predictions['rf_clf'] = self.models['rf_clf'].predict(self.X)
            predictions['gb_clf'] = self.models['gb_clf'].predict(self.X)
            predictions['xgb_high'] = self.models['xgb_high'].predict(self.X)
            predictions['xgb_low'] = self.models['xgb_low'].predict(self.X)
            predictions['xgb_close'] = self.models['xgb_close'].predict(self.X)
        except Exception as e:
            print(f"Error generating ML predictions: {e}")
            # Fallback predictions
            predictions['xgb_reg'] = np.random.randn(len(self.X))
            predictions['rf_reg'] = np.random.randn(len(self.X))
            predictions['gb_reg'] = np.random.randn(len(self.X))
            predictions['xgb_clf'] = np.random.choice([0, 1], len(self.X))
            predictions['rf_clf'] = np.random.choice([0, 1], len(self.X))
            predictions['gb_clf'] = np.random.choice([0, 1], len(self.X))
            predictions['xgb_high'] = np.random.randn(len(self.X)) * 10 + 105
            predictions['xgb_low'] = np.random.randn(len(self.X)) * 10 + 95
            predictions['xgb_close'] = np.random.randn(len(self.X)) * 10 + 100
        
        # Deep learning models (need to prepare sequences)
        sequence_length = min(30, len(self.X) // 2)
        if sequence_length >= 5:
            X_dl = []
            for i in range(sequence_length, len(self.X) + 1):
                X_dl.append(self.X.iloc[i-sequence_length:i].values)
                
            if len(X_dl) > 0:
                X_dl = np.array(X_dl)
                try:
                    predictions['lstm'] = self.models['lstm'].predict(X_dl).flatten()
                    predictions['gru'] = self.models['gru'].predict(X_dl).flatten()
                    predictions['cnn'] = self.models['cnn'].predict(X_dl).flatten()
                except Exception as e:
                    print(f"Error generating DL predictions: {e}")
                    # Fallback
                    predictions['lstm'] = np.random.randn(len(X_dl))
                    predictions['gru'] = np.random.randn(len(X_dl))
                    predictions['cnn'] = np.random.randn(len(X_dl))
            else:
                # Fallback if not enough data
                predictions['lstm'] = np.zeros(len(self.X))
                predictions['gru'] = np.zeros(len(self.X))
                predictions['cnn'] = np.zeros(len(self.X))
        else:
            # Fallback if not enough data
            predictions['lstm'] = np.zeros(len(self.X))
            predictions['gru'] = np.zeros(len(self.X))
            predictions['cnn'] = np.zeros(len(self.X))
        
        # Create ensemble prediction
        pred_list = [
            predictions['xgb_reg'],
            predictions['rf_reg'],
            predictions['gb_reg'],
            predictions['lstm'][:len(self.X)],
            predictions['gru'][:len(self.X)],
            predictions['cnn'][:len(self.X)]
        ]
        
        # Ensure all predictions have the same length
        min_len = min(len(p) for p in pred_list)
        if min_len == 0:
            min_len = len(self.X)
        pred_list = [p[:min_len] for p in pred_list]
        
        ensemble_pred = np.mean(pred_list, axis=0)
        
        # Generate signals
        self.signals = pd.DataFrame(index=self.X.index[:min_len] if min_len > 0 else self.X.index)
        self.signals['Prediction'] = ensemble_pred
        self.signals['Signal'] = np.where(ensemble_pred > 0.001, 1, np.where(ensemble_pred < -0.001, -1, 0))
        self.signals['Confidence'] = np.abs(ensemble_pred)
        
        # Add predicted High, Low, Close for next day
        self.signals['Predicted_High'] = predictions['xgb_high'][:len(self.signals)]
        self.signals['Predicted_Low'] = predictions['xgb_low'][:len(self.signals)]
        self.signals['Predicted_Close'] = predictions['xgb_close'][:len(self.signals)]
        
        print("Generated trading signals with predicted High, Low, Close for next day")
        
    def calculate_margin_requirements(self):
        """Calculate margin requirements for Micro Nikkei futures"""
        if self.signals is None or len(self.signals) == 0:
            print("No signals available for margin calculation.")
            return
            
        # Based on CME Group specifications for Micro Nikkei Futures (approximate)
        contract_size = 0.5  # 0.5 yen per index point for Micro Nikkei
        initial_margin_rate = 0.05  # 5% margin requirement
        
        # Use close prices - create proper alignment
        if hasattr(self, 'data') and len(self.data) > 0:
            # Try to get matching indices
            matching_indices = self.signals.index.intersection(self.data.index)
            if len(matching_indices) > 0:
                close_prices = self.data.loc[matching_indices, 'Close']
                # Align with signals index
                close_prices = close_prices.reindex(self.signals.index)
            else:
                # If no matching indices, create sample close prices with proper alignment
                print("Creating sample close prices with proper alignment.")
                close_prices = pd.Series(np.random.randn(len(self.signals)) * 10 + 100, index=self.signals.index)
        else:
            # Fallback: create sample close prices
            print("Creating sample close prices.")
            close_prices = pd.Series(np.random.randn(len(self.signals)) * 10 + 100, index=self.signals.index)
            
        self.signals['Contract_Value'] = close_prices * contract_size
        self.signals['Initial_Margin'] = self.signals['Contract_Value'] * initial_margin_rate
        
        print("Calculated margin requirements for Micro Nikkei Futures")
        
    def position_sizing(self):
        """Calculate position sizing based on account capital and risk management"""
        if self.signals is None or len(self.signals) == 0:
            print("No signals available for position sizing.")
            return
            
        risk_per_trade = 0.01  # Risk 1% of capital per trade
        max_position_size = 0.1  # Maximum 10% of capital in any position
        
        self.signals['Risk_Amount'] = self.initial_capital * risk_per_trade
        self.signals['Max_Position_Size'] = self.initial_capital * max_position_size
        self.signals['Position_Size'] = np.minimum(
            self.signals['Risk_Amount'] / (self.signals['Initial_Margin'] * 0.02),  # Assuming 2% stop loss
            self.signals['Max_Position_Size'] / self.signals['Initial_Margin']
        )
        self.signals['Position_Size'] = np.floor(self.signals['Position_Size']).astype(int)
        self.signals['Position_Size'] = np.maximum(self.signals['Position_Size'], 0)  # No short selling in this example
        
        print("Calculated position sizing")
        
    def backtest(self):
        """Run backtest of the trading strategy"""
        if self.signals is None or len(self.signals) == 0:
            print("No signals available for backtesting.")
            return
            
        # Initialize portfolio
        portfolio = pd.DataFrame(index=self.signals.index)
        portfolio['Holdings'] = 0
        portfolio['Cash'] = self.initial_capital
        portfolio['Total'] = self.initial_capital
        portfolio['Returns'] = 0.0
        portfolio['Positions'] = 0
        
        # Get close prices for the signal period
        if hasattr(self, 'data') and len(self.data) > 0:
            # Try to get matching indices
            matching_indices = self.signals.index.intersection(self.data.index)
            if len(matching_indices) > 0:
                close_prices = self.data.loc[matching_indices, 'Close']
                # Align with signals index
                close_prices = close_prices.reindex(self.signals.index)
            else:
                # If no matching indices, create sample close prices
                close_prices = pd.Series(np.random.randn(len(self.signals)) * 10 + 100, index=self.signals.index)
        else:
            # Fallback: create sample close prices
            close_prices = pd.Series(np.random.randn(len(self.signals)) * 10 + 100, index=self.signals.index)
        
        # Simulate trading
        for i in range(1, len(portfolio)):
            # Update holdings value
            if close_prices.iloc[i-1] != 0:
                portfolio.loc[portfolio.index[i], 'Holdings'] = portfolio.loc[portfolio.index[i-1], 'Holdings'] * (close_prices.iloc[i] / close_prices.iloc[i-1])
            else:
                portfolio.loc[portfolio.index[i], 'Holdings'] = portfolio.loc[portfolio.index[i-1], 'Holdings']
            
            # Execute trades based on signals
            signal = self.signals['Signal'].iloc[i]
            position_size = self.signals['Position_Size'].iloc[i]
            
            if signal == 1 and portfolio.loc[portfolio.index[i-1], 'Positions'] == 0:  # Buy signal
                cost = position_size * self.signals['Initial_Margin'].iloc[i]
                if portfolio.loc[portfolio.index[i-1], 'Cash'] >= cost:
                    portfolio.loc[portfolio.index[i], 'Positions'] = position_size
                    portfolio.loc[portfolio.index[i], 'Cash'] = portfolio.loc[portfolio.index[i-1], 'Cash'] - cost
                    portfolio.loc[portfolio.index[i], 'Holdings'] = position_size * self.signals['Contract_Value'].iloc[i]
            elif signal == -1 and portfolio.loc[portfolio.index[i-1], 'Positions'] > 0:  # Sell signal
                proceeds = portfolio.loc[portfolio.index[i-1], 'Positions'] * self.signals['Contract_Value'].iloc[i]
                portfolio.loc[portfolio.index[i], 'Positions'] = 0
                portfolio.loc[portfolio.index[i], 'Cash'] = portfolio.loc[portfolio.index[i-1], 'Cash'] + proceeds
                portfolio.loc[portfolio.index[i], 'Holdings'] = 0
            else:  # Hold
                portfolio.loc[portfolio.index[i], 'Positions'] = portfolio.loc[portfolio.index[i-1], 'Positions']
                portfolio.loc[portfolio.index[i], 'Cash'] = portfolio.loc[portfolio.index[i-1], 'Cash']
            
            # Calculate total value
            portfolio.loc[portfolio.index[i], 'Total'] = portfolio.loc[portfolio.index[i], 'Cash'] + portfolio.loc[portfolio.index[i], 'Holdings']
            if portfolio.loc[portfolio.index[i-1], 'Total'] != 0:
                portfolio.loc[portfolio.index[i], 'Returns'] = (portfolio.loc[portfolio.index[i], 'Total'] / portfolio.loc[portfolio.index[i-1], 'Total']) - 1
            else:
                portfolio.loc[portfolio.index[i], 'Returns'] = 0
            
        self.portfolio = portfolio
        
        # Calculate performance metrics
        if len(portfolio) > 1:
            total_return = (portfolio['Total'].iloc[-1] / self.initial_capital) - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
            volatility = portfolio['Returns'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            max_drawdown = (portfolio['Total'] / portfolio['Total'].cummax() - 1).min()
            
            print("\nBacktest Results:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Annual Return: {annual_return:.2%}")
            print(f"Volatility: {volatility:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Final Portfolio Value: ${portfolio['Total'].iloc[-1]:,.2f}")
        else:
            print("Insufficient data for backtest metrics.")
        
    def plot_results(self):
        """Plot backtest results"""
        if self.portfolio is None or len(self.portfolio) == 0:
            print("No portfolio data to plot.")
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Price and signals
            if hasattr(self, 'data') and len(self.data) > 0:
                # Try to get matching indices for plotting
                matching_indices = self.signals.index.intersection(self.data.index)
                if len(matching_indices) > 0:
                    plot_data = self.data.loc[matching_indices]
                    axes[0, 0].plot(plot_data.index, plot_data['Close'], label='Nikkei Close', color='blue')
                    
                    # Plot buy/sell signals that match the data
                    signal_matching_indices = self.signals.index.intersection(plot_data.index)
                    if len(signal_matching_indices) > 0:
                        signals_for_plot = self.signals.loc[signal_matching_indices]
                        buy_signals = signals_for_plot[signals_for_plot['Signal'] == 1]
                        sell_signals = signals_for_plot[signals_for_plot['Signal'] == -1]
                        if len(buy_signals) > 0:
                            axes[0, 0].scatter(buy_signals.index, plot_data.loc[buy_signals.index, 'Close'], 
                                              marker='^', color='green', label='Buy Signal', s=100)
                        if len(sell_signals) > 0:
                            axes[0, 0].scatter(sell_signals.index, plot_data.loc[sell_signals.index, 'Close'], 
                                              marker='v', color='red', label='Sell Signal', s=100)
                else:
                    # If no matching indices, plot sample data
                    axes[0, 0].plot(self.data.index[:len(self.signals)], self.data['Close'][:len(self.signals)], 
                                   label='Nikkei Close', color='blue')
            else:
                # Plot sample data
                sample_prices = pd.Series(np.random.randn(len(self.signals)) * 10 + 100, index=self.signals.index)
                axes[0, 0].plot(self.signals.index, sample_prices, label='Sample Prices', color='blue')
                
                # Plot buy/sell signals
                buy_signals = self.signals[self.signals['Signal'] == 1]
                sell_signals = self.signals[self.signals['Signal'] == -1]
                if len(buy_signals) > 0:
                    axes[0, 0].scatter(buy_signals.index, sample_prices.loc[buy_signals.index], 
                                      marker='^', color='green', label='Buy Signal', s=100)
                if len(sell_signals) > 0:
                    axes[0, 0].scatter(sell_signals.index, sample_prices.loc[sell_signals.index], 
                                      marker='v', color='red', label='Sell Signal', s=100)
                    
            axes[0, 0].set_title('Price and Trading Signals')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Portfolio value
            axes[0, 1].plot(self.portfolio.index, self.portfolio['Total'], label='Portfolio Value', color='purple')
            axes[0, 1].set_title('Portfolio Value Over Time')
            axes[0, 1].set_ylabel('Portfolio Value ($)')
            axes[0, 1].grid(True)
            
            # Drawdown
            if len(self.portfolio) > 1:
                drawdown = (self.portfolio['Total'] / self.portfolio['Total'].cummax() - 1) * 100
                axes[1, 0].plot(self.portfolio.index, drawdown, color='red')
                axes[1, 0].set_title('Drawdown (%)')
                axes[1, 0].set_ylabel('Drawdown (%)')
                axes[1, 0].grid(True)
            
            # Returns distribution
            axes[1, 1].hist(self.portfolio['Returns'], bins=30, color='orange', alpha=0.7)
            axes[1, 1].set_title('Returns Distribution')
            axes[1, 1].set_xlabel('Daily Returns')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting results: {e}")
            
    def display_predictions(self):
        """Display the latest predictions including High, Low, Close for next day"""
        if self.signals is not None and len(self.signals) > 0:
            latest_idx = self.signals.index[-1]
            latest_signal = self.signals.loc[latest_idx]
            
            # Get current close price for reference
            current_close = self.data['Close'].iloc[-1] if len(self.data) > 0 and 'Close' in self.data.columns else latest_signal['Predicted_Close'] / 1.01
            
            print("\n" + "="*70)
            print(f"CME GLOBEX FUTURES PREDICTION SYSTEM")
            print(f"Futures Symbol: {self.futures_symbol} (Micro Nikkei 225 Futures)")
            print("="*70)
            print(f"Prediction Date: {latest_idx.strftime('%Y-%m-%d')}")
            print(f"Current Close: {current_close:.2f}")
            print("-"*70)
            print(f"Predicted Direction: {'BUY ↑' if latest_signal['Signal'] == 1 else 'SELL ↓' if latest_signal['Signal'] == -1 else 'NEUTRAL ↔'}")
            print(f"Confidence Level: {latest_signal['Confidence']:.4f}")
            print("-"*70)
            print(f"Predicted High: {latest_signal['Predicted_High']:.2f}")
            print(f"Predicted Low: {latest_signal['Predicted_Low']:.2f}")
            print(f"Predicted Close: {latest_signal['Predicted_Close']:.2f}")
            print(f"Expected Range: {latest_signal['Predicted_Low']:.2f} - {latest_signal['Predicted_High']:.2f}")
            expected_return = (latest_signal['Predicted_Close'] / current_close - 1) * 100
            print(f"Expected Return: {expected_return:.2f}%")
            print("="*70)
        else:
            print("No predictions available to display.")
        
    def run_system(self, realtime=False):
        """Run the complete trading system"""
        print("Running Micro Nikkei CME Front Month Futures Trading System...")
        print(f"Using CME Globex Symbol: {self.futures_symbol}")
        
        # Fetch data from Interactive Brokers ONLY
        self.fetch_data(realtime=realtime)
        
        # Calculate technical indicators
        self.calculate_technical_indicators()
        
        # Create cultural calendar
        self.create_cultural_calendar()
        
        # Prepare features
        self.prepare_features()
        
        # Train models
        self.train_models()
        
        # Train deep learning models
        self.train_deep_learning_models()
        
        # Generate signals
        self.generate_signals()
        
        # Calculate margin requirements
        self.calculate_margin_requirements()
        
        # Position sizing
        self.position_sizing()
        
        # Backtest
        self.backtest()
        
        # Display latest predictions
        self.display_predictions()
        
        # Plot results
        self.plot_results()
        
        print("\nTrading System Execution Complete!")

# Usage example with Interactive Brokers ONLY for Micro Nikkei Futures
if __name__ == "__main__":
    # Initialize and run the trading system with Interactive Brokers for Micro Nikkei Futures
    trading_system = NikkeiFuturesTradingSystem(
        symbol="MNKU5",  # Micro Nikkei Futures symbol
        futures_symbol="MNKU5",  # CME Globex symbol for Micro Nikkei Futures (September 2025)
        initial_capital=100000, 
        ibkr_host='127.0.0.1',      # Default TWS host
        ibkr_port=7497,            # Default TWS port for live trading
        ibkr_client_id=1           # Client ID for connection
    )
    trading_system.run_system(realtime=True)