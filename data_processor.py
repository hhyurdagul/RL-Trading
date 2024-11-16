import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        """Load and prepare the XAUUSD data"""
        self.data = pd.read_csv(self.csv_path, sep="\t")
        self.data = self.data.rename(
            columns={"<CLOSE>": "Close", "<HIGH>": "High", "<LOW>": "Low", "<OPEN>": "Open", "<TICKVOL>": "Volume"}
        )
        self.data["Date"] = pd.to_datetime(self.data["<DATE>"] + " " + self.data["<TIME>"])
        self.data = self.data.set_index('Date')
        self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return self.data
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators for the dataset"""
        df = self.data.copy()
        
        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB1_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB1_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()

        df['BB2_upper'] = df['BB_middle'] + 3 * df['Close'].rolling(window=20).std()
        df['BB2_lower'] = df['BB_middle'] - 3 * df['Close'].rolling(window=20).std()
        
        # Calculate Stochastic Oscillator (5, 3, 3)
        df['%K'] = ((df['Close'] - df['Low'].rolling(window=5).min()) / (df['High'].rolling(window=5).max() - df['Low'].rolling(window=5).min())) * 100
        df['%D'] = df['%K'].rolling(window=3).mean()
        df['%D_signal'] = df['%D'].rolling(window=3).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        self.processed_data = df
        return df
    
    def normalize_data(self):
        """Normalize the features"""
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line',
                   'BB_middle', 'BB1_upper', 'BB1_lower', 'BB2_upper', 'BB2_lower', '%K', '%D', '%D_signal']
        
        self.processed_data[features] = self.scaler.fit_transform(self.processed_data[features])
        return self.processed_data
    
    def prepare_data(self):
        """Complete data preparation pipeline"""
        self.load_data()
        self.calculate_technical_indicators()
        # self.normalize_data()
        return self.processed_data



def get_data():
    df = DataProcessor("XAUUSD_M15.csv").prepare_data()
    # Lowercase column names
    df.columns = [col.lower() for col in df.columns]
    print(df.columns)
    # Prepend feature_ to all columns
    df.columns = ['feature_' + col if col != 'Date' else col for col in df.columns]
    df["close"] = df["feature_close"]
    df["high"] = df["feature_high"]
    df["low"] = df["feature_low"]
    df["open"] = df["feature_open"]
    df["volume"] = df["feature_volume"]
    return df

