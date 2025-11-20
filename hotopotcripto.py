import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import time
import datetime

# --- AYARLAR ---
tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
initial_capital = 1000  # Her coin için başlangıç sermayesi
commission = 0.001
n_states = 3
validation_days = 21
decision_threshold = 0.25  # Al/Sat eşiği
min_roi_threshold = 0.01  # %1 ROI minimum
slippage_pct = 0.0005  # %0.05 slippage simülasyonu
take_profit = 0.05  # %5 kazanç ile realize
stop_loss = 0.03    # %3 kayıp ile realize
TIME_FRAMES = {'GÜNLÜK':'1d','HAFTALIK':'1w','AYLIK':'1M'}
WEIGHT_CANDIDATES = np.linspace(0.1,0.9,9)

# --- YARDIMCI FONKSİYONLAR ---
def calculate_custom_score(df):
    if len(df)<5: return pd.Series(0,index=df.index)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1) if 'volume' in df.columns else 0
    s7 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    return s1+s2+s3+s4+s5+s6+s7

def get_ohlcv(ticker, timeframe, limit=1000):
    df = yf.download(ticker, period='max', interval=timeframe)
    df.dropna(inplace=True)
    return df

# --- DİNAMİK AĞIRLIK OPTİMİZASYONU ---
def optimize_dynamic_weights(df):
    df = df.copy()
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    
    if len(df) < validation_days + 5: return (0.7,0.3)
    
    train_df = df.iloc[:-validation_days]
    test_df = df.iloc[-validation_days:]
    
    X = train_df[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100, random_state=42)
    model.fit(X_s)
    
    state_stats = train_df.groupby(model.predict(X_s))['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    best_roi = -np.inf
    best_w = (0.5,0.5)
    
    for w_hmm in WEIGHT_CANDIDATES:
        w_score = 1-w_hmm
        cash_sim = initial_capital
        coin_amt_sim = 0
        
        for idx,row in test_df.iterrows():
            X_test = scaler.transform([[row['log_ret'], row['range']]])
            hmm_signal = 1 if model.predict(X_test)[0]==bull_state else (-1 if model.predict(X_test)[0]==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            price = row['close'] * (1 + np.random.uniform(-slippage_pct, slippage_pct))  # slippage
            
            if decision>decision_threshold: 
                coin_amt_sim = cash_sim / price
                cash_sim = 0
            elif decision<-decision_threshold: 
                cash_sim = coin_amt_sim * price
                coin_amt_sim = 0
        
        final_val = cash_sim + coin_amt_sim*test_df['close'].iloc[-1]
        roi = (final_val-initial_capital)/initial_capital
        
        if roi>best_roi: best_roi=roi; best_w=(w_hmm,w_score)
        
    return best_w

# --- MTF SİNYAL ÜRETİMİ ---
def analyze_mtf_signal(ticker, w_hmm, w_score):
    best_signal = "HOLD"
    best_tf = "N/A"
    
    for tf_name, tf_code in TIME_FRAMES.items():
        df = get_ohlcv(ticker, tf_code)
        df['log_ret'] = np.log(df['close']/df['close'].shift(1))
        df['range'] = (df['high']-df['low'])/df['close']
        df['custom_score'] = calculate_custom_score(df)
        df.dropna(inplace=True)
        if len(df)<50: continue
        
        X = df[['log_ret','range']].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100, random_state=42)
        model.fit(X_s)
        
        state_stats = df.groupby(model.predict(X_s))['log_ret'].mean()
        bull_state = state_stats.idxmax()
        bear_state = state_stats.idxmin()
        
        last_row = df.iloc[-1]
        hmm_signal = 1 if model.predict(scaler.transform([[last_row['log_ret'], last_row['range']]]))[0]==bull_state else (-1 if model.predict(scaler.transform([[last_row['log_ret'], last_row['range']]]))[0]==bear_state else 0)
        score_signal = 1 if last_row['custom_score']>=3 else (-1 if last_row['custom_score']<=-3 else 0)
        decision = w_hmm*hmm_signal + w_score*score_signal
        
        if decision>decision_threshold:
            best_signal = "BUY"
            best_tf = tf_name
            break
        elif decision<-decision_threshold:
            best_signal = "SELL"
            best_tf = tf_name
            break
            
    return best_signal, best_tf

# --- CANLI SIMÜLASYON BOTU ---
positions = {}
cash_balance = initial_capital * len(tickers)
capital_per_coin = initial_capital

print(f"Başlangıç Nakit: {cash_balance} USDT")

for day in range(5):  # 5 günlük simülasyon örneği
    print(f"\n=== Gün {day+1} | {datetime.datetime.now()} ===")
    
    for ticker in tickers:
        df_long = get_ohlcv(ticker, '1d')
        w_hmm, w_score = optimize_dynamic_weights(df_long)
        signal, tf = analyze_mtf_signal(ticker, w_hmm, w_score)
        
        price = df_long['close'].iloc[-1] * (1 + np.random.uniform(-slippage_pct, slippage_pct))
        pos = positions.get(ticker, 0)
        
        # BUY
        if signal=="BUY" and pos==0 and cash_balance>=capital_per_coin:
            qty = (capital_per_coin / price) * (1-commission)
            positions[ticker]=qty
            cash_balance -= capital_per_coin
            print(f"🟢 {tf} | BUY {qty:.4f} {ticker} @ {price:.2f}")
        
        # SELL
        elif signal=="SELL" and pos>0:
            sell_val = pos * price
            profit = sell_val * (1-commission)
            # Stop-loss / Take-profit
            entry_price = df_long['close'].iloc[-2]  # basit varsayılan giriş fiyatı
            change = (price-entry_price)/entry_price
            if change<=-stop_loss or change>=take_profit or True:  # her SELL sinyalinde
                cash_balance += profit
                positions[ticker]=0
                print(f"🔴 {tf} | SELL {pos:.4f} {ticker} @ {price:.2f} | Kasa: {cash_balance:.2f}")
        
        else:
            print(f"⚪ {tf} | HOLD {ticker}")
    
    total_val = cash_balance
    for t, q in positions.items():
        total_val += q * df_long['close'].iloc[-1]
    print(f"[PORTFÖY] Toplam Değer: {total_val:.2f} | Nakit: {cash_balance:.2f}")
    
    time.sleep(1)  # 1 saniye bekleme (gerçek günlük simülasyonda 86400 sn)
