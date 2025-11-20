import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import time
import datetime
import warnings

warnings.filterwarnings("ignore")

# --------------------------
# AYARLAR
# --------------------------
tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD"]
initial_capital = 1000  # Her coin için başlangıç sermayesi ($)
commission = 0.001       # %0.1 komisyon
n_states = 3
validation_days = 21
decision_threshold = 0.25
min_hold_hours = 6       # Minimum holding süresi
stop_loss_pct = 0.02     # %2 zarar
take_profit_pct = 0.05   # %5 kâr

TIME_FRAMES = {'GÜNLÜK':'1d','HAFTALIK':'7d','AYLIK':'30d'}
WEIGHT_CANDIDATES = np.linspace(0.1,0.9,9)

# --------------------------
# FONKSİYONLAR
# --------------------------
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

def get_data_yf(ticker, period='3y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    return df

def optimize_dynamic_weights(df):
    df = df.copy()
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    if len(df)<validation_days+5: return (0.7,0.3)
    
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
        w_score = 1 - w_hmm
        cash = initial_capital
        coin_amt = 0
        for idx,row in test_df.iterrows():
            X_test = scaler.transform([[row['log_ret'], row['range']]])
            hmm_signal = 1 if model.predict(X_test)[0]==bull_state else (-1 if model.predict(X_test)[0]==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            price = row['close']
            if decision>decision_threshold: coin_amt = cash/price; cash=0
            elif decision<-decision_threshold: cash = coin_amt*price; coin_amt=0
        final_val = cash + coin_amt*test_df['close'].iloc[-1]
        roi = (final_val-initial_capital)/initial_capital
        if roi>best_roi: best_roi=roi; best_w=(w_hmm,w_score)
    return best_w

def analyze_mtf_signal(df_dict, w_hmm, w_score):
    # df_dict: {timeframe: df}
    signal_score = 0
    selected_tf = "N/A"
    for tf, df in df_dict.items():
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
            signal_score += 1
            selected_tf = tf
        elif decision<-decision_threshold:
            signal_score -=1
            selected_tf = tf
    if signal_score>0: return "BUY", selected_tf
    elif signal_score<0: return "SELL", selected_tf
    else: return "HOLD", selected_tf

# --------------------------
# CANLI BOT / SIMÜLASYON
# --------------------------
positions = {}  # {ticker: {'qty':x, 'price':y, 'time':timestamp}}
portfolio_history = []

while True:
    now = datetime.datetime.now()
    print(f"\n=== {now} ===")
    total_value = 0
    for ticker in tickers:
        try:
            df_dict = {tf:get_data_yf(ticker, period='1y', interval='1d') if tf=='GÜNLÜK' else
                             get_data_yf(ticker, period='2y', interval='7d') if tf=='HAFTALIK' else
                             get_data_yf(ticker, period='3y', interval='30d') 
                       for tf in TIME_FRAMES}
            
            # Optimize
            w_hmm, w_score = optimize_dynamic_weights(df_dict['GÜNLÜK'])
            
            # Multi-Timeframe Signal
            signal, tf_used = analyze_mtf_signal(df_dict, w_hmm, w_score)
            
            # Pozisyon Yönetimi
            price = df_dict['GÜNLÜK']['close'].iloc[-1]
            pos = positions.get(ticker, {'qty':0, 'price':0, 'time':None})
            qty, entry_price, entry_time = pos['qty'], pos['price'], pos['time']
            
            # Minimum Holding ve Stop-Loss / Take-Profit
            can_trade = True
            if entry_time:
                delta_hours = (now - entry_time).total_seconds()/3600
                if delta_hours < min_hold_hours: can_trade=False
            if qty>0:
                if price<entry_price*(1-stop_loss_pct): signal='SELL'
                elif price>entry_price*(1+take_profit_pct): signal='SELL'
            
            # İşlem
            if signal=="BUY" and can_trade and qty==0:
                qty = (initial_capital / price)*(1-commission)
                positions[ticker] = {'qty':qty, 'price':price, 'time':now}
                print(f"🟢 BUY {ticker} qty={qty:.4f} price={price:.2f} tf={tf_used}")
            elif signal=="SELL" and qty>0 and can_trade:
                proceeds = qty*price*(1-commission)
                positions[ticker] = {'qty':0,'price':0,'time':None}
                print(f"🔴 SELL {ticker} qty={qty:.4f} price={price:.2f} proceeds={proceeds:.2f}")
            
            # Portföy Değeri Hesapla
            total_value += price*positions.get(ticker,{'qty':0})['qty']
            
        except Exception as e:
            print(f"🚨 {ticker} Hata: {e}")
            continue
    
    total_value += sum(initial_capital for t,p in positions.items() if p['qty']==0)  # Boş pozisyon için nakit
    portfolio_history.append({'time':now, 'total_value':total_value, 'positions':positions.copy()})
    print(f"[PORTFÖY DEĞERİ] {total_value:.2f}")
    
    # Saatlik güncelleme
    time.sleep(3600)  # 1 saat
