import ccxt
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import time
import datetime

# --- 1. AYARLAR ---
api_key = "YOUR_BINANCE_API_KEY"
api_secret = "YOUR_BINANCE_API_SECRET"

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

# Parametreler
tickers = ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT"]
initial_capital = 1000  # Her coin için USDT cinsinden başlangıç sermayesi
commission = 0.001
n_states = 3
validation_days = 21
decision_threshold = 0.25 # Al/Sat kesinleşme eşiği

# Gerekli Timeframe'ler (OHLCV çekerken kullanılacak)
TIME_FRAMES = {'GÜNLÜK': '1d', 'HAFTALIK': '1w', 'AYLIK': '1M'}
# HMM ve Score ağırlık adayları
WEIGHT_CANDIDATES = np.linspace(0.1, 0.9, 9)

# --- 2. YARDIMCI FONKSİYONLAR ---

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
    """Borsa API'den OHLCV verisini çeker."""
    ohlcv = exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.dropna(inplace=True)
    return df

# --- 3. DİNAMİK AĞIRLIK OPTİMİZASYONU ---
def optimize_dynamic_weights(df):
    """
    Son 21 günlük validation verisi üzerinde en iyi HMM/Puan ağırlığını bulur.
    Bu optimizasyon, Backtest V7'deki mantığın birebir aynısıdır.
    """
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
    best_w = (0.5,0.5) # Başlangıç değeri
    
    for w_hmm in WEIGHT_CANDIDATES:
        w_score = 1-w_hmm
        
        # Basit simülasyon (simülasyon kârını ölçmek için başlangıç sermayesini kullanır)
        cash_sim = initial_capital
        coin_amt_sim = 0
        
        for idx,row in test_df.iterrows():
            X_test = scaler.transform([[row['log_ret'], row['range']]])
            hmm_signal = 1 if model.predict(X_test)[0]==bull_state else (-1 if model.predict(X_test)[0]==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            price = row['close']
            
            if decision>decision_threshold: coin_amt_sim=cash_sim/price; cash_sim=0
            elif decision<-decision_threshold: cash_sim=coin_amt_sim*price; coin_amt_sim=0
            
        final_val = cash_sim + coin_amt_sim*test_df['close'].iloc[-1]
        roi = (final_val-initial_capital)/initial_capital
        
        if roi>best_roi: best_roi=roi; best_w=(w_hmm,w_score)
        
    return best_w

# --- 4. MTF KARAR MEKANİZMASI ---
def analyze_mtf_signal(ticker, w_hmm, w_score):
    """
    Çekilen veriyi kullanarak tüm timeframelerde sinyal üretir ve en iyi sinyali döndürür.
    Burada sadece son sinyale bakıldığı için Multi-Timeframe Turnuvası mantığı basitleştirilmiştir.
    """
    
    best_signal = "HOLD"
    best_timeframe = "N/A"
    
    for tf_name, tf_code in TIME_FRAMES.items():
        try:
            # Gerekli veriyi çek
            df = get_ohlcv(ticker, timeframe=tf_code, limit=500) # 500 mum çeker
            
            # Feature Engineering
            df['log_ret'] = np.log(df['close']/df['close'].shift(1))
            df['range'] = (df['high']-df['low'])/df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            
            if len(df) < 50: continue # Yeterli mum yoksa atla
            
            # HMM Eğitimi
            X = df[['log_ret','range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100, random_state=42)
            model.fit(X_s)
            
            # Boğa/Ayı State
            state_stats = df.groupby(model.predict(X_s))['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.idxmin()
            
            # Son Sinyal Üretimi
            last_row = df.iloc[-1]
            hmm_signal = 1 if model.predict(scaler.transform([[last_row['log_ret'], last_row['range']]]))[0]==bull_state else (-1 if model.predict(scaler.transform([[last_row['log_ret'], last_row['range']]]))[0]==bear_state else 0)
            score_signal = 1 if last_row['custom_score']>=3 else (-1 if last_row['custom_score']<=-3 else 0)
            
            decision = w_hmm*hmm_signal + w_score*score_signal
            
            if decision > decision_threshold:
                best_signal = "BUY"
                best_timeframe = tf_name
                break # En güçlü sinyali bulan ilk timeframe'i seç
            elif decision < -decision_threshold:
                best_signal = "SELL"
                best_timeframe = tf_name
                break # En güçlü sinyali bulan ilk timeframe'i seç
            
        except Exception as e:
            # print(f"MTF Analiz Hatası {tf_code}: {e}")
            continue
            
    return best_signal, best_timeframe

# --- 5. CANLI BOT DÖNGÜSÜ ---
# Mevcut pozisyon ve cüzdan bilgisini tutar
positions = {} # {ticker: miktar}
cash_balance = 0 # Gerçek bakiyeyi çekmek gerekir

# ⚠️ UYARI: Canlı işlemlerden önce cüzdan bilgisini çekmeniz gerekir.
try:
    balance = exchange.fetch_balance()
    # USDT bakiyesi üzerinden işlem yapılacağı varsayılıyor
    cash_balance = balance['total']['USDT'] 
    print(f"Başlangıç USDT Bakiyesi: {cash_balance}")
except Exception as e:
    print(f"BAKİYE ÇEKİLEMEDİ! Hata: {e}. Simülasyon için 1000 USDT varsayılıyor.")
    cash_balance = 1000

# Her coine ayrılan sermaye (Tekrar işlem yapıldığında bu sabit miktarı kullanacağız)
capital_per_coin = initial_capital 

while True:
    print(f"\n=== {datetime.datetime.now()} ===")
    
    for ticker in tickers:
        try:
            # --- Adım 1: Multi-Timeframe Verisini Çek ve Ağırlığı Optimize Et ---
            df_long = get_ohlcv(ticker, timeframe='1d', limit=1000) # Uzun dönem veri (1000 gün)
            
            # Optimizasyon: Son 21 günü en iyi açıklayan HMM/Puan ağırlığını bul
            w_hmm, w_score = optimize_dynamic_weights(df_long)
            
            # --- Adım 2: MTF Sinyali Üret (En iyi Timeframe'i bul) ---
            signal, timeframe = analyze_mtf_signal(ticker, w_hmm, w_score)
            
            # --- Adım 3: İşlem ve Pozisyon Yönetimi ---
            
            price = df_long['close'].iloc[-1]
            current_position = positions.get(ticker, 0)
            
            if signal == "BUY":
                # Alım yapılacak miktar: initial_capital (1000 USDT) ile alım yap (İlk alım)
                if current_position == 0 and cash_balance >= capital_per_coin:
                    qty = (capital_per_coin / price) * (1 - commission) # Komisyon düşüldü
                    
                    # ⚠️ Gerçek Emir:
                    # order = exchange.create_market_buy_order(ticker, qty)
                    
                    # Simülasyon:
                    positions[ticker] = qty
                    cash_balance -= capital_per_coin 
                    print(f"🟢 {timeframe} ({w_hmm:.2f} HMM): BUY {qty:.2f} {ticker} @ {price:.2f}")
                else:
                    print(f"🟡 {timeframe} ({w_hmm:.2f} HMM): HOLD - Zaten pozisyonda veya nakit yetersiz.")
            
            elif signal == "SELL":
                # Satış yapılacak miktar: Mevcut pozisyonu sat (Kar/Zarar realize edilir)
                if current_position > 0:
                    sell_usd = current_position * price
                    profit_after_fee = sell_usd * (1 - commission)
                    
                    # ⚠️ Gerçek Emir:
                    # order = exchange.create_market_sell_order(ticker, current_position)
                    
                    # Simülasyon:
                    cash_balance += profit_after_fee
                    positions[ticker] = 0
                    print(f"🔴 {timeframe} ({w_hmm:.2f} HMM): SELL {current_position:.2f} {ticker} @ {price:.2f}. Kasa: {cash_balance:.2f}")
                else:
                    print(f"🟡 {timeframe} ({w_hmm:.2f} HMM): HOLD - Satılacak pozisyon yok.")
            
            else:
                print(f"⚪ {timeframe} ({w_hmm:.2f} HMM): HOLD - Sinyal eşiği aşılmadı.")
            
        except Exception as e:
            print(f"🚨 {ticker} GENEL HATA: {e}")
            
    # Döngü sonunda genel portföy değerini yazdır
    total_value = cash_balance
    for ticker, qty in positions.items():
        try:
            price = exchange.fetch_ticker(ticker)['close']
            total_value += qty * price
        except:
            continue
            
    print(f"\n[PORTFÖY ÖZETİ] Toplam Değer: {total_value:.2f} USDT | Nakit: {cash_balance:.2f}")
    
    # Günlük strateji olduğu için günlük döngü yeterlidir.
    time.sleep(86400) # 24 saat (1 gün) bekle
