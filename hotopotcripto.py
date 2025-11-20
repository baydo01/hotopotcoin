import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager V6 (Ultimate)", layout="wide", initial_sidebar_state="expanded")

# --- CSS ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #6200EA;
        color: white;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 1. VERİ ÇEKME (CACHE)
# ---------------------------
@st.cache_data(ttl=21600)
def get_data_cached(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None, f"{ticker} için veri yok."
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
            
        if len(df) < 200: return None, "Yetersiz veri."

        # Temel Featurelar
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close']
        df['vol_20'] = df['log_ret'].rolling(window=20).std()
        df['sma_fast'] = df['close'].rolling(window=50).mean()

        df.dropna(inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

# ---------------------------
# 2. GELİŞMİŞ PUANLAMA (SCORING)
# ---------------------------
def compute_advanced_score(df):
    """
    Günlük, Haftalık, Aylık ve Yıllık Yükseliş Trendlerine Göre 0-1 Arası Skor Üretir.
    """
    # Günlük (5 Günlük Trend)
    daily_up = (df['close'].pct_change() > 0).astype(int)
    score_d = daily_up.rolling(5).mean()
    
    # Haftalık Trend (Resample ve Reindex ile)
    try:
        weekly = df['close'].resample('W-FRI').last()
        weekly_up = (weekly.pct_change() > 0).astype(int)
        score_w = weekly_up.rolling(5).mean().reindex(df.index, method='ffill')
    except: score_w = 0.5
    
    # Aylık Trend
    try:
        monthly = df['close'].resample('M').last()
        monthly_up = (monthly.pct_change() > 0).astype(int)
        score_m = monthly_up.rolling(5).mean().reindex(df.index, method='ffill')
    except: score_m = 0.5

    # Yıllık Trend
    try:
        yearly = df['close'].resample('Y').last()
        yearly_up = (yearly.pct_change() > 0).astype(int)
        score_y = yearly_up.rolling(5).mean().reindex(df.index, method='ffill')
    except: score_y = 0.5
    
    # Ortalama Skor (0 ile 1 arası)
    combined = pd.concat([score_d, score_w, score_m, score_y], axis=1).mean(axis=1)
    return combined.fillna(0.5) # Boşları nötrle

# ---------------------------
# 3. STRATEJİ MOTORU (ENSEMBLE)
# ---------------------------
def run_strategy(df_orig, params, alloc_capital, timeframe='daily', hmm_weight=0.7):
    try:
        # Zaman Dilimine Göre Resample
        if timeframe == 'daily':
            df = df_orig.copy()
        elif timeframe == 'weekly':
            df = df_orig.resample('W-FRI').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            # Featureları yeniden hesapla
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close']
            df.dropna(inplace=True)
        elif timeframe == 'monthly':
            df = df_orig.resample('M').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close']
            df.dropna(inplace=True)
            
        # Puanları Hesapla (Orijinal Günlük Veriden)
        raw_scores = compute_advanced_score(df_orig)
        # Timeframe'e uyırla (Son geçerli skoru al)
        aligned_scores = raw_scores.reindex(df.index, method='ffill')

        # HMM Hazırlık
        if len(df) < params['train_window'] + 10: return None, None, "Yetersiz Veri"
        
        X = df[['log_ret', 'range']].values
        scaler = StandardScaler()
        states_pred = np.full(len(df), -1)
        
        # Walk-Forward HMM
        for i in range(params['train_window'], len(df), params['retrain_every']):
            start_idx = max(0, i - params['train_window'])
            X_train = X[start_idx:i]
            if len(X_train) < 20: continue
            
            try:
                model = GaussianHMM(n_components=params['n_states'], covariance_type="full", n_iter=100, random_state=42)
                X_train_s = scaler.fit_transform(X_train)
                model.fit(X_train_s)
                
                end_idx = min(i + params['retrain_every'], len(df))
                X_pred = scaler.transform(X[i:end_idx])
                states_pred[i:end_idx] = model.predict(X_pred)
            except: continue
            
        df['state'] = states_pred
        df = df[df['state'] != -1]
        
        if df.empty: return None, None, "State Bulunamadı"

        # Boğa/Ayı Tespiti
        state_means = df.groupby('state')['log_ret'].mean()
        bull_state = state_means.idxmax()
        bear_state = state_means.idxmin()
        
        # Simülasyon
        cash = alloc_capital
        coin_amt = 0
        portfolio = []
        history = []
        
        for idx, row in df.iterrows():
            price = row['close']
            state = row['state']
            score = aligned_scores.get(idx, 0.5)
            
            # HMM Sinyali (0 ile 1 arası hedef)
            if state == bull_state: hmm_target = 1.0
            elif state == bear_state: hmm_target = 0.0
            else: hmm_target = 0.2 # Yatay piyasada az risk
            
            # Skor Sinyali (Zaten 0-1 arası)
            score_target = score
            
            # Ensemble Karar
            final_target = (hmm_target * hmm_weight) + (score_target * (1 - hmm_weight))
            
            # Eşik Değerler (Thresholds)
            if final_target > 0.6: position = 1.0 # AL
            elif final_target < 0.4: position = 0.0 # SAT
            else: position = 0.0 # Kararsızsa Nakitte Kal (Güvenli Mod)
            
            # İşlem Uygula
            current_val = cash + (coin_amt * price)
            if current_val <= 0: portfolio.append(0); continue
            current_pct = (coin_amt * price) / current_val
            
            if abs(position - current_pct) > 0.05: # %5 tolerans
                diff = (position - current_pct) * current_val
                fee = abs(diff) * params['commission']
                if diff > 0 and cash >= diff: # AL
                    coin_amt += (diff - fee) / price
                    cash -= diff
                elif diff < 0: # SAT
                    sell_val = abs(diff)
                    if (coin_amt * price) >= sell_val:
                        coin_amt -= sell_val / price
                        cash += (sell_val - fee)
            
            portfolio.append(cash + (coin_amt * price))
            
            if idx == df.index[-1]: # Son gün kaydı
                action = "AL" if position > 0.6 else ("SAT" if position < 0.4 else "BEKLE")
                regime = "BOĞA" if state == bull_state else ("AYI" if state == bear_state else "YATAY")
                history = {
                    "Fiyat": price, "Karar": action, "Zaman": timeframe,
                    "Ağırlık": f"%{int(hmm_weight*100)} HMM",
                    "Rejim": regime, "Skor": f"{score:.2f}"
                }

        return pd.Series(portfolio, index=df.index), history, None

    except Exception as e:
        return None, None, str(e)

# ---------------------------
# 4. TURNUVA MODU (Tüm Kombinasyonları Dene)
# ---------------------------
def run_tournament(df, ticker, capital, params):
    timeframes = ['daily', 'weekly', 'monthly']
    weights = [0.5, 0.7, 0.85, 0.9, 0.95] # HMM Ağırlıkları
    
    best_roi = -999
    best_result = None
    best_port = None
    
    for tf in timeframes:
        for w in weights:
            port, hist, err = run_strategy(df, params, capital, timeframe=tf, hmm_weight=w)
            if port is not None and not port.empty:
                final_val = port.iloc[-1]
                roi = (final_val - capital) / capital
                
                if roi > best_roi:
                    best_roi = roi
                    best_port = port
                    best_result = hist
                    best_result['ROI'] = roi * 100
                    best_result['Bakiye'] = final_val
                    best_result['Coin'] = ticker

    # HODL Hesabı
    try:
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        hodl_val = (capital / start_price) * end_price
    except: hodl_val = 0
    
    return best_result, best_port, hodl_val

# ---------------------------
# 5. ARAYÜZ
# ---------------------------
st.title("🏆 Hedge Fund Manager: V6 Ultimate")
st.markdown("### HMM + Trend Puanlama | Çoklu Zaman Turnuvası")

with st.sidebar:
    st.header("Ayarlar")
    tickers = st.multiselect("Coinler", ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"], default=["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD"])
    capital = st.number_input("Kasa ($)", 10000)
    st.divider()
    st.caption("Her coin için Günlük/Haftalık/Aylık ve %50-%95 ağırlık kombinasyonları test edilip şampiyon seçilir.")

if st.button("ANALİZİ BAŞLAT 🚀"):
    if not tickers: st.error("Coin Seçin")
    else:
        per_coin = capital / len(tickers)
        results = []
        total_bal = 0
        total_hodl = 0
        
        bar = st.progress(0)
        
        params = {'train_window': 180, 'retrain_every': 7, 'n_states': 3, 'commission': 0.001}
        
        for i, t in enumerate(tickers):
            df, err = get_data_cached(t, "2018-01-01")
            if df is not None:
                best_res, best_port, hodl_val = run_tournament(df, t, per_coin, params)
                if best_res:
                    results.append(best_res)
                    total_bal += best_res['Bakiye']
                    total_hodl += hodl_val
            bar.progress((i+1)/len(tickers))
            
        if results:
            # ÖZET
            total_roi = ((total_bal - capital) / capital) * 100
            alpha = total_bal - total_hodl
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Şampiyon Portföy", f"${total_bal:,.0f}", f"%{total_roi:.1f}")
            c2.metric("HODL Değeri", f"${total_hodl:,.0f}")
            c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
            
            # TABLO
            df_res = pd.DataFrame(results)
            st.subheader("🏆 Turnuva Sonuçları")
            
            def color_decision(val):
                if val == "AL": return 'background-color: #00c853; color: white'
                if val == "SAT": return 'background-color: #d50000; color: white'
                return 'background-color: #ffd600; color: black'

            cols = ['Coin', 'Fiyat', 'Karar', 'Zaman', 'Ağırlık', 'Rejim', 'Skor', 'ROI']
            st.dataframe(df_res[cols].style.applymap(color_decision, subset=['Karar']).format({
                "Fiyat": "${:,.2f}", "ROI": "%{:.1f}"
            }))
        else:
            st.error("Veri alınamadı.")
