import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
import datetime
import time

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund Bot Simulator", layout="wide")

# --- CSS ---
st.markdown("""
<style>
.stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #6200EA; color: white; font-weight: bold; }
div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --- Normalizasyon ---
def normalize_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if 'close' not in df.columns and 'adj close' in df.columns:
        df['close'] = df['adj close']
    if 'open' not in df.columns: df['open'] = df['close']
    if 'high' not in df.columns: df['high'] = df['close']
    if 'low' not in df.columns: df['low'] = df['close']
    if 'volume' not in df.columns: df['volume'] = 0
    df.dropna(inplace=True)
    return df

# --- Custom Score ---
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

# --- YFinance Veri Çekme ---
@st.cache_data(ttl=21600)
def get_data_cached(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        df = normalize_df(df)
        return df
    except:
        return None

# --- Dinamik Ağırlık Optimizasyonu ---
def optimize_dynamic_weights(df, alloc_capital, n_states=3, validation_days=21):
    df = df.copy()
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    if len(df) < validation_days + 5: return (0.7, 0.3)
    
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
    
    weight_candidates = np.linspace(0.1, 0.9, 9)
    best_roi = -np.inf
    best_w = (0.5, 0.5)
    
    for w_hmm in weight_candidates:
        w_score = 1-w_hmm
        cash = alloc_capital
        coin_amt = 0
        for idx,row in test_df.iterrows():
            X_test = scaler.transform([[row['log_ret'], row['range']]])
            hmm_signal = 1 if model.predict(X_test)[0]==bull_state else (-1 if model.predict(X_test)[0]==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            price = row['close']
            if decision>0.25: coin_amt = cash/price; cash=0
            elif decision<-0.25: cash = coin_amt*price; coin_amt=0
        final_val = cash + coin_amt*test_df['close'].iloc[-1]
        roi = (final_val - alloc_capital)/alloc_capital
        if roi>best_roi: best_roi=roi; best_w=(w_hmm,w_score)
    
    return best_w

# --- Canlı Bot Simülasyonu ---
def run_live_simulation(df, alloc_capital, n_states=3):
    w_hmm, w_score = optimize_dynamic_weights(df, alloc_capital, n_states)
    
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    
    X = df[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100, random_state=42)
    model.fit(X_s)
    df['state'] = model.predict(X_s)
    
    state_stats = df.groupby('state')['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    cash = alloc_capital
    coin_amt = 0
    history=[]
    for idx,row in df.iterrows():
        price=row['close']
        hmm_signal = 1 if row['state']==bull_state else (-1 if row['state']==bear_state else 0)
        score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
        decision = w_hmm*hmm_signal + w_score*score_signal
        
        if decision>0.25 and cash>0:
            coin_amt = cash/price
            cash=0
        elif decision<-0.25 and coin_amt>0:
            cash = coin_amt*price
            coin_amt=0
        total_val = cash + coin_amt*price
        history.append(total_val)
        
        # Son gün bilgisi
        if idx==df.index[-1]:
            regime = "BOĞA" if hmm_signal==1 else ("AYI" if hmm_signal==-1 else "YATAY")
            last_info={"Fiyat":price,"Regime":regime,"Öneri":"AL" if decision>0.25 else ("SAT" if decision<-0.25 else "BEKLE"),"Ağırlık":f"%{int(w_hmm*100)} HMM / %{int(w_score*100)} Puan"}
    
    return pd.Series(history, index=df.index), last_info

# --- ARAYÜZ ---
st.title("🤖 CCXT Live Bot Simulator (Simülasyon Modu)")
with st.sidebar:
    st.header("Ayarlar")
    default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD"]
    tickers = st.multiselect("Seçilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Toplam Kasa ($)",10000)

if st.button("Simülasyonu Başlat 🚀"):
    if not tickers: st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        results=[]
        total_balance=0
        bar=st.progress(0)
        status=st.empty()
        
        for i,ticker in enumerate(tickers):
            status.text(f"Analiz Ediliyor: {ticker}...")
            df=get_data_cached(ticker,"2018-01-01")
            if df is not None:
                portfolio_series,last_day_info=run_live_simulation(df, capital_per_coin)
                total_balance+=portfolio_series.iloc[-1]
                last_day_info.update({"Coin":ticker,"Bakiye":portfolio_series.iloc[-1],"ROI":((portfolio_series.iloc[-1]-capital_per_coin)/capital_per_coin)*100})
                results.append(last_day_info)
            bar.progress((i+1)/len(tickers))
        
        status.empty()
        if results:
            roi_total=((total_balance-initial_capital)/initial_capital)*100
            st.metric("Toplam Bakiye", f"${total_balance:,.0f}", f"%{roi_total:.1f}")
            
            df_res=pd.DataFrame(results)
            def highlight_decision(val):
                if val=="AL": return 'background-color:#00c853;color:white;font-weight:bold'
                if val=="SAT": return 'background-color:#d50000;color:white;font-weight:bold'
                return 'background-color:#ffd600;color:black'
            
            st.dataframe(df_res[['Coin','Fiyat','Öneri','Ağırlık','Regime','ROI']].style.applymap(highlight_decision, subset=['Öneri']).format({"Fiyat":"${:,.2f}","ROI":"%{:.1f}"}))
        else:
            st.error("Veri alınamadı veya hesaplanamadı.")
