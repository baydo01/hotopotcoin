import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Weekly Hedge Bot", layout="wide")

# --- PUAN SİNYALİ ---
def calculate_custom_score(df):
    if len(df) < 5: return pd.Series(0, index=df.index)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(21), 1, -1)
    vol = df['close'].pct_change().rolling(5).std()
    s3 = np.where(vol < vol.shift(5), 1, -1)
    s4 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1) if 'volume' in df.columns else 0
    s5 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    return s1 + s2 + s3 + s4 + s5

# --- VERİ ÇEKME ---
@st.cache_data(ttl=3600)
def get_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False)
    if df.empty: return None
    df.columns = [c.lower() for c in df.columns]
    if 'close' not in df.columns and 'adj close' in df.columns:
        df['close'] = df['adj close']
    df.dropna(inplace=True)
    return df

# --- HAFTALIK OPTİMİZASYON ---
def optimize_weekly_weights(df, params, alloc_capital):
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    if len(df) < 21: return (0.7,0.3)
    
    # HMM model
    X = df[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GaussianHMM(n_components=params['n_states'], covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_s)
    df['state'] = model.predict(X_s)
    state_stats = df.groupby('state')['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    # Test ağırlıkları
    weight_scenarios = [0.5,0.7,0.9]
    best_roi = -999; best_w=None
    test_data = df.iloc[-21:]  # Son 3 hafta
    for w_hmm in weight_scenarios:
        w_score = 1.0 - w_hmm
        cash = alloc_capital; coin_amt = 0
        for idx,row in test_data.iterrows():
            hmm_signal = 1 if row['state']==bull_state else (-1 if row['state']==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            price=row['close']
            if decision>0.25: coin_amt = cash/price; cash=0
            elif decision<-0.25: cash = coin_amt*price; coin_amt=0
        final_val = cash + coin_amt*test_data['close'].iloc[-1]
        roi = (final_val - alloc_capital)/alloc_capital
        if roi>best_roi: best_roi=roi; best_w=(w_hmm,w_score)
    return best_w if best_w else (0.7,0.3)

# --- HAFTALIK STRATEJİ ---
def run_weekly_strategy(df, params, alloc_capital):
    df = df.resample('W').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    
    w_hmm, w_score = optimize_weekly_weights(df, params, alloc_capital)
    
    X = df[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GaussianHMM(n_components=params['n_states'], covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_s)
    df['state'] = model.predict(X_s)
    
    state_stats = df.groupby('state')['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    cash = alloc_capital; coin_amt=0; portfolio=[]
    history=[]
    for idx,row in df.iterrows():
        price=row['close']
        hmm_signal = 1 if row['state']==bull_state else (-1 if row['state']==bear_state else 0)
        score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
        decision = w_hmm*hmm_signal + w_score*score_signal
        target_pct = 1.0 if decision>0.25 else (0.0 if decision<-0.25 else (coin_amt*price)/(cash+coin_amt*price))
        current_val = cash + coin_amt*price
        current_pct = (coin_amt*price)/current_val if current_val>0 else 0
        diff_usd = (target_pct - current_pct)*current_val
        fee = abs(diff_usd)*params['commission']
        if diff_usd>0 and cash>=diff_usd: coin_amt += (diff_usd-fee)/price; cash-=diff_usd
        elif diff_usd<0 and coin_amt*price>=abs(diff_usd): coin_amt -= abs(diff_usd)/price; cash += abs(diff_usd)-fee
        portfolio.append(cash+coin_amt*price)
        history.append({"Tarih":idx,"Fiyat":price,"AL/SAT":("AL" if decision>0.25 else ("SAT" if decision<-0.25 else "BEKLE")),"HMM":hmm_signal,"Puan":score_signal})
    return pd.Series(portfolio,index=df.index),history

# --- ARAYÜZ ---
st.title("📈 Weekly Hedge Bot")
st.sidebar.header("Ayarlar")
tickers=st.sidebar.multiselect("Coin Seç",["BTC-USD","ETH-USD","SOL-USD"],default=["BTC-USD"])
capital=st.sidebar.number_input("Başlangıç Kasa ($)",10000)
params={'n_states':3,'commission':0.001}

if st.button("Stratejiyi Çalıştır"):
    all_results=[]
    for ticker in tickers:
        df=get_data(ticker)
        if df is not None:
            portfolio,hist = run_weekly_strategy(df,params,capital/len(tickers))
            st.subheader(f"{ticker} - Portföy Değeri")
            st.line_chart(portfolio)
            st.write(pd.DataFrame(hist).tail(10))
