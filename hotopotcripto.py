import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager V6.4 (Stabil)", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #008080; /* Teal Buton */
        color: white;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# --- YARDIMCI FONKSİYONLAR ---

def calculate_custom_score(df):
    if len(df) < 366: return pd.Series(0, index=df.index)
    
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    
    if 'volume' in df.columns: s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1)
    else: s6 = 0
    
    if 'open' in df.columns: s7 = np.where(df['close'] > df['open'], 1, -1)
    else: s7 = 0
        
    total_score = s1 + s2 + s3 + s4 + s5 + s6 + s7
    return total_score

# --- 1. VERİ ÇEKME (KURALI YUMUŞATILDI) ---
@st.cache_data(ttl=21600)
def get_data_cached(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        
        df.dropna(inplace=True)
        # 730 günlük kısıt kalktı. Yalnızca 100 günlük temel kontrol kalır.
        return df
    except:
        return None

# --- 2. STRATEJİ MOTORU (TURNUVA) ---
def run_multi_timeframe_tournament(df_raw, params, alloc_capital):
    try:
        n_states, commission = params['n_states'], params['commission']
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
        best_roi = -999
        best_portfolio, best_config = None, None

        for tf_name, tf_code in timeframes.items():
            if tf_code == 'D': df = df_raw.copy()
            else:
                agg = {'close': 'last', 'high': 'max', 'low': 'min'}
                if 'open' in df_raw.columns: agg['open']='first'
                if 'volume' in df_raw.columns: agg['volume']='sum'
                df = df_raw.resample(tf_code).agg(agg).dropna()
            
            # Veri Filtreleri
            if len(df) < 200: continue
            
            df['log_ret'] = np.log(df['close']/df['close'].shift(1))
            df['range'] = (df['high']-df['low'])/df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            if len(df)<50: continue

            # HMM Eğitimi
            X = df[['log_ret','range']].values; X_s = StandardScaler().fit_transform(X)
            try:
                model = GaussianHMM(n_components=n_states,covariance_type="full",n_iter=100,random_state=42)
                model.fit(X_s); df['state'] = model.predict(X_s)
            except: continue

            state_stats = df.groupby('state')['log_ret'].mean()
            bull_state, bear_state = state_stats.idxmax(), state_stats.idxmin()

            for w_hmm in weight_scenarios:
                w_score = 1.0 - w_hmm
                cash, coin_amt, temp_portfolio = alloc_capital, 0, []

                for idx,row in df.iterrows():
                    price, state, score = row['close'], row['state'], row['custom_score']
                    hmm_signal = 1 if state==bull_state else (-1 if state==bear_state else 0)
                    score_signal = 1 if score>=3 else (-1 if score<=-3 else 0)
                    decision = w_hmm*hmm_signal + w_score*score_signal
                    target_pct = 1.0 if decision>0.25 else (0.0 if decision<-0.25 else 0.0)
                    action_text = 'AL' if decision>0.25 else ('SAT' if decision<-0.25 else 'BEKLE')
                    current_val = cash+coin_amt*price
                    if current_val<=0: temp_portfolio.append(0); continue
                    current_pct = coin_amt*price/current_val
                    
                    if abs(target_pct-current_pct)>0.05:
                        diff_usd = (target_pct-current_pct)*current_val; fee = abs(diff_usd)*commission
                        if diff_usd>0 and cash>=diff_usd: coin_amt += (diff_usd-fee)/price; cash-=diff_usd
                        elif diff_usd<0 and coin_amt*price>=abs(diff_usd): coin_amt-=abs(diff_usd)/price; cash+=abs(diff_usd)-fee
                    temp_portfolio.append(cash+coin_amt*price)
                    
                    if idx==df.index[-1]:
                        regime_label = 'BOĞA' if hmm_signal==1 else ('AYI' if hmm_signal==-1 else 'YATAY')
                        temp_history = {'Fiyat':price,'HMM':regime_label,'Puan':int(score),'Öneri':action_text,'Zaman':tf_name,'Ağırlık':f'%{int(w_hmm*100)} HMM / %{int(w_score*100)} Puan'}

                if temp_portfolio and (temp_portfolio[-1]-alloc_capital)/alloc_capital>best_roi:
                    best_roi = (temp_portfolio[-1]-alloc_capital)/alloc_capital
                    best_portfolio = pd.Series(temp_portfolio,index=df.index)
                    best_config = temp_history
        return best_portfolio, best_config
    except: return None, None

# --- ARAYÜZ ---
st.title("🏆 Hedge Fund Manager V6.4 (Start Date Tournament)")
st.markdown("### ⏱️ Hangi Tarihten Başlamak Kârlı? (2018 vs 2019 vs 2024)")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Kasa ($)",10000)
    st.info("Her coin için 2018, 2019 ve 2024 başlangıçları otomatik test edilir.")

if st.button("TÜM TURNUVALARI BAŞLAT 🚀"):
    if not tickers: st.error("Coin seçmelisin.")
    else:
        bar = st.progress(0); status = st.empty()
        start_dates = {'Uzun (2018)':'2018-01-01', 'Orta (2019)':'2019-01-01', 'Kısa (2024)':'2024-01-01'}
        params={'n_states':3,'commission':0.001}
        results_list = []
        
        for i,ticker in enumerate(tickers):
            status.text(f"Turnuva Oynanıyor: {ticker}...")
            best_roi_for_ticker = -999
            best_config_for_ticker = None
            df_final_data = None
            
            for sname, sdate in start_dates.items():
                df = get_data_cached(ticker,sdate)
                if df is not None:
                    df_final_data = df
                    res_series,best_conf = run_multi_timeframe_tournament(df,params,initial_capital/len(tickers))
                    
                    if res_series is not None:
                        roi = (res_series.iloc[-1]-(initial_capital/len(tickers)))/(initial_capital/len(tickers))
                        
                        if roi > best_roi_for_ticker:
                             best_roi_for_ticker = roi
                             best_config_for_ticker = best_conf
                             best_config_for_ticker.update({'Başlangıç': sname, 'Coin': ticker, 'Bakiye': res_series.iloc[-1]})
                             
            if best_config_for_ticker and df_final_data is not None:
                 start_price = df_final_data['close'].iloc[0]
                 end_price = df_final_data['close'].iloc[-1]
                 hodl_val = (initial_capital/len(tickers)/start_price)*end_price
                 best_config_for_ticker.update({'HODL': hodl_val})
                 results_list.append(best_config_for_ticker)
                 
            bar.progress((i+1)/len(tickers))
        status.empty()

        if results_list:
            df_res=pd.DataFrame(results_list)
            
            total_balance = df_res['Bakiye'].sum()
            total_hodl_balance = df_res['HODL'].sum()
            roi_total = ((total_balance - initial_capital)/initial_capital)*100
            alpha = total_balance - total_hodl_balance
            
            # Üst Metrikler
            c1,c2,c3 = st.columns(3)
            c1.metric("Şampiyon Bakiye",f"${total_balance:,.0f}",f"%{roi_total:.1f}")
            c2.metric("HODL Değeri",f"${total_hodl_balance:,.0f}")
            c3.metric("Alpha (Fark)",f"${alpha:,.0f}",delta_color="normal" if alpha>0 else "inverse")
            
            st.markdown("### 🏆 En Kârlı Kombinasyonlar ve Başlangıç Noktaları")
            
            def highlight_decision(val):
                if 'AL'==val: return 'background-color:#00c853;color:white;font-weight:bold'
                if 'SAT'==val: return 'background-color:#d50000;color:white;font-weight:bold'
                return 'background-color:#ffd600;color:black'

            cols=['Coin','Başlangıç','Fiyat','Öneri','Zaman','Ağırlık','HMM','Puan','ROI','HODL']
            st.dataframe(df_res[cols].style.applymap(highlight_decision,subset=['Öneri']).format({'Fiyat':'${:,.2f}','ROI':'{:.1f}%','HODL':'${:,.2f}'}))
        else: st.error("Veri alınamadı veya hesaplanamadı. Lütfen coin seçiminizi ve başlangıç tarihlerini kontrol edin.")
