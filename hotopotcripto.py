import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager V7.0 (Pure Score)", layout="wide", initial_sidebar_state="expanded")

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
    """
    7'li Puanlama Sistemi (-7 ile +7 arası)
    """
    if len(df) < 366: return pd.Series(0, index=df.index), pd.DataFrame()
    
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
    
    # Detaylı puanları da döndür (UI için)
    scores_df = pd.DataFrame({'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6, 's7': s7}, index=df.index)
    return total_score, scores_df

# --- 1. VERİ ÇEKME ---
@st.cache_data(ttl=21600)
def get_data_cached(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        
        df.dropna(inplace=True)
        if len(df) < 100: return None
        return df
    except Exception: return None

# --- 2. STRATEJİ MOTORU (SADECE PUAN) ---
def run_pure_score_tournament(df_raw, params, alloc_capital):
    try:
        commission = params['commission']
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        score_thresholds = [3, 4, 5] # +3, +4, +5 puan eşiklerini test et
        
        best_roi = -np.inf
        best_portfolio, best_config = None, None

        for tf_name, tf_code in timeframes.items():
            if tf_code == 'D': df = df_raw.copy()
            else:
                agg = {'close': 'last', 'high': 'max', 'low': 'min'}
                if 'open' in df_raw.columns: agg['open']='first'
                if 'volume' in df_raw.columns: agg['volume']='sum'
                df = df_raw.resample(tf_code).agg(agg).dropna()
            
            if len(df) < 200: continue
            
            df['custom_score'], score_details_df = calculate_custom_score(df_raw)
            # Timeframe'e hizala
            df['custom_score'] = df['custom_score'].reindex(df.index, method='ffill')

            # Hız Testleri (Farklı eşik değerleri)
            for threshold in score_thresholds:
                
                cash, coin_amt, temp_portfolio = alloc_capital, 0, []

                for idx,row in df.iterrows():
                    price = row['close']
                    score = row['custom_score']
                    
                    # Puan bazlı karar: Puan eşiği geçerse AL
                    if score >= threshold: target_pct = 1.0; action_text = "AL"
                    elif score <= -threshold: target_pct = 0.0; action_text = "SAT"
                    else: target_pct = 0.0; action_text = "BEKLE"
                    
                    # İşlem Uygula
                    current_val = cash + coin_amt * price
                    if current_val<=0: temp_portfolio.append(0); continue
                    current_pct = coin_amt * price / current_val
                    
                    if abs(target_pct - current_pct) > 0.05:
                        diff_usd = (target_pct - current_pct) * current_val; fee = abs(diff_usd) * commission
                        if diff_usd>0 and cash>=diff_usd: coin_amt += (diff_usd-fee)/price; cash-=diff_usd
                        elif diff_usd<0 and coin_amt*price>=abs(diff_usd): coin_amt-=abs(diff_usd)/price; cash+=abs(diff_usd)-fee
                    
                    temp_portfolio.append(cash + coin_amt * price)
                    
                    if idx == df.index[-1]:
                        temp_history = {'Fiyat':price,'Puan':int(score),'Öneri':action_text,'Zaman':tf_name,'Eşik':threshold}

                if temp_portfolio and (temp_portfolio[-1]-alloc_capital)/alloc_capital > best_roi:
                    best_roi = (temp_portfolio[-1]-alloc_capital)/alloc_capital
                    best_portfolio = pd.Series(temp_portfolio,index=df.index)
                    best_config = temp_history
        return best_portfolio, best_config
    except Exception as e: return None, None

# --- ARAYÜZ ---
st.title("🏆 Hedge Fund Manager V7.0 (Pure Score)")
st.markdown("### 🥇 HMM Çıkarıldı | %100 Stabil Puanlama Sistemi")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Kasa ($)",10000)
    st.info("Sistem HMM kullanmaz. Sadece 7'li Momentum Puanına göre işlem yapar.")

if st.button("SAF ANALİZİ BAŞLAT 🚀"):
    if not tickers: st.error("Coin seçmelisin.")
    else:
        bar = st.progress(0); status = st.empty()
        params={'commission':0.001}
        results_list = []
        
        for i,ticker in enumerate(tickers):
            status.text(f"Puan Turnuvası: {ticker}...")
            df = get_data_cached(ticker,"2018-01-01")
            
            if df is not None:
                res_series,best_conf = run_pure_score_tournament(df,params,initial_capital/len(tickers))
                if res_series is not None:
                    final_val = res_series.iloc[-1]
                    hodl_val = (initial_capital/len(tickers)/df['close'].iloc[0])*df['close'].iloc[-1]
                    
                    best_conf.update({'Coin':ticker,'Bakiye':final_val,'ROI':(final_val - initial_capital/len(tickers))/(initial_capital/len(tickers))*100,'HODL':hodl_val})
                    results_list.append(best_conf)
                 
            bar.progress((i+1)/len(tickers))
        status.empty()

        if results_list:
            df_res=pd.DataFrame(results_list)
            total_balance = df_res['Bakiye'].sum(); total_hodl_balance = df_res['HODL'].sum()
            roi_total = ((total_balance - initial_capital)/initial_capital)*100
            alpha = total_balance - total_hodl_balance
            
            c1,c2,c3 = st.columns(3)
            c1.metric("Şampiyon Bakiye",f"${total_balance:,.0f}",f"%{roi_total:.1f}")
            c2.metric("HODL Değeri",f"${total_hodl_balance:,.0f}")
            c3.metric("Alpha (Fark)",f"${alpha:,.0f}",delta_color="normal" if alpha>0 else "inverse")
            
            st.markdown("### 🏆 En Kârlı Kombinasyonlar ve Başlangıç Noktaları")
            
            def highlight_decision(val):
                if 'AL'==val: return 'background-color:#00c853;color:white;font-weight:bold'
                if 'SAT'==val: return 'background-color:#d50000;color:white;font-weight:bold'
                return 'background-color:#ffd600;color:black'

            cols=['Coin','Fiyat','Öneri','Zaman','Eşik','Puan','ROI','HODL']
            st.dataframe(df_res[cols].style.applymap(highlight_decision,subset=['Öneri']).format({'Fiyat':'${:,.2f}','ROI':'{:.1f}%','HODL':'${:,.2f}'}))
        else: st.error("Analiz tamamlandı, ancak hiçbir strateji kârlı sonuç üretmedi.")
