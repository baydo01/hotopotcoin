import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager: Multi-Year V5", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
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
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# --- ÖZEL PUAN HESABI ---
def calculate_custom_score(df):
    if len(df) < 5:  # artık minimum kısa
        return pd.Series(0, index=df.index)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1) if 'volume' in df.columns else 0
    s7 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    return s1 + s2 + s3 + s4 + s5 + s6 + s7

# --- VERİ ÇEKME ---
@st.cache_data(ttl=21600)
def get_data_cached(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        df.dropna(inplace=True)
        return df
    except:
        return None

# --- TURNUVA ---
def run_multi_timeframe_tournament(df_raw, params, alloc_capital):
    try:
        n_states = params['n_states']
        commission = params['commission']
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
        best_roi = -999
        best_portfolio = []
        best_config = {}
        
        for tf_name, tf_code in timeframes.items():
            if tf_code == 'D':
                df = df_raw.copy()
            else:
                agg_dict = {'close':'last', 'high':'max', 'low':'min'}
                if 'open' in df_raw.columns: agg_dict['open']='first'
                if 'volume' in df_raw.columns: agg_dict['volume']='sum'
                df = df_raw.resample(tf_code).agg(agg_dict).dropna()
            if len(df) < 5: continue
            
            df['log_ret'] = np.log(df['close']/df['close'].shift(1))
            df['range'] = (df['high']-df['low'])/df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            if len(df) < 5: continue
            
            X = df[['log_ret','range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X_s)
                states = model.predict(X_s)
                df['state'] = states
            except:
                continue
            
            state_stats = df.groupby('state')['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.idxmin()
            
            for w_hmm in weight_scenarios:
                w_score = 1.0 - w_hmm
                cash = alloc_capital
                coin_amt = 0
                temp_portfolio = []
                temp_history = {}
                
                for idx,row in df.iterrows():
                    price=row['close']
                    state=row['state']
                    score=row['custom_score']
                    hmm_signal = 1 if state==bull_state else (-1 if state==bear_state else 0)
                    score_signal = 1 if score>=3 else (-1 if score<=-3 else 0)
                    weighted_decision = (w_hmm*hmm_signal)+(w_score*score_signal)
                    
                    target_pct=0.0
                    action_text="BEKLE"
                    if weighted_decision>0.25: target_pct=1.0; action_text="AL"
                    elif weighted_decision<-0.25: target_pct=0.0; action_text="SAT"
                    
                    current_val=cash + coin_amt*price
                    if current_val<=0: temp_portfolio.append(0); continue
                    current_pct=(coin_amt*price)/current_val
                    if abs(target_pct-current_pct)>0.05:
                        diff_usd=(target_pct-current_pct)*current_val
                        fee=abs(diff_usd)*commission
                        if diff_usd>0:
                            if cash>=diff_usd: coin_amt+=(diff_usd-fee)/price; cash-=diff_usd
                        else:
                            sell_usd=abs(diff_usd)
                            if (coin_amt*price)>=sell_usd: coin_amt-=sell_usd/price; cash+=(sell_usd-fee)
                    
                    val=cash+coin_amt*price
                    temp_portfolio.append(val)
                    
                    if idx==df.index[-1]:
                        regime_label="BOĞA" if hmm_signal==1 else ("AYI" if hmm_signal==-1 else "YATAY")
                        temp_history={"Fiyat":price,"HMM":regime_label,"Puan":int(score),
                                      "Öneri":action_text,"Zaman":tf_name,
                                      "Ağırlık":f"%{int(w_hmm*100)} HMM / %{int(w_score*100)} Puan"}
                
                if len(temp_portfolio)>0:
                    final_bal=temp_portfolio[-1]
                    roi=(final_bal-alloc_capital)/alloc_capital
                    if roi>best_roi:
                        best_roi=roi
                        best_portfolio=pd.Series(temp_portfolio,index=df.index)
                        best_config=temp_history
        return best_portfolio,best_config
    except:
        return None,None

# --- ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Multi-Year V5")
st.markdown("### ⚔️ Tüm Başlangıç Yılları Tek Tablo")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
    tickers=st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital=st.number_input("Kasa ($)",10000)
    years=[2018,2019,2020,2022,2023,2024,2025]
    selected_years=st.multiselect("Başlangıç Yılları",years,default=years)
    st.info("Sistem her coin için seçilen yılları tek tek test eder ve en iyi stratejiyi hesaplar.")

if st.button("BÜYÜK TURNUVAYI BAŞLAT 🚀"):
    if not tickers: st.error("Coin seçmelisin.")
    else:
        results_list=[]
        bar=st.progress(0)
        status=st.empty()
        params={'n_states':3,'commission':0.001}
        
        total_tasks=len(tickers)*len(selected_years)
        task_count=0
        
        for year in selected_years:
            start_date=f"{year}-01-01"
            for ticker in tickers:
                task_count+=1
                status.text(f"{ticker} - {start_date} test ediliyor ({task_count}/{total_tasks})")
                df=get_data_cached(ticker,start_date)
                if df is not None:
                    res_series,best_conf=run_multi_timeframe_tournament(df,params,initial_capital/len(tickers))
                    if res_series is not None:
                        final_val=res_series.iloc[-1]
                        start_price=df['close'].iloc[0]
                        end_price=df['close'].iloc[-1]
                        hodl_val=(initial_capital/len(tickers)/start_price)*end_price
                        alpha=final_val-hodl_val
                        if best_conf:
                            best_conf.update({"Coin":ticker,"Başlangıç Tarihi":start_date,
                                              "Bakiye":final_val,"HODL":hodl_val,
                                              "Alpha":alpha,"ROI":((final_val-initial_capital/len(tickers))/(initial_capital/len(tickers))*100)})
                            results_list.append(best_conf)
                bar.progress(task_count/total_tasks)
        status.empty()
        
        if results_list:
            df_res=pd.DataFrame(results_list)
            cols=['Coin','Başlangıç Tarihi','Fiyat','Öneri','Zaman','Ağırlık','HMM','Puan','Bakiye','HODL','Alpha','ROI']
            st.dataframe(df_res[cols].sort_values(by=['Başlangıç Tarihi','Coin']),height=600)
        else:
            st.error("Veri alınamadı veya hesaplanamadı.")
