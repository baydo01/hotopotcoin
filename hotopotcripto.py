import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager V5 (Tournament)", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #6200EA; /* Mor Buton */
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
    if len(df) < 366:
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

# --- 1. VERİ ÇEKME ---
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
        if len(df) < 730: return None
        df.dropna(inplace=True)
        return df
    except:
        return None

# --- 2. STRATEJİ MOTORU ---
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
            df = df_raw.copy() if tf_code=='D' else df_raw.resample(tf_code).agg(
                {k: 'last' if k=='close' else ('first' if k=='open' else 'max' if k=='high' else 'min' if k=='low' else 'sum') 
                 for k in df_raw.columns if k in ['open','high','low','close','volume']}).dropna()
            if len(df) < 200: continue
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            if len(df) < 50: continue
            X = df[['log_ret','range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X_s)
                df['state'] = model.predict(X_s)
            except:
                continue
            state_stats = df.groupby('state')['log_ret'].mean()
            bull_state, bear_state = state_stats.idxmax(), state_stats.idxmin()
            for w_hmm in weight_scenarios:
                w_score = 1.0 - w_hmm
                cash, coin_amt, temp_portfolio, temp_history = alloc_capital, 0, [], {}
                for idx, row in df.iterrows():
                    price, state, score = row['close'], row['state'], row['custom_score']
                    hmm_signal = 1 if state==bull_state else (-1 if state==bear_state else 0)
                    score_signal = 1 if score>=3 else (-1 if score<=-3 else 0)
                    weighted_decision = w_hmm*hmm_signal + w_score*score_signal
                    target_pct, action_text = 0.0, "BEKLE"
                    if weighted_decision>0.25: target_pct, action_text = 1.0, "AL"
                    elif weighted_decision<-0.25: target_pct, action_text = 0.0, "SAT"
                    current_val = cash + (coin_amt*price)
                    if current_val<=0: temp_portfolio.append(0); continue
                    current_pct = (coin_amt*price)/current_val
                    if abs(target_pct-current_pct)>0.05:
                        diff_usd = (target_pct-current_pct)*current_val
                        fee = abs(diff_usd)*commission
                        if diff_usd>0 and cash>=diff_usd:
                            coin_amt += (diff_usd-fee)/price; cash -= diff_usd
                        elif diff_usd<0 and (coin_amt*price)>=abs(diff_usd):
                            coin_amt -= abs(diff_usd)/price; cash += abs(diff_usd)-fee
                    temp_portfolio.append(cash + coin_amt*price)
                    if idx==df.index[-1]:
                        regime_label = "BOĞA" if hmm_signal==1 else ("AYI" if hmm_signal==-1 else "YATAY")
                        temp_history = {"Fiyat":price,"HMM":regime_label,"Puan":int(score),
                                        "Öneri":action_text,"Zaman":tf_name,
                                        "Ağırlık":f"%{int(w_hmm*100)} HMM / %{int(w_score*100)} Puan"}
                if temp_portfolio and (temp_portfolio[-1]-alloc_capital)/alloc_capital>best_roi:
                    best_roi=(temp_portfolio[-1]-alloc_capital)/alloc_capital
                    best_portfolio=pd.Series(temp_portfolio,index=df.index)
                    best_config=temp_history
        return best_portfolio, best_config
    except:
        return None, None

# --- 3. ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Timeframe Tournament (V5)")
st.markdown("### ⚔️ Günlük vs Haftalık vs Aylık | En İyi Strateji Seçiliyor...")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Kasa ($)", 10000)
    st.info("Sistem her coin için Günlük, Haftalık ve Aylık verileri ayrı ayrı test eder. Ayrıca 5 farklı ağırlık senaryosunu dener. En çok kazandıran kombinasyonu uygular.")

if st.button("BÜYÜK TURNUVAYI BAŞLAT 🚀"):
    if not tickers:
        st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        results_list, total_balance, total_hodl_balance = [], 0, 0
        bar, status = st.progress(0), st.empty()
        params = {'n_states':3,'commission':0.001}

        # --- ORİJİNAL 2018 BAŞLANGIÇLI TURNUVA ---
        for i, ticker in enumerate(tickers):
            status.text(f"Turnuva Oynanıyor: {ticker}...")
            df = get_data_cached(ticker,"2018-01-01")
            if df is not None:
                res_series, best_conf = run_multi_timeframe_tournament(df, params, capital_per_coin)
                if res_series is not None:
                    final_val=res_series.iloc[-1]
                    total_balance+=final_val
                    start_price, end_price=df['close'].iloc[0], df['close'].iloc[-1]
                    hodl_val=(capital_per_coin/start_price)*end_price
                    total_hodl_balance+=hodl_val
                    if best_conf:
                        best_conf.update({'Coin':ticker,'Bakiye':final_val,
                                          'ROI':((final_val-capital_per_coin)/capital_per_coin)*100})
                        results_list.append(best_conf)
            bar.progress((i+1)/len(tickers))
        status.empty()

        if results_list:
            roi_total = ((total_balance-initial_capital)/initial_capital)*100
            alpha = total_balance - total_hodl_balance
            c1,c2,c3=st.columns(3)
            c1.metric("Turnuva Şampiyonu Bakiye",f"${total_balance:,.0f}",f"%{roi_total:.1f}")
            c2.metric("HODL Değeri",f"${total_hodl_balance:,.0f}")
            c3.metric("Alpha (Fark)",f"${alpha:,.0f}",delta_color="normal" if alpha>0 else "inverse")
            st.markdown("### 🏆 ŞAMPİYONLAR LİGİ VE KARARLAR")
            st.info("Her coin için en iyi çalışan 'Zaman Dilimi' ve 'Strateji Ağırlığı' aşağıdadır.")
            df_res=pd.DataFrame(results_list)
            def highlight_decision(val):
                if val=="AL": return 'background-color:#00c853;color:white;font-weight:bold'
                if "SAT" in val: return 'background-color:#d50000;color:white;font-weight:bold'
                return 'background-color:#ffd600;color:black'
            cols=['Coin','Fiyat','Öneri','Zaman','Ağırlık','HMM','Puan','ROI']
            st.dataframe(df_res[cols].style.applymap(highlight_decision,subset=['Öneri']).format({"Fiyat":"${:,.2f}","ROI":"%{:.1f}"}))
        else:
            st.error("Veri alınamadı veya hesaplanamadı.")

        # --- 4. SENARYO ANALİZİ: 2023 ve 2024 ---
        with st.expander("📊 2023 ve 2024 Başlangıç Senaryoları ile Sonuçlar"):
            start_years=["2023-01-01","2024-01-01"]
            for start in start_years:
                st.markdown(f"### Başlangıç Tarihi: {start}")
                scenario_balance, scenario_hodl, scenario_list=0,0,[]
                for ticker in tickers:
                    df_scn=get_data_cached(ticker,start)
                    if df_scn is not None:
                        res_series, best_conf=run_multi_timeframe_tournament(df_scn,params,capital_per_coin)
                        if res_series is not None:
                            final_val=res_series.iloc[-1]
                            scenario_balance+=final_val
                            start_price, end_price=df_scn['close'].iloc[0], df_scn['close'].iloc[-1]
                            hodl_val=(capital_per_coin/start_price)*end_price
                            scenario_hodl+=hodl_val
                            if best_conf:
                                best_conf.update({'Coin':ticker,'Bakiye':final_val,
                                                  'ROI':((final_val-capital_per_coin)/capital_per_coin)*100})
                                scenario_list.append(best_conf)
                if scenario_list:
                    roi_total=((scenario_balance-initial_capital)/initial_capital)*100
                    alpha=scenario_balance-scenario_hodl
                    c1,c2,c3=st.columns(3)
                    c1.metric("Bakiye",f"${scenario_balance:,.0f}",f"%{roi_total:.1f}")
                    c2.metric("HODL",f"${scenario_hodl:,.0f}")
                    c3.metric("Alpha",f"${alpha:,.0f}",delta_color="normal" if alpha>0 else "inverse")
                    df_scn_res=pd.DataFrame(scenario_list)
                    st.dataframe(df_scn_res[cols].style.applymap(highlight_decision,subset=['Öneri']).format({"Fiyat":"${:,.2f}","ROI":"%{:.1f}"}))
                else:
                    st.warning(f"{start} başlangıç tarihi için veri alınamadı veya hesaplanamadı.")
