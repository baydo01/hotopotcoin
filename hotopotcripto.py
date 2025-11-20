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
st.set_page_config(page_title="Hedge Fund Manager V7.1 (Fix)", layout="wide", initial_sidebar_state="expanded")

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
        # Veriyi indir
        df = yf.download(ticker, start=start_date, progress=False)
        
        if df.empty: return None, "Veri boş döndü (API sorunu olabilir)."
        
        # MultiIndex Temizliği
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Timezone Temizliği (Çok Kritik!)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
            
        if len(df) < 200: return None, f"Yetersiz veri satırı: {len(df)}"

        # Feature Engineering
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close']
        
        df.dropna(inplace=True)
        return df, None
    except Exception as e:
        return None, f"Veri indirme hatası: {str(e)}"

# ---------------------------
# 2. PUANLAMA SİSTEMİ
# ---------------------------
def compute_advanced_score(df):
    # Günlük
    daily_up = (df['close'].pct_change() > 0).astype(int)
    score_d = daily_up.rolling(5).mean()
    
    # Haftalık (Ortalama 7 gün)
    score_w = daily_up.rolling(35).mean() 
    
    # Aylık (Ortalama 30 gün)
    score_m = daily_up.rolling(150).mean()

    # Yıllık (Ortalama 365 gün)
    score_y = daily_up.rolling(365).mean()
    
    combined = pd.concat([score_d, score_w, score_m, score_y], axis=1).mean(axis=1)
    return combined.fillna(0.5)

# ---------------------------
# 3. TURNUVA MOTORU
# ---------------------------
def run_tournament_logic(df, ticker, capital, params):
    try:
        timeframes = ['daily', 'weekly', 'monthly']
        weights = [0.5, 0.7, 0.85, 0.9, 0.95]
        
        best_roi = -999
        best_result = None
        
        # Puanları önden hesapla
        raw_scores = compute_advanced_score(df)

        for tf in timeframes:
            # Resample
            if tf == 'daily':
                sub_df = df.copy()
            elif tf == 'weekly':
                sub_df = df.resample('W-FRI').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
                sub_df['log_ret'] = np.log(sub_df['close'] / sub_df['close'].shift(1))
                sub_df['range'] = (sub_df['high'] - sub_df['low']) / sub_df['close']
                sub_df.dropna(inplace=True)
            else: # monthly
                sub_df = df.resample('M').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
                sub_df['log_ret'] = np.log(sub_df['close'] / sub_df['close'].shift(1))
                sub_df['range'] = (sub_df['high'] - sub_df['low']) / sub_df['close']
                sub_df.dropna(inplace=True)
                
            if len(sub_df) < 50: continue

            # Puanları hizala
            aligned_scores = raw_scores.reindex(sub_df.index, method='ffill').fillna(0.5)
            
            # HMM Eğitimi
            X = sub_df[['log_ret', 'range']].values
            scaler = StandardScaler()
            
            try:
                model = GaussianHMM(n_components=params['n_states'], covariance_type="full", n_iter=100, random_state=42)
                X_s = scaler.fit_transform(X)
                model.fit(X_s)
                states = model.predict(X_s)
                sub_df['state'] = states
            except: continue

            # Rejim Tespiti
            state_means = sub_df.groupby('state')['log_ret'].mean()
            bull_state = state_means.idxmax()
            bear_state = state_means.idxmin()
            
            # Ağırlık Testleri (Vektörize)
            for w in weights:
                hmm_sig = np.where(sub_df['state'] == bull_state, 1.0, np.where(sub_df['state'] == bear_state, 0.0, 0.2))
                score_sig = aligned_scores.values
                
                final_sig = (hmm_sig * w) + (score_sig * (1-w))
                
                # Pozisyon: >0.6 AL, <0.4 SAT
                positions = np.where(final_sig > 0.6, 1.0, np.where(final_sig < 0.4, 0.0, 0.0))
                
                # Backtest
                strat_ret = pd.Series(positions).shift(1).fillna(0) * sub_df['close'].pct_change().fillna(0)
                cum_ret = (1 + strat_ret).cumprod()
                
                final_roi = cum_ret.iloc[-1] - 1
                
                if final_roi > best_roi:
                    best_roi = final_roi
                    last_pos = positions[-1]
                    
                    action = "AL" if last_pos == 1 else "SAT"
                    if last_pos == 0 and 0.4 <= final_sig[-1] <= 0.6: action = "BEKLE"
                        
                    regime = "BOĞA" if states[-1] == bull_state else ("AYI" if states[-1] == bear_state else "YATAY")
                    
                    best_result = {
                        'Coin': ticker,
                        'Fiyat': sub_df['close'].iloc[-1],
                        'Karar': action,
                        'Zaman': tf,
                        'Ağırlık': f"%{int(w*100)} HMM",
                        'Rejim': regime,
                        'Skor': f"{aligned_scores.iloc[-1]:.2f}",
                        'ROI': final_roi * 100,
                        'Bakiye': (1+final_roi) * capital
                    }

        # HODL Hesabı
        try:
            start_p = df['close'].iloc[0]
            end_p = df['close'].iloc[-1]
            hodl_val = (capital / start_p) * end_p
        except: hodl_val = capital

        return best_result, hodl_val, None

    except Exception as e:
        return None, None, f"Strateji Hatası: {str(e)}"

# ---------------------------
# 4. ARAYÜZ VE SESSION STATE
# ---------------------------
st.title("🏆 Hedge Fund Manager: V7.1 (Fix)")
st.markdown("### HMM + Puanlama | Hata Ayıklama Modu")

with st.sidebar:
    st.header("Ayarlar")
    tickers = st.multiselect("Coinler", ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD"], default=["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD"])
    capital = st.number_input("Kasa ($)", 10000)
    
    analyze_btn = st.button("ANALİZİ BAŞLAT 🚀")
    reset_btn = st.button("Hafızayı Temizle 🗑️")

if reset_btn:
    st.cache_data.clear()
    if 'results' in st.session_state: del st.session_state['results']
    st.success("Hafıza temizlendi!")

if analyze_btn:
    if not tickers:
        st.error("Coin seçin.")
    else:
        per_coin = capital / len(tickers)
        results_data = []
        total_bal = 0
        total_hodl = 0
        
        bar = st.progress(0)
        status = st.empty()
        
        params = {'n_states': 3}
        
        for i, t in enumerate(tickers):
            status.text(f"{t} analiz ediliyor...")
            # Veri Çek
            df, err_data = get_data_cached(t, "2018-01-01")
            
            if df is not None:
                # Strateji Çalıştır
                best_res, hodl_val, err_strat = run_tournament_logic(df, t, per_coin, params)
                
                if best_res:
                    results_data.append(best_res)
                    total_bal += best_res['Bakiye']
                    total_hodl += hodl_val
                else:
                    # HATA VARSA GÖSTER
                    st.error(f"{t} Analiz Hatası: {err_strat}")
            else:
                # VERİ HATASI VARSA GÖSTER
                st.error(f"{t} Veri Hatası: {err_data}")
            
            bar.progress((i+1)/len(tickers))
        
        status.empty()
        
        # Sonuçları Kaydet
        st.session_state['results'] = {
            'data': results_data,
            'total_bal': total_bal,
            'total_hodl': total_hodl
        }

# Sonuçları Göster
if 'results' in st.session_state:
    res = st.session_state['results']
    data = res['data']
    
    if data:
        start_cap = capital
        end_cap = res['total_bal']
        roi_total = ((end_cap - start_cap) / start_cap) * 100 if start_cap > 0 else 0
        alpha = end_cap - res['total_hodl']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Şampiyon Portföy", f"${end_cap:,.0f}", f"%{roi_total:.1f}")
        c2.metric("HODL Değeri", f"${res['total_hodl']:,.0f}")
        c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
        
        st.subheader("🏆 Turnuva Sonuçları")
        df_res = pd.DataFrame(data)
        
        def color_decision(val):
            if val == "AL": return 'background-color: #00c853; color: white'
            if val == "SAT": return 'background-color: #d50000; color: white'
            return 'background-color: #ffd600; color: black'

        cols = ['Coin', 'Fiyat', 'Karar', 'Zaman', 'Ağırlık', 'Rejim', 'Skor', 'ROI']
        st.dataframe(df_res[cols].style.applymap(color_decision, subset=['Karar']).format({
            "Fiyat": "${:,.2f}", "ROI": "%{:.1f}"
        }))
    else:
        st.warning("Sonuç listesi boş. Lütfen yukarıdaki kırmızı hataları kontrol edin.")
