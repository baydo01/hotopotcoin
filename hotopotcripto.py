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
    """
    5'li Puanlama Sistemi (-7 ile +7 arası)
    """
    # Veri yeterli mi kontrolü
    if len(df) < 366:
        return pd.Series(0, index=df.index)

    # 1. Kısa Vade (Son 5 Mum)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    
    # 2. Orta Vade (Son 35 Mum)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    
    # 3. Uzun Vade (Son 150 Mum)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    
    # 4. Makro Vade (Son 365 Mum)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    
    # 5. Volatilite Yönü
    # Not: Oynaklık artıyorsa genelde düşüş trendi sertleşir -> Risk (-1)
    # Oynaklık azalıyorsa stabilizasyon -> Güven (+1)
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    
    # 6. Hacim Trendi
    # Eğer sütunlarda volume yoksa hata vermesin
    if 'volume' in df.columns:
        s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1)
    else:
        s6 = 0
    
    # 7. Mum Yapısı
    if 'open' in df.columns:
        s7 = np.where(df['close'] > df['open'], 1, -1)
    else:
        s7 = 0
    
    # Toplam Skor
    total_score = s1 + s2 + s3 + s4 + s5 + s6 + s7
    return total_score

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
            
        # En az 2 yıllık veri olsun
        if len(df) < 730: return None 
        
        # Ham veriyi temizle
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# --- 2. STRATEJİ MOTORU (ÇOKLU ZAMAN DİLİMİ + TURNUVA) ---
def run_multi_timeframe_tournament(df_raw, params, alloc_capital):
    try:
        n_states = params['n_states']
        commission = params['commission']
        
        # Test edilecek Zaman Dilimleri
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        # Ağırlık Senaryoları (HMM Ağırlığı)
        weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
        
        best_roi = -999
        best_portfolio = []
        best_config = {} 
        
        # --- TURNUVA DÖNGÜSÜ ---
        for tf_name, tf_code in timeframes.items():
            
            # 1. Veriyi İlgili Zaman Dilimine Çevir (Resample)
            if tf_code == 'D':
                df = df_raw.copy()
            else:
                agg_dict = {'close': 'last', 'high': 'max', 'low': 'min'}
                if 'open' in df_raw.columns: agg_dict['open'] = 'first'
                if 'volume' in df_raw.columns: agg_dict['volume'] = 'sum'
                
                df = df_raw.resample(tf_code).agg(agg_dict).dropna()
            
            if len(df) < 200: continue
            
            # Feature Engineering
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            
            if len(df) < 50: continue

            # HMM Eğitimi
            X = df[['log_ret', 'range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X_s)
                states = model.predict(X_s)
                df['state'] = states
            except:
                continue 
            
            # Boğa/Ayı Tespiti
            state_stats = df.groupby('state')['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.idxmin()
            
            # Ağırlık Testleri
            for w_hmm in weight_scenarios:
                w_score = 1.0 - w_hmm
                
                cash = alloc_capital
                coin_amt = 0
                temp_portfolio = []
                temp_history = {}
                
                # Backtest
                for idx, row in df.iterrows():
                    price = row['close']
                    state = row['state']
                    score = row['custom_score']
                    
                    # Sinyaller
                    hmm_signal = 0
                    if state == bull_state: hmm_signal = 1
                    elif state == bear_state: hmm_signal = -1
                    
                    score_signal = 0
                    if score >= 3: score_signal = 1
                    elif score <= -3: score_signal = -1
                    
                    # Karar (Ağırlıklı)
                    weighted_decision = (w_hmm * hmm_signal) + (w_score * score_signal)
                    
                    target_pct = 0.0
                    action_text = "BEKLE"
                    
                    if weighted_decision > 0.25: 
                        target_pct = 1.0; action_text = "AL"
                    elif weighted_decision < -0.25:
                        target_pct = 0.0; action_text = "SAT"
                    
                    # İşlem
                    current_val = cash + (coin_amt * price)
                    if current_val <= 0: temp_portfolio.append(0); continue
                    current_pct = (coin_amt * price) / current_val
                    
                    if abs(target_pct - current_pct) > 0.05:
                        diff_usd = (target_pct - current_pct) * current_val
                        fee = abs(diff_usd) * commission
                        if diff_usd > 0:
                            if cash >= diff_usd:
                                coin_amt += (diff_usd - fee) / price
                                cash -= diff_usd
                        else:
                            sell_usd = abs(diff_usd)
                            if (coin_amt * price) >= sell_usd:
                                coin_amt -= sell_usd / price
                                cash += (sell_usd - fee)
                    
                    val = cash + (coin_amt * price)
                    temp_portfolio.append(val)
                    
                    # Son gün verisi (HATA DÜZELTME BURADA YAPILDI: Key ismi 'Öneri' oldu)
                    if idx == df.index[-1]:
                        regime_label = "BOĞA" if hmm_signal==1 else ("AYI" if hmm_signal==-1 else "YATAY")
                        temp_history = {
                            "Fiyat": price, "HMM": regime_label, "Puan": int(score), 
                            "Öneri": action_text, "Zaman": tf_name, 
                            "Ağırlık": f"%{int(w_hmm*100)} HMM / %{int(w_score*100)} Puan"
                        }
                
                if len(temp_portfolio) > 0:
                    final_bal = temp_portfolio[-1]
                    roi = (final_bal - alloc_capital) / alloc_capital
                    
                    if roi > best_roi:
                        best_roi = roi
                        best_portfolio = pd.Series(temp_portfolio, index=df.index)
                        best_config = temp_history

        return best_portfolio, best_config

    except Exception as e:
        return None, None

# --- 3. ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Timeframe Tournament (V5)")
st.markdown("### ⚔️ Günlük vs Haftalık vs Aylık | En İyi Strateji Seçiliyor...")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Kasa ($)", 10000)
    st.info("Sistem her coin için Günlük, Haftalık ve Aylık verileri ayrı ayrı test eder. Ayrıca 5 farklı ağırlık senaryosunu dener. En çok kazandıran kombinasyonu uygular.")

if st.button("BÜYÜK TURNUVAYI BAŞLAT 🚀"):
    if not tickers:
        st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        
        results_list = []
        total_balance = 0
        total_hodl_balance = 0
        
        bar = st.progress(0)
        status = st.empty()
        
        params = {'n_states': 3, 'commission': 0.001}
        
        for i, ticker in enumerate(tickers):
            status.text(f"Turnuva Oynanıyor: {ticker}...")
            df = get_data_cached(ticker, "2018-01-01")
            
            if df is not None:
                res_series, best_conf = run_multi_timeframe_tournament(df, params, capital_per_coin)
                
                if res_series is not None:
                    final_val = res_series.iloc[-1]
                    total_balance += final_val
                    
                    # HODL Değeri
                    start_price = df['close'].iloc[0]
                    end_price = df['close'].iloc[-1]
                    hodl_val = (capital_per_coin / start_price) * end_price
                    total_hodl_balance += hodl_val
                    
                    if best_conf:
                        best_conf['Coin'] = ticker
                        best_conf['Bakiye'] = final_val
                        best_conf['ROI'] = ((final_val - capital_per_coin) / capital_per_coin) * 100
                        results_list.append(best_conf)
            
            bar.progress((i+1)/len(tickers))
        
        status.empty()

        if results_list:
            # GENEL METRİKLER
            roi_total = ((total_balance - initial_capital) / initial_capital) * 100
            alpha = total_balance - total_hodl_balance
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Turnuva Şampiyonu Bakiye", f"${total_balance:,.0f}", f"%{roi_total:.1f}")
            c2.metric("HODL Değeri", f"${total_hodl_balance:,.0f}")
            c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
            
            # SONUÇ TABLOSU
            st.markdown("### 🏆 ŞAMPİYONLAR LİGİ VE KARARLAR")
            st.info("Her coin için en iyi çalışan 'Zaman Dilimi' ve 'Strateji Ağırlığı' aşağıdadır.")
            
            df_res = pd.DataFrame(results_list)
            
            def highlight_decision(val):
                val_str = str(val)
                if 'AL' == val_str: return 'background-color: #00c853; color: white; font-weight: bold'
                if 'SAT' in val_str: return 'background-color: #d50000; color: white; font-weight: bold'
                return 'background-color: #ffd600; color: black'
            
            # Tabloyu Düzenle
            cols = ['Coin', 'Fiyat', 'Öneri', 'Zaman', 'Ağırlık', 'HMM', 'Puan', 'ROI']
            
            # HATA ALINAN KISIM DÜZELTİLDİ:
            # 'Karar' key'i yerine artık 'Öneri' kullanıyoruz, sütunlar eşleşti.
            st.dataframe(df_res[cols].style.applymap(highlight_decision, subset=['Öneri']).format({
                "Fiyat": "${:,.2f}",
                "ROI": "%{:.1f}"
            }))
            
        else:
            st.error("Veri alınamadı veya hesaplanamadı.")
