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
st.set_page_config(page_title="Hedge Fund Manager V4 (Tournament)", layout="wide", initial_sidebar_state="expanded")

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
    Senin istediğin 5'li Puanlama Sistemi (-7 ile +7 arası)
    """
    # 1. Kısa Vade (Son 5 Gün)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    
    # 2. Orta Vade (Son 5 Hafta ~ 35 Gün)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    
    # 3. Uzun Vade (Son 5 Ay ~ 150 Gün)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    
    # 4. Makro Vade (Son 1 Yıl - 5 Yıl verisi yoksa 1 Yıl kullanır)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    
    # 5. Volatilite Yönü (Son 5 gün volatilite düşüyorsa iyidir +1, artıyorsa risk -1)
    # Volatilite genelde düşüşte artar (Kriptoda)
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    
    # 6. Hacim Trendi (Hacim artıyorsa +1)
    s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1)
    
    # 7. Mum Yapısı (Kapanış > Açılış ise +1)
    s7 = np.where(df['close'] > df['open'], 1, -1)
    
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
            
        if len(df) < 370: return None # 1 Yıllık veri şart
        
        # Feature Engineering (HMM İçin)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close']
        
        # Feature Engineering (Puanlama İçin)
        df['custom_score'] = calculate_custom_score(df)
        
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# --- 2. STRATEJİ MOTORU (TURNUVA MODU) ---
def run_tournament_strategy(df, params, alloc_capital):
    try:
        n_states = params['n_states']
        
        # --- ADIM 1: HMM ANALİZİ (HAFTALIK) ---
        df_weekly = df.resample('W').agg({'close': 'last', 'high': 'max', 'low': 'min'}).dropna()
        df_weekly['log_ret'] = np.log(df_weekly['close'] / df_weekly['close'].shift(1))
        df_weekly['range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
        df_weekly.dropna(inplace=True)
        
        if len(df_weekly) < 50: return None, None, None

        # HMM Eğitimi
        X_w = df_weekly[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_w_s = scaler.fit_transform(X_w)
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X_w_s)
        states_weekly = model.predict(X_w_s)
        df_weekly['state'] = states_weekly
        
        # Boğa/Ayı Tespiti
        state_stats = df_weekly.groupby('state')['log_ret'].mean()
        bull_state = state_stats.idxmax()
        bear_state = state_stats.idxmin()
        
        # Günlüğe Eşle
        df_weekly['week_start'] = df_weekly.index.to_period('W').start_time
        df_merged = pd.merge_asof(df.sort_index(), df_weekly[['state']].sort_index(), left_index=True, right_index=True, direction='backward')
        
        # --- ADIM 2: AĞIRLIKLI TESTLER (TURNUVA) ---
        # HMM Ağırlıkları: %50, %70, %85, %90, %95
        # Puan Ağırlıkları: %50, %30, %15, %10, %5
        weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
        
        best_roi = -999
        best_portfolio = []
        best_weight = 0.50
        best_history = []
        
        commission = params['commission']
        
        # Her senaryoyu test et
        for w_hmm in weight_scenarios:
            w_score = 1.0 - w_hmm
            
            cash = alloc_capital
            coin_amt = 0
            temp_portfolio = []
            temp_history = []
            
            for idx, row in df_merged.iterrows():
                price = row['close']
                state = row['state']
                score = row['custom_score'] # -7 ile +7 arası
                
                # 1. HMM Sinyali (-1, 0, +1'e çevir)
                hmm_signal = 0
                if state == bull_state: hmm_signal = 1
                elif state == bear_state: hmm_signal = -1
                else: hmm_signal = 0 # Yatay
                
                # 2. Puan Sinyali (Normalize et: -1 ile +1 arasına sıkıştır)
                # Skor -7 ile +7 arasında. Bunu basitçe -1, 0, +1 yapalım
                score_signal = 0
                if score >= 3: score_signal = 1   # Güçlü Pozitif
                elif score <= -3: score_signal = -1 # Güçlü Negatif
                else: score_signal = 0
                
                # 3. HİBRİT KARAR (Ağırlıklı Ortalama)
                # Örn: (0.7 * 1) + (0.3 * -1) = 0.4 (Hafif Al)
                weighted_decision = (w_hmm * hmm_signal) + (w_score * score_signal)
                
                # Pozisyon Belirle
                target_pct = 0.0
                action_text = "BEKLE"
                
                if weighted_decision > 0.3: # Eşik Değer (Threshold)
                    target_pct = 1.0 # Full Gir
                    action_text = "AL"
                elif weighted_decision < -0.3:
                    target_pct = 0.0 # Sat
                    action_text = "SAT"
                else:
                    # Kararsız bölge (önceki pozisyonu koru veya %50 gir)
                    # Risk almamak için nakit
                    target_pct = 0.0 
                    action_text = "NAKİT (Kararsız)"

                # Trade İşlemi
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
                
                # Log sadece son senaryo için tutulur, burada geçici tutalım
                if idx == df_merged.index[-1]: # Sadece son gün verisi lazım
                    regime_label = "BOĞA" if hmm_signal==1 else ("AYI" if hmm_signal==-1 else "YATAY")
                    temp_history.append({
                        "Fiyat": price, "HMM": regime_label, "Puan": int(score), 
                        "Karar": action_text, "FinalSkor": round(weighted_decision, 2)
                    })
            
            # Performans Ölç
            final_bal = temp_portfolio[-1]
            roi = (final_bal - alloc_capital) / alloc_capital
            
            if roi > best_roi:
                best_roi = roi
                best_portfolio = temp_portfolio
                best_weight = w_hmm
                # Son günün karar verisini al
                best_history = temp_history[0]

        # En iyi sonucu döndür
        portfolio_series = pd.Series(best_portfolio, index=df_merged.index)
        
        signal_data = {
            "Fiyat": best_history["Fiyat"],
            "HMM Durumu": best_history["HMM"],
            "Puan (7 üzerinden)": best_history["Puan"],
            "Kazanan Ağırlık": f"%{int(best_weight*100)} HMM / %{int((1-best_weight)*100)} Puan",
            "Öneri": best_history["Karar"]
        }
        
        return portfolio_series, signal_data

    except Exception as e:
        return None, None

# --- 3. ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Tournament Edition (V4)")
st.markdown("### ⚔️ 5 Farklı Strateji Yarışıyor -> Kazanan Uygulanıyor")

with st.sidebar:
    st.header("Ayarlar")
    # Varsayılan olarak hepsi seçili
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Kasa ($)", 10000)
    st.info("Sistem %50-50 ile %95-5 arasındaki tüm oranları dener, en kârlısını seçer.")

if st.button("TURNUVAYI BAŞLAT VE ANALİZ ET 🚀"):
    if not tickers:
        st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        portfolio_df = pd.DataFrame()
        hodl_df = pd.DataFrame()
        signal_list = []
        
        bar = st.progress(0)
        status = st.empty()
        
        params = {'n_states': 3, 'commission': 0.001}
        
        for i, ticker in enumerate(tickers):
            status.text(f"Turnuva Oynanıyor: {ticker}...")
            df = get_data_cached(ticker, "2020-01-01")
            
            if df is not None:
                res, sig_data = run_tournament_strategy(df, params, capital_per_coin)
                
                if res is not None:
                    portfolio_df[ticker] = res
                    start_p = df.loc[res.index[0], 'close']
                    hodl_df[ticker] = (capital_per_coin / start_p) * df.loc[res.index, 'close']
                    
                    if sig_data:
                        sig_data['Coin'] = ticker
                        signal_list.append(sig_data)
            
            bar.progress((i+1)/len(tickers))
        
        status.empty()

        if not portfolio_df.empty:
            portfolio_df.fillna(method='ffill', inplace=True).fillna(0, inplace=True)
            hodl_df.fillna(method='ffill', inplace=True).fillna(0, inplace=True)
            
            common_idx = portfolio_df.index.intersection(hodl_df.index)
            total_port = portfolio_df.loc[common_idx].sum(axis=1)
            total_hodl = hodl_df.loc[common_idx].sum(axis=1)
            
            final_bal = total_port.iloc[-1]
            roi = ((final_bal - initial_capital)/initial_capital)*100
            alpha = final_bal - total_hodl.iloc[-1]
            
            # METRİKLER
            c1, c2, c3 = st.columns(3)
            c1.metric("Şampiyon Model Bakiyesi", f"${final_bal:,.0f}", f"%{roi:.1f}")
            c2.metric("HODL Değeri", f"${total_hodl.iloc[-1]:,.0f}")
            c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
            
            # --- ANA TABLO: TURNUVA SONUÇLARI ---
            st.markdown("### 🏆 KAZANAN STRATEJİ VE BUGÜNÜN EMRİ")
            st.info("Her coin için geçmişte en çok kazandıran 'Ağırlık Oranı' otomatik seçildi.")
            
            if signal_list:
                s_df = pd.DataFrame(signal_list)
                
                def highlight_decision(val):
                    if 'AL' == str(val): return 'background-color: #00c853; color: white; font-weight: bold'
                    if 'SAT' in str(val): return 'background-color: #d50000; color: white; font-weight: bold'
                    return 'background-color: #ffd600; color: black'
                
                cols = ['Coin', 'Fiyat', 'Öneri', 'Kazanan Ağırlık', 'HMM Durumu', 'Puan (7 üzerinden)']
                st.dataframe(s_df[cols].style.applymap(highlight_decision, subset=['Öneri']).format({"Fiyat": "${:,.2f}"}))
            
            st.line_chart(pd.concat([total_port.rename("Şampiyon Bot"), total_hodl.rename("HODL")], axis=1))
        else:
            st.error("Veri alınamadı.")
