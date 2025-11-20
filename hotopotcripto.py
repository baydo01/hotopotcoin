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
st.set_page_config(page_title="Hedge Fund Manager V3 (Scoring)", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2962FF;
        color: white;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- YARDIMCI FONKSİYONLAR ---

def calculate_momentum_score(df):
    """
    Kullanıcının istediği 'Geçmiş Artış/Azalış' Puanlaması (0 veya 1)
    """
    # 1. Kısa Vade (Son 5 Gün)
    # Fiyat 5 gün öncesinden yüksekse 1, değilse 0
    score_5d = (df['close'] > df['close'].shift(5)).astype(int)
    
    # 2. Orta Vade (Son 5 Hafta ~ 35 Gün)
    score_5w = (df['close'] > df['close'].shift(35)).astype(int)
    
    # 3. Uzun Vade (Son 5 Ay ~ 150 Gün)
    score_5m = (df['close'] > df['close'].shift(150)).astype(int)
    
    # Toplam Puan (0 ile 3 arası)
    total_score = score_5d + score_5w + score_5m
    
    return total_score, score_5d, score_5w, score_5m

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 1. VERİ ÇEKME (ÖNBELLEKLİ) ---
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
            
        if len(df) < 200: return None # 5 aylık veri için en az 200 gün lazım
        
        # --- Feature Engineering ---
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close']
        df['sma_fast'] = df['close'].rolling(window=50).mean()
        df['rsi'] = calculate_rsi(df['close'], 14)
        
        # --- YENİ: MOMENTUM PUANLAMA SİSTEMİ ---
        df['total_score'], df['score_5d'], df['score_5w'], df['score_5m'] = calculate_momentum_score(df)
        
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# --- 2. STRATEJİ MOTORU (HMM + PUANLAMA) ---
def run_scoring_strategy(df, params, alloc_capital):
    try:
        n_states = params['n_states']
        
        # --- HAFTALIK HMM (Genel Rejim İçin) ---
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
        
        # Günlük Veriye Eşle
        df_weekly['week_start'] = df_weekly.index.to_period('W').start_time
        df_merged = pd.merge_asof(df.sort_index(), df_weekly[['state']].sort_index(), left_index=True, right_index=True, direction='backward')
        
        # --- GÜNLÜK İŞLEM DÖNGÜSÜ ---
        cash = alloc_capital
        coin_amt = 0
        portfolio = []
        decision_history = []
        
        commission = params['commission']
        max_alloc = params['max_alloc']
        
        for idx, row in df_merged.iterrows():
            price = row['close']
            state = row['state']
            score = row['total_score'] # 0, 1, 2 veya 3
            rsi = row['rsi']
            
            is_hmm_bull = (state == bull_state)
            
            target_pct = 0.0
            action_text = "BEKLE"
            
            # --- PUANLI KARAR MEKANİZMASI ---
            
            # 1. Süper Trend (Puan 3/3 + HMM Boğa)
            if score == 3 and is_hmm_bull:
                target_pct = max_alloc
                action_text = "GÜÇLÜ AL (3/3 Puan)"
                
            # 2. Güçlü Yükseliş (Puan 2/3 veya 3/3 ama HMM Ayı)
            elif score >= 2:
                target_pct = max_alloc * 0.7
                action_text = "AL (Momentum Yüksek)"
                
            # 3. Zayıf/Yatay (Puan 1/3)
            elif score == 1:
                if is_hmm_bull and rsi < 50: # Destek atılabilir
                    target_pct = max_alloc * 0.3
                    action_text = "TUT/EKLE (1/3 Puan)"
                else:
                    target_pct = 0.0
                    action_text = "NAKİT (Zayıf)"
            
            # 4. Çöküş (Puan 0/3)
            else:
                target_pct = 0.0
                action_text = "SAT/KAÇ (0/3 Puan)"

            # Trade İşlemi
            current_val = cash + (coin_amt * price)
            if current_val <= 0: portfolio.append(0); continue
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
            
            portfolio.append(cash + (coin_amt * price))
            
            decision_history.append({
                "Tarih": idx, "Fiyat": price, "Puan": f"{int(score)}/3", 
                "HMM": "BOĞA" if is_hmm_bull else "AYI", "Karar": action_text,
                "Detay": f"5G:{int(row['score_5d'])} 5H:{int(row['score_5w'])} 5A:{int(row['score_5m'])}"
            })
            
        portfolio_series = pd.Series(portfolio, index=df_merged.index)
        history_df = pd.DataFrame(decision_history).set_index("Tarih")
        
        last = decision_history[-1]
        signal_data = {
            "Fiyat": last["Fiyat"], "Puan": last["Puan"], 
            "Öneri": last["Karar"], "Detay": last["Detay"]
        }
        return portfolio_series, signal_data, history_df

    except Exception:
        return None, None, None

# --- 3. ARAYÜZ ---
st.title("Pro Hedge Fund: Puanlama Modeli (V3)")
st.markdown("### 📊 HMM + Momentum Puanı (5 Gün / 5 Hafta / 5 Ay)")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", 
                             ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD", "LINK-USD", "SHIB-USD"], 
                             default=default_tickers)
    initial_capital = st.number_input("Kasa ($)", 10000)

if st.button("TÜM COİNLERİ ANALİZ ET 🚀"):
    if not tickers:
        st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        portfolio_df = pd.DataFrame()
        hodl_df = pd.DataFrame()
        signal_list = []
        all_histories = {}
        
        bar = st.progress(0)
        status = st.empty()
        
        params = {'n_states': 3, 'commission': 0.001, 'max_alloc': 1.0}
        
        for i, ticker in enumerate(tickers):
            status.text(f"Hesaplanıyor: {ticker} (Puanlar Çıkarılıyor...)")
            df = get_data_cached(ticker, "2020-01-01")
            
            if df is not None:
                res, sig_data, history_df = run_scoring_strategy(df, params, capital_per_coin)
                
                if res is not None:
                    portfolio_df[ticker] = res
                    start_p = df.loc[res.index[0], 'close']
                    hodl_df[ticker] = (capital_per_coin / start_p) * df.loc[res.index, 'close']
                    
                    if sig_data:
                        sig_data['Coin'] = ticker
                        signal_list.append(sig_data)
                        all_histories[ticker] = history_df
            
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
            c1.metric("V3 Model Bakiyesi", f"${final_bal:,.0f}", f"%{roi:.1f}")
            c2.metric("HODL Değeri", f"${total_hodl.iloc[-1]:,.0f}")
            c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
            
            # --- ANA TABLO: PUANLAMA ÖZETİ ---
            st.markdown("### 🏆 COIN PUAN TABLOSU")
            st.info("Puan 3/3 ise çok güçlü yükseliş trendidir. 0/3 ise güçlü düşüştür.")
            
            if signal_list:
                s_df = pd.DataFrame(signal_list)
                
                def highlight_score(val):
                    if '3/3' in str(val): return 'background-color: #00c853; color: white; font-weight: bold'
                    if '2/3' in str(val): return 'background-color: #b2ff59; color: black; font-weight: bold'
                    if '0/3' in str(val): return 'background-color: #d50000; color: white; font-weight: bold'
                    return ''
                
                cols = ['Coin', 'Fiyat', 'Puan', 'Öneri', 'Detay']
                st.dataframe(s_df[cols].style.applymap(highlight_score, subset=['Puan']).format({"Fiyat": "${:,.2f}"}))
            
            # DETAYLI GEÇMİŞ
            st.markdown("---")
            st.subheader("📜 Geçmiş Analiz")
            sel = st.selectbox("Detayını Göster:", list(all_histories.keys()))
            if sel:
                st.dataframe(all_histories[sel].tail(15).sort_index(ascending=False).style.format({"Fiyat": "${:,.2f}"}))
                
            st.line_chart(pd.concat([total_port.rename("V3 Model"), total_hodl.rename("HODL")], axis=1))
        else:
            st.error("Veri alınamadı.")
