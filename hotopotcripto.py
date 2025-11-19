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
st.set_page_config(page_title="Hedge Fund Manager Pro", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. VERİ ÇEKME (ÖNBELLEKLİ & HIZLI) ---
# Bu fonksiyon veriyi bir kez çeker ve 6 saat boyunca hafızada tutar.
# Böylece sayfayı her yenilediğinde beklemek zorunda kalmazsın.
@st.cache_data(ttl=21600) 
def get_data_cached(ticker, start_date):
    try:
        # Veriyi indir
        df = yf.download(ticker, start=start_date, progress=False)
        
        # Veri boşsa veya hata varsa None döndür (Hata kalkanı)
        if df.empty: return None

        # Sütun isimlerini düzelt
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
            
        # Yetersiz veri kontrolü
        if len(df) < 100: return None
        
        # Feature Engineering
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close']
        df['vol_20'] = df['log_ret'].rolling(window=20).std()
        df['sma_fast'] = df['close'].rolling(window=50).mean()
        
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# --- 2. STRATEJİ MOTORU (GÜNLÜK KARAR) ---
def run_strategy_single(df, params, alloc_capital):
    try:
        train_window = params['train_window']
        retrain_every = params['retrain_every']
        n_states = params['n_states']
        
        # Veri kontrolü (Hata almamak için kritik)
        if df is None or len(df) < train_window + 50: 
            return None, None, None
        
        feature_cols = ['log_ret', 'range', 'vol_20']
        X = df[feature_cols].values
        states_pred = np.full(len(df), -1)
        
        scaler = StandardScaler()
        
        # HMM Modeli (Günlük Döngü)
        for i in range(train_window, len(df), retrain_every):
            start_idx = max(0, i - train_window)
            X_train = X[start_idx:i]
            if len(X_train) < 50: continue

            try:
                X_train_s = scaler.fit_transform(X_train)
                model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X_train_s)
                
                pred_end = min(i + retrain_every, len(df))
                X_pred = X[i:pred_end]
                X_pred_s = scaler.transform(X_pred)
                states_pred[i:pred_end] = model.predict(X_pred_s)
            except:
                continue
                
        df_res = df.copy()
        df_res['state'] = states_pred
        df_res = df_res[df_res['state'] != -1]
        
        if df_res.empty: return None, None, None

        # Rejimleri Tanı (Boğa/Ayı)
        state_stats = df_res.groupby('state')['log_ret'].mean()
        bull_state = state_stats.idxmax()
        bear_state = state_stats.idxmin()
        
        cash = alloc_capital
        coin_amt = 0
        portfolio = []
        decision_history = [] # Günlük log defteri
        
        commission = params['commission']
        max_alloc = params['max_alloc']
        
        # Backtest Döngüsü
        for idx, row in df_res.iterrows():
            price = row['close']
            state = row['state']
            sma_fast = row['sma_fast']
            
            is_uptrend = price > sma_fast
            is_hmm_bull = (state == bull_state)
            is_hmm_bear = (state == bear_state)
            
            target_pct = 0.0
            action_text = "BEKLE"
            
            # Strateji Mantığı
            if is_uptrend:
                if is_hmm_bear: 
                    target_pct = max_alloc * 0.8; action_text="AL (Riskli)"
                else: 
                    target_pct = max_alloc; action_text="GÜÇLÜ AL"
            else:
                if is_hmm_bull: 
                    target_pct = max_alloc * 0.2; action_text="DİP ALIMI"
                else: 
                    target_pct = 0.0; action_text="SAT/NAKİT"
                
            current_val = cash + (coin_amt * price)
            if current_val <= 0: portfolio.append(0); continue
                
            current_pct = (coin_amt * price) / current_val
            
            # İşlem Yap
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
            
            # Günlük Log Kaydı
            regime_label = "BOĞA 🐂" if is_hmm_bull else ("AYI 🐻" if is_hmm_bear else "YATAY 🦀")
            trend_label = "YÜKSELİŞ 📈" if is_uptrend else "DÜŞÜŞ 📉"
            
            decision_history.append({
                "Tarih": idx, 
                "Fiyat": price, 
                "Trend": trend_label,
                "Rejim": regime_label, 
                "Karar": action_text
            })
        
        # Sonuçları Paketle
        portfolio_series = pd.Series(portfolio, index=df_res.index)
        history_df = pd.DataFrame(decision_history).set_index("Tarih")
        
        last_rec = decision_history[-1]
        signal_data = {
            "Fiyat": last_rec["Fiyat"],
            "HMM Rejimi": last_rec["Rejim"],
            "Öneri": last_rec["Karar"],
            "Trend": last_rec["Trend"]
        }
            
        return portfolio_series, signal_data, history_df
    except Exception:
        return None, None, None

# --- 3. ARAYÜZ ---
st.title("🏦 Hedge Fund Manager (Turbo Mod ⚡)")
st.markdown("HMM Destekli Yapay Zeka Botu - Günlük Karar Destek Sistemi")

with st.sidebar:
    st.header("Ayarlar")
    # Varsayılan coin listesi
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    tickers = st.multiselect("Coinler", 
                             ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"], 
                             default=default_tickers)
    
    initial_capital = st.number_input("Kasa ($)", 10000)
    st.info("Veriler önbelleğe alınır. Sayfayı yenilediğinizde tekrar beklemezsiniz.")

# Ana Buton
if st.button("GÜNLÜK ANALİZİ BAŞLAT 🚀"):
    
    if not tickers:
        st.error("Lütfen en az bir coin seçin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        portfolio_df = pd.DataFrame()
        hodl_df = pd.DataFrame()
        signal_list = []
        all_histories = {}
        
        bar = st.progress(0)
        status = st.empty()
        
        # Strateji Parametreleri
        params = {
            'train_window': 180, 
            'retrain_every': 1,  # HER GÜN güncelle
            'n_states': 3, 
            'commission': 0.001, 
            'max_alloc': 1.0
        }
        
        for i, ticker in enumerate(tickers):
            status.text(f"Analiz ediliyor: {ticker}...")
            
            # Veriyi Cache'den hızlıca al
            df = get_data_cached(ticker, "2021-01-01")
            
            if df is not None:
                res, sig_data, history_df = run_strategy_single(df, params, capital_per_coin)
                
                if res is not None:
                    portfolio_df[ticker] = res
                    # Hodl hesapla
                    start_p = df.loc[res.index[0], 'close']
                    hodl_val = (capital_per_coin / start_p) * df.loc[res.index, 'close']
                    hodl_val = hodl_val.reindex(res.index, method='ffill')
                    hodl_df[ticker] = hodl_val
                    
                    if sig_data:
                        sig_data['Coin'] = ticker
                        signal_list.append(sig_data)
                        all_histories[ticker] = history_df
            
            bar.progress((i+1)/len(tickers))
        
        status.empty() # Yazıyı temizle

        # --- SONUÇLARI GÖSTER ---
        if not portfolio_df.empty:
            # Güvenli Birleştirme (Hata Önleyici)
            portfolio_df.fillna(method='ffill', inplace=True)
            portfolio_df.fillna(0, inplace=True)
            hodl_df.fillna(method='ffill', inplace=True)
            hodl_df.fillna(0, inplace=True)
            
            # Ortak indexi bul
            common_idx = portfolio_df.index.intersection(hodl_df.index)
            total_port = portfolio_df.loc[common_idx].sum(axis=1)
            total_hodl = hodl_df.loc[common_idx].sum(axis=1)
            
            # Metrikler
            final_bal = total_port.iloc[-1]
            roi = ((final_bal - initial_capital)/initial_capital)*100
            hodl_end = total_hodl.iloc[-1]
            alpha = final_bal - hodl_end
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Bot Bakiyesi", f"${final_bal:,.0f}", f"%{roi:.1f}")
            c2.metric("HODL Bakiyesi", f"${hodl_end:,.0f}")
            c3.metric("Bot Farkı (Alpha)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
            
            # 1. BUGÜNÜN KARAR TABLOSU
            st.markdown("---")
            st.subheader("📢 BUGÜNÜN SİNYALLERİ (Son Kapanış)")
            if signal_list:
                s_df = pd.DataFrame(signal_list)
                # Renklendirme fonksiyonu
                def color_coding(val):
                    if 'AL' in str(val): return 'background-color: #d4edda; color: green; font-weight: bold'
                    if 'SAT' in str(val): return 'background-color: #f8d7da; color: red; font-weight: bold'
                    if 'DİP' in str(val): return 'background-color: #fff3cd; color: orange; font-weight: bold'
                    return ''
                
                cols = ['Coin', 'Fiyat', 'Öneri', 'HMM Rejimi', 'Trend']
                st.dataframe(s_df[cols].style.applymap(color_coding, subset=['Öneri']).format({"Fiyat": "${:,.2f}"}))
                
            # 2. DETAYLI GÜNLÜK (SON 10 GÜN)
            st.markdown("---")
            st.subheader("📜 Detaylı Günlük İşlem Defteri")
            st.info("Botun son 10 günde fikrinin nasıl değiştiğini görmek için coin seçin:")
            
            sel_coin = st.selectbox("Coin Seçin:", list(all_histories.keys()))
            if sel_coin:
                history_view = all_histories[sel_coin].tail(10).sort_index(ascending=False)
                st.dataframe(history_view.style.format({"Fiyat": "${:,.2f}"}).applymap(
                    lambda x: 'color: green; font-weight: bold' if 'AL' in str(x) else ('color: red; font-weight: bold' if 'SAT' in str(x) else ''),
                    subset=['Karar']
                ))
                
            # 3. GRAFİK
            st.markdown("---")
            st.subheader("📈 Performans Karşılaştırması")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(total_port.index, total_port, label="Hedge Fund Bot", color="#4B0082", linewidth=2)
            ax.plot(total_hodl.index, total_hodl, label="HODL (Bekle)", color="gray", alpha=0.5, linestyle="--")
            ax.set_ylabel("Dolar ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        else:
            st.error("Veri alınamadı. Lütfen sayfayı yenileyip tekrar deneyin veya farklı coinler seçin.")
