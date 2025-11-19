import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager Pro", layout="wide", initial_sidebar_state="expanded")

# --- CSS GÖRSELLİK ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #4B0082;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- FONKSİYONLAR ---

@st.cache_data(ttl=3600) # 1 saatte bir veriyi tazele
def get_data(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        
        # Feature Engineering (Modelin Girdileri)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close']
        df['vol_20'] = df['log_ret'].rolling(window=20).std()
        df['sma_fast'] = df['close'].rolling(window=50).mean()
        
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

def run_strategy_single(df, params, alloc_capital):
    """
    Modeli çalıştırır ve hem portföyü hem de detaylı karar geçmişini döner.
    """
    train_window = params['train_window']
    retrain_every = params['retrain_every']
    n_states = params['n_states']
    
    feature_cols = ['log_ret', 'range', 'vol_20']
    X = df[feature_cols].values
    states_pred = np.full(len(df), -1)
    
    if len(df) < train_window + retrain_every: return None, None, None

    scaler = StandardScaler()
    
    # --- GÜNLÜK EĞİTİM DÖNGÜSÜ ---
    # retrain_every=1 olduğu için bu döngü her gün çalışır.
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

    # Rejim Analizi (Hangi state Boğa, hangisi Ayı?)
    state_stats = df_res.groupby('state')['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    cash = alloc_capital
    coin_amt = 0
    portfolio = []
    commission = params['commission']
    max_alloc = params['max_alloc']
    
    # Karar Geçmişini Kaydetmek İçin Liste
    decision_history = []

    for idx, row in df_res.iterrows():
        price = row['close']
        state = row['state']
        sma_fast = row['sma_fast']
        
        is_uptrend = price > sma_fast
        is_hmm_bull = (state == bull_state)
        is_hmm_bear = (state == bear_state)
        
        # Hedef Pozisyon Belirleme
        target_pct = 0.0
        action_note = "BEKLE"
        
        if is_uptrend:
            if is_hmm_bear: 
                target_pct = max_alloc * 0.8
                action_note = "AL (Riskli)"
            else: 
                target_pct = max_alloc 
                action_note = "GÜÇLÜ AL"
        else:
            if is_hmm_bull: 
                target_pct = max_alloc * 0.2 
                action_note = "DİP ALIMI"
            else: 
                target_pct = 0.0 
                action_note = "SAT/NAKİT"
            
        # Portföy Değeri
        current_val = cash + (coin_amt * price)
        if current_val <= 0: 
            portfolio.append(0); continue
            
        current_pct = (coin_amt * price) / current_val
        
        # Alım-Satım İşlemi
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
        
        # Tarihsel Log Kaydı
        regime_label = "BOĞA 🐂" if is_hmm_bull else ("AYI 🐻" if is_hmm_bear else "YATAY 🦀")
        decision_history.append({
            "Tarih": idx,
            "Fiyat": price,
            "Trend": "YÜKSELİŞ" if is_uptrend else "DÜŞÜŞ",
            "Rejim": regime_label,
            "Karar": action_note
        })
    
    # Veri Çerçevelerini Hazırla
    portfolio_series = pd.Series(portfolio, index=df_res.index)
    history_df = pd.DataFrame(decision_history).set_index("Tarih")
    
    # --- SON GÜN SİNYALİ ---
    last_rec = decision_history[-1]
    
    signal_data = {
        "Fiyat": last_rec["Fiyat"],
        "Trend": "YÜKSELİŞ 📈" if last_rec["Trend"] == "YÜKSELİŞ" else "DÜŞÜŞ 📉",
        "HMM Rejimi": last_rec["Rejim"],
        "Öneri": last_rec["Karar"],
        "Hedef Pozisyon": f"%{target_pct*100:.0f}",
        "Mantık": "Model Analizi"
    }
        
    return portfolio_series, signal_data, history_df

def calculate_short_term_projection(df, days_to_project=30):
    recent_df = df.iloc[-30:]
    daily_change = recent_df['close'].diff().mean()
    daily_std = recent_df['close'].diff().std()
    
    last_price = df.iloc[-1]['close']
    last_date = df.index[-1]
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_project + 1)]
    projections = {'Date': future_dates, 'Optimistic': [], 'Realistic': [], 'Pessimistic': []}
    
    curr_opt = last_price
    curr_real = last_price
    curr_pess = last_price
    
    for i in range(days_to_project):
        curr_opt += daily_change + (daily_std * 0.5)
        curr_real += daily_change
        curr_pess += daily_change - (daily_std * 0.5)
        
        projections['Optimistic'].append(curr_opt)
        projections['Realistic'].append(curr_real)
        projections['Pessimistic'].append(curr_pess)
        
    return pd.DataFrame(projections)

# --- ARAYÜZ KISMI ---

st.title("🏦 Hedge Fund Manager (Günlük Bot)")
st.markdown("### 🧠 Yapay Zeka Destekli Günlük Karar Sistemi")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    default_coins = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    tickers = st.multiselect("Takip Edilecek Coinler", 
                             ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LTC-USD"],
                             default=default_coins)
    
    st.divider()
    start_date = st.date_input("Veri Başlangıcı", pd.to_datetime("2021-01-01"))
    initial_capital = st.number_input("Başlangıç Kasası ($)", 10000)
    commission = st.number_input("Komisyon Oranı", 0.001, format="%.4f")
    max_alloc = st.slider("Max Pozisyon (%)", 0.1, 1.0, 1.0)
    
    st.info("ℹ️ Bu modda sistem HER GÜN verileri yeniden işler ve günlük rapor üretir.")

if len(tickers) > 0:
    
    main_tab1, main_tab2 = st.tabs(["📊 Günlük Sinyaller & Rapor", "🔮 Gelecek Simülasyonu"])
    
    with main_tab1:
        if st.button("GÜNLÜK ANALİZİ BAŞLAT 🚀", type="primary"):
            st.write("🔄 Piyasalar taranıyor, Yapay Zeka modelleri eğitiliyor...")
            
            capital_per_coin = initial_capital / len(tickers)
            portfolio_df = pd.DataFrame()
            hodl_df = pd.DataFrame()
            signal_list = []
            
            # Detaylı logları tutmak için sözlük
            all_histories = {} 
            
            progress_bar = st.progress(0)
            
            try:
                # --- KRİTİK: GÜNLÜK EĞİTİM AYARLARI ---
                params = {
                    'train_window': 180,  # Son 6 ayın hafızası
                    'retrain_every': 1,   # HER GÜN YENİDEN KARAR VER
                    'n_states': 3, 
                    'commission': commission, 
                    'max_alloc': max_alloc
                }
                
                for i, ticker in enumerate(tickers):
                    df = get_data(ticker, str(start_date))
                    if df.empty: continue
                    
                    # Fonksiyon artık 3 değer dönüyor: Sonuç, Sinyal, Detaylı Geçmiş
                    res, sig_data, history_df = run_strategy_single(df, params, capital_per_coin)
                    
                    if res is not None:
                        portfolio_df[ticker] = res
                        
                        # HODL Hesapla
                        start_price = df.loc[res.index[0], 'close']
                        relevant_prices = df.loc[res.index[0]:, 'close']
                        hodl_val = (capital_per_coin / start_price) * relevant_prices
                        hodl_val = hodl_val.reindex(res.index, method='ffill')
                        hodl_df[ticker] = hodl_val
                        
                        if sig_data:
                            sig_data['Coin'] = ticker
                            signal_list.append(sig_data)
                        
                        # Detaylı geçmişi sakla
                        all_histories[ticker] = history_df
                    
                    progress_bar.progress((i + 1) / len(tickers))
                
                # --- SONUÇ GÖSTERİMİ ---
                if not portfolio_df.empty:
                    portfolio_df.fillna(method='ffill', inplace=True).fillna(0, inplace=True)
                    hodl_df.fillna(method='ffill', inplace=True).fillna(0, inplace=True)
                    
                    total_portfolio = portfolio_df.sum(axis=1)
                    total_hodl = hodl_df.sum(axis=1)
                    
                    # Ortak index
                    common_idx = total_portfolio.index.intersection(total_hodl.index)
                    total_portfolio = total_portfolio.loc[common_idx]
                    total_hodl = total_hodl.loc[common_idx]
                    
                    final_bal = total_portfolio.iloc[-1]
                    roi = ((final_bal - initial_capital) / initial_capital) * 100
                    hodl_final = total_hodl.iloc[-1]
                    alpha = final_bal - hodl_final
                    
                    # METRİKLER
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Bot Bakiyesi", f"${final_bal:,.0f}", f"{roi:.1f}%")
                    c2.metric("Sepet (HODL) Değeri", f"${hodl_final:,.0f}")
                    c3.metric("Bot Farkı (Alpha)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
                    
                    # BUGÜNÜN ÖZETİ
                    st.markdown("---")
                    st.subheader("📢 BUGÜNÜN KARARLARI (Son Kapanış)")
                    if signal_list:
                        sig_df = pd.DataFrame(signal_list)
                        cols = ['Coin', 'Fiyat', 'Öneri', 'Trend', 'HMM Rejimi']
                        
                        def highlight(val):
                            if 'AL' in val: return 'background-color: #d4edda; color: green; font-weight: bold'
                            if 'SAT' in val: return 'background-color: #f8d7da; color: red; font-weight: bold'
                            return ''
                            
                        st.dataframe(sig_df[cols].style.applymap(highlight, subset=['Öneri']).format({"Fiyat": "${:,.2f}"}))
                    
                    # --- YENİ ÖZELLİK: DETAYLI GÜNLÜK LOG ---
                    st.markdown("---")
                    st.subheader("📜 Detaylı Günlük Karar Defteri (Son 10 Gün)")
                    st.info("Botun son 10 gündeki fikir değişikliklerini aşağıdan inceleyebilirsin.")
                    
                    selected_history_coin = st.selectbox("Hangi Coin'in Günlüğünü Görmek İstersin?", tickers)
                    
                    if selected_history_coin in all_histories:
                        # Son 10 günü al ve ters çevir (Bugün en üstte olsun)
                        daily_log = all_histories[selected_history_coin].tail(10).sort_index(ascending=False)
                        
                        # Tabloyu güzelleştir
                        st.dataframe(daily_log.style.format({"Fiyat": "${:,.2f}"}).applymap(
                            lambda v: 'color: green; font-weight: bold' if 'AL' in str(v) else ('color: red; font-weight: bold' if 'SAT' in str(v) else ''), 
                            subset=['Karar']
                        ))
                    else:
                        st.warning("Bu coin için yeterli veri oluşmadı.")
                        
                    # GRAFİK
                    st.markdown("---")
                    st.subheader("📈 Performans Grafiği")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(total_portfolio.index, total_portfolio, label="Bot Stratejisi", color="#4B0082", linewidth=2)
                    ax.plot(total_hodl.index, total_hodl, label="HODL (Bekle)", color="gray", alpha=0.5, linestyle="--")
                    ax.set_ylabel("Dolar ($)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                else:
                    st.error("Veri hesaplanamadı.")
            except Exception as e:
                st.error(f"Hata: {e}")
    
    with main_tab2:
        st.header("🔮 30 Günlük Projeksiyon")
        coin_f = st.selectbox("Coin Seç", tickers, key="forecast")
        if st.button("Tahmin Et"):
            df_f = get_data(coin_f, "2021-01-01")
            if not df_f.empty:
                proj = calculate_short_term_projection(df_f)
                fig_f, ax_f = plt.subplots(figsize=(12, 5))
                ax_f.plot(df_f.index[-60:], df_f['close'].iloc[-60:], color='black', label='Geçmiş')
                ax_f.plot(proj['Date'], proj['Realistic'], color='blue', label='Tahmin')
                ax_f.fill_between(proj['Date'], proj['Pessimistic'], proj['Optimistic'], color='gray', alpha=0.2)
                ax_f.legend()
                st.pyplot(fig_f)
                st.dataframe(proj.set_index('Date'))
            else:
                st.error("Veri yok.")
else:
    st.info("Lütfen coin seçin.")
