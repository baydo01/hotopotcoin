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

# --- CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #4B0082;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- FONKSİYONLAR ---

@st.cache_data
def get_data(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        
        # Feature Engineering
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
    Tek bir coin için stratejiyi çalıştırır ve günlük bakiyeyi döner.
    Ayrıca son güncel sinyali de döndürür.
    """
    train_window = params['train_window']
    retrain_every = params['retrain_every']
    n_states = params['n_states']
    
    feature_cols = ['log_ret', 'range', 'vol_20']
    X = df[feature_cols].values
    states_pred = np.full(len(df), -1)
    
    if len(df) < train_window + retrain_every: return None, None

    scaler = StandardScaler()
    
    # HMM Walk-Forward
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
    
    if df_res.empty: return None, None

    state_stats = df_res.groupby('state')['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    cash = alloc_capital
    coin_amt = 0
    portfolio = []
    commission = params['commission']
    max_alloc = params['max_alloc']
    
    for idx, row in df_res.iterrows():
        price = row['close']
        state = row['state']
        sma_fast = row['sma_fast']
        
        is_uptrend = price > sma_fast
        is_hmm_bull = (state == bull_state)
        is_hmm_bear = (state == bear_state)
        
        target_pct = 0.0
        
        if is_uptrend:
            if is_hmm_bear: target_pct = max_alloc * 0.8 
            else: target_pct = max_alloc 
        else:
            if is_hmm_bull: target_pct = max_alloc * 0.2 
            else: target_pct = 0.0 
            
        current_val = cash + (coin_amt * price)
        if current_val <= 0: 
            portfolio.append(0); continue
            
        current_pct = (coin_amt * price) / current_val
        
        if abs(target_pct - current_pct) > 0.05:
            diff_usd = (target_pct - current_pct) * current_val
            fee = abs(diff_usd) * commission
            
            if diff_usd > 0: # AL
                if cash >= diff_usd:
                    coin_amt += (diff_usd - fee) / price
                    cash -= diff_usd
            else: # SAT
                sell_usd = abs(diff_usd)
                if (coin_amt * price) >= sell_usd:
                    coin_amt -= sell_usd / price
                    cash += (sell_usd - fee)
                    
        portfolio.append(cash + (coin_amt * price))
    
    # --- GÜNCEL SİNYAL ANALİZİ (SON GÜN) ---
    last_row = df_res.iloc[-1]
    last_price = last_row['close']
    last_state = last_row['state']
    last_sma = last_row['sma_fast']
    
    is_uptrend_now = last_price > last_sma
    is_bull_now = (last_state == bull_state)
    is_bear_now = (last_state == bear_state)
    
    final_action = "NÖTR"
    final_target = 0.0
    reason = ""
    
    if is_uptrend_now:
        if is_bear_now:
            final_target = max_alloc * 0.8
            final_action = "AL / TUT (Temkinli)"
            reason = "Trend Yukarı ama Risk Yüksek"
        else:
            final_target = max_alloc
            final_action = "AL / TUT (Güçlü)"
            reason = "Trend Yukarı ve Güvenli"
    else:
        if is_bull_now:
            final_target = max_alloc * 0.2
            final_action = "DİP ALIMI (%20)"
            reason = "Trend Aşağı ama Dip Sinyali"
        else:
            final_target = 0.0
            final_action = "SAT / NAKİT"
            reason = "Trend Aşağı ve Riskli"
            
    signal_data = {
        "Fiyat": last_price,
        "Trend": "YÜKSELİŞ 📈" if is_uptrend_now else "DÜŞÜŞ 📉",
        "HMM Rejimi": "BOĞA 🐂" if is_bull_now else ("AYI 🐻" if is_bear_now else "YATAY 🦀"),
        "Öneri": final_action,
        "Hedef Pozisyon": f"%{final_target*100:.0f}",
        "Mantık": reason
    }
        
    return pd.Series(portfolio, index=df_res.index), signal_data

def calculate_short_term_projection(df, days_to_project=30):
    """
    Kısa Vadeli Tahmin (Gelecek 30 Gün)
    """
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

# --- ARAYÜZ ---

st.title("🏦 Hedge Fund Manager (Multi-Asset)")
st.markdown("Portföy Yönetimi ve Gelecek Projeksiyonu Sistemi")

with st.sidebar:
    st.header("⚙️ Portföy Ayarları")
    
    default_coins = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    tickers = st.multiselect("Portföye Eklenecek Coinler", 
                             ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LTC-USD"],
                             default=default_coins)
    
    start_date = st.date_input("Başlangıç", pd.to_datetime("2020-01-01"))
    initial_capital = st.number_input("Toplam Sermaye ($)", 10000)
    commission = st.number_input("Komisyon", 0.001, format="%.4f")
    max_alloc = st.slider("Max Pozisyon (Her Coin İçin)", 0.1, 1.0, 1.0)
    
    st.divider()
    st.info("Gelecek tahminleri artık günlük bazda (Kısa Vadeli) çalışır.")

if len(tickers) > 0:
    
    main_tab1, main_tab2 = st.tabs(["📊 Portföy Yönetimi (Backtest)", "🔮 Kısa Vadeli Tahmin (30 Gün)"])
    
    # --- SEKME 1: BACKTEST ---
    with main_tab1:
        if st.button("Portföyü Yönet 🚀", type="primary"):
            st.write("🔄 İşlem başladı, lütfen bekleyin...")
            capital_per_coin = initial_capital / len(tickers)
            st.info(f"Coin Başına Sermaye: ${capital_per_coin:,.2f}")
            
            portfolio_df = pd.DataFrame()
            hodl_df = pd.DataFrame()
            signal_list = [] 
            
            progress_bar = st.progress(0)
            
            try:
                with st.spinner("Hedge Fonu Çalışıyor..."):
                    params = {
                        'train_window': 365, 'retrain_every': 30, 
                        'n_states': 3, 'commission': commission, 'max_alloc': max_alloc
                    }
                    
                    for i, ticker in enumerate(tickers):
                        df = get_data(ticker, str(start_date))
                        if df.empty: continue
                        
                        res, sig_data = run_strategy_single(df, params, capital_per_coin)
                        
                        if res is not None:
                            portfolio_df[ticker] = res
                            start_price = df.loc[res.index[0], 'close']
                            hodl_val = (capital_per_coin / start_price) * df.loc[res.index, 'close']
                            hodl_df[ticker] = hodl_val
                            
                            if sig_data:
                                sig_data['Coin'] = ticker
                                signal_list.append(sig_data)
                        
                        progress_bar.progress((i + 1) / len(tickers))
                    
                    # Sonuçları Birleştir
                    if not portfolio_df.empty:
                        portfolio_df.fillna(method='ffill', inplace=True)
                        portfolio_df.fillna(0, inplace=True)
                        hodl_df.fillna(method='ffill', inplace=True)
                        hodl_df.fillna(0, inplace=True)
                        
                        total_portfolio = portfolio_df.sum(axis=1)
                        total_hodl = hodl_df.sum(axis=1)
                        
                        common_idx = total_portfolio[total_portfolio > 0].index
                        total_portfolio = total_portfolio.loc[common_idx]
                        total_hodl = total_hodl.loc[common_idx]
                        
                        final_bal = total_portfolio.iloc[-1]
                        roi = ((final_bal - initial_capital) / initial_capital) * 100
                        hodl_final = total_hodl.iloc[-1]
                        alpha = final_bal - hodl_final
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Toplam Fon Değeri", f"${final_bal:,.0f}", f"{roi:.1f}%")
                        c2.metric("Sepet HODL Değeri", f"${hodl_final:,.0f}")
                        c3.metric("Alpha (Fark)", f"${alpha:,.0f}", 
                                  delta_color="normal" if alpha > 0 else "inverse")
                        
                        sub_tab1, sub_tab2 = st.tabs(["📈 Genel Performans", "🧩 Coin Bazlı Detay"])
                        
                        with sub_tab1:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(total_portfolio.index, total_portfolio, label="Hedge Fund (Model)", color="#4B0082", linewidth=2)
                            ax.plot(total_hodl.index, total_hodl, label="Sepet HODL", color="gray", alpha=0.5, linestyle="--")
                            ax.set_title("Portföy vs Sepet HODL")
                            ax.set_ylabel("Değer ($)")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        with sub_tab2:
                            st.markdown("### Hangi Coin Ne Kazandırdı?")
                            last_vals = portfolio_df.iloc[-1].copy()
                            last_vals.name = "Bakiye ($)"
                            last_vals = last_vals.sort_values(ascending=False)
                            st.bar_chart(last_vals)
                            st.dataframe(portfolio_df.tail())
                            
                        st.markdown("---")
                        st.subheader("📢 Güncel Al/Sat Sinyalleri (Yapay Zeka Kararı)")
                        
                        if signal_list:
                            sig_df = pd.DataFrame(signal_list)
                            cols = ['Coin', 'Fiyat', 'Öneri', 'Hedef Pozisyon', 'Trend', 'HMM Rejimi', 'Mantık']
                            sig_df = sig_df[cols]
                            
                            def highlight_action(val):
                                color = ''
                                if 'AL' in val or 'TUT' in val:
                                    color = 'background-color: #d4edda; color: #155724' 
                                elif 'SAT' in val or 'NAKİT' in val:
                                    color = 'background-color: #f8d7da; color: #721c24' 
                                elif 'DİP' in val:
                                     color = 'background-color: #fff3cd; color: #856404' 
                                return color

                            st.dataframe(sig_df.style.applymap(highlight_action, subset=['Öneri']).format({"Fiyat": "${:,.2f}"}))
                        else:
                            st.warning("Sinyal verisi oluşturulamadı.")
                    else:
                         st.error("Hiçbir coinden veri alınamadı.")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
    
    # --- SEKME 2: KISA VADELİ TAHMİN ---
    with main_tab2:
        st.header("🔮 Kısa Vadeli Tahmin (30 Gün)")
        
        selected_coin_forecast = st.selectbox("Tahmin Edilecek Coin Seç", tickers)
        
        if st.button(f"{selected_coin_forecast} İçin 30 Günlük Tahmin 🔮"):
            df_coin = get_data(selected_coin_forecast, "2020-01-01")
            
            if not df_coin.empty:
                proj_df = calculate_short_term_projection(df_coin)
                
                st.subheader("📄 Günlük Tahminler")
                st.dataframe(proj_df.set_index('Date').style.format("${:,.2f}"))
                
                fig_f, ax_f = plt.subplots(figsize=(12, 6))
                recent_history = df_coin['close'].iloc[-60:]
                ax_f.plot(recent_history.index, recent_history.values, label='Geçmiş Fiyat (Son 60 Gün)', color='black', linewidth=1.5)
                
                dates = proj_df['Date']
                ax_f.plot(dates, proj_df['Optimistic'], label='İyimser Senaryo', linestyle='--', color='green', alpha=0.7)
                ax_f.plot(dates, proj_df['Realistic'], label='Gerçekçi Tahmin (Momentum)', linestyle='-', color='blue', linewidth=2)
                ax_f.plot(dates, proj_df['Pessimistic'], label='Kötümser Senaryo', linestyle='--', color='red', alpha=0.7)
                
                ax_f.fill_between(dates, proj_df['Pessimistic'], proj_df['Optimistic'], color='gray', alpha=0.1, label='Olasılık Aralığı')
                
                ax_f.set_title(f"{selected_coin_forecast}: 30 Günlük Fiyat Projeksiyonu")
                ax_f.set_ylabel("Fiyat ($)")
                ax_f.legend()
                ax_f.grid(True, alpha=0.3)
                st.pyplot(fig_f)
            else:
                st.error("Seçilen coin için yeterli veri bulunamadı.")

else:
    st.info("Lütfen soldaki menüden en az bir coin seçin.")