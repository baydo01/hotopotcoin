import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
import datetime

# Hataları gizle
warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager: V14 - Optimize Edilmiş Başlangıç", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #00897B; color: white; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --- SABİT BOT PARAMETRELERİ (GLOBAL) ---
BOT_PARAMS = {
    'n_states': 3,
    'commission': 0.001,
    'train_days': 252,    # V14 DÜZELTMESİ: 5 yıldan 1 yıla düşürüldü
    'optimize_days': 21,
    'rebalance_days': 5,
}

# --- AĞIRLIKLANDIRMA SENARYOLARI (GLOBAL) ---
WEIGHT_SCENARIOS = {
    'A': [2.0, 1.0, 0.5], 
    'B': [1.5, 1.0, 0.7], 
    'C': [1.0, 1.0, 1.0], 
    'D': [3.0, 1.0, 0.2],
}

# --- ÖZEL PUAN HESABI ---
def calculate_custom_score(df):
    if len(df) < 5: return pd.Series(0, index=df.index)
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
        df = df[['close', 'open', 'high', 'low', 'volume']].copy()
        return df
    except:
        return None

# --- VERİ AĞIRLIĞI OPTİMİZASYONU FONKSİYONU ---
def optimize_data_weights(train_data_all, optim_data_all, n_states, weight_scenarios, current_date, tickers):
    
    if train_data_all.empty or len(train_data_all) < n_states:
        return 'C', weight_scenarios['C']

    best_w_set = 'C'
    best_optim_roi = -np.inf
    
    one_year_ago = current_date - pd.Timedelta(days=365)
    three_years_ago = current_date - pd.Timedelta(days=365*3)
    
    w_hmm, w_score = 0.7, 0.3

    for set_name, weights in weight_scenarios.items():
        w_latest, w_mid, w_old = weights
        
        # 1. Eğitim Verisi İçin sample_weight Hesaplama
        train_data = train_data_all.copy()
        
        train_data['weight'] = 1.0
        train_data['Date'] = train_data.index.get_level_values('Date')
        
        train_data['weight'] = np.where(train_data['Date'] >= one_year_ago, w_latest, train_data['weight'])
        train_data['weight'] = np.where((train_data['Date'] >= three_years_ago) & (train_data['Date'] < one_year_ago), w_mid, train_data['weight'])
        train_data['weight'] = np.where(train_data['Date'] < three_years_ago, w_old, train_data['weight'])
        train_data.drop(columns=['Date'], inplace=True)
        
        # 2. HMM Eğitimi (sample_weight kullanarak)
        X_train = train_data[['log_ret', 'range']].values
        
        if X_train.shape[0] < n_states:
            continue

        scaler = StandardScaler()
        X_s_train = scaler.fit_transform(X_train)
        
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
            weights_float = train_data['weight'].values.astype(np.float64)
            model.fit(X_s_train, sample_weight=weights_float)
            
            state_stats = train_data.groupby(model.predict(X_s_train))['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.idxmin()
        except Exception:
            continue
        
        # 3. Optimizasyon Penceresinde Simülasyon
        total_optim_roi = 0
        optim_data_processed = optim_data_all.copy()
        
        for ticker in tickers:
            try:
                coin_optim_data = optim_data_processed.xs(ticker, level='ticker')
            except KeyError:
                continue
            
            if coin_optim_data.empty or len(coin_optim_data) < 5: continue
            
            temp_cash = 100 
            temp_coin_amt = 0
            
            for _, row in coin_optim_data.iterrows():
                X_optim_point = scaler.transform([[row['log_ret'], row['range']]])
                hmm_signal = 1 if model.predict(X_optim_point)[0] == bull_state else (-1 if model.predict(X_optim_point)[0] == bear_state else 0)
                score_signal = 1 if row['custom_score'] >= 3 else (-1 if row['custom_score'] <= -3 else 0)
                weighted_decision = (w_hmm * hmm_signal) + (w_score * score_signal)
                
                price = row['close']
                if weighted_decision > 0.25: temp_coin_amt = temp_cash / price; temp_cash = 0
                elif weighted_decision < -0.25: temp_cash = temp_coin_amt * price; temp_coin_amt = 0
            
            if not coin_optim_data.empty:
                final_optim_val = temp_cash + temp_coin_amt * coin_optim_data['close'].iloc[-1]
                total_optim_roi += (final_optim_val - 100) / 100

        if total_optim_roi > best_optim_roi:
            best_optim_roi = total_optim_roi
            best_w_set = set_name
            
    return best_w_set, weight_scenarios[best_w_set]


# --- TEMEL FONKSİYON: DİNAMİK PORTFÖY BACKTESTİ ---
def run_dynamic_portfolio_backtest_v10(df_combined, tickers, params, initial_capital, start_year):
    
    train_window = params['train_days']
    optim_window = params['optimize_days']
    rebalance_window = params['rebalance_days']
    n_states = params['n_states']
    commission = params['commission']
    
    cash = initial_capital
    coin_amounts = {t: 0 for t in tickers}
    portfolio_history = pd.Series(dtype='float64')
    coin_decisions = {}

    # Veriyi kullanıcı başlangıç yılına göre filtrele
    start_date = pd.to_datetime(f'{start_year}-01-01')
    df_combined = df_combined[df_combined.index.get_level_values('Date') >= start_date].copy()


    dates = df_combined.index.get_level_values('Date').unique().sort_values()
    df_clean = df_combined.reindex(pd.MultiIndex.from_product([dates, tickers], names=['Date', 'ticker'])).dropna(subset=['close']).copy()
    dates = df_clean.index.get_level_values('Date').unique().sort_values()
    
    min_data_required = train_window + optim_window + rebalance_window
    
    if len(dates) < min_data_required:
        st.error(f"⚠️ Yetersiz veri. Seçilen {start_year} başlangıç yılı için sadece {len(dates)} gün mevcut. Minimum gereksinim: {min_data_required} gün.")
        return None, None
    
    
    for i in range(train_window + optim_window, len(dates), rebalance_window):
        
        # 1. Pencere Tarihlerini Tanımla
        rebalance_execution_date = dates[i - rebalance_window]
        trade_end_date = dates[i - 1] 
        optim_end_date = dates[i - rebalance_window - 1]
        optim_start_date = dates[i - rebalance_window - optim_window]
        train_start_date = dates[i - rebalance_window - optim_window - train_window]
        
        # 2. Özellik Hesaplama
        train_optim_data_all = df_clean.loc[train_start_date:optim_end_date].copy()
        
        for col in ['log_ret', 'range', 'custom_score']:
            train_optim_data_all[col] = np.nan
        
        for t in tickers:
            try:
                df_t = train_optim_data_all.xs(t, level='ticker').copy()
            except KeyError:
                continue

            if not df_t.empty:
                df_t['log_ret'] = df_t['close'].pct_change().apply(lambda x: np.log(1+x))
                df_t['range'] = (df_t['high'] - df_t['low']) / df_t['close']
                df_t['custom_score'] = calculate_custom_score(df_t)
                
                idx = df_t.index
                train_optim_data_all.loc[(idx, t), ['log_ret', 'range', 'custom_score']] = df_t[['log_ret', 'range', 'custom_score']].values

        train_data_all = train_optim_data_all.loc[train_start_date:optim_end_date].dropna(subset=['log_ret', 'range', 'custom_score']).copy()
        optim_data_all = train_optim_data_all.loc[optim_start_date:optim_end_date].dropna(subset=['log_ret', 'range', 'custom_score']).copy()

        # En iyi veri ağırlıklandırma setini bul
        best_w_set, weights = optimize_data_weights(train_data_all, optim_data_all, n_states, WEIGHT_SCENARIOS, rebalance_execution_date, tickers)
        w_latest, w_mid, w_old = weights
        w_hmm, w_score = 0.7, 0.3

        # 3. Eğitim (En iyi ağırlık seti ile)
        one_year_ago = rebalance_execution_date - pd.Timedelta(days=365)
        three_years_ago = rebalance_execution_date - pd.Timedelta(days=365*3)

        train_data_final = train_data_all.copy()
        train_data_final['weight'] = 1.0
        
        train_data_final['Date'] = train_data_final.index.get_level_values('Date')
        train_data_final['weight'] = np.where(train_data_final['Date'] >= one_year_ago, w_latest, train_data_final['weight'])
        train_data_final['weight'] = np.where((train_data_final['Date'] >= three_years_ago) & (train_data_final['Date'] < one_year_ago), w_mid, train_data_final['weight'])
        train_data_final['weight'] = np.where(train_data_final['Date'] < three_years_ago, w_old, train_data_final['weight'])
        train_data_final.drop(columns=['Date'], inplace=True)

        # Hata koruması
        if len(train_data_final) < n_states:
             continue
        
        X_train = train_data_final[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s_train = scaler.fit_transform(X_train)
        
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
            weights_float = train_data_final['weight'].values.astype(np.float64)
            model.fit(X_s_train, sample_weight=weights_float)
            
            state_stats = train_data_final.groupby(model.predict(X_s_train))['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.min()
        except Exception:
             continue

        # 4. Sinyal Hesaplama (Rebalance Karar Gününde)
        coin_decisions = {}
        rebalance_decision_date = optim_end_date
        
        for ticker in tickers:
            try:
                last_day_data = df_clean.loc[rebalance_decision_date].xs(ticker, level='ticker').iloc[-1]
                
                prev_close = df_clean.loc[:rebalance_decision_date].xs(ticker, level='ticker')['close'].iloc[-2]
                log_ret = np.log(last_day_data['close'] / prev_close)
                range_ = (last_day_data['high'] - last_day_data['low']) / last_day_data['close']
                
                X_point = scaler.transform([[log_ret, range_]])
                hmm_signal = 1 if model.predict(X_point)[0] == bull_state else (-1 if model.predict(X_point)[0] == bear_state else 0)
                
                weighted_decision = (w_hmm * hmm_signal)
                
                coin_decisions[ticker] = {
                    'signal': weighted_decision,
                    'price': last_day_data['close'],
                    'action': "AL" if weighted_decision > 0.25 else ("SAT" if weighted_decision < -0.25 else "BEKLE"),
                    'weight_set': best_w_set
                }
            except Exception: 
                coin_decisions[ticker] = {'signal': 0, 'price': 0, 'action': "BEKLE", 'weight_set': best_w_set}
        
        # 5. Portföy Yeniden Dengeleme
        rebalance_execution_date = rebalance_execution_date
        
        total_value = cash
        for t in tickers:
            if coin_amounts[t] > 0:
                try:
                    current_price = df_clean.loc[(rebalance_execution_date, t), 'close']
                    total_value += coin_amounts[t] * current_price
                except KeyError:
                    pass

        # SATIŞ işlemlerini yap
        for t in tickers:
            if t in coin_decisions and coin_decisions[t]['action'] == 'SAT' and coin_amounts[t] > 0:
                try:
                    current_price = df_clean.loc[(rebalance_execution_date, t), 'close']
                    sell_usd = coin_amounts[t] * current_price
                    fee = sell_usd * commission
                    
                    cash += (sell_usd - fee)
                    coin_amounts[t] = 0
                except KeyError: pass 

        # ALIM işlemlerini yap
        buy_signals = [t for t, d in coin_decisions.items() if d['action'] == 'AL']
        if buy_signals and cash > 0:
            target_pct = 1.0 / len(buy_signals)
            buyable_cash = cash
            
            for t in buy_signals:
                try:
                    buy_amount = buyable_cash * target_pct
                    current_price = df_clean.loc[(rebalance_execution_date, t), 'close']
                    fee = buy_amount * commission
                    
                    coin_amounts[t] += (buy_amount - fee) / current_price
                    cash -= buy_amount
                except KeyError: pass

        # 6. İşlem Penceresi boyunca pozisyonları tut ve bakiye kaydet
        trade_df_multi = df_clean.loc[rebalance_execution_date:trade_end_date]
        
        for date, group in trade_df_multi.groupby(level='Date'):
            current_day_value = cash
            
            for t in tickers:
                if coin_amounts[t] > 0:
                    try:
                        current_price = group.loc[(date, t), 'close']
                        current_day_value += coin_amounts[t] * current_price
                    except KeyError: pass
            
            portfolio_history.loc[date] = float(current_day_value)
            
    return portfolio_history.sort_index(), coin_decisions

# ----------------------------------------------------------------------
# --- ARAYÜZ VE VERİ BİRLEŞTİRME ---
# ----------------------------------------------------------------------
st.title("💰 Hedge Fund Manager: V13 - Veri Ağırlığı Optimizasyonu")
st.markdown("### 🗓️ Hangi Geçmiş Verinin Daha Önemli Olduğunu BOT Belirliyor")

with st.sidebar:
    st.header("Ayarlar (Otonom)")
    default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
    tickers=st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital=st.number_input("Kasa ($)", 10000, step=1000)
    
    # V13 Düzeltmesi: Başlangıç yılı seçeneği geri getirildi
    years=[2018, 2019, 2020, 2021, 2022]
    selected_year=st.selectbox("Simülasyon Başlangıç Yılı", years, index=3)
    
    st.info(f"""
        **Bot Parametreleri:**
        * Eğitim Penceresi: {BOT_PARAMS['train_days']} gün (~1 Yıl)
        * Komisyon: {BOT_PARAMS['commission']*100}%
        * Yeniden Dengeleme: {BOT_PARAMS['rebalance_days']} günde bir (Haftalık)
    """)

if st.button("DİNAMİK PORTFÖY BOTU ÇALIŞTIR 🚀"):
    if not tickers: st.error("Lütfen en az bir coin seçin.")
    else:
        all_dfs = []
        status = st.empty()
        
        # Tüm geçmiş veriyi çekmek için çok erken bir başlangıç tarihi kullan
        start_date_yf = "2018-01-01" 
        
        for ticker in tickers:
            status.text(f"⚙️ {ticker} verisi çekiliyor...")
            df = get_data_cached(ticker, start_date_yf)
            if df is not None:
                df['ticker'] = ticker
                all_dfs.append(df)
            
        if not all_dfs:
            st.error("Hiçbir coin için veri bulunamadı.")
        else:
            df_combined = pd.concat(all_dfs, keys=tickers, names=['ticker', 'Date'])
            df_combined = df_combined.swaplevel(0, 1).sort_index()

            status.text(f"⚙️ Dinamik Portföy Simülasyonu Başlatılıyor...")
            
            # Seçilen başlangıç yılını fonksiyona geçir
            history_series, last_signals = run_dynamic_portfolio_backtest_v10(df_combined, tickers, BOT_PARAMS, initial_capital, selected_year)
            
            status.empty()

            if history_series is not None and len(history_series) > 0:
                final_val = history_series.iloc[-1]
                roi = ((final_val - initial_capital) / initial_capital) * 100
                
                # HODL Karşılaştırması
                hodl_val = 0
                for ticker in tickers:
                    df_ticker = df_combined.xs(ticker, level='ticker')
                    # HODL başlangıç fiyatını selected_year'a göre bul
                    start_price_hodl = df_ticker[df_ticker.index >= pd.to_datetime(f'{selected_year}-01-01')]['close'].iloc[0]
                    end_price = df_ticker['close'].iloc[-1]
                    hodl_val += (initial_capital / len(tickers) / start_price_hodl) * end_price
                
                alpha = final_val - hodl_val
                
                st.success(f"✅ BOT SİMÜLASYONU BAŞARILI!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("BOT Final Bakiye", f"${final_val:,.2f}", f"{roi:,.2f}% ROI")
                col2.metric("Eşit Ağırlıklı HODL", f"${hodl_val:,.2f}")
                col3.metric("Alpha (Bot Getirisi - HODL)", f"${alpha:,.2f}")
                
                st.subheader("Portföy Değer Eğrisi")
                st.line_chart(history_series.rename("Bot Portföy Değeri"), use_container_width=True)
                
                st.subheader("Son Haftalık Sinyaller")
                st.json(last_signals)
                
            else:
                st.error("Simülasyon sonuç vermedi. Lütfen başlangıç verilerinizi ve pencere boyutlarınızı kontrol edin.")
