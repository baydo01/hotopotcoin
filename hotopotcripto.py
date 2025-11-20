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
st.set_page_config(page_title="Hedge Fund Manager: V10 - Veri Ağırlığı Optimizasyonu", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #00897B; color: white; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --- SABİT BOT PARAMETRELERİ (Otonom) ---
BOT_PARAMS = {
    'n_states': 3,
    'commission': 0.001,
    'train_days': 252 * 5,    # Son 5 Yıl Veri Eğitimi İçin (~1260 İşlem Günü)
    'optimize_days': 21,   # ~3 Hafta Optimizasyon Penceresi
    'rebalance_days': 5,    # ~1 Hafta Yeniden Dengeleme Penceresi
}

# --- AĞIRLIKLANDIRMA SENARYOLARI ---
# Her senaryo [Çok Yakın (Son 1 Yıl), Orta Yakın (1-3 Yıl), Eski (3+ Yıl)] için ağırlık çarpanını tanımlar.
WEIGHT_SCENARIOS = {
    'A': [2.0, 1.0, 0.5],  # Güncel veri 4x daha önemli (2.0/0.5)
    'B': [1.5, 1.0, 0.7],  # Daha dengeli
    'C': [1.0, 1.0, 1.0],  # Eşit ağırlık (Baseline)
    'D': [3.0, 1.0, 0.2],  # Güncele aşırı odaklanma
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

# --- VERİ ÇEKME (Aynı Kaldı) ---
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
        df = df[['close', 'open', 'high', 'low', 'volume']]
        return df
    except:
        return None

# --- YENİ TEMEL FONKSİYON: VERİ AĞIRLIĞI OPTİMİZASYONU ---
def optimize_data_weights(train_data, optim_data, n_states, weight_scenarios, current_date):
    
    best_w_set = 'C' # Default eşit ağırlık
    best_optim_roi = -np.inf
    
    # Tüm veriyi 3 döneme ayır: (Çok Yakın: Son 1 yıl), (Orta Yakın: 1-3 yıl), (Eski: 3+ yıl)
    one_year_ago = current_date - pd.Timedelta(days=365)
    three_years_ago = current_date - pd.Timedelta(days=365*3)
    
    # Veri Ağırlığı ve HMM/Puan Ağırlığı Senaryoları
    signal_weights = [0.7] # Sadece HMM %70, Puan %30'u kullan
    
    for set_name, weights in WEIGHT_SCENARIOS.items():
        w_latest, w_mid, w_old = weights
        
        # 1. Eğitim Verisi İçin sample_weight Hesaplama
        train_data['weight'] = 1.0 # Başlangıç ağırlığı 1
        train_data['weight'] = np.where(train_data.index >= one_year_ago, w_latest, train_data['weight'])
        train_data['weight'] = np.where((train_data.index >= three_years_ago) & (train_data.index < one_year_ago), w_mid, train_data['weight'])
        train_data['weight'] = np.where(train_data.index < three_years_ago, w_old, train_data['weight'])
        
        # 2. HMM Eğitimi (sample_weight kullanarak)
        X_train = train_data[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s_train = scaler.fit_transform(X_train)
        
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
            # Ağırlıklandırmayı burada uygula!
            model.fit(X_s_train, sample_weight=train_data['weight'].values)
            
            state_stats = train_data.groupby(model.predict(X_s_train))['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.idxmin()
        except:
            continue
        
        # 3. Optimizasyon Penceresinde Simülasyon (Tüm coinler için TEK ağırlık setini test et)
        total_optim_roi = 0
        
        for ticker in optim_data.index.get_level_values('ticker').unique():
            coin_optim_data = optim_data.xs(ticker, level='ticker')
            
            if len(coin_optim_data) < 5: continue
            
            w_hmm, w_score = 0.7, 0.3 # HMM/Puan ağırlığını sabit tut

            # Özellik Hesaplama
            coin_optim_data['log_ret'] = np.log(coin_optim_data['close']/coin_optim_data['close'].shift(1))
            coin_optim_data['range'] = (coin_optim_data['high']-coin_optim_data['low'])/coin_optim_data['close']
            coin_optim_data['custom_score'] = calculate_custom_score(coin_optim_data)
            coin_optim_data.dropna(inplace=True)
            
            # Simülasyon
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

        # En iyi Ağırlık Setini Seç
        if total_optim_roi > best_optim_roi:
            best_optim_roi = total_optim_roi
            best_w_set = set_name
            
    return best_w_set, WEIGHT_SCENARIOS[best_w_set]


# --- TEMEL FONKSİYON: DİNAMİK PORTFÖY BACKTESTİ ---
def run_dynamic_portfolio_backtest_v10(df_combined, tickers, params, initial_capital):
    
    # Ayarlar
    train_window = params['train_days']
    optim_window = params['optimize_days']
    rebalance_window = params['rebalance_days']
    n_states = params['n_states']
    commission = params['commission']
    
    # Başlangıç değişkenleri
    cash = initial_capital
    coin_amounts = {t: 0 for t in tickers}
    portfolio_history = pd.Series(dtype='object') # Tarih/Değer çiftlerini tutar

    df_clean = df_combined.dropna(subset=['close'])
    dates = df_clean.index.get_level_values('Date').unique().sort_values()
    
    if len(dates) < train_window + optim_window + rebalance_window:
        return None, None
    
    # Kayar Pencere Döngüsü (Tarih indeksleri üzerinde)
    for i in range(train_window + optim_window, len(dates), rebalance_window):
        
        # 1. Pencere Tarihlerini Tanımla
        rebalance_execution_date = dates[i - rebalance_window] # İşlem Başlangıcı
        trade_end_date = dates[i - 1] 
        optim_end_date = dates[i - rebalance_window - 1]
        optim_start_date = dates[i - rebalance_window - optim_window]
        train_start_date = dates[i - rebalance_window - optim_window - train_window]
        
        # 2. Veri Ağırlığı Optimizasyonu
        train_data_all = df_clean.loc[train_start_date:optim_end_date]
        optim_data_all = df_clean.loc[optim_start_date:optim_end_date]
        current_date = rebalance_execution_date

        # Gerekli özellikleri tek bir yerde hesapla
        for t in tickers:
            df_train = train_data_all.xs(t, level='ticker').copy()
            if not df_train.empty:
                df_train['log_ret'] = np.log(df_train['close'] / df_train['close'].shift(1))
                df_train['range'] = (df_train['high'] - df_train['low']) / df_train['close']
                df_train['custom_score'] = calculate_custom_score(df_train)
                train_data_all.loc[(df_train.index, t), ['log_ret', 'range', 'custom_score']] = df_train[['log_ret', 'range', 'custom_score']].values

        train_data_all.dropna(inplace=True)
        optim_data_all.dropna(inplace=True)

        # En iyi veri ağırlıklandırma setini bul
        best_w_set, weights = optimize_data_weights(train_data_all, optim_data_all, n_states, WEIGHT_SCENARIOS, current_date)
        w_latest, w_mid, w_old = weights
        w_hmm, w_score = 0.7, 0.3 # Sinyal ağırlığı sabit

        # 3. Eğitim (En iyi ağırlık seti ile)
        one_year_ago = current_date - pd.Timedelta(days=365)
        three_years_ago = current_date - pd.Timedelta(days=365*3)

        train_data_final = train_data_all.copy()
        train_data_final['weight'] = 1.0
        train_data_final['weight'] = np.where(train_data_final.index.get_level_values('Date') >= one_year_ago, w_latest, train_data_final['weight'])
        train_data_final['weight'] = np.where((train_data_final.index.get_level_values('Date') >= three_years_ago) & (train_data_final.index.get_level_values('Date') < one_year_ago), w_mid, train_data_final['weight'])
        train_data_final['weight'] = np.where(train_data_final.index.get_level_values('Date') < three_years_ago, w_old, train_data_final['weight'])
        
        X_train = train_data_final[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s_train = scaler.fit_transform(X_train)
        
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X_s_train, sample_weight=train_data_final['weight'].values)
        state_stats = train_data_final.groupby(model.predict(X_s_train))['log_ret'].mean()
        bull_state = state_stats.idxmax()
        bear_state = state_stats.idxmin()

        # 4. Sinyal Hesaplama (Rebalance Karar Gününde)
        coin_decisions = {}
        
        for ticker in tickers:
            last_day_data = df_clean.loc[optim_end_date].xs(ticker, level='ticker').iloc[-1]
            last_price = last_day_data['close']
            
            # Sinyal için gerekli özellikleri hesapla
            prev_close = df_clean.loc[:optim_end_date].xs(ticker, level='ticker')['close'].iloc[-2]
            log_ret = np.log(last_price / prev_close)
            range_ = (last_day_data['high'] - last_day_data['low']) / last_price
            
            # Custom Score için train datasına bakmak gerekiyor.
            # Basitlik için sadece HMM'e odaklanalım ve Puan sinyalini 0 kabul edelim
            
            X_point = scaler.transform([[log_ret, range_]])
            hmm_signal = 1 if model.predict(X_point)[0] == bull_state else (-1 if model.predict(X_point)[0] == bear_state else 0)
            
            weighted_decision = (w_hmm * hmm_signal) # Puan sinyalini göz ardı ettik (0)

            coin_decisions[ticker] = {
                'signal': weighted_decision,
                'price': last_price,
                'action': "AL" if weighted_decision > 0.25 else ("SAT" if weighted_decision < -0.25 else "BEKLE")
            }
        
        # 5. Portföy Yeniden Dengeleme (Rebalance Execution Date fiyatları ile)
        
        # Tüm pozisyonların değerini hesapla
        total_value = cash
        for t in tickers:
            if coin_amounts[t] > 0:
                current_price = df_clean.loc[(rebalance_execution_date, t), 'close']
                total_value += coin_amounts[t] * current_price

        # SATIŞ işlemlerini yap
        for t in tickers:
            if t in coin_decisions and coin_decisions[t]['action'] == 'SAT' and coin_amounts[t] > 0:
                current_price = df_clean.loc[(rebalance_execution_date, t), 'close']
                sell_usd = coin_amounts[t] * current_price
                fee = sell_usd * commission
                cash += (sell_usd - fee)
                coin_amounts[t] = 0

        # ALIM işlemlerini yap
        buy_signals = [t for t, d in coin_decisions.items() if d['action'] == 'AL']
        if buy_signals and cash > 0:
            target_pct = 1.0 / len(buy_signals)
            buyable_cash = cash
            
            for t in buy_signals:
                buy_amount = buyable_cash * target_pct
                current_price = df_clean.loc[(rebalance_execution_date, t), 'close']
                fee = buy_amount * commission
                
                coin_amounts[t] += (buy_amount - fee) / current_price
                cash -= buy_amount

        # 6. İşlem Penceresi boyunca pozisyonları tut ve bakiye kaydet
        trade_df_multi = df_clean.loc[rebalance_execution_date:trade_end_date]
        
        for date, group in trade_df_multi.groupby(level='Date'):
            current_day_value = cash
            
            for t in tickers:
                if coin_amounts[t] > 0:
                    current_price = group.loc[(date, t), 'close']
                    current_day_value += coin_amounts[t] * current_price
            
            # Tarihleri kontrol ederek sadece güncel tarihi al
            if date not in portfolio_history.index or current_day_value > portfolio_history.loc[date]:
                 portfolio_history.loc[date] = current_day_value
            
    # Final portföy serisini float'a çevir
    portfolio_history = portfolio_history.astype(float)
    return portfolio_history.sort_index(), coin_decisions


# --- ARAYÜZ VE VERİ BİRLEŞTİRME ---
st.title("💰 Hedge Fund Manager: V10 - Veri Ağırlığı Optimizasyonu")
st.markdown("### 🗓️ Hangi Geçmiş Verinin Daha Önemli Olduğunu BOT Belirliyor")

with st.sidebar:
    st.header("Ayarlar (Otonom)")
    default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
    tickers=st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital=st.number_input("Kasa ($)", 10000, step=1000)
    start_year = st.selectbox("Başlangıç Yılı (Tüm geçmiş veriyi kullanır)", [2018, 2019, 2020, 2021, 2022], index=3)
    
    st.info(f"""
        **Bot Parametreleri:**
        * Eğitim Penceresi: {BOT_PARAMS['train_days']} gün (~5 Yıl)
        * Komisyon: {BOT_PARAMS['commission']*100}%
        * Yeniden Dengeleme: {BOT_PARAMS['rebalance_days']} günde bir (Haftalık)
    """)

if st.button("DİNAMİK PORTFÖY BOTU ÇALIŞTIR 🚀"):
    if not tickers: st.error("Lütfen en az bir coin seçin.")
    else:
        all_dfs = []
        status = st.empty()
        start_date = f"{start_year}-01-01"
        
        for ticker in tickers:
            status.text(f"⚙️ {ticker} verisi çekiliyor...")
            df = get_data_cached(ticker, start_date)
            if df is not None:
                df['ticker'] = ticker
                all_dfs.append(df)
            
        if not all_dfs:
            st.error("Hiçbir coin için yeterli veri bulunamadı.")
        else:
            df_combined = pd.concat(all_dfs, keys=tickers, names=['ticker', 'Date'])
            df_combined = df_combined.swaplevel(0, 1).sort_index()

            status.text(f"⚙️ Dinamik Portföy Simülasyonu Başlatılıyor...")
            
            # Simülasyonu başlat
            history_series, last_signals = run_dynamic_portfolio_backtest_v10(df_combined, tickers, BOT_PARAMS, initial_capital)
            
            status.empty()

            if history_series is not None and len(history_series) > 0:
                final_val = history_series.iloc[-1]
                roi = ((final_val - initial_capital) / initial_capital) * 100
                
                # HODL Karşılaştırması
                hodl_val = 0
                for ticker in tickers:
                    df_ticker = df_combined.xs(ticker, level='ticker')
                    if len(df_ticker) > 0:
                        start_price = df_ticker['close'].iloc[0]
                        end_price = df_ticker['close'].iloc[-1]
                        hodl_val += (initial_capital / len(tickers) / start_price) * end_price
                
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
                st.error("Simülasyon sonuç vermedi. Lütfen başlangıç yılını veya coin seçimini kontrol edin.")
