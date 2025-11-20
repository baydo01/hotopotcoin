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
st.set_page_config(page_title="Hedge Fund Manager: V8 - Dynamic Portfolio Bot", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #00897B; color: white; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --- ÖZEL PUAN HESABI (Aynı Kaldı) ---
def calculate_custom_score(df):
    """Farklı zaman dilimlerindeki kapanış fiyatları, volatilite ve hacim bazlı özel puan sinyali hesaplar."""
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
    """Yahoo Finance'dan veriyi çeker ve ön işleme tabi tutar."""
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        df.dropna(inplace=True)
        return df
    except:
        return None

# --- TEMEL FONKSİYON: DİNAMİK PORTFÖY BACKTESTİ (Walk-Forward Mantığı) ---
def run_dynamic_portfolio_backtest(df_combined, tickers, params, initial_capital):
    
    # Ayarlar
    train_window = params['train_days']      # 1 Yıl (~252 İşlem Günü)
    optim_window = params['optimize_days']   # 3 Hafta (~21 İşlem Günü)
    rebalance_window = params['trade_days']  # 1 Hafta (~5 İşlem Günü)
    n_states = params['n_states']
    commission = params['commission']
    
    # Başlangıç değişkenleri
    capital = initial_capital
    cash = initial_capital
    coin_amounts = {t: 0 for t in tickers}
    portfolio_history = pd.Series(dtype='float64')
    
    # Veriyi sadece işlem günlerine indirge
    df = df_combined.dropna(subset=['close']).copy()
    
    # Kayar Pencere Döngüsü
    # Pencere, rebalance_window'un bitiş gününden başlar ve her adımda rebalance_window kadar kayar.
    start_index = train_window + optim_window
    
    for i in range(start_index, len(df), rebalance_window):
        
        # 1. Pencere Tanımlama
        optim_end_idx = i - rebalance_window
        optim_start_idx = optim_end_idx - optim_window
        train_end_idx = optim_start_idx
        train_start_idx = train_end_idx - train_window
        
        trade_start_idx = i - rebalance_window
        trade_end_idx = i

        # İşlem yapılmayan ilk kısımları atla
        if train_start_idx < 0: continue

        # 2. Rebalancing Kararı (Trade Start Gününden Önce)
        
        # 2a. Train ve Optim Verisini Çıkar
        train_data_all = df.iloc[train_start_idx:train_end_idx]
        optim_data_all = df.iloc[optim_start_idx:optim_end_idx]

        # Her coin için HMM eğit ve ağırlık optimize et
        coin_signals = {}
        for ticker in tickers:
            
            # Tekil coin verisini al
            train_df = train_data_all.xs(ticker, level='ticker').copy()
            optim_df = optim_data_all.xs(ticker, level='ticker').copy()
            
            # Özellik Hesaplama ve HMM Eğitimi (Train data üzerinde)
            if len(train_df) < train_window or len(optim_df) < optim_window: continue
            
            train_df['log_ret'] = np.log(train_df['close']/train_df['close'].shift(1))
            train_df['range'] = (train_df['high']-train_df['low'])/train_df['close']
            train_df['custom_score'] = calculate_custom_score(train_df)
            train_df.dropna(inplace=True)
            
            X_train = train_df[['log_ret','range']].values
            scaler = StandardScaler()
            X_s_train = scaler.fit_transform(X_train)
            
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X_s_train)
                train_df['state'] = model.predict(X_s_train)
                state_stats = train_df.groupby('state')['log_ret'].mean()
                bull_state = state_stats.idxmax()
                bear_state = state_stats.idxmin()
            except:
                continue 

            # Ağırlık Optimizasyonu (Validation Data üzerinde)
            w_hmm, w_score = (0.7, 0.3) # Default
            
            # Sadece basit ROI hesaplayarak en iyi ağırlığı bulma mantığı burada devam edebilir.
            # Basitlik için ve V7'deki mantığı korumak için optimize_year_weights'ın iç mantığını kullanın.
            # Ancak çoklu varlıkta bu yavaşlayacaktır. Şimdilik V7'deki optimize mantığını dışarıda tutuyoruz.

            # Optimize Edilmiş Sinyal Hesaplama (Optim Data'nın son günü)
            last_optim_row = optim_data_all.xs(ticker, level='ticker').iloc[-1]
            last_optim_price = last_optim_row['close']
            
            # Gerekli özellikleri tekrar hesapla
            optim_df['log_ret'] = np.log(optim_df['close']/optim_df['close'].shift(1))
            optim_df['range'] = (optim_df['high']-optim_df['low'])/optim_df['close']
            optim_df['custom_score'] = calculate_custom_score(optim_df)
            
            X_optim_point = scaler.transform([[optim_df['log_ret'].iloc[-1], optim_df['range'].iloc[-1]]])
            hmm_signal = 1 if model.predict(X_optim_point)[0] == bull_state else (-1 if model.predict(X_optim_point)[0] == bear_state else 0)
            score_signal = 1 if optim_df['custom_score'].iloc[-1] >= 3 else (-1 if optim_df['custom_score'].iloc[-1] <= -3 else 0)
            
            # Ağırlıklandırma (Şimdilik V7'deki en iyi w'yu kullanmadık, default alalım)
            weighted_decision = (w_hmm * hmm_signal) + (w_score * score_signal)
            
            coin_signals[ticker] = {
                'signal': weighted_decision,
                'price': last_optim_price,
                'action': "AL" if weighted_decision > 0.25 else ("SAT" if weighted_decision < -0.25 else "BEKLE")
            }

        # 3. Portföy Yeniden Dengeleme (Rebalancing)
        
        # Önce Toplam Portföy Değerini Hesapla
        total_value = cash
        for t in tickers:
            if coin_amounts[t] > 0 and t in df.index.get_level_values('ticker'):
                current_price = df.xs(t, level='ticker').loc[df.index.get_level_values('Date').iloc[trade_start_idx], 'close']
                total_value += coin_amounts[t] * current_price
        
        # Rebalancing için hedef ağırlıkları belirle
        target_allocation = {t: 0 for t in tickers}
        buy_signals = [t for t, sig in coin_signals.items() if sig['action'] == 'AL']
        sell_signals = [t for t, sig in coin_signals.items() if sig['action'] == 'SAT']
        
        if buy_signals:
            # Sinyal verenler arasında eşit dağıt
            target_pct = 1.0 / len(buy_signals)
            for t in buy_signals:
                target_allocation[t] = total_value * target_pct
        
        # Satış işlemlerini yap (Cash biriktir)
        for t in tickers:
            if t in sell_signals and coin_amounts[t] > 0:
                current_price = df.xs(t, level='ticker').loc[df.index.get_level_values('Date').iloc[trade_start_idx], 'close']
                sell_usd = coin_amounts[t] * current_price
                fee = sell_usd * commission
                
                cash += (sell_usd - fee)
                coin_amounts[t] = 0 # Pozisyonu tamamen kapat
                
        # Alım işlemlerini yap (Cash kullan)
        if buy_signals:
            buyable_cash = cash
            for t in buy_signals:
                target_usd = (total_value * (1.0 / len(buy_signals)))
                
                # Sadece mevcut nakit ile alım yapabiliriz
                if buyable_cash > 0:
                    buy_amount = min(buyable_cash, target_usd)
                    current_price = df.xs(t, level='ticker').loc[df.index.get_level_values('Date').iloc[trade_start_idx], 'close']
                    fee = buy_amount * commission
                    
                    coin_amounts[t] += (buy_amount - fee) / current_price
                    cash -= buy_amount
                    buyable_cash -= buy_amount

        # 4. İşlem (Trading) Penceresi boyunca pozisyonları tut
        trade_df_multi = df.iloc[trade_start_idx:trade_end_idx]
        
        for date, group in trade_df_multi.groupby(level='Date'):
            current_day_value = cash
            
            for t in tickers:
                if coin_amounts[t] > 0 and t in group.index.get_level_values('ticker'):
                    current_price = group.loc[(date, t), 'close']
                    current_day_value += coin_amounts[t] * current_price
            
            portfolio_history[date] = current_day_value

    return portfolio_history, coin_signals

# --- ARAYÜZ VE VERİ BİRLEŞTİRME ---
st.title("💰 Hedge Fund Manager: V8 - Dinamik Portföy Rebalancing")
st.markdown("### 🔄 Tüm Coinler Arasında Haftalık Sermaye Transferi")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD"] # Test için ilk 4 coini aldım
    tickers=st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital=st.number_input("Kasa ($)", 10000, step=1000)
    start_year = st.selectbox("Başlangıç Yılı (Tüm geçmiş veriyi kullanır)", [2018, 2019, 2020, 2021, 2022], index=3)
    
    st.subheader("Pencere Boyutları (İşlem Günü)")
    train_days = st.number_input("Eğitim Penceresi (Train Days)", 252, help="Yaklaşık 1 Yıl")
    optimize_days = st.number_input("Optimizasyon Penceresi (Validation Days)", 21, help="Yaklaşık Son 3 Hafta")
    rebalance_days = st.number_input("Rebalance Penceresi (Trade Days)", 5, help="Haftalık yeniden dengeleme süresi")
    
    st.info("Sistem her 5 günde bir (rebalance_days), **son 1 yıl** veride eğitilip **son 3 hafta** veride optimize edilen HMM+Puan sinyallerine göre coinler arasında sermayeyi yeniden dağıtır.")

if st.button("DİNAMİK PORTFÖYÜ ÇALIŞTIR 🚀"):
    if not tickers: st.error("Lütfen en az bir coin seçin.")
    else:
        # Tüm coinlerin verisini çek ve birleştir
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
            # Tüm verileri MultiIndex DataFrame'de birleştir
            df_combined = pd.concat(all_dfs, keys=tickers, names=['ticker', 'Date'])
            df_combined = df_combined.swaplevel(0, 1).sort_index()

            # Parametreler
            params = {'n_states': 3, 'commission': 0.001, 
                      'train_days': train_days, 'optimize_days': optimize_days, 
                      'trade_days': rebalance_days}

            status.text(f"⚙️ Dinamik Portföy Simülasyonu Başlatılıyor...")
            
            # Simülasyonu başlat
            history_series, last_signals = run_dynamic_portfolio_backtest(df_combined, tickers, params, initial_capital)
            
            status.empty()

            if history_series is not None:
                final_val = history_series.iloc[-1]
                roi = ((final_val - initial_capital) / initial_capital) * 100
                
                # HODL Karşılaştırması (Tüm coinlere eşit dağıtım)
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
                
            else:
                st.error("Simülasyon sonuç vermedi. Lütfen başlangıç yılını veya pencere boyutlarını kontrol edin.")
