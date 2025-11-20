import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# hmmlearn ve sklearn uyarılarını ve versiyon farklarını yönetmek için
import warnings
warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager V6 (Time Travel)", layout="wide", initial_sidebar_state="expanded")

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
    if len(df) < 50: # Daha esnek olması için düşürüldü, ancak mantık aynı
        return pd.Series(0, index=df.index)

    # 1. Kısa Vade (Son 5 Mum)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    
    # 2. Orta Vade (Son 35 Mum)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    
    # 3. Uzun Vade (Son 150 Mum) - Veri kısaysa 0 döner
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    
    # 4. Makro Vade (Son 365 Mum) - Veri kısaysa 0 döner
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    
    # 5. Volatilite Yönü
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    
    # 6. Hacim Trendi
    if 'volume' in df.columns:
        s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1)
    else:
        s6 = 0
    
    # 7. Mum Yapısı
    if 'open' in df.columns:
        s7 = np.where(df['close'] > df['open'], 1, -1)
    else:
        s7 = 0
    
    # Toplam Skor (NaN değerleri 0 kabul et)
    total_score = pd.Series(s1 + s2 + s3 + s4 + s5 + s6 + s7).fillna(0)
    return total_score

# --- 1. VERİ ÇEKME ---
@st.cache_data(ttl=21600) 
def get_data_cached(ticker, start_date):
    try:
        # İndirme işlemi
        df = yf.download(ticker, start=start_date, progress=False)
        
        if df.empty: return None

        # MultiIndex sütun düzeltmesi (yeni yfinance sürümleri için)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [c.lower() for c in df.columns]
        
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
            
        df.dropna(inplace=True)
        
        # Veri çok kısaysa (örneğin yeni listelenmiş coin)
        if len(df) < 100: return None 
        
        return df
    except Exception as e:
        return None

# --- 2. STRATEJİ MOTORU (ÇOKLU ZAMAN DİLİMİ + TURNUVA) ---
def run_multi_timeframe_tournament(df_raw, params, alloc_capital):
    try:
        n_states = params['n_states']
        commission = params['commission']
        
        # Test edilecek Zaman Dilimleri
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        # Ağırlık Senaryoları
        weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
        
        best_roi = -9999
        best_portfolio = None
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
            
            # HMM için yeterli veri kontrolü
            if len(df) < 50: continue
            
            # Feature Engineering
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close']
            
            # Skorlamayı burada yapıyoruz
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            
            if len(df) < 30: continue

            # HMM Eğitimi
            X = df[['log_ret', 'range']].values
            scaler = StandardScaler()
            try:
                X_s = scaler.fit_transform(X)
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
                
                # Backtest Hızı için Sinyalleri önceden hesapla
                # Loop yerine vektörize etmek daha hızlı olur ama okunaklılık için loop bırakıldı
                
                regime_label = "YATAY"
                action_text = "BEKLE"
                hmm_signal_last = 0

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
                    
                    if weighted_decision > 0.25: 
                        target_pct = 1.0
                    elif weighted_decision < -0.25:
                        target_pct = 0.0
                    
                    # İşlem
                    current_val = cash + (coin_amt * price)
                    
                    # İflas kontrolü
                    if current_val <= 0: 
                        temp_portfolio.append(0)
                        continue

                    current_pct = (coin_amt * price) / current_val
                    
                    # Rebalance (Threshold %5)
                    if abs(target_pct - current_pct) > 0.05:
                        diff_usd = (target_pct - current_pct) * current_val
                        fee = abs(diff_usd) * commission
                        
                        if diff_usd > 0: # AL
                            cost = diff_usd + fee
                            if cash >= diff_usd: # Basit nakit kontrolü (fee içinden düşülür mantığıyla)
                                # Fee'yi işlemden düşelim
                                buy_amt = (diff_usd - fee) / price
                                if buy_amt > 0:
                                    coin_amt += buy_amt
                                    cash -= diff_usd
                        else: # SAT
                            sell_usd = abs(diff_usd)
                            if (coin_amt * price) >= sell_usd * 0.99: # Toleranslı
                                coin_amt -= sell_usd / price
                                cash += (sell_usd - fee)
                    
                    val = cash + (coin_amt * price)
                    temp_portfolio.append(val)
                    
                    # Son iterasyon değerleri
                    if idx == df.index[-1]:
                        hmm_signal_last = hmm_signal
                        action_text = "AL" if target_pct > 0.5 else "SAT"
                        if target_pct == 0 and coin_amt == 0: action_text = "NAKİTTE BEKLE"
                
                if len(temp_portfolio) > 0:
                    final_bal = temp_portfolio[-1]
                    roi = (final_bal - alloc_capital) / alloc_capital
                    
                    if roi > best_roi:
                        best_roi = roi
                        best_portfolio = pd.Series(temp_portfolio, index=df.index)
                        
                        regime_label = "BOĞA" if hmm_signal_last==1 else ("AYI" if hmm_signal_last==-1 else "YATAY")
                        best_config = {
                            "Fiyat": df['close'].iloc[-1], 
                            "HMM": regime_label, 
                            "Puan": int(df['custom_score'].iloc[-1]), 
                            "Öneri": action_text, 
                            "Zaman": tf_name, 
                            "Ağırlık": f"%{int(w_hmm*100)} HMM"
                        }

        return best_portfolio, best_config

    except Exception as e:
        return None, None

# --- 3. ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Time Travel Edition (V6)")
st.markdown("### ⚔️ Strateji Turnuvası ve Yıllık Karşılaştırma")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Başlangıç Kasası ($)", 10000)
    st.info("Sistem verileri 2018'den itibaren çeker. Aşağıdaki tablo ise 'Eğer botu o yılın 1 Ocak tarihinde başlatsaydım ne olurdu?' sorusunu yanıtlar.")

if st.button("ANALİZİ BAŞLAT 🚀"):
    if not tickers:
        st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        
        results_list = []
        yearly_comparison_data = [] # Yıllık karşılaştırma için liste
        
        total_balance = 0
        total_hodl_balance = 0
        
        bar = st.progress(0)
        status = st.empty()
        
        params = {'n_states': 3, 'commission': 0.001}
        years_to_test = [2020, 2021, 2022, 2023, 2024]
        
        for i, ticker in enumerate(tickers):
            status.text(f"İşleniyor: {ticker}...")
            
            # 1. Ana Veriyi Çek (2018'den bugüne - Modelin öğrenmesi için)
            df_full = get_data_cached(ticker, "2018-01-01")
            
            if df_full is not None:
                # A) MEVCUT DURUM (Tüm veriyi kullanıp en iyi stratejiyi bulur)
                res_series, best_conf = run_multi_timeframe_tournament(df_full, params, capital_per_coin)
                
                if res_series is not None:
                    final_val = res_series.iloc[-1]
                    total_balance += final_val
                    
                    start_price = df_full['close'].iloc[0]
                    end_price = df_full['close'].iloc[-1]
                    hodl_val = (capital_per_coin / start_price) * end_price
                    total_hodl_balance += hodl_val
                    
                    if best_conf:
                        best_conf['Coin'] = ticker
                        best_conf['Bakiye'] = final_val
                        best_conf['ROI'] = ((final_val - capital_per_coin) / capital_per_coin) * 100
                        results_list.append(best_conf)
                
                # B) YILLIK KARŞILAŞTIRMA (Zaman Yolculuğu)
                coin_yearly_stats = {'Coin': ticker}
                
                for year in years_to_test:
                    start_date_str = f"{year}-01-01"
                    # Veriyi o yıldan itibaren kes (Simülasyon: O gün veriyi okumaya başlasaydık)
                    df_slice = df_full[df_full.index >= start_date_str].copy()
                    
                    # Eğer veri çok kısaysa (örn: Coin 2022'de çıktıysa, 2020 hücresi boş kalmalı)
                    if len(df_slice) > 100:
                        res_slice, _ = run_multi_timeframe_tournament(df_slice, params, capital_per_coin)
                        if res_slice is not None:
                            end_val = res_slice.iloc[-1]
                            roi_slice = ((end_val - capital_per_coin) / capital_per_coin) * 100
                            coin_yearly_stats[str(year)] = roi_slice
                        else:
                            coin_yearly_stats[str(year)] = None # Strateji çalışmadı
                    else:
                        coin_yearly_stats[str(year)] = None # Veri yok
                
                yearly_comparison_data.append(coin_yearly_stats)
            
            bar.progress((i+1)/len(tickers))
        
        status.empty()

        if results_list:
            # --- 1. GENEL SONUÇLAR ---
            roi_total = ((total_balance - initial_capital) / initial_capital) * 100
            alpha = total_balance - total_hodl_balance
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam Bakiye (Tüm Zamanlar)", f"${total_balance:,.0f}", f"%{roi_total:.1f}")
            c2.metric("HODL Değeri", f"${total_hodl_balance:,.0f}")
            c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
            
            # --- 2. ANA TABLO ---
            st.subheader("📋 Güncel Durum ve Sinyaller")
            df_res = pd.DataFrame(results_list)
            
            def highlight_decision(val):
                val_str = str(val)
                if 'AL' == val_str: return 'background-color: #00c853; color: white; font-weight: bold'
                if 'SAT' in val_str: return 'background-color: #d50000; color: white; font-weight: bold'
                return 'background-color: #ffd600; color: black'
            
            cols = ['Coin', 'Fiyat', 'Öneri', 'Zaman', 'Ağırlık', 'HMM', 'Puan', 'ROI']
            st.dataframe(df_res[cols].style.applymap(highlight_decision, subset=['Öneri']).format({
                "Fiyat": "${:,.2f}",
                "ROI": "%{:.1f}"
            }))
            
            # --- 3. YENİ ÖZELLİK: YILLIK KARŞILAŞTIRMA ---
            st.markdown("---")
            st.subheader("📅 Yıllara Göre Performans Simülasyonu (% ROI)")
            st.markdown("""
            Bu tablo, botu belirtilen yılın **1 Ocak** tarihinde çalıştırmaya başlasaydınız bugün ne kadar kâr (%) edeceğinizi gösterir.
            *Not: Veri yetersizse (örn. Coin o tarihte yoksa) değer boş (NaN) görünür.*
            """)
            
            df_yearly = pd.DataFrame(yearly_comparison_data)
            df_yearly.set_index('Coin', inplace=True)
            
            # Renklendirme Fonksiyonu
            def color_roi(val):
                if pd.isna(val): return ''
                color = '#00c853' if val > 0 else '#d50000'
                return f'color: {color}; font-weight: bold'

            st.dataframe(df_yearly.style.applymap(color_roi).format("{:.1f}%"), use_container_width=True)
            
        else:
            st.error("Veri alınamadı veya hesaplanamadı.")
