import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

# --- Hata Yönetimi ve Kütüphane Kontrolü ---
warnings.filterwarnings("ignore")
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    st.error("Lütfen 'hmmlearn' kütüphanesini kurun: pip install hmmlearn")
    st.stop()

from sklearn.preprocessing import StandardScaler

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager V8 (Stable)", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #6200EA;
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
    Basit Puanlama Sistemi
    """
    # Veri çok kısaysa (örn: 2024 başı), hesaplama hata vermesin diye dolduruyoruz
    if len(df) < 5: return pd.Series(0, index=df.index)

    # 1. Kısa Vade
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    
    # 2. Orta Vade (Veri yetiyorsa)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1) if len(df) > 35 else 0
    
    # 3. Uzun Vade
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1) if len(df) > 150 else 0
    
    # 4. Makro Vade
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1) if len(df) > 365 else 0
    
    # 5. Volatilite
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    
    # 6. Hacim
    if 'volume' in df.columns:
        s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1)
    else:
        s6 = 0
    
    # 7. Mum
    if 'open' in df.columns:
        s7 = np.where(df['close'] > df['open'], 1, -1)
    else:
        s7 = 0
    
    total_score = pd.Series(s1 + s2 + s3 + s4 + s5 + s6 + s7).fillna(0)
    return total_score

# --- 1. VERİ ÇEKME (EN SAĞLAM YÖNTEM) ---
@st.cache_data(ttl=21600) 
def get_data_cached(ticker, start_date):
    try:
        # Yfinance'in son sürüm hatalarını önlemek için auto_adjust=False deniyoruz
        df = yf.download(ticker, start=start_date, progress=False)
        
        if df.empty: return None

        # MultiIndex Sütun Düzeltmesi (yfinance güncellemesi kaynaklı sorunlar için)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Close sütunu kontrolü
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
            
        if 'close' not in df.columns: return None
        
        df.dropna(inplace=True)
        
        # Veri çok kısaysa (30 günden az veriyle analiz olmaz)
        if len(df) < 30: return None 
        
        return df
    except Exception:
        return None

# --- 2. STRATEJİ MOTORU ---
def run_multi_timeframe_tournament(df_raw, params, alloc_capital):
    """
    Bu fonksiyon, veri seti üzerinde Günlük, Haftalık ve Aylık testleri yapar.
    Veri kısa olsa bile (örn. 2024) hata vermeden en uygun zaman dilimini bulur.
    """
    try:
        n_states = params['n_states']
        commission = params['commission']
        
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
        
        best_roi = -999999
        best_portfolio = None
        best_config = {} 
        
        # --- TURNUVA DÖNGÜSÜ ---
        for tf_name, tf_code in timeframes.items():
            
            # RESAMPLE (Zaman Dilimi Dönüşümü)
            if tf_code == 'D':
                df = df_raw.copy()
            else:
                agg_dict = {'close': 'last', 'high': 'max', 'low': 'min'}
                if 'open' in df_raw.columns: agg_dict['open'] = 'first'
                if 'volume' in df_raw.columns: agg_dict['volume'] = 'sum'
                df = df_raw.resample(tf_code).agg(agg_dict).dropna()
            
            # 2024 gibi kısa yıllarda AYLIK veri çok az olur (örn 10 mum).
            # HMM algoritması 10 veri ile çalışamaz. Bu yüzden kontrol koyuyoruz.
            # Günlük ve Haftalık muhtemelen çalışacaktır.
            if len(df) < 20: 
                continue
            
            # Feature Engineering
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # HMM Eğitimi
            X = df[['log_ret', 'range']].values
            scaler = StandardScaler()
            
            try:
                X_s = scaler.fit_transform(X)
                # n_iter düşük tutularak hız sağlanır
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
            
            # Ağırlık Testleri Loop'u
            for w_hmm in weight_scenarios:
                w_score = 1.0 - w_hmm
                
                cash = alloc_capital
                coin_amt = 0
                temp_portfolio = []
                
                # Sinyal Değişkenleri
                regime_label = "YATAY"
                action_text = "BEKLE"
                hmm_signal_last = 0
                
                for idx, row in df.iterrows():
                    price = row['close']
                    state = row['state']
                    score = row['custom_score']
                    
                    hmm_signal = 0
                    if state == bull_state: hmm_signal = 1
                    elif state == bear_state: hmm_signal = -1
                    
                    score_signal = 0
                    if score >= 3: score_signal = 1
                    elif score <= -3: score_signal = -1
                    
                    weighted_decision = (w_hmm * hmm_signal) + (w_score * score_signal)
                    
                    target_pct = 0.0
                    if weighted_decision > 0.25: target_pct = 1.0
                    elif weighted_decision < -0.25: target_pct = 0.0
                    
                    # Cüzdan Değeri
                    current_val = cash + (coin_amt * price)
                    if current_val <= 0: # İflas
                        temp_portfolio.append(0)
                        continue
                        
                    current_pct = (coin_amt * price) / current_val
                    
                    # Al-Sat İşlemi (Rebalance)
                    if abs(target_pct - current_pct) > 0.05:
                        diff_usd = (target_pct - current_pct) * current_val
                        fee = abs(diff_usd) * commission
                        
                        if diff_usd > 0:
                            if cash >= diff_usd:
                                buy_amt = (diff_usd - fee) / price
                                if buy_amt > 0:
                                    coin_amt += buy_amt
                                    cash -= diff_usd
                        else:
                            sell_usd = abs(diff_usd)
                            if (coin_amt * price) >= sell_usd * 0.99:
                                coin_amt -= sell_usd / price
                                cash += (sell_usd - fee)
                    
                    val = cash + (coin_amt * price)
                    temp_portfolio.append(val)
                    
                    # Son gün bilgisi (Rapor için)
                    if idx == df.index[-1]:
                        hmm_signal_last = hmm_signal
                        action_text = "AL" if target_pct > 0.5 else ("SAT" if target_pct < 0.1 else "BEKLE")
                        if target_pct == 0 and coin_amt == 0: action_text = "NAKİTTE"
                
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

    except Exception:
        return None, None

# --- 3. ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Time Travel (Revize V8)")
st.markdown("### ⚔️ Günlük vs Haftalık vs Aylık | Yıllık Performans Testi")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Kasa ($)", 10000)

if st.button("ANALİZİ BAŞLAT 🚀"):
    if not tickers:
        st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        
        results_list = []
        yearly_data = [] # Yıllık verileri tutacak
        
        total_balance = 0
        total_hodl_balance = 0
        
        bar = st.progress(0)
        status = st.empty()
        
        params = {'n_states': 3, 'commission': 0.001}
        years_to_test = [2020, 2021, 2022, 2023, 2024]
        
        for i, ticker in enumerate(tickers):
            status.text(f"İşleniyor: {ticker}...")
            
            # 1. ANA VERİ (2018'den itibaren çekiyoruz ki HMM modeli iyi öğrensin)
            df_full = get_data_cached(ticker, "2018-01-01")
            
            if df_full is not None:
                # --- A) GÜNCEL EN İYİ STRATEJİ ---
                # Tüm veriyi kullanarak şu an ne yapmalı?
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
                
                # --- B) YILLIK PERFORMANS TESTİ ---
                # Burada stratejiyi "Eğer X yılında başlatsaydık" diye simüle ediyoruz.
                # 2024 gibi yakın yıllar için "Aylık" strateji veri yetersizliğinden çalışmazsa,
                # kod otomatik olarak "Haftalık" veya "Günlük" olana geçip sonucu getirecektir.
                coin_stats = {'Coin': ticker}
                
                for year in years_to_test:
                    start_dt = f"{year}-01-01"
                    # Sadece o tarihten sonraki veriyi al (Geleceği görme yok)
                    df_slice = df_full[df_full.index >= start_dt].copy()
                    
                    # Eğer o tarihte coin varsa ve yeterli veri oluşmuşsa
                    if len(df_slice) > 50: 
                        res_slice, _ = run_multi_timeframe_tournament(df_slice, params, capital_per_coin)
                        if res_slice is not None:
                            end_val = res_slice.iloc[-1]
                            roi_year = ((end_val - capital_per_coin) / capital_per_coin) * 100
                            coin_stats[str(year)] = roi_year
                        else:
                            coin_stats[str(year)] = None
                    else:
                        coin_stats[str(year)] = None # Veri yok
                
                yearly_data.append(coin_stats)

            else:
                # Veri çekilemediyse loglama yapmıyoruz, sadece geçiyoruz
                pass
            
            bar.progress((i+1)/len(tickers))
        
        status.empty()

        if results_list:
            # --- ÖZET METRİKLER ---
            roi_total = ((total_balance - initial_capital) / initial_capital) * 100
            alpha = total_balance - total_hodl_balance
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Şampiyon Strateji Bakiye", f"${total_balance:,.0f}", f"%{roi_total:.1f}")
            c2.metric("HODL Değeri (2018+)", f"${total_hodl_balance:,.0f}")
            c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
            
            # --- ANA TABLO ---
            st.subheader("📋 Güncel Durum ve Kararlar")
            df_res = pd.DataFrame(results_list)
            
            def highlight_decision(val):
                val_str = str(val)
                if 'AL' == val_str: return 'background-color: #00c853; color: white; font-weight: bold'
                if 'SAT' in val_str: return 'background-color: #d50000; color: white; font-weight: bold'
                if 'NAKİTTE' in val_str: return 'background-color: #6200EA; color: white; font-weight: bold'
                return 'background-color: #ffd600; color: black'
            
            cols = ['Coin', 'Fiyat', 'Öneri', 'Zaman', 'Ağırlık', 'HMM', 'Puan', 'ROI']
            st.dataframe(df_res[cols].style.applymap(highlight_decision, subset=['Öneri']).format({
                "Fiyat": "${:,.2f}",
                "ROI": "%{:.1f}"
            }))
            
            # --- YILLIK TABLO ---
            st.markdown("---")
            st.subheader("📅 Yıllara Göre Kâr Simülasyonu (% ROI)")
            st.markdown("*Eğer botu o yılın başında başlatsaydınız, bugün kâr oranınız ne olurdu?*")
            
            if yearly_data:
                df_yearly = pd.DataFrame(yearly_data)
                df_yearly.set_index('Coin', inplace=True)
                
                def color_roi(val):
                    if pd.isna(val): return 'color: grey; opacity: 0.5'
                    color = '#00c853' if val > 0 else '#d50000'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(df_yearly.style.applymap(color_roi).format("{:.1f}%"), use_container_width=True)
            
        else:
            st.error("Veriler çekilemedi. Yahoo Finance bağlantısında geçici bir sorun olabilir veya kütüphane versiyonları uyumsuzdur.")
