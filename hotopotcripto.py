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
st.set_page_config(page_title="Hedge Fund Manager V6 (Tournament + Weighted)", layout="wide", initial_sidebar_state="expanded")

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
    7'li Puanlama Sistemi (değerler -7 ... +7)
    Puanların kaynağı açıkça satırlarda tutulur ve sonuç raporunda gösterilir.
    """
    # Veri yeterli mi kontrolü
    if len(df) < 366:
        return pd.Series(0, index=df.index), pd.DataFrame(0, index=df.index, columns=['s1','s2','s3','s4','s5','s6','s7'])

    # 1. Kısa Vade (Son 5 Mum)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    # 2. Orta Vade (Son 35 Mum)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    # 3. Uzun Vade (Son 150 Mum)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    # 4. Makro Vade (Son 365 Mum)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    # 5. Volatilite Yönü (son 5'lik oynaklık)
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    # 6. Hacim Trendi
    if 'volume' in df.columns:
        s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1)
    else:
        s6 = np.zeros(len(df), dtype=int)
    # 7. Mum Yapısı
    if 'open' in df.columns:
        s7 = np.where(df['close'] > df['open'], 1, -1)
    else:
        s7 = np.zeros(len(df), dtype=int)

    scores_df = pd.DataFrame({
        's1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6, 's7': s7
    }, index=df.index)

    total_score = scores_df.sum(axis=1)
    return total_score, scores_df


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

        # En az 2 yıllık veri olsun
        if len(df) < 730: return None

        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.warning(f"get_data_cached hata: {e}")
        return None


# --- V6: Çoklu Zaman Dilimi + Ağırlıklı Model Testleri + Puan Kaynağı Gösterimi ---

def run_multi_timeframe_tournament_v6(df_raw, params, alloc_capital, weight_scenarios, decision_threshold=0.25):
    try:
        n_states = params.get('n_states', 3)
        commission = params.get('commission', 0.001)

        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}

        best_roi = -np.inf
        best_portfolio = None
        best_config = None
        best_details = None

        for tf_name, tf_code in timeframes.items():
            # Resample
            if tf_code == 'D':
                df = df_raw.copy()
            else:
                agg_dict = {'close': 'last', 'high': 'max', 'low': 'min'}
                if 'open' in df_raw.columns: agg_dict['open'] = 'first'
                if 'volume' in df_raw.columns: agg_dict['volume'] = 'sum'
                df = df_raw.resample(tf_code).agg(agg_dict).dropna()

            if len(df) < 200: continue

            # Feature engineering + score
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close']
            total_score, scores_breakdown = calculate_custom_score(df)
            df['custom_score'] = total_score
            # attach breakdown for later reporting
            for c in scores_breakdown.columns:
                df[c] = scores_breakdown[c]

            df.dropna(inplace=True)
            if len(df) < 50: continue

            # HMM
            try:
                X = df[['log_ret', 'range']].values
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)
                model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, random_state=42)
                model.fit(X_s)
                df['state'] = model.predict(X_s)
            except Exception as e:
                st.warning(f"HMM eğitiminde hata ({tf_name}): {e}")
                continue

            # identify bull/bear states
            try:
                state_stats = df.groupby('state')['log_ret'].mean()
                bull_state = state_stats.idxmax()
                bear_state = state_stats.idxmin()
            except Exception as e:
                st.warning(f"State analiz hatası: {e}")
                continue

            # Test weight scenarios
            for w_hmm in weight_scenarios:
                w_score = 1.0 - w_hmm

                cash = alloc_capital
                coin_amt = 0.0
                port_vals = []

                # record last-day details (for UI)
                last_detail = None

                for idx, row in df.iterrows():
                    price = row['close']
                    state = row['state']
                    score = int(row['custom_score'])

                    # hmm_signal: +1 bull, -1 bear, 0 neutral
                    hmm_signal = 0
                    if state == bull_state: hmm_signal = 1
                    elif state == bear_state: hmm_signal = -1

                    # score_signal based on thresholds (user can tune)
                    score_signal = 0
                    if score >= 3: score_signal = 1
                    elif score <= -3: score_signal = -1

                    # Weighted final signal (note: smaller weight = score model)
                    weighted_decision = (w_hmm * hmm_signal) + (w_score * score_signal)

                    # mapping to target allocations
                    target_pct = 0.0
                    action_text = 'BEKLE'
                    if weighted_decision > decision_threshold:
                        target_pct = 1.0
                        action_text = 'AL'
                    elif weighted_decision < -decision_threshold:
                        target_pct = 0.0
                        action_text = 'SAT'

                    current_val = cash + (coin_amt * price)
                    if current_val <= 0:
                        port_vals.append(0)
                        continue
                    current_pct = (coin_amt * price) / current_val if current_val > 0 else 0

                    # execution with commission
                    if abs(target_pct - current_pct) > 0.05:
                        diff_usd = (target_pct - current_pct) * current_val
                        fee = abs(diff_usd) * commission
                        if diff_usd > 0:
                            # buy
                            buy_amount = min(diff_usd, cash)
                            coin_amt += (buy_amount - buy_amount * commission) / price
                            cash -= buy_amount
                        else:
                            # sell
                            sell_usd = abs(diff_usd)
                            sell_amount = min(sell_usd, coin_amt * price)
                            coin_amt -= sell_amount / price
                            cash += (sell_amount - sell_amount * commission)

                    port_vals.append(cash + coin_amt * price)

                    # fill last_detail with rich info for UI
                    last_detail = {
                        'Tarih': idx,
                        'Fiyat': price,
                        'HMM_state': state,
                        'HMM_signal': hmm_signal,
                        'Score': score,
                        'Score_breakdown': {c: int(row[c]) for c in scores_breakdown.columns},
                        'WeightedDecision': weighted_decision,
                        'HMM_weight': w_hmm,
                        'Score_weight': w_score,
                        'Öneri': action_text,
                        'ZamanDilimi': tf_name
                    }

                if len(port_vals) == 0:
                    continue

                final_bal = port_vals[-1]
                roi = (final_bal - alloc_capital) / alloc_capital

                if roi > best_roi:
                    best_roi = roi
                    best_portfolio = pd.Series(port_vals, index=df.index)
                    best_config = {
                        'ZamanDilimi': tf_name,
                        'HMM_weight': w_hmm,
                        'Score_weight': w_score,
                        'FinalBalance': final_bal,
                        'ROI': roi
                    }
                    best_details = last_detail

        return best_portfolio, best_config, best_details

    except Exception as e:
        st.warning(f"run_multi_timeframe_tournament_v6 hata: {e}")
        return None, None, None


# --- 3. ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Timeframe Tournament (V6)")
st.markdown("### ⚔️ Günlük vs Haftalık vs Aylık | Ağırlıklı Model Testleri (V6)")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Kasa ($)", 10000)
    st.markdown("---")
    st.subheader("Ağırlık Senaryoları (HMM ağırlığı)")
    # Kullanıcıya hazır seçimler ver
    preset_weights = st.multiselect("Hazır Senaryolar", ["50/50","70/30","85/15","90/10","95/5"], default=["50/50","70/30","85/15","90/10","95/5"])
    # map to numeric
    weight_map = {"50/50":0.50, "70/30":0.70, "85/15":0.85, "90/10":0.90, "95/5":0.95}
    weight_scenarios = [weight_map[w] for w in preset_weights]

    st.markdown("---")
    st.subheader("Algoritma Ayarları")
    n_states = st.slider("HMM state sayısı", 2, 6, 3)
    commission = st.number_input("Komisyon (oran)", 0.0, 0.01, 0.001, step=0.0005)
    decision_threshold = st.slider("Karar eşiği (weighted) (küçük->daha agresif)", 0.05, 0.5, 0.25)
    st.info("Sistem her coin için Günlük/Haftalık/Aylık verileri test eder, her senaryonun son değerini raporlar ve en yüksek ROI veren kombinasyonu seçer.")

if st.button("BÜYÜK TURNUVAYI BAŞLAT (V6) 🚀"):
    if not tickers:
        st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        results_list = []
        total_balance = 0.0
        total_hodl_balance = 0.0

        bar = st.progress(0)
        status = st.empty()

        params = {'n_states': n_states, 'commission': commission}

        for i, ticker in enumerate(tickers):
            status.text(f"Turnuva Oynanıyor: {ticker}...")
            df = get_data_cached(ticker, "2016-01-01")
            if df is None:
                status.text(f"{ticker} için veri alınamadı. Geçiliyor...")
                bar.progress((i+1)/len(tickers))
                continue

            best_series, best_conf, best_details = run_multi_timeframe_tournament_v6(df, params, capital_per_coin, weight_scenarios, decision_threshold)

            if best_series is not None and best_conf is not None:
                final_val = best_conf['FinalBalance']
                total_balance += final_val

                # HODL hesapla
                start_price = df['close'].iloc[0]
                end_price = df['close'].iloc[-1]
                hodl_val = (capital_per_coin / start_price) * end_price
                total_hodl_balance += hodl_val

                # prepare result entry
                entry = {
                    'Coin': ticker,
                    'Fiyat': best_details['Fiyat'] if best_details is not None else df['close'].iloc[-1],
                    'Öneri': best_details['Öneri'] if best_details is not None else 'BEKLE',
                    'Zaman': best_conf['ZamanDilimi'],
                    'Ağırlık': f"%{int(best_conf['HMM_weight']*100)} HMM / %{int(best_conf['Score_weight']*100)} Puan",
                    'HMM': best_details['HMM_state'] if best_details is not None else None,
                    'Puan': best_details['Score'] if best_details is not None else None,
                    'ROI': best_conf['ROI']*100,
                    'Details': best_details
                }
                results_list.append(entry)

            bar.progress((i+1)/len(tickers))

        status.empty()

        if results_list:
            roi_total = ((total_balance - initial_capital) / initial_capital) * 100
            alpha = total_balance - total_hodl_balance

            c1, c2, c3 = st.columns(3)
            c1.metric("Turnuva Şampiyonu Bakiye", f"${total_balance:,.0f}", f"%{roi_total:.1f}")
            c2.metric("HODL Değeri", f"${total_hodl_balance:,.0f}")
            c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")

            st.markdown("### 🏆 ŞAMPİYONLAR LİGİ VE KARARLAR (V6)")
            st.info("Her coin için en iyi çalışan 'Zaman Dilimi' ve 'Strateji Ağırlığı' aşağıdadır. 'Details' sütununda puan kırılımı ve son gün bilgileri bulunur.")

            df_res = pd.DataFrame(results_list)

            def highlight_decision(val):
                val_str = str(val)
                if val_str == 'AL': return 'background-color: #00c853; color: white; font-weight: bold'
                if 'SAT' in val_str: return 'background-color: #d50000; color: white; font-weight: bold'
                return 'background-color: #ffd600; color: black'

            cols = ['Coin', 'Fiyat', 'Öneri', 'Zaman', 'Ağırlık', 'HMM', 'Puan', 'ROI']
            st.dataframe(df_res[cols].style.applymap(highlight_decision, subset=['Öneri']).format({
                'Fiyat': '${:,.2f}', 'ROI': '{:.1f}%'
            }))

            # Ayrıntılı detay gösterimi için seçim kutusu
            st.markdown('---')
            st.subheader('Detaylı İnceleme (Seçili Coin)')
            sel = st.selectbox('Detay için coin seçin', [r['Coin'] for r in results_list])
            detail = next((r for r in results_list if r['Coin'] == sel), None)
            if detail:
                st.write('Son Gün Detayları:')
                st.json(detail['Details'])

        else:
            st.error('Veri alınamadı veya hesaplanamadı.')
