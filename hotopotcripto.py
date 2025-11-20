import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

# Hataları gizle
warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager: Multi-Year V6", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #6200EA; color: white; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --- ÖZEL PUAN HESABI ---
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
    # Puan: -7 (en ayı) ile +7 (en boğa) arasında değişir
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

# --- YIL AĞIRLIK OPTİMİZASYONU ---
def optimize_year_weights(df_raw, params, alloc_capital, test_days=21):
    """
    Verinin son 21 günlük (3 haftalık) bölümünü kullanarak
    en iyi HMM/Puan ağırlık kombinasyonunu (w_hmm, w_score) belirler.
    Bu, her yılın piyasa dinamiğine en uygun sinyal karışımını bulur.
    """
    df = df_raw.copy()
    
    # Gerekli kolonları hesapla
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    
    # Model eğitimi için yeterli veri yoksa default ağırlık döner
    if len(df) < test_days + 5: return (0.7, 0.3)  # default: HMM %70, Puan %30

    # Test (Validation) ve Train setlerini ayır
    test_data = df.iloc[-test_days:]
    train_data = df.iloc[:-test_days]

    # HMM Modelini Train datası üzerinde eğit
    X = train_data[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GaussianHMM(n_components=params['n_states'], covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_s)
    
    # Boğa/Ayı rejimlerini belirle
    train_data['state'] = model.predict(X_s)
    state_stats = train_data.groupby('state')['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()

    # Denenecek ağırlık senaryoları (HMM ağırlığı)
    weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
    best_roi = -999
    best_w = None
    
    # Test setinde en iyi performansı veren ağırlığı bul
    for w_hmm in weight_scenarios:
        w_score = 1.0 - w_hmm
        cash = alloc_capital
        coin_amt = 0
        
        # Test (Validation) seti üzerinde simülasyon
        for idx,row in test_data.iterrows():
            # HMM tahminini (State) Test datası için yap
            X_test_point = scaler.transform([[row['log_ret'], row['range']]])
            hmm_signal = 1 if model.predict(X_test_point)[0] == bull_state else (-1 if model.predict(X_test_point)[0] == bear_state else 0)
            
            # Puan sinyalini al
            score_signal = 1 if row['custom_score'] >= 3 else (-1 if row['custom_score'] <= -3 else 0)
            
            # Ağırlıklı karar
            weighted_decision = (w_hmm * hmm_signal) + (w_score * score_signal)
            price = row['close']
            
            # Basit al-sat simülasyonu (komisyonsuz)
            if weighted_decision > 0.25: # Güçlü AL
                coin_amt = cash / price
                cash = 0
            elif weighted_decision < -0.25: # Güçlü SAT
                cash = coin_amt * price
                coin_amt = 0
        
        final_val = cash + coin_amt * test_data['close'].iloc[-1]
        roi = (final_val - alloc_capital) / alloc_capital
        
        # En iyi ROI veren ağırlığı seç
        if roi > best_roi:
            best_roi = roi
            best_w = (w_hmm, w_score)
            
    # Eğer hiçbir senaryo ROI üretmezse (best_w None kalırsa) default döndür
    return best_w if best_w is not None else (0.7, 0.3) 

# --- MULTI-TIMEFRAME TURNUVA ---
def run_multi_timeframe_tournament(df_raw, params, alloc_capital):
    """Farklı zaman dilimlerinde (Günlük, Haftalık, Aylık) HMM+Puan stratejisini uygular."""
    try:
        n_states = params['n_states']
        commission = params['commission']
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        best_roi = -999
        best_portfolio = []
        best_config = {}
        
        # Yıl bazlı en iyi ağırlığı bul (Validation)
        w_hmm, w_score = optimize_year_weights(df_raw, params, alloc_capital)

        for tf_name, tf_code in timeframes.items():
            # Zaman dilimi resampling
            if tf_code == 'D':
                df = df_raw.copy()
            else:
                agg_dict = {'close':'last','high':'max','low':'min'}
                if 'open' in df_raw.columns: agg_dict['open']='first'
                if 'volume' in df_raw.columns: agg_dict['volume']='sum'
                df = df_raw.resample(tf_code).agg(agg_dict).dropna()
            if len(df) < 5: continue

            # Özellik (Feature) hesaplamaları
            df['log_ret'] = np.log(df['close']/df['close'].shift(1))
            df['range'] = (df['high']-df['low'])/df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            if len(df) < 5: continue

            # HMM Modelini eğit
            X = df[['log_ret','range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
            model.fit(X_s)
            df['state'] = model.predict(X_s)
            
            # Boğa/Ayı rejimlerini belirle
            state_stats = df.groupby('state')['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.idxmin()

            cash = alloc_capital
            coin_amt = 0
            temp_portfolio = []
            temp_history = {}

            # Stratejiyi uygula (tüm data üzerinde)
            for idx,row in df.iterrows():
                price=row['close']
                state=row['state']
                score=row['custom_score']
                
                hmm_signal = 1 if state==bull_state else (-1 if state==bear_state else 0)
                score_signal = 1 if score>=3 else (-1 if score<=-3 else 0)
                
                # OPTİMİZE EDİLMİŞ AĞIRLIK KULLANILARAK KARAR
                weighted_decision = (w_hmm*hmm_signal) + (w_score*score_signal)

                target_pct=0.0
                action_text="BEKLE"
                if weighted_decision>0.25: target_pct=1.0; action_text="AL"
                elif weighted_decision<-0.25: target_pct=0.0; action_text="SAT"

                # Portföy Yeniden Dengeleme (Komisyon dahil)
                current_val=cash + coin_amt*price
                if current_val<=0: temp_portfolio.append(0); continue
                current_pct=(coin_amt*price)/current_val
                if abs(target_pct-current_pct)>0.05:
                    diff_usd=(target_pct-current_pct)*current_val
                    fee=abs(diff_usd)*commission
                    if diff_usd>0: # ALIYORUZ
                        if cash>=diff_usd: coin_amt+=(diff_usd-fee)/price; cash-=diff_usd
                    else: # SATIYORUZ
                        sell_usd=abs(diff_usd)
                        if (coin_amt*price)>=sell_usd: coin_amt-=sell_usd/price; cash+=(sell_usd-fee)
                
                val=cash+coin_amt*price
                temp_portfolio.append(val)

                if idx==df.index[-1]:
                    regime_label="BOĞA" if hmm_signal==1 else ("AYI" if hmm_signal==-1 else "YATAY")
                    temp_history={"Fiyat":price,"HMM":regime_label,"Puan":int(score),
                                  "Öneri":action_text,"Zaman":tf_name,
                                  "Ağırlık":f"%{int(w_hmm*100)} HMM / %{int(w_score*100)} Puan"}

            # En iyi zaman dilimi performansı
            if len(temp_portfolio)>0:
                final_bal=temp_portfolio[-1]
                roi=(final_bal-alloc_capital)/alloc_capital
                if roi>best_roi:
                    best_roi=roi
                    best_portfolio=pd.Series(temp_portfolio,index=df.index)
                    best_config=temp_history
        return best_portfolio,best_config
    except Exception as e:
        # print(f"Hata oluştu: {e}") # Debug amaçlı
        return None,None

# --- ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Multi-Year V6")
st.markdown("### ⚔️ Yıllara Göre Ağırlık Optimizasyonu ve Tek Tablo Çıktı")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
    tickers=st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital=st.number_input("Kasa ($)",10000)
    years=[2018,2019,2020,2021,2022,2023,2024,2025] # 2021 EKLENDİ
    selected_years=st.multiselect("Başlangıç Yılları",years,default=years)
    st.info("Sistem her coin için seçilen yılları test eder, **son 3 haftayı validation olarak kullanır ve en iyi HMM/Puan ağırlığını otomatik seçer.**")

if st.button("BÜYÜK TURNUVAYI BAŞLAT 🚀"):
    if not tickers: st.error("Coin seçmelisin.")
    else:
        results_list=[]
        bar=st.progress(0)
        status=st.empty()
        params={'n_states':3,'commission':0.001}
        total_tasks=len(tickers)*len(selected_years)
        task_count=0
        
        for year in selected_years:
            start_date=f"{year}-01-01"
            for ticker in tickers:
                task_count+=1
                status.text(f"⚙️ {ticker} - {start_date} test ediliyor ({task_count}/{total_tasks})")
                
                # Veriyi çek
                df=get_data_cached(ticker,start_date)
                
                if df is not None:
                    # Multi-Timeframe Turnuvasını başlat
                    res_series,best_conf=run_multi_timeframe_tournament(df,params,initial_capital/len(tickers))
                    
                    if res_series is not None:
                        # Sonuçları hesapla
                        final_val=res_series.iloc[-1]
                        start_price=df['close'].iloc[0]
                        end_price=df['close'].iloc[-1]
                        hodl_val=(initial_capital/len(tickers)/start_price)*end_price
                        alpha=final_val-hodl_val
                        
                        if best_conf:
                            best_conf.update({"Coin":ticker,"Başlangıç Tarihi":start_date,
                                              "Bakiye":final_val,"HODL":hodl_val,
                                              "Alpha":alpha,"ROI":((final_val-initial_capital/len(tickers))/(initial_capital/len(tickers))*100)})
                            results_list.append(best_conf)
                
                bar.progress(task_count/total_tasks)
        
        status.empty()

        if results_list:
            df_res=pd.DataFrame(results_list)
            # Sonuç tablosu
            cols=['Coin','Başlangıç Tarihi','Fiyat','Öneri','Zaman','Ağırlık','HMM','Puan','Bakiye','HODL','Alpha','ROI']
            st.dataframe(df_res[cols].sort_values(by=['Başlangıç Tarihi','Coin']),height=600)
        else:
            st.error("Veri alınamadı veya hesaplanamadı.")
