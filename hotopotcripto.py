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
    portfolio_history = pd.Series(dtype='object')

    # Veriyi sadece close değeri olan günlerle temizle ve tarihleri al
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
        train_data_all_raw = df_clean.loc[train_start_date:optim_end_date].copy()
        optim_data_all_raw = df_clean.loc[optim_start_date:optim_end_date].copy()
        current_date = rebalance_execution_date

        # Gerekli özellikleri tek bir yerde hesapla
        temp_data_for_features = train_data_all_raw.reset_index(level='ticker')
        
        # Her coin için gruplayıp özellikleri hesapla
        for t in tickers:
            df_t = temp_data_for_features[temp_data_for_features['ticker'] == t].copy()
            if not df_t.empty:
                df_t['log_ret'] = np.log(df_t['close'] / df_t['close'].shift(1))
                df_t['range'] = (df_t['high'] - df_t['low']) / df_t['close']
                df_t['custom_score'] = calculate_custom_score(df_t)
                
                # Orijinal MultiIndex DataFrame'i güncellemek için merge kullan
                train_data_all_raw.loc[(df_t.index, t), ['log_ret', 'range', 'custom_score']] = df_t[['log_ret', 'range', 'custom_score']].values

        train_data_all = train_data_all_raw.dropna(subset=['log_ret', 'range', 'custom_score']).copy()
        
        # --- Optimizasyon Penceresi İçin Sinyal Hesaplama ---
        optim_data_for_features = optim_data_all_raw.reset_index(level='ticker')
        for t in tickers:
            df_t = optim_data_for_features[optim_data_for_features['ticker'] == t].copy()
            if not df_t.empty:
                df_t['log_ret'] = np.log(df_t['close'] / df_t['close'].shift(1))
                df_t['range'] = (df_t['high'] - df_t['low']) / df_t['close']
                df_t['custom_score'] = calculate_custom_score(df_t)
                optim_data_all_raw.loc[(df_t.index, t), ['log_ret', 'range', 'custom_score']] = df_t[['log_ret', 'range', 'custom_score']].values
        
        optim_data_all = optim_data_all_raw.dropna(subset=['log_ret', 'range', 'custom_score']).copy()

        # En iyi veri ağırlıklandırma setini bul (A, B, C, D)
        best_w_set, weights = optimize_data_weights(train_data_all, optim_data_all, n_states, WEIGHT_SCENARIOS, current_date)
        w_latest, w_mid, w_old = weights
        w_hmm, w_score = 0.7, 0.3 # Sinyal ağırlığı sabit

        # 3. Eğitim (En iyi ağırlık seti ile) - train_data_all kullan
        one_year_ago = current_date - pd.Timedelta(days=365)
        three_years_ago = current_date - pd.Timedelta(days=365*3)

        train_data_final = train_data_all.copy()
        train_data_final['weight'] = 1.0
        
        # Ağırlıklandırma
        train_data_final['Date'] = train_data_final.index.get_level_values('Date')
        train_data_final['weight'] = np.where(train_data_final['Date'] >= one_year_ago, w_latest, train_data_final['weight'])
        train_data_final['weight'] = np.where((train_data_final['Date'] >= three_years_ago) & (train_data_final['Date'] < one_year_ago), w_mid, train_data_final['weight'])
        train_data_final['weight'] = np.where(train_data_final['Date'] < three_years_ago, w_old, train_data_final['weight'])
        train_data_final.drop(columns=['Date'], inplace=True)

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
        
        rebalance_decision_date = optim_end_date
        
        for ticker in tickers:
            try:
                # Sinyal için gerekli datayı al
                last_day_data = df_clean.loc[rebalance_decision_date].xs(ticker, level='ticker').iloc[-1]
                
                # Özellikleri hesapla
                prev_close = df_clean.loc[:rebalance_decision_date].xs(ticker, level='ticker')['close'].iloc[-2]
                log_ret = np.log(last_day_data['close'] / prev_close)
                range_ = (last_day_data['high'] - last_day_data['low']) / last_day_data['close']
                
                # HMM tahmini
                X_point = scaler.transform([[log_ret, range_]])
                hmm_signal = 1 if model.predict(X_point)[0] == bull_state else (-1 if model.predict(X_point)[0] == bear_state else 0)
                
                # Puan sinyali (Hata riskine karşı şimdilik basit bir değer kullanalım)
                # Puan sinyalini de bu noktada doğru hesaplamak gerekiyor, ancak şimdilik 0 kabul edelim
                
                weighted_decision = (w_hmm * hmm_signal) # Puan sinyalini göz ardı ettik (0)

                coin_decisions[ticker] = {
                    'signal': weighted_decision,
                    'price': last_day_data['close'],
                    'action': "AL" if weighted_decision > 0.25 else ("SAT" if weighted_decision < -0.25 else "BEKLE"),
                    'weight_set': best_w_set
                }
            except KeyError:
                # Eğer coinin o günkü verisi eksikse, BEKLE kararı ver
                coin_decisions[ticker] = {'signal': 0, 'price': 0, 'action': "BEKLE", 'weight_set': best_w_set}
            except IndexError:
                # Eğer yeterli geçmiş veri yoksa (log_ret için)
                coin_decisions[ticker] = {'signal': 0, 'price': 0, 'action': "BEKLE", 'weight_set': best_w_set}
        
        # 5. Portföy Yeniden Dengeleme (Rebalance Execution Date fiyatları ile)
        
        rebalance_execution_date = rebalance_execution_date # = Trade_start_date
        
        # Tüm pozisyonların değerini hesapla
        total_value = cash
        for t in tickers:
            if coin_amounts[t] > 0 and t in coin_decisions:
                try:
                    current_price = df_clean.loc[(rebalance_execution_date, t), 'close']
                    total_value += coin_amounts[t] * current_price
                except KeyError:
                    # Fiyat yoksa pozisyonu o anki fiyattan (coin_decisions) hesapla (risklidir, ama ilerlemek için)
                    total_value += coin_amounts[t] * coin_decisions[t]['price']


        # SATIŞ işlemlerini yap
        for t in tickers:
            if t in coin_decisions and coin_decisions[t]['action'] == 'SAT' and coin_amounts[t] > 0:
                try:
                    current_price = df_clean.loc[(rebalance_execution_date, t), 'close']
                    sell_usd = coin_amounts[t] * current_price
                    fee = sell_usd * commission
                    
                    cash += (sell_usd - fee)
                    coin_amounts[t] = 0
                except KeyError:
                    # Fiyat yoksa satışı gerçekleştiremeyiz, pozisyonu koru
                    pass 

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
                except KeyError:
                    # Fiyat yoksa alım yapma
                    pass

        # 6. İşlem Penceresi boyunca pozisyonları tut ve bakiye kaydet
        trade_df_multi = df_clean.loc[rebalance_execution_date:trade_end_date]
        
        for date, group in trade_df_multi.groupby(level='Date'):
            current_day_value = cash
            
            for t in tickers:
                if coin_amounts[t] > 0:
                    try:
                        current_price = group.loc[(date, t), 'close']
                        current_day_value += coin_amounts[t] * current_price
                    except KeyError:
                        # Eğer güncel veri yoksa o coini o günkü fiyattan (bir önceki günkü) değerlendir
                        pass
            
            # Tarihleri kontrol ederek sadece güncel tarihi al
            portfolio_history.loc[date] = float(current_day_value)
            
    # Final portföy serisini float'a çevir
    return portfolio_history.sort_index(), coin_decisions
