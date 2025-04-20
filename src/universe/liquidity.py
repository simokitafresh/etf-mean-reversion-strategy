"""流動性に基づくETFスクリーニング"""
import yfinance as yf
import time
import pandas as pd
from ..data.cache import DataCache

# キャッシュのインスタンス化
cache = DataCache()

def screen_liquidity(etf_base_list):
    """流動性条件に基づいてETFをスクリーニングする"""
    # キャッシュから取得を試みる
    cache_key = f"liquidity_screened_etfs_{len(etf_base_list)}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュから流動性スクリーニング結果を取得しました")
        return cached_data
    
    qualified_etfs = []
    
    # バッチ処理でAPI制限に対応
    for i in range(0, len(etf_base_list), 5):
        batch = etf_base_list[i:i+5]
        symbols = [etf['symbol'] for etf in batch]
        
        try:
            # 複数銘柄のデータを一度に取得
            data = yf.download(symbols, period="3mo", group_by="ticker", progress=False)
            
            for etf in batch:
                symbol = etf['symbol']
                
                try:
                    # 該当銘柄のデータを抽出
                    if len(symbols) > 1:
                        if symbol in data:
                            etf_data = data[symbol]
                        else:
                            print(f"データがありません: {symbol}")
                            continue
                    else:
                        etf_data = data
                    
                    # 平均出来高の計算
                    avg_volume = etf_data['Volume'].mean() if 'Volume' in etf_data else 0
                    
                    # AUMの取得（直接取得が難しいため個別にAPIを使用）
                    ticker_obj = yf.Ticker(symbol)
                    aum = ticker_obj.info.get('totalAssets', 0)
                    
                    # Bid-Askスプレッドの推定（直接取得できない場合）
                    # 日中のボラティリティをプロキシとして使用
                    if 'High' in etf_data and 'Low' in etf_data and 'Close' in etf_data:
                        volatility = (etf_data['High'] - etf_data['Low']).mean() / etf_data['Close'].mean()
                        estimated_spread = volatility * 0.1  # 簡易推定
                    else:
                        estimated_spread = 0.005  # デフォルト値
                    
                    # 設立後1年以上経過しているか確認（データ長でプロキシ）
                    history_length = len(etf_data)
                    age_in_years = history_length / 252 if history_length > 0 else 0
                    
                    # 条件を満たすか確認
                    if (avg_volume >= 100000 and 
                        aum >= 1000000000 and  # 10億USD
                        estimated_spread <= 0.001 and  # 0.10%
                        age_in_years >= 1):
                        
                        qualified_etfs.append({
                            'symbol': symbol,
                            'name': etf['name'],
                            'avg_volume': float(avg_volume),
                            'aum': float(aum) if isinstance(aum, (int, float)) else 0,
                            'estimated_spread': float(estimated_spread),
                            'age_in_years': float(age_in_years)
                        })
                        
                        print(f"適格: {symbol} (出来高: {avg_volume:.0f}, AUM: ${aum/1e9:.1f}B)")
                
                except Exception as e:
                    print(f"エラー ({symbol}): {str(e)}")
            
            # API制限対策
            time.sleep(1)
        
        except Exception as e:
            print(f"バッチダウンロードエラー: {str(e)}")
    
    # キャッシュに保存
    cache.set_json(cache_key, qualified_etfs)
    
    return qualified_etfs
