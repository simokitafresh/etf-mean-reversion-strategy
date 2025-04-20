# src/universe/correlation.py
"""相関に基づくETFフィルタリング"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
from ..data.cache import DataCache

# キャッシュのインスタンス化
cache = DataCache()

def correlation_filtering(liquid_etfs, target_count=50, correlation_threshold=0.98):
    """相関に基づいてETFをフィルタリングする
    
    Args:
        liquid_etfs (list): 流動性スクリーニングを通過したETFのリスト
        target_count (int): 目標ETF数
        correlation_threshold (float): 相関係数の閾値
    
    Returns:
        list: フィルタリング後のETFリスト
    """
    # キャッシュから取得を試みる
    cache_key = f"correlation_filtered_etfs_{len(liquid_etfs)}_{correlation_threshold}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュから相関フィルタリング結果を取得しました")
        return cached_data
    
    print(f"相関フィルタリングを開始します（閾値: {correlation_threshold}）...")
    
    symbols = [etf['symbol'] for etf in liquid_etfs]
    
    # 直近1年間の日次リターンを取得
    prices_df = pd.DataFrame()
    
    # 10銘柄ずつバッチ処理
    for i in range(0, len(symbols), 10):
        batch_symbols = symbols[i:i+10]
        try:
            print(f"価格データ取得中: {batch_symbols}")
            batch_data = yf.download(
                batch_symbols, 
                period="1y", 
                interval="1d",
                group_by="ticker",
                progress=False
            )['Adj Close']
            
            # 単一銘柄の場合は構造が異なる
            if len(batch_symbols) == 1:
                prices_df[batch_symbols[0]] = batch_data
            else:
                # 複数銘柄の場合
                for symbol in batch_symbols:
                    if symbol in batch_data:
                        prices_df[symbol] = batch_data[symbol]
            
            time.sleep(1)  # API制限対策
        except Exception as e:
            print(f"エラー - 相関データバッチ {i}: {str(e)}")
    
    # 欠損値を前方補完
    prices_df.fillna(method='ffill', inplace=True)
    
    # 80%以上のデータがある銘柄のみ保持
    min_data_points = 0.8 * len(prices_df)
    valid_columns = prices_df.columns[prices_df.count() >= min_data_points]
    prices_df = prices_df[valid_columns]
    
    print(f"有効なデータを持つ銘柄: {len(valid_columns)}/{len(symbols)}")
    
    # 日次リターンに変換
    returns_df = prices_df.pct_change().dropna()
    
    # 相関行列の計算
    corr_matrix = returns_df.corr()
    
    # 相関行列の可視化を保存（オプション）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
        plt.title('ETF相関ヒートマップ')
        plt.savefig('data/results/correlation_heatmap.png')
        plt.close()
        print("相関ヒートマップを保存しました: data/results/correlation_heatmap.png")
    except Exception as e:
        print(f"ヒートマップ生成エラー: {str(e)}")
    
    # 相関が高い銘柄ペアを特定し、流動性の低い方を除外
    to_exclude = set()
    remaining_symbols = list(returns_df.columns)
    
    print(f"相関フィルタリング前の銘柄数: {len(remaining_symbols)}")
    
    for i in range(len(remaining_symbols)):
        for j in range(i+1, len(remaining_symbols)):
            if i not in to_exclude and j not in to_exclude:
                sym_i = remaining_symbols[i]
                sym_j = remaining_symbols[j]
                
                if abs(corr_matrix.loc[sym_i, sym_j]) > correlation_threshold:
                    # 流動性情報を取得
                    etf_i = next(etf for etf in liquid_etfs if etf['symbol'] == sym_i)
                    etf_j = next(etf for etf in liquid_etfs if etf['symbol'] == sym_j)
                    
                    # 流動性（出来高×AUM）で比較
                    liquidity_i = etf_i['avg_volume'] * etf_i.get('aum', 0)
                    liquidity_j = etf_j['avg_volume'] * etf_j.get('aum', 0)
                    
                    # 流動性の低い方を除外
                    if liquidity_i < liquidity_j:
                        to_exclude.add(i)
                        print(f"除外: {sym_i} (高相関ペア: {sym_i} - {sym_j}, 相関: {corr_matrix.loc[sym_i, sym_j]:.2f})")
                    else:
                        to_exclude.add(j)
                        print(f"除外: {sym_j} (高相関ペア: {sym_i} - {sym_j}, 相関: {corr_matrix.loc[sym_i, sym_j]:.2f})")
    
    # 除外されなかった銘柄を取得
    filtered_symbols = [remaining_symbols[i] for i in range(len(remaining_symbols)) if i not in to_exclude]
    filtered_etfs = [etf for etf in liquid_etfs if etf['symbol'] in filtered_symbols]
    
    print(f"相関フィルタリング後の銘柄数: {len(filtered_etfs)}")
    
    # 目標数調整
    if len(filtered_etfs) > target_count * 1.5:
        print(f"銘柄数が目標より多いため、より厳しい閾値でフィルタリングを再実行します")
        # 再帰的に閾値を下げて実行
        return correlation_filtering(liquid_etfs, target_count, correlation_threshold - 0.05)
    elif len(filtered_etfs) < target_count * 0.7 and correlation_threshold < 0.95:
        print(f"銘柄数が目標より少ないため、より緩い閾値でフィルタリングを再実行します")
        # 再帰的に閾値を上げて実行
        return correlation_filtering(liquid_etfs, target_count, correlation_threshold + 0.05)
    
    # 結果をキャッシュに保存
    cache.set_json(cache_key, filtered_etfs)
    
    return filtered_etfs
