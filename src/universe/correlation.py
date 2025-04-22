# src/universe/correlation.py
"""相関に基づくETFフィルタリング"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
import warnings
from ..data.cache import DataCache

# キャッシュのインスタンス化
cache = DataCache()

def correlation_filtering(liquid_etfs, target_count=50, correlation_threshold=0.98, max_attempts=5):
    """相関に基づいてETFをフィルタリングする
    
    Args:
        liquid_etfs (list): 流動性スクリーニングを通過したETFのリスト
        target_count (int): 目標ETF数
        correlation_threshold (float): 相関係数の閾値
        max_attempts (int): 再帰的な調整の最大試行回数
        
    Returns:
        list: フィルタリング後のETFリスト
    """
    # 入力チェック
    if not liquid_etfs:
        warnings.warn("相関フィルタリングの入力が空です")
        return []
    
    if not isinstance(liquid_etfs, list):
        try:
            liquid_etfs = list(liquid_etfs)
        except:
            warnings.warn("相関フィルタリングの入力が正しい形式ではありません")
            return []
    
    # キャッシュから取得を試みる
    cache_key = f"correlation_filtered_etfs_{len(liquid_etfs)}_{correlation_threshold}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュから相関フィルタリング結果を取得しました")
        return cached_data
    
    # 再帰深度チェック（無限ループ防止）
    if max_attempts <= 0:
        print(f"警告: 相関フィルタリングの最大試行回数に達しました。現在の結果を返します。")
        return liquid_etfs
    
    print(f"相関フィルタリングを開始します（閾値: {correlation_threshold}）...")
    
    # 結果ディレクトリの確認
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
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
                    progress=False,
                    threads=True  # 並列ダウンロードを使用
                )
                
                # 単一銘柄の場合は構造が異なる
                if len(batch_symbols) == 1:
                    symbol = batch_symbols[0]
                    if 'Close' in batch_data.columns:
                        prices_df[symbol] = batch_data['Close']
                else:
                    # 複数銘柄の場合
                    for symbol in batch_symbols:
                        if symbol in batch_data and 'Close' in batch_data[symbol]:
                            prices_df[symbol] = batch_data[symbol]['Close']
                
                # APIレート制限対策
                time.sleep(1)
            except Exception as e:
                print(f"バッチ{i}のデータ取得エラー: {str(e)}")
                time.sleep(5)  # エラー時は長めに待機
                continue
        
        # データ処理の基本チェック
        if prices_df.empty:
            print("エラー: 価格データを取得できませんでした。元のETFリストを返します。")
            return liquid_etfs
        
        if len(prices_df.columns) < 2:
            print("エラー: 2つ以上の有効なETFデータが必要です。元のETFリストを返します。")
            return liquid_etfs
        
        # 欠損値を前方補完してから後方補完（より堅牢）
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        
        # 80%以上のデータがある銘柄のみ保持
        min_data_points = 0.8 * len(prices_df)
        valid_columns = prices_df.columns[prices_df.count() >= min_data_points]
        
        if len(valid_columns) < 2:
            print("エラー: 有効なデータを持つETFが不足しています。元のETFリストを返します。")
            return liquid_etfs
        
        prices_df = prices_df[valid_columns]
        
        print(f"有効なデータを持つ銘柄: {len(valid_columns)}/{len(symbols)}")
        
        # 日次リターンに変換
        returns_df = prices_df.pct_change().dropna()
        
        # 相関行列の計算（NaN値が存在する場合に備えたハンドリング）
        corr_matrix = returns_df.corr()
        
        # NaN値チェック（相関計算に失敗した場合）
        if corr_matrix.isna().any().any():
            print("警告: 相関行列にNaN値が含まれています。これらは0で置き換えられます。")
            corr_matrix = corr_matrix.fillna(0)
        
        # 対角成分を1に設定（自己相関）
        np.fill_diagonal(corr_matrix.values, 1)
        
        # 相関行列の可視化を保存
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
            plt.title('ETF相関ヒートマップ')
            plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
            plt.close()
            print(f"相関ヒートマップを保存しました: {os.path.join(results_dir, 'correlation_heatmap.png')}")
        except Exception as e:
            print(f"ヒートマップ生成エラー: {str(e)}")
        
        # 相関が高い銘柄ペアを特定し、流動性の低い方を除外
        to_exclude = set()
        remaining_symbols = list(returns_df.columns)
        
        print(f"相関フィルタリング前の銘柄数: {len(remaining_symbols)}")
        
        # 処理の安全性向上のため、手続きを関数化
        def find_high_correlation_pairs():
            """相関が高いペアを検出して除外リストを生成"""
            exclude_set = set()
            
            for i in range(len(remaining_symbols)):
                if i in to_exclude:
                    continue
                    
                sym_i = remaining_symbols[i]
                
                for j in range(i+1, len(remaining_symbols)):
                    if j in to_exclude:
                        continue
                        
                    sym_j = remaining_symbols[j]
                    
                    try:
                        # 相関係数の取得と閾値との比較
                        correlation = corr_matrix.loc[sym_i, sym_j]
                        
                        if abs(correlation) > correlation_threshold:
                            # ETF情報を取得
                            etf_i = next((etf for etf in liquid_etfs if etf['symbol'] == sym_i), None)
                            etf_j = next((etf for etf in liquid_etfs if etf['symbol'] == sym_j), None)
                            
                            if etf_i is None or etf_j is None:
                                # ETF情報が見つからない場合はスキップ
                                continue
                            
                            # 流動性（出来高×AUM）で比較
                            liquidity_i = etf_i.get('avg_volume', 0) * etf_i.get('aum', 0)
                            liquidity_j = etf_j.get('avg_volume', 0) * etf_j.get('aum', 0)
                            
                            # 流動性の低い方を除外
                            if liquidity_i < liquidity_j:
                                exclude_set.add(i)
                                print(f"除外: {sym_i} (高相関ペア: {sym_i} - {sym_j}, 相関: {correlation:.2f})")
                            else:
                                exclude_set.add(j)
                                print(f"除外: {sym_j} (高相関ペア: {sym_i} - {sym_j}, 相関: {correlation:.2f})")
                    except Exception as e:
                        print(f"ペア ({sym_i}, {sym_j}) の処理中にエラー: {str(e)}")
                        continue
            
            return exclude_set
        
        # 高相関ペアの検出
        to_exclude = find_high_correlation_pairs()
        
        # 除外されなかった銘柄を取得
        filtered_symbols = [remaining_symbols[i] for i in range(len(remaining_symbols)) if i not in to_exclude]
        filtered_etfs = [etf for etf in liquid_etfs if etf['symbol'] in filtered_symbols]
        
        print(f"相関フィルタリング後の銘柄数: {len(filtered_etfs)}")
        
        # 目標数調整のロジック改善
        if len(filtered_etfs) > target_count * 1.5:
            print(f"銘柄数が目標より多いため、より厳しい閾値でフィルタリングを再実行します")
            # 再帰的に閾値を下げて実行
            new_threshold = max(0.8, correlation_threshold - 0.05)  # 下限値を0.8に制限
            return correlation_filtering(liquid_etfs, target_count, new_threshold, max_attempts - 1)
        elif len(filtered_etfs) < target_count * 0.7 and correlation_threshold < 0.98:
            print(f"銘柄数が目標より少ないため、より緩い閾値でフィルタリングを再実行します")
            # 再帰的に閾値を上げて実行
            new_threshold = min(0.99, correlation_threshold + 0.05)  # 上限値を0.99に制限
            return correlation_filtering(liquid_etfs, target_count, new_threshold, max_attempts - 1)
        
        # 結果をキャッシュに保存
        cache.set_json(cache_key, filtered_etfs)
        
        return filtered_etfs
        
    except Exception as e:
        print(f"相関フィルタリング処理全体のエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        # エラー時は入力データをそのまま返す
        return liquid_etfs
