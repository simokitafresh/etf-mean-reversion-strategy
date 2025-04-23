"""相関に基づくETFフィルタリング"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
import warnings
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from src.data.cache_manager import CacheManager

# ロガーの設定
logger = logging.getLogger(__name__)

# キャッシュマネージャーのシングルトンインスタンスを取得
cache_manager = CacheManager.get_instance()

def correlation_filtering(
    liquid_etfs: List[Dict[str, Any]], 
    target_count: int = 50, 
    correlation_threshold: float = 0.98, 
    max_attempts: int = 5
) -> List[Dict[str, Any]]:
    """相関に基づいてETFをフィルタリングする
    
    Args:
        liquid_etfs: 流動性スクリーニングを通過したETFのリスト
        target_count: 目標ETF数
        correlation_threshold: 相関係数の閾値
        max_attempts: 再帰的な調整の最大試行回数
        
    Returns:
        list: フィルタリング後のETFリスト
    """
    # 入力チェックの強化
    if not liquid_etfs:
        logger.warning("相関フィルタリングの入力が空です")
        return []
    
    if not isinstance(liquid_etfs, list):
        try:
            liquid_etfs = list(liquid_etfs)
            logger.warning("入力をリストに変換しました")
        except:
            logger.error("相関フィルタリングの入力が正しい形式ではありません")
            return []
    
    # 入力の検証 - 最低限必要な項目があるか
    valid_etfs = [
        etf for etf in liquid_etfs 
        if isinstance(etf, dict) and 'symbol' in etf and etf['symbol']
    ]
    
    if len(valid_etfs) != len(liquid_etfs):
        invalid_count = len(liquid_etfs) - len(valid_etfs)
        logger.warning(f"{invalid_count}個の無効なETFを除外しました")
        liquid_etfs = valid_etfs
        
    if len(liquid_etfs) < 2:
        logger.warning("相関フィルタリングには少なくとも2つのETFが必要です")
        return liquid_etfs
    
    # 閾値の検証
    if correlation_threshold < 0 or correlation_threshold > 1:
        logger.warning(f"無効な相関閾値 ({correlation_threshold})。0.8に調整します。")
        correlation_threshold = 0.8
    
    # キャッシュから取得を試みる
    cache_key = f"correlation_filtered_etfs_{len(liquid_etfs)}_{correlation_threshold}"
    cached_data = cache_manager.get_json(cache_key)
    if cached_data:
        logger.info("キャッシュから相関フィルタリング結果を取得しました")
        return cached_data
    
    # 再帰深度チェック（無限ループ防止）
    if max_attempts <= 0:
        logger.warning(f"相関フィルタリングの最大試行回数に達しました。現在の結果を返します。")
        return liquid_etfs
    
    logger.info(f"相関フィルタリングを開始します（閾値: {correlation_threshold}）...")
    
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
                logger.info(f"価格データ取得中: {batch_symbols}")
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
                logger.error(f"バッチ{i}のデータ取得エラー: {str(e)}")
                time.sleep(5)  # エラー時は長めに待機
                continue
        
        # データ処理の基本チェック強化
        if prices_df.empty:
            logger.error("価格データを取得できませんでした。元のETFリストを返します。")
            return liquid_etfs
        
        if len(prices_df.columns) < 2:
            logger.error("2つ以上の有効なETFデータが必要です。元のETFリストを返します。")
            return liquid_etfs
        
        # NaNの前処理強化 - 前後データを参照する前に重複を削除
        prices_df = prices_df[~prices_df.index.duplicated(keep='first')]
        
        # 欠損値処理の強化 - 段階的処理で可能な限りデータを保持
        # 1. まず前方方向に補完
        prices_df_filled = prices_df.fillna(method='ffill')
        # 2. その後、後方方向に補完
        prices_df_filled = prices_df_filled.fillna(method='bfill')
        
        # 完全に欠損しているカラムの特定
        null_columns = prices_df_filled.columns[prices_df_filled.isna().all()]
        if len(null_columns) > 0:
            logger.warning(f"完全に欠損しているカラムを削除します: {list(null_columns)}")
            prices_df_filled = prices_df_filled.drop(columns=null_columns)
        
        # データ量が十分な銘柄のみ保持
        # 80%以上のデータがある銘柄のみ保持
        min_data_points = 0.8 * len(prices_df_filled)
        valid_columns = prices_df_filled.columns[prices_df_filled.count() >= min_data_points]
        
        if len(valid_columns) < 2:
            logger.error("有効なデータを持つETFが不足しています。元のETFリストを返します。")
            return liquid_etfs
        
        prices_df_filled = prices_df_filled[valid_columns]
        
        logger.info(f"有効なデータを持つ銘柄: {len(valid_columns)}/{len(symbols)}")
        
        # 日次リターンに変換
        returns_df = prices_df_filled.pct_change().dropna()
        
        # リターンデータが空でないことを確認
        if returns_df.empty:
            logger.error("リターンデータの生成に失敗しました。元のETFリストを返します。")
            return liquid_etfs
        
        # 異常値の処理 - 極端なリターン値をクリッピング
        # リターンの最大・最小範囲を制限（例：-30%〜+30%）
        returns_df = returns_df.clip(-0.3, 0.3)
        
        # 相関行列の計算（NaN値が存在する場合に備えたハンドリング）
        # 安全な相関計算
        try:
            # 全銘柄のペアワイズ相関を計算（パールソン）
            corr_matrix = calculate_robust_correlation(returns_df)
            
            # 相関行列に問題がないか確認
            if corr_matrix is None or corr_matrix.empty:
                logger.error("相関行列の計算に失敗しました。元のETFリストを返します。")
                return liquid_etfs
            
        except Exception as e:
            logger.error(f"相関行列計算エラー: {str(e)}")
            return liquid_etfs
        
        # NaN値チェック（相関計算に失敗した場合）
        nan_mask = corr_matrix.isna()
        has_nans = nan_mask.any().any()
        
        if has_nans:
            logger.warning("相関行列にNaN値が含まれています。これらは0で置き換えられます。")
            nan_count = nan_mask.sum().sum()
            total_values = corr_matrix.size
            logger.warning(f"NaN値の割合: {nan_count}/{total_values} ({nan_count/total_values:.1%})")
            
            # NaN値を0に置換
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
            logger.info(f"相関ヒートマップを保存しました: {os.path.join(results_dir, 'correlation_heatmap.png')}")
        except Exception as e:
            logger.warning(f"ヒートマップ生成エラー: {str(e)}")
        
        # 相関が高い銘柄ペアを特定し、流動性の低い方を除外
        to_exclude = set()
        remaining_symbols = list(returns_df.columns)
        
        logger.info(f"相関フィルタリング前の銘柄数: {len(remaining_symbols)}")
        
        # 処理の安全性向上のため、手続きを関数化
        def find_high_correlation_pairs():
            """相関が高いペアを検出して除外リストを生成"""
            exclude_set = set()
            
            # シンボル有効性チェック
            valid_symbols = [s for s in remaining_symbols if s in corr_matrix.index and s in corr_matrix.columns]
            if len(valid_symbols) != len(remaining_symbols):
                logger.warning(f"{len(remaining_symbols) - len(valid_symbols)}個の無効なシンボルをスキップします")
            
            for i in range(len(valid_symbols)):
                if i in to_exclude:
                    continue
                    
                sym_i = valid_symbols[i]
                
                for j in range(i+1, len(valid_symbols)):
                    if j in to_exclude:
                        continue
                        
                    sym_j = valid_symbols[j]
                    
                    try:
                        # 相関係数の取得と閾値との比較
                        correlation = corr_matrix.loc[sym_i, sym_j]
                        
                        # 相関値の検証
                        if not is_valid_correlation(correlation):
                            continue
                        
                        if abs(correlation) > correlation_threshold:
                            # ETF情報を取得
                            etf_i = next((etf for etf in liquid_etfs if etf['symbol'] == sym_i), None)
                            etf_j = next((etf for etf in liquid_etfs if etf['symbol'] == sym_j), None)
                            
                            if etf_i is None or etf_j is None:
                                # ETF情報が見つからない場合はスキップ
                                continue
                            
                            # 流動性（出来高×AUM）で比較
                            # 安全な流動性スコア計算
                            liquidity_i = safe_multiply(etf_i.get('avg_volume', 0), etf_i.get('aum', 0))
                            liquidity_j = safe_multiply(etf_j.get('avg_volume', 0), etf_j.get('aum', 0))
                            
                            # 流動性の低い方を除外
                            if liquidity_i < liquidity_j:
                                exclude_set.add(i)
                                logger.info(f"除外: {sym_i} (高相関ペア: {sym_i} - {sym_j}, 相関: {correlation:.2f})")
                            else:
                                exclude_set.add(j)
                                logger.info(f"除外: {sym_j} (高相関ペア: {sym_i} - {sym_j}, 相関: {correlation:.2f})")
                    except Exception as e:
                        logger.warning(f"ペア ({sym_i}, {sym_j}) の処理中にエラー: {str(e)}")
                        continue
            
            return exclude_set
        
        # 高相関ペアの検出
        to_exclude = find_high_correlation_pairs()
        
        # 除外されなかった銘柄を取得
        valid_indices = [i for i in range(len(remaining_symbols)) if i not in to_exclude]
        filtered_symbols = [remaining_symbols[i] for i in valid_indices]
        filtered_etfs = [etf for etf in liquid_etfs if etf['symbol'] in filtered_symbols]
        
        logger.info(f"相関フィルタリング後の銘柄数: {len(filtered_etfs)}")
        
        # 目標数調整のロジック改善
        if len(filtered_etfs) > target_count * 1.5:
            logger.info(f"銘柄数が目標より多いため、より厳しい閾値でフィルタリングを再実行します")
            # 再帰的に閾値を下げて実行
            new_threshold = max(0.8, correlation_threshold - 0.05)  # 下限値を0.8に制限
            return correlation_filtering(liquid_etfs, target_count, new_threshold, max_attempts - 1)
        elif len(filtered_etfs) < target_count * 0.7 and correlation_threshold < 0.98:
            logger.info(f"銘柄数が目標より少ないため、より緩い閾値でフィルタリングを再実行します")
            # 再帰的に閾値を上げて実行
            new_threshold = min(0.99, correlation_threshold + 0.05)  # 上限値を0.99に制限
            return correlation_filtering(liquid_etfs, target_count, new_threshold, max_attempts - 1)
        
        # キャッシュに保存
        cache_manager.set_json(cache_key, filtered_etfs)
        
        return filtered_etfs
        
    except Exception as e:
        logger.error(f"相関フィルタリング処理全体のエラー: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        # エラー時は入力データをそのまま返す
        return liquid_etfs

# ヘルパー関数
def is_valid_correlation(corr_value):
    """相関値が有効かどうかチェック"""
    if corr_value is None:
        return False
    if np.isnan(corr_value) or np.isinf(corr_value):
        return False
    if corr_value < -1 or corr_value > 1:
        return False
    return True

def safe_multiply(a, b):
    """安全な乗算 (NaNや無限大を回避)"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return 0
    if np.isnan(a) or np.isnan(b) or np.isinf(a) or np.isinf(b):
        return 0
    return a * b

def calculate_robust_correlation(returns_df, method='pearson'):
    """より堅牢な相関計算
    
    Args:
        returns_df: リターンデータのデータフレーム
        method: 相関計算方法 ('pearson', 'spearman', 'kendall')
        
    Returns:
        pd.DataFrame: 相関行列
    """
    if returns_df.empty:
        return None
    
    # データの準備 - 極端な値のクリッピング
    clipped_df = returns_df.clip(-0.5, 0.5)
    
    # 欠損値処理
    clean_df = clipped_df.fillna(method='ffill').fillna(method='bfill')
    
    # 欠損値が多すぎる列を削除
    min_data_points = 0.8 * len(clean_df)
    valid_columns = clean_df.columns[clean_df.count() >= min_data_points]
    if len(valid_columns) < 2:
        return None
    
    clean_df = clean_df[valid_columns]
    
    try:
        # 相関行列の計算
        corr_matrix = clean_df.corr(method=method)
        
        # NaNを処理
        corr_matrix = corr_matrix.fillna(0)
        
        # 極端な値のクリッピング
        corr_matrix = corr_matrix.clip(-1, 1)
        
        return corr_matrix
    
    except Exception as e:
        logger.error(f"堅牢な相関計算エラー: {str(e)}")
        return None
