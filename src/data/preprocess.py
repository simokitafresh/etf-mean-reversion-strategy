# src/data/preprocess.py
"""データ前処理とデータ品質確認のためのユーティリティ"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
import logging  # 追加: ロギング用

# ロガーの設定
logger = logging.getLogger(__name__)

# キャッシュのインスタンス化 - エラー処理強化
try:
    from ..data.cache import DataCache
    cache = DataCache()
    cache_available = True
except Exception as e:
    logger.warning(f"キャッシュ初期化エラー: {str(e)}")
    logger.info("キャッシュなしで続行します")
    cache_available = False

def data_quality_check(selected_etfs):
    """ETFデータの品質を確認し、基準を満たす銘柄だけを残す
    
    Args:
        selected_etfs (list): クラスタリングで選出されたETFのリスト
        
    Returns:
        list: 品質チェックを通過したETFのリスト
    """
    # キャッシュから取得を試みる - エラー処理強化
    if cache_available:
        cache_key = f"quality_checked_etfs_{len(selected_etfs)}"
        try:
            cached_data = cache.get_json(cache_key)
            if cached_data:
                print("キャッシュからデータ品質確認結果を取得しました")
                return cached_data
        except Exception as e:
            logger.warning(f"キャッシュ読み込みエラー: {str(e)}")
            logger.info("キャッシュ読み込みに失敗しました。データを再計算します。")
    
    print("データ品質確認を開始します...")
    
    final_etfs = []
    
    for etf in selected_etfs:
        symbol = etf['symbol']
        
        try:
            print(f"データ品質確認中: {symbol}")
            
            # 調整済み価格データを取得（できるだけ長期間）
            data = yf.download(symbol, period="5y", progress=False)
            
            if len(data) == 0:
                print(f"  警告: {symbol}のデータが取得できません")
                continue
            
            # 異常値検出（IQR法）
            returns = data['Close'].pct_change().dropna()
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((returns < lower_bound) | (returns > upper_bound)).sum()
            outlier_percentage = outliers / len(returns)
            
            # 欠損値の確認
            missing_values = data['Close'].isna().sum()
            missing_percentage = missing_values / len(data)
            
            # 連続する欠損値を検出
            consecutive_missing = 0
            max_consecutive_missing = 0
            
            for val in data['Close'].isna():
                if val:
                    consecutive_missing += 1
                    max_consecutive_missing = max(max_consecutive_missing, consecutive_missing)
                else:
                    consecutive_missing = 0
            
            # 分割・配当調整の確認（一貫性のためRenameも変更）
            try:
                adjustment_ratio = data['Close'] / data['Open']
                has_adjustments = (adjustment_ratio.std() > 0.0001)  # 調整がある場合は標準偏差が大きい
            except:
                has_adjustments = False
            
            # データ品質情報を追加
            etf['data_quality'] = {
                'outlier_percentage': float(outlier_percentage),
                'missing_percentage': float(missing_percentage),
                'max_consecutive_missing': int(max_consecutive_missing),
                'has_adjustments': bool(has_adjustments),
                'data_length_years': float(len(data) / 252),  # 取引日数から年数を概算
                'data_start': data.index[0].strftime('%Y-%m-%d'),
                'data_end': data.index[-1].strftime('%Y-%m-%d')
            }
            
            # 品質基準を適用
            if (outlier_percentage < 0.01 and  # 異常値が1%未満
                missing_percentage < 0.05 and  # 欠損値が5%未満
                max_consecutive_missing < 3):   # 3日以上の連続欠損がない
                
                final_etfs.append(etf)
                print(f"  合格: {symbol} (異常値: {outlier_percentage:.1%}, 欠損値: {missing_percentage:.1%})")
            else:
                print(f"  不合格: {symbol} - 基準を満たしません "
                      f"(異常値: {outlier_percentage:.1%}, 欠損値: {missing_percentage:.1%}, "
                      f"連続欠損最大: {max_consecutive_missing}日)")
            
            # API制限対策
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  エラー - {symbol}のデータ品質確認: {str(e)}")
    
    print(f"データ品質確認完了: {len(final_etfs)}/{len(selected_etfs)}銘柄が通過")
    
    # キャッシュに保存 - エラー処理強化
    if cache_available:
        try:
            cache.set_json(cache_key, final_etfs)
        except Exception as e:
            logger.warning(f"キャッシュ保存エラー: {str(e)}")
    
    return final_etfs
