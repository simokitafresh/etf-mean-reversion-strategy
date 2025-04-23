# src/data/preprocess.py
"""データ前処理とデータ品質確認のためのユーティリティ"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
import logging
from typing import List, Dict, Any, Optional

# ロガーの設定
logger = logging.getLogger(__name__)

# キャッシュマネージャーのインポート
from src.data.cache_manager import CacheManager

# キャッシュマネージャーのシングルトンインスタンスを取得
cache_manager = CacheManager.get_instance()

def data_quality_check(selected_etfs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ETFデータの品質を確認し、基準を満たす銘柄だけを残す
    
    Args:
        selected_etfs: クラスタリングで選出されたETFのリスト
        
    Returns:
        List[Dict[str, Any]]: 品質チェックを通過したETFのリスト
    """
    # 入力チェック
    if not selected_etfs:
        logger.warning("空のETFリストが渡されました")
        return []
    
    # キャッシュから取得を試みる
    cache_key = f"quality_checked_etfs_{len(selected_etfs)}"
    cached_data = cache_manager.get_json(cache_key)
    if cached_data:
        logger.info("キャッシュからデータ品質確認結果を取得しました")
        return cached_data
    
    logger.info("データ品質確認を開始します...")
    
    final_etfs = []
    
    for etf in selected_etfs:
        symbol = etf.get('symbol')
        if not symbol:
            logger.warning(f"シンボルがないETFをスキップします: {etf}")
            continue
            
        try:
            logger.info(f"データ品質確認中: {symbol}")
            
            # 調整済み価格データを取得（できるだけ長期間）
            data = yf.download(symbol, period="5y", progress=False)
            
            if len(data) == 0:
                logger.warning(f"警告: {symbol}のデータが取得できません")
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
            
            # 分割・配当調整の確認
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
                logger.info(f"合格: {symbol} (異常値: {outlier_percentage:.1%}, 欠損値: {missing_percentage:.1%})")
            else:
                logger.info(f"不合格: {symbol} - 基準を満たしません "
                          f"(異常値: {outlier_percentage:.1%}, 欠損値: {missing_percentage:.1%}, "
                          f"連続欠損最大: {max_consecutive_missing}日)")
            
            # API制限対策
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"エラー - {symbol}のデータ品質確認: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
    
    logger.info(f"データ品質確認完了: {len(final_etfs)}/{len(selected_etfs)}銘柄が通過")
    
    # キャッシュに保存
    if final_etfs:
        cache_manager.set_json(cache_key, final_etfs)
    
    return final_etfs

def clean_price_data(
    df: pd.DataFrame, 
    fill_method: str = 'ffill',
    handle_outliers: bool = True,
    min_price: float = 0.01
) -> pd.DataFrame:
    """価格データのクリーニング
    
    Args:
        df: 価格データフレーム（OHLCV形式）
        fill_method: 欠損値の補完方法 ('ffill', 'bfill', 'interpolate')
        handle_outliers: 異常値を処理するかどうか
        min_price: 最小有効価格
        
    Returns:
        pd.DataFrame: クリーニング済みのデータフレーム
    """
    if df is None or df.empty:
        logger.warning("空のデータフレームが渡されました")
        return pd.DataFrame()
    
    # 結果用にコピー
    result = df.copy()
    
    # 重複インデックスの削除
    if isinstance(result.index, pd.DatetimeIndex):
        result = result[~result.index.duplicated(keep='first')]
    
    # 必須カラムの確認
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in result.columns]
    
    if missing_columns:
        logger.warning(f"必須カラムが欠けています: {missing_columns}")
        # 欠けているカラムを作成（Closeを基準に）
        for col in missing_columns:
            if col != 'Volume' and 'Close' in result.columns:
                result[col] = result['Close']
            elif col == 'Volume':
                result[col] = 0
    
    # 無効な価格の処理
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in result.columns:
            # ゼロや負の値を置換
            zero_mask = result[col] <= min_price
            if zero_mask.any():
                logger.warning(f"{col}に{zero_mask.sum()}件の無効な価格があります")
                result.loc[zero_mask, col] = np.nan
    
    # 異常値処理
    if handle_outliers:
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in result.columns:
                # IQRによる異常値検出
                valid_values = result[col].dropna()
                if len(valid_values) > 0:
                    Q1 = valid_values.quantile(0.25)
                    Q3 = valid_values.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 5 * IQR  # 5 IQRの範囲を許容
                    upper_bound = Q3 + 5 * IQR
                    
                    outliers = ((result[col] < lower_bound) | (result[col] > upper_bound)) & result[col].notna()
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        logger.warning(f"{col}に{outlier_count}件の異常値があります")
                        result.loc[outliers, col] = np.nan
    
    # 欠損値の処理
    if fill_method == 'ffill':
        # 前方向補完の後に後方向補完
        result = result.fillna(method='ffill').fillna(method='bfill')
    elif fill_method == 'bfill':
        # 後方向補完の後に前方向補完
        result = result.fillna(method='bfill').fillna(method='ffill')
    elif fill_method == 'interpolate':
        # 線形補間
        result = result.interpolate(method='linear', axis=0).fillna(method='ffill').fillna(method='bfill')
    else:
        logger.warning(f"不明な補完方法: {fill_method}, 'ffill'を使用します")
        result = result.fillna(method='ffill').fillna(method='bfill')
    
    # 高値・安値の整合性を確保
    if all(col in result.columns for col in ['Open', 'High', 'Low', 'Close']):
        # 高値は他のすべての価格以上であるべき
        result['High'] = result[['Open', 'High', 'Low', 'Close']].max(axis=1)
        # 安値は他のすべての価格以下であるべき
        result['Low'] = result[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    # 出来高の処理（負の値を0に）
    if 'Volume' in result.columns:
        result.loc[result['Volume'] < 0, 'Volume'] = 0
    
    return result

def resample_price_data(
    df: pd.DataFrame,
    freq: str = 'D',
    fill_gaps: bool = True
) -> pd.DataFrame:
    """価格データのリサンプリング
    
    Args:
        df: 価格データフレーム（OHLCV形式）
        freq: リサンプリング頻度 ('D'=日次, 'W'=週次, 'M'=月次)
        fill_gaps: ギャップを補完するかどうか
        
    Returns:
        pd.DataFrame: リサンプリングされたデータフレーム
    """
    if df is None or df.empty:
        logger.warning("空のデータフレームが渡されました")
        return pd.DataFrame()
    
    # 日付インデックスでない場合は変換
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            logger.error("インデックスを日付型に変換できません")
            return df
    
    # OHLCVカラムの確認
    ohlc_columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
    volume_column = 'Volume' if 'Volume' in df.columns else None
    
    # リサンプリングの集約ルール
    agg_dict = {}
    
    if 'Open' in ohlc_columns:
        agg_dict['Open'] = 'first'
    if 'High' in ohlc_columns:
        agg_dict['High'] = 'max'
    if 'Low' in ohlc_columns:
        agg_dict['Low'] = 'min'
    if 'Close' in ohlc_columns:
        agg_dict['Close'] = 'last'
    if volume_column:
        agg_dict[volume_column] = 'sum'
    
    # その他の列はリサンプリングから除外
    other_columns = [col for col in df.columns if col not in ohlc_columns + ([volume_column] if volume_column else [])]
    
    # リサンプリング実行
    if not agg_dict:
        logger.warning("リサンプリング対象のカラムが見つかりません")
        return df
    
    resampled = df[list(agg_dict.keys())].resample(freq).agg(agg_dict)
    
    # ギャップの補完
    if fill_gaps:
        resampled = clean_price_data(resampled)
    
    # その他の列をマージ（最後の値を使用）
    if other_columns:
        for col in other_columns:
            other_resampled = df[col].resample(freq).last()
            resampled[col] = other_resampled
    
    return resampled
