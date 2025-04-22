# src/signals/rules.py
"""シグナルルール定義モジュール"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def identify_trend(data: pd.DataFrame) -> pd.Series:
    """トレンドを識別する（上昇・下降・横ばい）
    
    Args:
        data: テクニカル指標を含むデータフレーム
        
    Returns:
        pd.Series: トレンドを示すシリーズ（1=上昇、-1=下降、0=横ばい）
    """
    # EMAの傾きと価格位置から判断
    # 上昇トレンド: EMAが上昇中 & 価格 > EMA
    # 下降トレンド: EMAが下降中 & 価格 < EMA
    # その他は横ばい
    
    trend = pd.Series(0, index=data.index)  # デフォルトは横ばい
    
    # 上昇トレンド
    up_trend = (data['EMA200_Slope'] > 0) & (data['Price_Rel_EMA'] > 0)
    trend[up_trend] = 1
    
    # 下降トレンド
    down_trend = (data['EMA200_Slope'] < 0) & (data['Price_Rel_EMA'] < 0)
    trend[down_trend] = -1
    
    return trend

def detect_overshoots(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """価格のオーバーシュート（行き過ぎ）を検出する
    
    Args:
        data: テクニカル指標を含むデータフレーム
        
    Returns:
        Tuple: (買いオーバーシュート, 売りオーバーシュート)のタプル
    """
    # Bollingerバンドのオーバーシュート
    bb_buy = data['Close'] < data['BB_Lower']
    bb_sell = data['Close'] > data['BB_Upper']
    
    # Stochasticオーバーシュート（クロスオーバー）
    stoch_buy = (data['Stoch_K'] < 20) & (data['Stoch_K'] > data['Stoch_D']) & (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1))
    stoch_sell = (data['Stoch_K'] > 80) & (data['Stoch_K'] < data['Stoch_D']) & (data['Stoch_K'].shift(1) >= data['Stoch_D'].shift(1))
    
    # 最終的なオーバーシュートシグナル
    buy_overshoot = bb_buy & stoch_buy
    sell_overshoot = bb_sell & stoch_sell
    
    return buy_overshoot, sell_overshoot

def generate_signals(
    data: pd.DataFrame,
    min_samples: int = 30
) -> pd.DataFrame:
    """トレンドとオーバーシュートに基づいてシグナルを生成する
    
    Args:
        data: テクニカル指標を含むデータフレーム
        min_samples: 有効なシグナルとみなす最小のサンプル数
        
    Returns:
        pd.DataFrame: シグナルを含むデータフレーム
    """
    # 結果用にデータフレームをコピー
    result = data.copy()
    
    # トレンドの識別
    result['Trend'] = identify_trend(data)
    
    # オーバーシュートの検出
    buy_overshoot, sell_overshoot = detect_overshoots(data)
    result['Buy_Overshoot'] = buy_overshoot
    result['Sell_Overshoot'] = sell_overshoot
    
    # シグナル生成 - 修正：NaN値の処理と条件強化
    # NaN値を明示的に処理
    trend_is_up = result['Trend'].fillna(0) == 1
    trend_is_down = result['Trend'].fillna(0) == -1
    buy_overshoot_valid = result['Buy_Overshoot'].fillna(False)
    sell_overshoot_valid = result['Sell_Overshoot'].fillna(False)
    
    # 連続シグナルのフィルタリング（前日もシグナルだった場合は除外）
    buy_overshoot_not_repeated = ~buy_overshoot_valid.shift(1).fillna(False)
    sell_overshoot_not_repeated = ~sell_overshoot_valid.shift(1).fillna(False)
    
    # 最終的なシグナル生成（修正された条件）
    result['Buy_Signal'] = trend_is_up & buy_overshoot_valid & buy_overshoot_not_repeated
    result['Sell_Signal'] = trend_is_down & sell_overshoot_valid & sell_overshoot_not_repeated
    
    # オリジナルのシグナル数を保存
    buy_signal_count = result['Buy_Signal'].sum()
    sell_signal_count = result['Sell_Signal'].sum()
    
    # シグナル数が最小サンプル数を下回る場合は無効化フラグを設定
    buy_signals_valid = buy_signal_count >= min_samples
    sell_signals_valid = sell_signal_count >= min_samples
    
    # 無効な場合は警告を表示
    if not buy_signals_valid:
        print(f"警告: 買いシグナルのサンプル数({buy_signal_count})が最小数({min_samples})を下回るため無効フラグを設定しました")
    
    if not sell_signals_valid:
        print(f"警告: 売りシグナルのサンプル数({sell_signal_count})が最小数({min_samples})を下回るため無効フラグを設定しました")
    
    # 有効性フラグを追加
    result['Buy_Signal_Valid'] = buy_signals_valid
    result['Sell_Signal_Valid'] = sell_signals_valid
    
    # シグナル強度スコアの計算
    # Bollingerバンド逸脱度 + Stochastic極値度を数値化
    result['Buy_Signal_Strength'] = 0.0
    result['Sell_Signal_Strength'] = 0.0
    
    # 買いシグナル強度 - シグナルが存在する場合のみ計算
    buy_mask = result['Buy_Signal']
    if buy_mask.any():
        try:
            # Bollingerバンド逸脱度（0〜1のスケール）
            bb_deviation = (result['BB_Lower'] - result['Close']) / result['BB_Lower']
            # 0以上の値に制限（割り算の結果がマイナスになる可能性があるため）
            bb_deviation = bb_deviation.clip(0, None)
            
            # Stochastic極値度（0〜1のスケール、20に近いほど0に近づく）
            stoch_extremity = ((20 - result['Stoch_K']) / 20).clip(0, 1)
            
            # 正規化して合成（両方とも0〜1のスケール）
            # NaNが発生する可能性があるため、fillnaで対処
            result.loc[buy_mask, 'Buy_Signal_Strength'] = (
                0.7 * bb_deviation.loc[buy_mask].fillna(0) + 
                0.3 * stoch_extremity.loc[buy_mask].fillna(0)
            )
        except Exception as e:
            print(f"買いシグナル強度の計算中にエラーが発生しました: {str(e)}")
    
    # 売りシグナル強度 - シグナルが存在する場合のみ計算
    sell_mask = result['Sell_Signal']
    if sell_mask.any():
        try:
            # Bollingerバンド逸脱度（0〜1のスケール）
            bb_deviation = (result['Close'] - result['BB_Upper']) / result['BB_Upper']
            # 0以上の値に制限
            bb_deviation = bb_deviation.clip(0, None)
            
            # Stochastic極値度（0〜1のスケール、80に近いほど0に近づく）
            stoch_extremity = ((result['Stoch_K'] - 80) / 20).clip(0, 1)
            
            # 正規化して合成（両方とも0〜1のスケール）
            # NaNが発生する可能性があるため、fillnaで対処
            result.loc[sell_mask, 'Sell_Signal_Strength'] = (
                0.7 * bb_deviation.loc[sell_mask].fillna(0) + 
                0.3 * stoch_extremity.loc[sell_mask].fillna(0)
            )
        except Exception as e:
            print(f"売りシグナル強度の計算中にエラーが発生しました: {str(e)}")
    
    return result
