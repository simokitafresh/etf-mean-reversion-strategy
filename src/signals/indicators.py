# src/signals/indicators.py
"""テクニカル指標計算モジュール"""
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple

def calculate_bollinger_bands(
    prices: pd.Series, 
    window: int = 20, 
    num_std: float = 2.0
) -> Dict[str, pd.Series]:
    """ボリンジャーバンドを計算する
    
    Args:
        prices: 価格のTimeSeries
        window: 移動平均の期間
        num_std: 標準偏差の倍率
        
    Returns:
        Dict: 'middle', 'upper', 'lower', 'width'のキーを持つ辞書
    """
    # 移動平均の計算
    middle_band = prices.rolling(window=window).mean()
    
    # 標準偏差の計算
    rolling_std = prices.rolling(window=window).std()
    
    # 上下のバンドを計算
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    # バンド幅（ボラティリティ指標）
    band_width = (upper_band - lower_band) / middle_band
    
    return {
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band,
        'width': band_width
    }

def calculate_stochastic(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    k_period: int = 14, 
    d_period: int = 3
) -> Dict[str, pd.Series]:
    """ストキャスティクスオシレーターを計算する
    
    Args:
        high: 高値のTimeSeries
        low: 安値のTimeSeries
        close: 終値のTimeSeries
        k_period: %Kの期間
        d_period: %Dの期間
        
    Returns:
        Dict: 'k', 'd'のキーを持つ辞書
    """
    # 最高値と最安値を計算
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # %Kを計算
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # %Dを計算（%Kの移動平均）
    d = k.rolling(window=d_period).mean()
    
    return {
        'k': k,
        'd': d
    }

def calculate_ema_slope(
    prices: pd.Series, 
    period: int = 200, 
    slope_period: int = 20
) -> pd.Series:
    """指数移動平均（EMA）の傾きを計算する
    
    Args:
        prices: 価格のTimeSeries
        period: EMAの期間
        slope_period: 傾きを計算する期間
        
    Returns:
        pd.Series: EMAの傾きを表すTimeSeries
    """
    # EMAを計算
    ema = prices.ewm(span=period, adjust=False).mean()
    
    # 傾きの計算（移動平均線の変化率）
    slope = (ema / ema.shift(slope_period) - 1) * 100
    
    return {
        'ema': ema,
        'slope': slope
    }

def calculate_all_indicators(
    data: pd.DataFrame,
    bb_window: int = 20,
    bb_std: float = 2.0,
    stoch_k: int = 14,
    stoch_d: int = 3,
    ema_period: int = 200,
    ema_slope_period: int = 20
) -> pd.DataFrame:
    """すべてのテクニカル指標を計算し、元のデータフレームに追加する
    
    Args:
        data: OHLCV形式のデータフレーム
        bb_window: ボリンジャーバンドの期間
        bb_std: ボリンジャーバンドの標準偏差倍率
        stoch_k: ストキャスティクスの%K期間
        stoch_d: ストキャスティクスの%D期間
        ema_period: EMAの期間
        ema_slope_period: EMA傾きの計算期間
        
    Returns:
        pd.DataFrame: 指標を追加したデータフレーム
    """
    # 入力確認
    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"入力データには '{col}' 列が必要です")
    
    # 結果用にデータフレームをコピー
    result = data.copy()
    
    # ボリンジャーバンドの計算
    bb = calculate_bollinger_bands(
        result['Adj Close'], 
        window=bb_window, 
        num_std=bb_std
    )
    
    result['BB_Middle'] = bb['middle']
    result['BB_Upper'] = bb['upper']
    result['BB_Lower'] = bb['lower']
    result['BB_Width'] = bb['width']
    
    # ストキャスティクスの計算
    stoch = calculate_stochastic(
        result['High'], 
        result['Low'], 
        result['Close'], 
        k_period=stoch_k, 
        d_period=stoch_d
    )
    
    result['Stoch_K'] = stoch['k']
    result['Stoch_D'] = stoch['d']
    
    # EMAと傾きの計算
    ema = calculate_ema_slope(
        result['Adj Close'], 
        period=ema_period, 
        slope_period=ema_slope_period
    )
    
    result['EMA200'] = ema['ema']
    result['EMA200_Slope'] = ema['slope']
    
    # 価格位置（EMAに対する相対位置）を追加
    result['Price_Rel_EMA'] = (result['Adj Close'] / result['EMA200'] - 1) * 100
    
    return result
