# src/utils/param_utils.py
"""パラメータ処理用ユーティリティ関数"""
from typing import Tuple

def parse_param_key(param_key: str) -> Tuple[int, float, int, int, int, int]:
    """パラメータキーからパラメータ値を抽出する
    
    Args:
        param_key: パラメータキー（例："BB20-2.0_Stoch14-3_EMA200_Hold5"）
        
    Returns:
        Tuple: (bb_window, bb_std, stoch_k, stoch_d, ema_period, holding_period)
    """
    try:
        # BBパラメータ
        bb_part = param_key.split('_')[0]
        bb_values = bb_part.replace('BB', '').split('-')
        bb_window = int(bb_values[0])
        bb_std = float(bb_values[1])
        
        # Stochパラメータ
        stoch_part = param_key.split('_')[1]
        stoch_values = stoch_part.replace('Stoch', '').split('-')
        stoch_k = int(stoch_values[0])
        stoch_d = int(stoch_values[1])
        
        # EMAパラメータ
        ema_part = param_key.split('_')[2]
        ema_period = int(ema_part.replace('EMA', ''))
        
        # 保有期間
        hold_part = param_key.split('_')[3]
        holding_period = int(hold_part.replace('Hold', ''))
        
        return bb_window, bb_std, stoch_k, stoch_d, ema_period, holding_period
    
    except Exception as e:
        print(f"パラメータキーの解析エラー ({param_key}): {str(e)}")
        return 0, 0, 0, 0, 0, 0
