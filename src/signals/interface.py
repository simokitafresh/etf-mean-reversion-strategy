# src/signals/interface.py（新規ファイル）

"""シグナル計算のインターフェース定義"""
from typing import Dict, List, Any, Optional
import pandas as pd

def calculate_signals_for_etf(
    symbol: str,
    bb_window: int = 20,
    bb_std: float = 2.0,
    stoch_k: int = 14,
    stoch_d: int = 3,
    ema_period: int = 200,
    ema_slope_period: int = 20,
    period: str = "5y",
    min_samples: int = 30,
    price_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """指定されたETFのシグナルを計算する
    
    Args:
        symbol: ETFのティッカーシンボル
        bb_window: ボリンジャーバンドの期間
        bb_std: ボリンジャーバンドの標準偏差倍率
        stoch_k: ストキャスティクスの%K期間
        stoch_d: ストキャスティクスの%D期間
        ema_period: EMAの期間
        ema_slope_period: EMA傾きの計算期間
        period: データ取得期間
        min_samples: 有効なシグナルとみなす最小のサンプル数
        price_data: 既存の価格データ（Noneの場合はダウンロード）
        
    Returns:
        pd.DataFrame: シグナルを含むデータフレーム
    """
    # 実装をインポート
    from . import _implementation
    return _implementation.calculate_signals_for_etf(
        symbol, bb_window, bb_std, stoch_k, stoch_d, 
        ema_period, ema_slope_period, period, min_samples, price_data
    )

def calculate_signals_for_universe(
    universe: List[Dict[str, Any]],
    parameter_sets: List[Dict[str, Any]],
    period: str = "5y",
    min_samples: int = 30
) -> Dict[str, Dict[str, Any]]:
    """ETFユニバース全体のシグナルを計算する
    
    Args:
        universe: ETFユニバースのリスト
        parameter_sets: パラメータセットのリスト
        period: データ取得期間
        min_samples: 有効なシグナルとみなす最小のサンプル数
        
    Returns:
        Dict: シンボルとパラメータセットごとのシグナル結果
    """
    # 実装をインポート
    from . import _implementation
    return _implementation.calculate_signals_for_universe(
        universe, parameter_sets, period, min_samples
    )
