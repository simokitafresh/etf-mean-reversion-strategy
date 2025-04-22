# src/universe/interface.py（新規ファイル）

"""ユニバース選定のインターフェース定義"""
from typing import List, Dict, Any, Optional

def select_universe(
    base_list: Optional[List[Dict[str, Any]]] = None, 
    target_count: int = 50, 
    clustering_method: str = 'tda_optics'
) -> List[Dict[str, Any]]:
    """ETFユニバースを選定する
    
    Args:
        base_list: 基礎ETFリスト。Noneの場合は自動取得
        target_count: 目標ETF数
        clustering_method: クラスタリング手法
        
    Returns:
        List[Dict[str, Any]]: 選定されたETFユニバース
    """
    # 実装をインポート
    from . import _implementation
    return _implementation.select_universe(
        base_list, target_count, clustering_method
    )

def get_sample_etfs() -> List[Dict[str, Any]]:
    """サンプルETFリストを取得する
    
    Returns:
        List[Dict[str, Any]]: サンプルETFのリスト
    """
    from .sample_etfs import get_sample_etfs as get_samples
    return get_samples()
