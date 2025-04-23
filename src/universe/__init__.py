"""ETFユニバース選定モジュール - パブリックAPI

このモジュールではインポート戦略のベストプラクティスに従っています：
1. 相対インポート: 同一パッケージ内のモジュール参照に使用
2. 絶対インポート: 異なるパッケージからのインポートに使用
3. 公開APIの明示: __all__リストでパッケージの公開インターフェースを定義
"""
from .interface import select_universe, get_sample_etfs

# 公開APIのみをエクスポート
__all__ = [
    'select_universe', 
    'get_sample_etfs',
    'create_universe_provider'
]

def create_universe_provider(provider_type: str = 'default'):
    """ユニバース選定プロバイダーを作成するファクトリー関数
    
    Args:
        provider_type: プロバイダーの種類（'default', 'sample', 'mock'）
        
    Returns:
        UniverseProvider: ユニバース選定プロバイダーのインスタンス
    """
    from typing import Protocol, List, Dict, Any, Optional
    
    class UniverseProvider(Protocol):
        """ユニバース選定プロバイダーのインターフェース"""
        def select_universe(
            self,
            base_list: Optional[List[Dict[str, Any]]] = None, 
            target_count: int = 50, 
            clustering_method: str = 'tda_optics'
        ) -> List[Dict[str, Any]]:
            """ETFユニバースを選定する"""
            ...
    
    class DefaultUniverseProvider:
        def select_universe(
            self,
            base_list: Optional[List[Dict[str, Any]]] = None, 
            target_count: int = 50, 
            clustering_method: str = 'tda_optics'
        ) -> List[Dict[str, Any]]:
            """標準の選定ロジックを使用"""
            from ._implementation import select_universe as impl_select
            return impl_select(base_list, target_count, clustering_method)
    
    class SampleUniverseProvider:
        def select_universe(
            self,
            base_list: Optional[List[Dict[str, Any]]] = None, 
            target_count: int = 50, 
            clustering_method: str = 'tda_optics'
        ) -> List[Dict[str, Any]]:
            """サンプルETFリストを使用"""
            from .sample_etfs import get_sample_etfs
            return get_sample_etfs()
    
    class MockUniverseProvider:
        def select_universe(
            self,
            base_list: Optional[List[Dict[str, Any]]] = None, 
            target_count: int = 50, 
            clustering_method: str = 'tda_optics'
        ) -> List[Dict[str, Any]]:
            """テスト用モックデータを使用"""
            return [
                {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "category": "US Large Cap", "avg_volume": 75000000},
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "category": "US Tech", "avg_volume": 50000000},
                {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "category": "US Small Cap", "avg_volume": 25000000}
            ]
    
    if provider_type == 'default':
        return DefaultUniverseProvider()
    elif provider_type == 'sample':
        return SampleUniverseProvider()
    elif provider_type == 'mock':
        return MockUniverseProvider()
    else:
        raise ValueError(f"不明なプロバイダータイプ: {provider_type}")
