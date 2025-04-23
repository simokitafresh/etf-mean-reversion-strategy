"""シグナル生成モジュール - パブリックAPI

このモジュールではインポート戦略のベストプラクティスに従っています：
1. 相対インポート: 同一パッケージ内のモジュール参照に使用
2. 絶対インポート: 異なるパッケージからのインポートに使用
3. 公開APIの明示: __all__リストでパッケージの公開インターフェースを定義
"""
from .interface import calculate_signals_for_etf, calculate_signals_for_universe

# 公開APIのみをエクスポート
__all__ = [
    'calculate_signals_for_etf', 
    'calculate_signals_for_universe',
    'create_signal_provider'
]

def create_signal_provider(provider_type: str = 'default'):
    """シグナルプロバイダーを作成するファクトリー関数
    
    Args:
        provider_type: プロバイダーの種類（'default', 'mock', 'custom'）
        
    Returns:
        SignalProvider: シグナル計算プロバイダーのインスタンス
    """
    if provider_type == 'default':
        from src.parameters.context import SignalProviderAdapter
        from ._implementation import calculate_signals_for_universe as calc_func
        return SignalProviderAdapter(calc_func)
    elif provider_type == 'mock':
        # モックプロバイダー（テスト用）
        from src.parameters.context import SignalProviderAdapter
        
        def mock_calculate_signals(*args, **kwargs):
            """シグナル計算のモック関数"""
            return {"mock_etf": {"mock_param": {"buy_signals": 5, "sell_signals": 5}}}
            
        return SignalProviderAdapter(mock_calculate_signals)
    elif provider_type == 'custom':
        # カスタムプロバイダー（将来的な拡張用）
        raise NotImplementedError("カスタムシグナルプロバイダーはまだ実装されていません")
    else:
        raise ValueError(f"不明なプロバイダータイプ: {provider_type}")
