# src/parameters/context.py（新規ファイル）

"""パラメータ評価のコンテキスト管理"""
from typing import Protocol, Dict, List, Any, Optional, Callable
import pandas as pd

class SignalProvider(Protocol):
    """シグナル計算プロバイダーのインターフェース"""
    
    def calculate_signals_for_universe(
        self,
        universe: List[Dict[str, Any]],
        parameter_sets: List[Dict[str, Any]],
        period: str = "5y",
        min_samples: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """ETFユニバース全体のシグナルを計算する"""
        ...

class ParameterContext:
    """パラメータ評価の設定とコンテキスト"""
    
    def __init__(self, signal_provider: Optional[SignalProvider] = None):
        """コンテキストの初期化
        
        Args:
            signal_provider: シグナル計算プロバイダー
        """
        self.signal_provider = signal_provider
        
        # デフォルトのプロバイダーを設定
        if self.signal_provider is None:
            from src.signals import calculate_signals_for_universe
            self.signal_provider = SignalProviderAdapter(calculate_signals_for_universe)
    
    def run_grid_search(
        self,
        universe: List[Dict],
        param_grid: List[Dict],
        signals_data: Dict = None,
        recalculate: bool = False
    ) -> Dict[str, Any]:
        """パラメータグリッドサーチを実行する
        
        Args:
            universe: ETFユニバース
            param_grid: パラメータグリッド
            signals_data: 既存のシグナルデータ（なければ新規計算）
            recalculate: 既存の結果を再計算するかどうか
            
        Returns:
            Dict: グリッドサーチ結果
        """
        from .grid_search import run_grid_search as _run_grid_search
        
        # シグナルデータが提供されていない場合は計算
        if signals_data is None and self.signal_provider is not None:
            signals_data = self.signal_provider.calculate_signals_for_universe(universe, param_grid)
        
        return _run_grid_search(universe, param_grid, signals_data, recalculate)

class SignalProviderAdapter:
    """関数をSignalProviderインターフェースに適合させるアダプター"""
    
    def __init__(self, calculate_func: Callable):
        """アダプターの初期化
        
        Args:
            calculate_func: シグナル計算関数
        """
        self.calculate_func = calculate_func
    
    def calculate_signals_for_universe(
        self,
        universe: List[Dict[str, Any]],
        parameter_sets: List[Dict[str, Any]],
        period: str = "5y",
        min_samples: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """ETFユニバース全体のシグナルを計算する"""
        return self.calculate_func(universe, parameter_sets, period, min_samples)
