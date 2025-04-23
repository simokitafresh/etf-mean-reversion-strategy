"""ETFクラスタリングモジュール - 基底クラスと戦略パターン実装"""
import numpy as np
import pandas as pd
import os
import time
import warnings
import logging
from typing import List, Dict, Any, Optional, Union, Type

# 相対インポートの統一（プロジェクト内の別パッケージへの参照）
from src.data.cache_manager import CacheManager

# ロガーの設定
logger = logging.getLogger(__name__)

# キャッシュマネージャーのインスタンスを取得
cache_manager = CacheManager.get_instance()

def create_clusterer(method='tda_optics', **kwargs):
    """クラスタリング戦略のファクトリー関数
    
    Args:
        method: クラスタリング手法
        **kwargs: 追加パラメータ
        
    Returns:
        BaseClusterer: クラスタリング戦略のインスタンス
    """
    # 同一パッケージ内のモジュールには明示的な相対インポートを使用
    from .base_clusterer import BaseClusterer
    
    # 利用可能なクラスタリング戦略の登録
    clusterer_registry = _get_clusterer_registry()
    
    # 指定された方法が登録されているか確認
    if method in clusterer_registry:
        clusterer_class = clusterer_registry[method]
        return clusterer_class(**kwargs)
    else:
        # 登録されていない場合はfallbackを使用
        fallback_method = 'tda_optics'
        warnings.warn(f"未知のクラスタリング手法: {method}、{fallback_method}を使用します")
        
        if fallback_method in clusterer_registry:
            return clusterer_registry[fallback_method](**kwargs)
        else:
            # 最後の手段としてOPTICSクラスタラーを使用
            from .optics_clusterer import OPTICSClusterer
            return OPTICSClusterer(**kwargs)

def _get_clusterer_registry():
    """利用可能なクラスタリング戦略の辞書を取得する
    
    Returns:
        Dict[str, Type[BaseClusterer]]: クラスタリング戦略の辞書
    """
    # 必要なクラスのインポート
    from .base_clusterer import BaseClusterer
    from .tda_clusterer import TDAClusterer
    from .optics_clusterer import OPTICSClusterer
    from .pca_optics_clusterer import PCAOPTICSClusterer
    from .tsne_optics_clusterer import TSNEOPTICSClusterer
    from .tda_optics_clusterer import TDAOPTICSClusterer
    from .ensemble_clusterer import EnsembleClusterer
    
    # 戦略の辞書を作成
    registry = {
        'tda': TDAClusterer,
        'optics': OPTICSClusterer,
        'pca_optics': PCAOPTICSClusterer,
        'tsne_optics': TSNEOPTICSClusterer,
        'tda_optics': TDAOPTICSClusterer,
        'ensemble': EnsembleClusterer
    }
    
    return registry

def register_clusterer(name: str, clusterer_class: Type):
    """新しいクラスタリング戦略を登録する
    
    Args:
        name: 戦略の名前
        clusterer_class: クラスタラーのクラス
    """
    registry = _get_clusterer_registry()
    registry[name] = clusterer_class
    logger.info(f"新しいクラスタリング戦略が登録されました: {name}")

def cluster_etfs(etfs, method='tda_optics', **kwargs):
    """ETFのクラスタリングを実行する利便性関数
    
    Args:
        etfs: ETF情報のリスト
        method: クラスタリング手法
        **kwargs: 追加パラメータ
        
    Returns:
        list: 選択されたETFのリスト
    """
    # 効率化：小さいリストの場合はクラスタリングをスキップ
    if len(etfs) <= 5:
        logger.info(f"銘柄数が少ないため ({len(etfs)} <= 5)、クラスタリングをスキップします")
        return etfs
    
    # キャッシュキーの生成
    symbols = [etf.get('symbol', '') for etf in etfs]
    symbols_str = '-'.join(sorted(symbols[:5])) + f"_{len(symbols)}"
    cache_key = f"clustered_etfs_{symbols_str}_{method}"
    
    # キャッシュから取得を試みる
    cached_data = cache_manager.get_json(cache_key)
    if cached_data:
        logger.info(f"キャッシュからクラスタリング結果を取得しました ({method})")
        return cached_data
    
    # クラスタリング実行
    clusterer = create_clusterer(method, **kwargs)
    result = clusterer.cluster(etfs)
    
    # キャッシュに保存
    if result:
        cache_manager.set_json(cache_key, result)
    
    return result
