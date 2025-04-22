# src/universe/clustering/__init__.py

"""ETFクラスタリングモジュール - 基底クラスと戦略パターン実装"""
import numpy as np
import pandas as pd
import os
import time
import warnings
from typing import List, Dict, Any, Optional, Union

from ...data.cache import DataCache

# キャッシュのシングルトンインスタンス
cache = DataCache()

def create_clusterer(method='tda_optics', **kwargs):
    """クラスタリング戦略のファクトリー関数
    
    Args:
        method: クラスタリング手法
        **kwargs: 追加パラメータ
        
    Returns:
        BaseClusterer: クラスタリング戦略のインスタンス
    """
    from .tda_clusterer import TDAClusterer
    from .optics_clusterer import OPTICSClusterer
    from .pca_optics_clusterer import PCAOPTICSClusterer
    from .tsne_optics_clusterer import TSNEOPTICSClusterer
    from .tda_optics_clusterer import TDAOPTICSClusterer
    from .ensemble_clusterer import EnsembleClusterer
    
    if method == 'tda':
        return TDAClusterer(**kwargs)
    elif method == 'optics':
        return OPTICSClusterer(**kwargs)
    elif method == 'pca_optics':
        return PCAOPTICSClusterer(**kwargs)
    elif method == 'tsne_optics':
        return TSNEOPTICSClusterer(**kwargs)
    elif method == 'tda_optics':
        return TDAOPTICSClusterer(**kwargs)
    elif method == 'ensemble':
        return EnsembleClusterer(**kwargs)
    else:
        warnings.warn(f"未知のクラスタリング手法: {method}、tda_opticsを使用します")
        return TDAOPTICSClusterer(**kwargs)

def cluster_etfs(etfs, method='tda_optics', **kwargs):
    """ETFのクラスタリングを実行する利便性関数
    
    Args:
        etfs: ETF情報のリスト
        method: クラスタリング手法
        **kwargs: 追加パラメータ
        
    Returns:
        list: 選択されたETFのリスト
    """
    clusterer = create_clusterer(method, **kwargs)
    return clusterer.cluster(etfs)
