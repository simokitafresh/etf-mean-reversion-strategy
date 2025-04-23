# src/universe/_implementation.py

"""ユニバース選定の実装モジュール（内部使用）"""
from typing import List, Dict, Any, Optional
import warnings
import logging

# 内部モジュールの参照にはパッケージの相対インポートを使用
from .liquidity import screen_liquidity
from .correlation import correlation_filtering
from .clustering import create_clusterer
from .sample_etfs import get_sample_etfs as get_samples

# 別パッケージのインポートには絶対インポートを使用
from src.data.fetch import get_base_etf_list
from src.data.preprocess import data_quality_check
from src.data.cache_manager import CacheManager

# ロガーの設定
logger = logging.getLogger(__name__)

# キャッシュマネージャーの取得（シングルトンアクセス）
cache_manager = CacheManager.get_instance()

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
    # キャッシュキーの生成
    cache_key = f"selected_universe_{target_count}_{clustering_method}"
    
    # キャッシュから取得を試みる
    cached_data = cache_manager.get_json(cache_key)
    if cached_data:
        logger.info("キャッシュからユニバース選定結果を取得しました")
        return cached_data
    
    logger.info("ETFユニバースの選定を開始します...")
    
    # 1. 基礎ETFリストの取得
    if base_list is None:
        try:
            logger.info("基礎ETFリストを取得しています...")
            base_list = get_base_etf_list()
        except Exception as e:
            logger.error(f"基礎ETFリストの取得に失敗しました: {str(e)}")
            logger.info("サンプルETFリストを使用します")
            base_list = get_samples()
    
    if not base_list:
        logger.warning("空のETFリストが提供されました。サンプルETFリストを使用します。")
        return get_samples()
    
    # 2. 流動性によるスクリーニング
    logger.info(f"流動性スクリーニングを開始します ({len(base_list)}銘柄)...")
    try:
        # 流動性基準を調整して適切な数のETFを得る
        min_volume = 100000  # 10万株/日
        min_aum = 1000000000  # 10億ドル
        
        liquid_etfs = screen_liquidity(
            base_list,
            min_volume=min_volume,
            min_aum=min_aum
        )
        
        # ETFが少なすぎる場合は基準を緩和
        if len(liquid_etfs) < target_count * 2:
            logger.info(f"流動性基準を緩和します ({len(liquid_etfs)} < {target_count * 2}銘柄)...")
            min_volume = 50000  # 5万株/日
            min_aum = 500000000  # 5億ドル
            
            liquid_etfs = screen_liquidity(
                base_list,
                min_volume=min_volume,
                min_aum=min_aum
            )
        
        logger.info(f"流動性スクリーニング完了: {len(liquid_etfs)}/{len(base_list)}銘柄が通過")
        
    except Exception as e:
        logger.error(f"流動性スクリーニングエラー: {str(e)}")
        # サンプルETFにフォールバック
        logger.info("サンプルETFリストを使用します")
        return get_samples()
    
    # 3. 相関に基づくフィルタリング
    logger.info(f"相関フィルタリングを開始します ({len(liquid_etfs)}銘柄)...")
    try:
        correlation_threshold = 0.95  # 高相関の閾値
        
        filtered_etfs = correlation_filtering(
            liquid_etfs, 
            target_count=target_count,
            correlation_threshold=correlation_threshold
        )
        
        logger.info(f"相関フィルタリング完了: {len(filtered_etfs)}/{len(liquid_etfs)}銘柄が通過")
        
    except Exception as e:
        logger.error(f"相関フィルタリングエラー: {str(e)}")
        filtered_etfs = liquid_etfs
    
    # 4. データ品質チェック
    logger.info(f"データ品質確認を開始します ({len(filtered_etfs)}銘柄)...")
    try:
        quality_etfs = data_quality_check(filtered_etfs)
        logger.info(f"データ品質確認完了: {len(quality_etfs)}/{len(filtered_etfs)}銘柄が通過")
    except Exception as e:
        logger.error(f"データ品質確認エラー: {str(e)}")
        quality_etfs = filtered_etfs
    
    # 5. クラスタリングによる最終選択
    logger.info(f"クラスタリングを開始します ({len(quality_etfs)}銘柄)...")
    try:
        clusterer = create_clusterer(method=clustering_method)
        selected_etfs = clusterer.cluster(quality_etfs)
        
        logger.info(f"クラスタリング完了: {len(selected_etfs)}銘柄が選択されました")
        
    except Exception as e:
        logger.error(f"クラスタリングエラー: {str(e)}")
        # 単純な出来高でのソートとスライスでフォールバック
        selected_etfs = sorted(
            quality_etfs, 
            key=lambda x: x.get('avg_volume', 0) * x.get('aum', 0), 
            reverse=True
        )[:target_count]
        
        logger.info(f"フォールバック選択: 出来高×AUMで上位{len(selected_etfs)}銘柄を選択")
    
    # 目標数に調整
    if len(selected_etfs) > target_count:
        # 出来高×AUMでソートして上位を選択
        selected_etfs = sorted(
            selected_etfs, 
            key=lambda x: x.get('avg_volume', 0) * x.get('aum', 0), 
            reverse=True
        )[:target_count]
        
        logger.info(f"目標数に調整: {target_count}銘柄")
    
    # キャッシュに保存
    cache_manager.set_json(cache_key, selected_etfs)
    
    return selected_etfs
