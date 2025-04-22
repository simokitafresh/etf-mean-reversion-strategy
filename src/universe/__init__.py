"""ETFユニバース選定モジュール"""
import os
import pandas as pd
import json
import warnings
from typing import List, Dict, Any, Optional, Callable

from .liquidity import screen_liquidity
from .correlation import correlation_filtering
from ..data.preprocess import data_quality_check
from ..data.cache import DataCache

# キャッシュのグローバルインスタンス
cache = DataCache()

def select_universe(
    base_list: Optional[List[Dict[str, Any]]] = None, 
    target_count: int = 50, 
    clustering_method: str = 'tda'
) -> List[Dict[str, Any]]:
    """ETFユニバースを選定する統合関数
    
    Args:
        base_list: 基礎ETFリスト。Noneの場合は自動取得
        target_count: 目標ETF数
        clustering_method: クラスタリング手法（'tda', 'optics', 'tsne_optics', 'pca_optics'）
        
    Returns:
        List[Dict[str, Any]]: 選定されたETFユニバース
    """
    from ..data.fetch import get_base_etf_list
    
    # クラスタリング手法に応じたモジュールをインポート
    from .clustering import perform_clustering, tda_clustering
    
    # 互換性のための処理
    if clustering_method == 'tda':
        clustering_func = tda_clustering
    elif clustering_method in ['optics', 'pca_optics', 'tsne_optics']:
        clustering_func = lambda etfs, **kwargs: perform_clustering(etfs, method=clustering_method, **kwargs)
    else:
        print(f"警告: 不明なクラスタリング手法 '{clustering_method}'。'tda'を使用します。")
        clustering_func = tda_clustering
    
    # キャッシュから取得を試みる
    cache_key = f"final_etf_universe_{target_count}_{clustering_method}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print(f"キャッシュから最終ETFユニバース({clustering_method})を取得しました")
        return cached_data
    
    # 結果ディレクトリの作成
    os.makedirs("data/results", exist_ok=True)
    
    print("ステップ1: 基礎候補リスト取得中...")
    if base_list is None:
        try:
            base_list = get_base_etf_list()
            print(f"  基礎候補リスト: {len(base_list)}銘柄")
        except Exception as e:
            print(f"エラー: 基礎候補リストの取得に失敗しました: {str(e)}")
            return []
    else:
        print(f"  提供された基礎候補リスト: {len(base_list)}銘柄")
    
    # 入力チェック
    if not base_list:
        print("エラー: 基礎候補リストが空です")
        return []
    
    print("ステップ2: 流動性スクリーニング中...")
    try:
        liquid_etfs = screen_liquidity(base_list)
        print(f"  流動性条件通過: {len(liquid_etfs)}銘柄")
    except Exception as e:
        print(f"エラー: 流動性スクリーニングに失敗しました: {str(e)}")
        liquid_etfs = base_list
        print("  警告: 流動性スクリーニングをスキップして基礎リストを使用します")
    
    # 流動性チェック後の確認
    if not liquid_etfs:
        print("警告: 流動性条件を満たすETFがありません。基礎リストを使用します。")
        liquid_etfs = base_list
    
    print("ステップ3: 相関フィルタリング中...")
    try:
        filtered_etfs = correlation_filtering(liquid_etfs, target_count)
        print(f"  相関フィルタ通過: {len(filtered_etfs)}銘柄")
    except Exception as e:
        print(f"エラー: 相関フィルタリングに失敗しました: {str(e)}")
        filtered_etfs = liquid_etfs
        print("  警告: 相関フィルタリングをスキップして流動性リストを使用します")
    
    # 相関フィルタリング後の確認
    if not filtered_etfs:
        print("警告: 相関フィルタリング後にETFがありません。流動性リストを使用します。")
        filtered_etfs = liquid_etfs
    
    print(f"ステップ4: 構造的分散確保（{clustering_method.upper()}アプローチ）...")
    try:
        clustered_etfs = clustering_func(filtered_etfs)
        print(f"  クラスタリング後: {len(clustered_etfs)}銘柄")
    except Exception as e:
        print(f"エラー: クラスタリングに失敗しました: {str(e)}")
        clustered_etfs = filtered_etfs
        print("  警告: クラスタリングをスキップして相関フィルタリングリストを使用します")
    
    # クラスタリング後の確認
    if not clustered_etfs:
        print("警告: クラスタリング後にETFがありません。相関フィルタリングリストを使用します。")
        clustered_etfs = filtered_etfs
    
    print("ステップ5: データ品質確認...")
    try:
        final_etfs = data_quality_check(clustered_etfs)
        print(f"  最終ETFユニバース: {len(final_etfs)}銘柄")
    except Exception as e:
        print(f"エラー: データ品質確認に失敗しました: {str(e)}")
        final_etfs = clustered_etfs
        print("  警告: データ品質確認をスキップしてクラスタリングリストを使用します")
    
    # データ品質確認後の確認
    if not final_etfs:
        print("警告: データ品質確認後にETFがありません。クラスタリングリストを使用します。")
        final_etfs = clustered_etfs
    
    # 結果をJSONとCSVに保存
    try:
        # JSON保存
        json_path = f"data/results/final_etf_universe_{clustering_method}.json"
        with open(json_path, 'w') as f:
            json.dump(final_etfs, f, indent=2)
        
        # CSV保存
        csv_path = f"data/results/final_etf_universe_{clustering_method}.csv"
        df = pd.DataFrame(final_etfs)
        df.to_csv(csv_path, index=False)
        
        print(f"最終ETFユニバースを保存しました:")
        print(f"  {json_path}")
        print(f"  {csv_path}")
    except Exception as e:
        print(f"警告: 結果ファイルの保存に失敗しました: {str(e)}")
    
    # 最終リストの概要を表示
    print("\n最終ETFユニバース概要:")
    if len(final_etfs) > 0:
        if all(isinstance(etf, dict) for etf in final_etfs):
            for i, etf in enumerate(final_etfs[:min(10, len(final_etfs))]):
                symbol = etf.get('symbol', 'N/A')
                name = etf.get('name', 'N/A')
                print(f"  {i+1}. {symbol}: {name}")
            
            if len(final_etfs) > 10:
                print(f"  ... 他 {len(final_etfs) - 10} 銘柄")
    
    # キャッシュに保存
    cache.set_json(cache_key, final_etfs)
    
    return final_etfs
