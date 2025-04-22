"""ETFユニバース選定モジュール"""
from .liquidity import screen_liquidity
from .correlation import correlation_filtering
from ..data.preprocess import data_quality_check

def select_universe(base_list=None, target_count=50, clustering_method='optics'):
    """ETFユニバースを選定する統合関数
    
    Args:
        base_list: 基礎ETFリスト。Noneの場合は自動取得
        target_count: 目標ETF数
        clustering_method: クラスタリング手法。'optics'または'tda'
    """
    from ..data.fetch import get_base_etf_list
    import os
    import pandas as pd
    import json
    
    # クラスタリング手法に応じたモジュールをインポート
    if clustering_method == 'optics':
        from .optics_clustering import perform_clustering as clustering_func
    else:  # 'tda' (デフォルト)
        from .clustering import tda_clustering as clustering_func
    
    # キャッシュから取得を試みる
    from ..data.cache import DataCache
    cache = DataCache()
    cache_key = f"final_etf_universe_{target_count}_{clustering_method}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print(f"キャッシュから最終ETFユニバース({clustering_method})を取得しました")
        return cached_data
    
    # 結果ディレクトリの作成
    os.makedirs("data/results", exist_ok=True)
    
    print("ステップ1: 基礎候補リスト取得中...")
    if base_list is None:
        base_list = get_base_etf_list()
    print(f"  基礎候補リスト: {len(base_list)}銘柄")
    
    print("ステップ2: 流動性スクリーニング中...")
    liquid_etfs = screen_liquidity(base_list)
    print(f"  流動性条件通過: {len(liquid_etfs)}銘柄")
    
    print("ステップ3: 相関フィルタリング中...")
    filtered_etfs = correlation_filtering(liquid_etfs, target_count)
    print(f"  相関フィルタ通過: {len(filtered_etfs)}銘柄")
    
    print(f"ステップ4: 構造的分散確保（{clustering_method.upper()}アプローチ）...")
    clustered_etfs = clustering_func(filtered_etfs)
    print(f"  クラスタリング後: {len(clustered_etfs)}銘柄")
    
    print("ステップ5: データ品質確認...")
    final_etfs = data_quality_check(clustered_etfs)
    print(f"  最終ETFユニバース: {len(final_etfs)}銘柄")
    
    # 結果をJSONとCSVに保存
    with open(f"data/results/final_etf_universe_{clustering_method}.json", 'w') as f:
        json.dump(final_etfs, f, indent=2)
    
    # 読みやすいCSV形式でも保存
    df = pd.DataFrame(final_etfs)
    df.to_csv(f"data/results/final_etf_universe_{clustering_method}.csv", index=False)
    
    print(f"最終ETFユニバースを保存しました:")
    print(f"  data/results/final_etf_universe_{clustering_method}.json")
    print(f"  data/results/final_etf_universe_{clustering_method}.csv")
    
    # 最終リストの概要を表示
    print("\n最終ETFユニバース概要:")
    if len(final_etfs) > 0:
        if 'symbol' in final_etfs[0] and 'name' in final_etfs[0]:
            for etf in final_etfs:
                print(f"  {etf['symbol']} - {etf['name']}")
    
    # キャッシュに保存
    cache.set_json(cache_key, final_etfs)
    
    return final_etfs
