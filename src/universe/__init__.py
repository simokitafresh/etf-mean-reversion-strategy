# src/universe/__init__.py を更新
"""ETFユニバース選定モジュール"""
from .liquidity import screen_liquidity
from .correlation import correlation_filtering
# 他のサブモジュールも必要に応じてインポート

def select_universe(base_list=None, target_count=50):
    """ETFユニバースを選定する統合関数"""
    from ..data.fetch import get_base_etf_list
    import os
    import pandas as pd
    
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
    
    # 結果を保存
    df = pd.DataFrame(filtered_etfs)
    df.to_csv("data/results/filtered_etfs.csv", index=False)
    print(f"結果を保存しました: data/results/filtered_etfs.csv")
    
    # 後続のステップは実装途中
    # TODO: クラスタリング
    
    return filtered_etfs
