"""ETFユニバース選定モジュール"""
from .liquidity import screen_liquidity
# 他のサブモジュールも必要に応じてインポート

def select_universe(base_list=None, target_count=50):
    """ETFユニバースを選定する統合関数"""
    from ..data.fetch import get_base_etf_list
    import os
    
    # 結果ディレクトリの作成
    os.makedirs("data/results", exist_ok=True)
    
    print("ステップ1: 基礎候補リスト取得中...")
    if base_list is None:
        base_list = get_base_etf_list()
    print(f"  基礎候補リスト: {len(base_list)}銘柄")
    
    print("ステップ2: 流動性スクリーニング中...")
    liquid_etfs = screen_liquidity(base_list)
    print(f"  流動性条件通過: {len(liquid_etfs)}銘柄")
    
    # 後続のステップは実装途中としておく
    # TODO: 相関フィルタリング
    # TODO: クラスタリング
    
    # 現時点では流動性スクリーニング結果を返す
    return liquid_etfs
