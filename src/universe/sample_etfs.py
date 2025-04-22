"""MVPテスト用のサンプルETFモジュール"""

def get_sample_etfs():
    """MVPテスト用の厳選ETFリスト"""
    return [
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "category": "US Large Cap"},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust", "category": "US Tech"},
        {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "category": "US Small Cap"},
        {"symbol": "EFA", "name": "iShares MSCI EAFE ETF", "category": "Developed Markets"},
        {"symbol": "EEM", "name": "iShares MSCI Emerging Markets ETF", "category": "Emerging Markets"},
        {"symbol": "GLD", "name": "SPDR Gold Shares", "category": "Commodities"},
        {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "category": "US Treasury Bonds"},
        {"symbol": "VNQ", "name": "Vanguard Real Estate Index Fund", "category": "Real Estate"}
    ]

# src/universe/sample_etfs.py の変更部分
def override_universe_selection(module_path):
    """ETFユニバース選定関数をオーバーライドして高速サンプルモードにする
    
    Args:
        module_path: src.universeモジュールパス（通常は'src.universe'）
    """
    import importlib
    import types
    
    # モジュールをインポート
    universe_module = importlib.import_module(module_path)
    
    # オリジナルの関数を保存
    original_select_universe = universe_module.select_universe
    
    # サンプルETFを返す新しい関数
    def sample_select_universe(*args, **kwargs):
        print("🔍 サンプルETFモードを使用します（処理時間短縮のため）")
        sample_etfs = get_sample_etfs()
        
        # clustering_methodパラメータを取得（デフォルトは'stable'）
        clustering_method = kwargs.get('clustering_method', 'stable')
        
        print(f"📊 選択された{len(sample_etfs)}銘柄のETF (方式: {clustering_method}):")
        for etf in sample_etfs:
            print(f"  • {etf['symbol']}: {etf['name']} ({etf['category']})")
        return sample_etfs
    
    # 関数をオーバーライド
    universe_module.select_universe = sample_select_universe
    universe_module.original_select_universe = original_select_universe
    
    print("✅ ETFユニバース選定関数をサンプルモードに切り替えました")
    print("💡 ヒント: 完全なユニバース選定に戻すには、以下を実行してください:")
    print("   import src.universe; src.universe.select_universe = src.universe.original_select_universe")
    
    return sample_etfs
