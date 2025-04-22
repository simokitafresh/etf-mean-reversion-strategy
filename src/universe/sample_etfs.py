"""MVPテスト用のサンプルETFモジュール"""
import importlib
import types
import warnings
from typing import List, Dict, Any, Optional, Callable

def get_sample_etfs() -> List[Dict[str, Any]]:
    """MVPテスト用の厳選ETFリスト
    
    Returns:
        List[Dict[str, Any]]: サンプルETFのリスト
    """
    return [
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "category": "US Large Cap", "avg_volume": 75000000, "aum": 380000000000, "estimated_spread": 0.0001},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust", "category": "US Tech", "avg_volume": 50000000, "aum": 180000000000, "estimated_spread": 0.0001},
        {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "category": "US Small Cap", "avg_volume": 25000000, "aum": 60000000000, "estimated_spread": 0.0002},
        {"symbol": "EFA", "name": "iShares MSCI EAFE ETF", "category": "Developed Markets", "avg_volume": 18000000, "aum": 50000000000, "estimated_spread": 0.0002},
        {"symbol": "EEM", "name": "iShares MSCI Emerging Markets ETF", "category": "Emerging Markets", "avg_volume": 35000000, "aum": 28000000000, "estimated_spread": 0.0003},
        {"symbol": "GLD", "name": "SPDR Gold Shares", "category": "Commodities", "avg_volume": 8000000, "aum": 55000000000, "estimated_spread": 0.0002},
        {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "category": "US Treasury Bonds", "avg_volume": 12000000, "aum": 25000000000, "estimated_spread": 0.0002},
        {"symbol": "VNQ", "name": "Vanguard Real Estate Index Fund", "category": "Real Estate", "avg_volume": 5000000, "aum": 30000000000, "estimated_spread": 0.0002}
    ]

def override_universe_selection(module_path: str) -> List[Dict[str, Any]]:
    """ETFユニバース選定関数をオーバーライドして高速サンプルモードにする
    
    Args:
        module_path: src.universeモジュールパス（通常は'src.universe'）
        
    Returns:
        List[Dict[str, Any]]: サンプルETFのリスト
    """
    try:
        # モジュールをインポート
        universe_module = importlib.import_module(module_path)
        
        # オリジナルの関数を保存
        if hasattr(universe_module, 'select_universe'):
            original_select_universe = universe_module.select_universe
        else:
            warnings.warn(f"モジュール {module_path} に select_universe 関数が見つかりません")
            return get_sample_etfs()
        
        # サンプルETFを返す新しい関数
        def sample_select_universe(*args, **kwargs):
            print("🔍 サンプルETFモードを使用します（処理時間短縮のため）")
            sample_etfs = get_sample_etfs()
            
            # clustering_methodパラメータを取得（デフォルトは'tda'）
            clustering_method = kwargs.get('clustering_method', 'tda')
            
            # クラスタ情報を追加（各ETFにクラスタIDを設定）
            for i, etf in enumerate(sample_etfs):
                etf['cluster'] = i % 4  # 0, 1, 2, 3 のクラスタに均等に分配
            
            print(f"📊 選択された{len(sample_etfs)}銘柄のETF (方式: {clustering_method}):")
            for etf in sample_etfs:
                print(f"  • {etf['symbol']}: {etf['name']} ({etf['category']})")
            return sample_etfs
        
        # 関数をオーバーライド
        universe_module.select_universe = sample_select_universe
        
        # 元の関数を保持（復元用）
        if not hasattr(universe_module, 'original_select_universe'):
            universe_module.original_select_universe = original_select_universe
        
        print("✅ ETFユニバース選定関数をサンプルモードに切り替えました")
        print("💡 ヒント: 完全なユニバース選定に戻すには、以下を実行してください:")
        print("   import src.universe; src.universe.select_universe = src.universe.original_select_universe")
        
        return get_sample_etfs()
        
    except Exception as e:
        warnings.warn(f"ユニバース選定関数のオーバーライドに失敗しました: {str(e)}")
        print("サンプルETFリストを代わりに返します")
        return get_sample_etfs()

def restore_universe_selection(module_path: str) -> bool:
    """ETFユニバース選定関数を元に戻す
    
    Args:
        module_path: src.universeモジュールパス（通常は'src.universe'）
        
    Returns:
        bool: 復元が成功したかどうか
    """
    try:
        # モジュールをインポート
        universe_module = importlib.import_module(module_path)
        
        # 元の関数が保存されているか確認
        if hasattr(universe_module, 'original_select_universe'):
            # 元の関数に戻す
            universe_module.select_universe = universe_module.original_select_universe
            print("✅ ETFユニバース選定関数を元に戻しました")
            return True
        else:
            warnings.warn("元のselect_universe関数が保存されていません")
            return False
            
    except Exception as e:
        warnings.warn(f"ユニバース選定関数の復元に失敗しました: {str(e)}")
        return False
