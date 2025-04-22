"""ETFユニバース選定モジュール - パブリックAPI

このモジュールではインポート戦略のベストプラクティスに従っています：
1. 相対インポート: 同一パッケージ内のモジュール参照に使用
2. 絶対インポート: 異なるパッケージからのインポートに使用
3. 公開APIの明示: __all__リストでパッケージの公開インターフェースを定義
"""
from .interface import select_universe, get_sample_etfs

# 公開APIのみをエクスポート
__all__ = ['select_universe', 'get_sample_etfs']
