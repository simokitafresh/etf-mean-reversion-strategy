# src/data/cache_manager.py（新規ファイル）

"""キャッシュマネージャーモジュール - 統一的なキャッシュアクセスを提供"""
from typing import Optional, Any, Dict, Union
from .cache import DataCache

class CacheManager:
    """キャッシュアクセスを一元管理するファサードクラス
    
    このクラスは、プロジェクト全体でのキャッシュアクセスを統一し、
    依存関係を単純化します。
    """
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'CacheManager':
        """シングルトンインスタンスへのアクセスを提供"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """内部使用のみ - get_instance()を使用してください"""
        self._cache = DataCache()
    
    def get_json(self, key: str, default: Any = None) -> Any:
        """JSONデータをキャッシュから取得する
        
        Args:
            key: キャッシュキー
            default: キーが存在しない場合のデフォルト値
            
        Returns:
            Any: キャッシュされたデータまたはデフォルト値
        """
        return self._cache.get_json(key, default)
    
    def set_json(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """JSONデータをキャッシュに保存する
        
        Args:
            key: キャッシュキー
            data: 保存するデータ
            ttl: 有効期限（秒）、Noneの場合はデフォルト
        """
        self._cache.set_json(key, data, ttl)
    
    def get_file(self, file_path: str, binary: bool = True) -> Optional[Union[bytes, str]]:
        """ファイルをキャッシュから取得する
        
        Args:
            file_path: ファイルパス
            binary: バイナリモードで読み込むかどうか
            
        Returns:
            Optional[Union[bytes, str]]: ファイルデータまたはNone
        """
        return self._cache.get_file(file_path, binary)
    
    def set_file(self, file_path: str, data: Union[bytes, str], 
                 ttl: Optional[int] = None, binary: bool = True) -> bool:
        """ファイルをキャッシュに保存する
        
        Args:
            file_path: ファイルパス
            data: ファイルデータ
            ttl: 有効期限（秒）、Noneの場合はデフォルト
            binary: バイナリモードで書き込むかどうか
            
        Returns:
            bool: 保存が成功したかどうか
        """
        return self._cache.set_file(file_path, data, ttl, binary)
    
    def clear(self) -> None:
        """キャッシュを完全にクリア"""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュの統計情報を取得する
        
        Returns:
            Dict[str, Any]: 統計情報
        """
        return self._cache.get_stats()
