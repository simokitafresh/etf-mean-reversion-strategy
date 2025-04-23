# src/data/cache_manager.py

"""キャッシュマネージャーモジュール - 統一的なキャッシュアクセスを提供"""
import logging
from typing import Optional, Any, Dict, Union, List
from .cache import DataCache

# ロガーの設定
logger = logging.getLogger(__name__)

class CacheManager:
    """キャッシュアクセスを一元管理するファサードクラス
    
    このクラスは、プロジェクト全体でのキャッシュアクセスを統一し、
    依存関係を単純化します。シングルトンパターンを使用して、
    アプリケーション全体で一貫したキャッシュインスタンスを提供します。
    """
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'CacheManager':
        """シングルトンインスタンスへのアクセスを提供
        
        Returns:
            CacheManager: キャッシュマネージャーのシングルトンインスタンス
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """内部使用のみ - get_instance()を使用してください"""
        try:
            self._cache = DataCache()
            self._available = True
        except Exception as e:
            logger.error(f"キャッシュシステムの初期化に失敗しました: {str(e)}")
            self._available = False
    
    @property
    def is_available(self) -> bool:
        """キャッシュが利用可能かどうかを確認
        
        Returns:
            bool: キャッシュが利用可能な場合はTrue
        """
        return self._available
    
    def get_json(self, key: str, default: Any = None) -> Any:
        """JSONデータをキャッシュから取得する
        
        Args:
            key: キャッシュキー
            default: キーが存在しない場合のデフォルト値
            
        Returns:
            Any: キャッシュされたデータまたはデフォルト値
        """
        if not self._available:
            return default
            
        try:
            return self._cache.get_json(key, default)
        except Exception as e:
            logger.warning(f"キャッシュからの読み込みに失敗しました (キー={key}): {str(e)}")
            return default
    
    def set_json(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """JSONデータをキャッシュに保存する
        
        Args:
            key: キャッシュキー
            data: 保存するデータ
            ttl: 有効期限（秒）、Noneの場合はデフォルト
            
        Returns:
            bool: 保存が成功したかどうか
        """
        if not self._available:
            return False
            
        try:
            self._cache.set_json(key, data, ttl)
            return True
        except Exception as e:
            logger.warning(f"キャッシュへの保存に失敗しました (キー={key}): {str(e)}")
            return False
    
    def get_file(self, file_path: str, binary: bool = True) -> Optional[Union[bytes, str]]:
        """ファイルをキャッシュから取得する
        
        Args:
            file_path: ファイルパス
            binary: バイナリモードで読み込むかどうか
            
        Returns:
            Optional[Union[bytes, str]]: ファイルデータまたはNone
        """
        if not self._available:
            return None
            
        try:
            return self._cache.get_file(file_path, binary)
        except Exception as e:
            logger.warning(f"キャッシュからのファイル読み込みに失敗しました (パス={file_path}): {str(e)}")
            return None
    
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
        if not self._available:
            return False
            
        try:
            return self._cache.set_file(file_path, data, ttl, binary)
        except Exception as e:
            logger.warning(f"キャッシュへのファイル保存に失敗しました (パス={file_path}): {str(e)}")
            return False
    
    def clear(self) -> bool:
        """キャッシュを完全にクリア
        
        Returns:
            bool: クリアが成功したかどうか
        """
        if not self._available:
            return False
            
        try:
            self._cache.clear()
            return True
        except Exception as e:
            logger.warning(f"キャッシュのクリアに失敗しました: {str(e)}")
            return False
    
    def clear_key(self, key: str) -> bool:
        """特定のキーのキャッシュをクリアする
        
        Args:
            key: クリアするキー
            
        Returns:
            bool: クリアが成功したかどうか
        """
        if not self._available:
            return False
            
        try:
            return self._cache.clear_key(key)
        except Exception as e:
            logger.warning(f"キーのクリアに失敗しました (キー={key}): {str(e)}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """パターンに一致するキーのキャッシュをクリアする
        
        Args:
            pattern: 検索パターン（部分一致）
            
        Returns:
            int: クリアされたキャッシュの数
        """
        if not self._available:
            return 0
            
        try:
            return self._cache.clear_pattern(pattern)
        except Exception as e:
            logger.warning(f"パターンによるクリアに失敗しました (パターン={pattern}): {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュの統計情報を取得する
        
        Returns:
            Dict[str, Any]: 統計情報
        """
        if not self._available:
            return {
                'available': False,
                'error': 'キャッシュシステムが利用できません'
            }
            
        try:
            stats = self._cache.get_stats()
            stats['available'] = True
            return stats
        except Exception as e:
            logger.warning(f"統計情報の取得に失敗しました: {str(e)}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """キャッシュのキー一覧を取得する
        
        Args:
            pattern: 検索パターン（部分一致）
            
        Returns:
            List[str]: キャッシュキーのリスト
        """
        if not self._available:
            return []
            
        try:
            return self._cache.get_keys(pattern)
        except Exception as e:
            logger.warning(f"キー一覧の取得に失敗しました: {str(e)}")
            return []
    
    def optimize(self) -> Dict[str, Any]:
        """キャッシュを最適化する（期限切れ削除と永続化）
        
        Returns:
            Dict[str, Any]: 最適化結果
        """
        if not self._available:
            return {
                'success': False,
                'error': 'キャッシュシステムが利用できません'
            }
            
        try:
            result = self._cache.optimize()
            result['success'] = True
            return result
        except Exception as e:
            logger.warning(f"キャッシュの最適化に失敗しました: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
