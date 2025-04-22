"""キャッシュユーティリティモジュール - 中央集権的なキャッシュアクセス"""
from ..data.cache import DataCache
from typing import Dict, Any, Optional, Union, List

# シングルトンインスタンスを取得する関数
def get_cache() -> DataCache:
    """データキャッシュのシングルトンインスタンスを取得する
    
    Returns:
        DataCache: キャッシュのシングルトンインスタンス
    """
    return DataCache()

def clear_cache() -> None:
    """キャッシュを完全にクリアする"""
    cache = get_cache()
    cache.clear()
    print("キャッシュをクリアしました")

def clear_cache_key(key: str) -> bool:
    """特定のキーのキャッシュをクリアする
    
    Args:
        key: クリアするキャッシュキー
        
    Returns:
        bool: キーが存在してクリアされたかどうか
    """
    cache = get_cache()
    return cache.clear_key(key)

def clear_cache_pattern(pattern: str) -> int:
    """パターンに一致するキャッシュエントリをクリアする
    
    Args:
        pattern: 検索パターン（部分一致）
        
    Returns:
        int: クリアされたエントリ数
    """
    cache = get_cache()
    count = 0
    for key in cache.get_keys():
        if pattern in key:
            if cache.clear_key(key):
                count += 1
    return count

def get_cache_stats() -> Dict[str, Any]:
    """キャッシュの統計情報を取得する
    
    Returns:
        Dict[str, Any]: 統計情報
    """
    cache = get_cache()
    return cache.get_stats()

def get_json_from_cache(key: str, default: Any = None) -> Any:
    """JSONデータをキャッシュから取得する
    
    Args:
        key: キャッシュキー
        default: キーが存在しない場合のデフォルト値
        
    Returns:
        Any: キャッシュされたデータまたはデフォルト値
    """
    cache = get_cache()
    return cache.get_json(key, default)

def set_json_in_cache(key: str, data: Any, ttl: Optional[int] = None) -> None:
    """JSONデータをキャッシュに保存する
    
    Args:
        key: キャッシュキー
        data: 保存するデータ
        ttl: 有効期限（秒）、Noneの場合はデフォルト
    """
    cache = get_cache()
    cache.set_json(key, data, ttl)

def get_file_from_cache(file_path: str, binary: bool = True) -> Optional[Union[bytes, str]]:
    """ファイルをキャッシュから取得する
    
    Args:
        file_path: ファイルパス
        binary: バイナリモードで読み込むかどうか
        
    Returns:
        Optional[Union[bytes, str]]: ファイルデータまたはNone
    """
    cache = get_cache()
    return cache.get_file(file_path, binary)

def set_file_in_cache(file_path: str, data: Union[bytes, str], 
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
    cache = get_cache()
    return cache.set_file(file_path, data, ttl, binary)

def optimize_cache() -> Dict[str, Any]:
    """キャッシュを最適化する（期限切れ削除と永続化）
    
    Returns:
        Dict[str, Any]: 最適化結果
    """
    cache = get_cache()
    return cache.optimize()

def get_cache_keys(pattern: Optional[str] = None) -> List[str]:
    """キャッシュのキー一覧を取得する
    
    Args:
        pattern: 検索パターン（部分一致）
        
    Returns:
        List[str]: キャッシュキーのリスト
    """
    cache = get_cache()
    keys = cache.get_keys()
    
    if pattern:
        return [k for k in keys if pattern in k]
    return keys
