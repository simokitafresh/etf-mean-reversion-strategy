"""データキャッシュモジュール - シングルトン実装"""
import json
import os
import time
import datetime
import hashlib
import pickle
import warnings
import shutil
from typing import Dict, Any, Optional, Union, List, Tuple

class DataCache:
    """データキャッシュシングルトンクラス"""
    _instance = None
    
    def __new__(cls):
        """シングルトンパターン実装のための __new__ メソッド"""
        if cls._instance is None:
            cls._instance = super(DataCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """インスタンスの初期化（最初のインスタンス作成時のみ実行）"""
        if getattr(self, '_initialized', False):
            return
        
        # キャッシュ保存先ディレクトリ
        self._cache_dir = os.path.join("data", "cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # インメモリキャッシュ
        self._memory_cache = {}
        self._file_cache = {}
        
        # キャッシュメタデータ
        self._cache_metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "last_access": {},
            "ttl": {}  # キー: 有効期限（秒）
        }
        
        # 統計情報
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "expired": 0,
            "evictions": 0
        }
        
        # 設定
        self._config = {
            "max_memory_entries": 1000,  # メモリキャッシュの最大エントリ数
            "max_file_entries": 5000,    # ファイルキャッシュの最大エントリ数
            "default_ttl": 86400,        # デフォルトの有効期限（1日）
            "persist_on_exit": True,     # 終了時にキャッシュを永続化
            "eviction_policy": "lru"     # 削除ポリシー（"lru" または "fifo"）
        }
        
        # 永続化されたキャッシュの読み込み
        self._load_cache()
        
        self._initialized = True
        print(f"DataCacheシングルトンを初期化しました（メモリエントリ: {len(self._memory_cache)}, ファイルエントリ: {len(self._file_cache)}）")
    
    def _load_cache(self):
        """永続化されたキャッシュを読み込む"""
        try:
            metadata_path = os.path.join(self._cache_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self._cache_metadata = json.load(f)
            
            memory_cache_path = os.path.join(self._cache_dir, "memory_cache.pickle")
            if os.path.exists(memory_cache_path):
                with open(memory_cache_path, 'rb') as f:
                    self._memory_cache = pickle.load(f)
            
            file_index_path = os.path.join(self._cache_dir, "file_index.json")
            if os.path.exists(file_index_path):
                with open(file_index_path, 'r', encoding='utf-8') as f:
                    self._file_cache = json.load(f)
            
            # 有効期限チェック
            self._check_expirations()
            
        except Exception as e:
            warnings.warn(f"キャッシュ読み込みエラー: {str(e)}")
            # 読み込みに失敗した場合は空のキャッシュで開始
            self._memory_cache = {}
            self._file_cache = {}
            self._cache_metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "last_access": {},
                "ttl": {}
            }
    
    def _persist_cache(self):
        """キャッシュを永続化する"""
        try:
            metadata_path = os.path.join(self._cache_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache_metadata, f, indent=2)
            
            memory_cache_path = os.path.join(self._cache_dir, "memory_cache.pickle")
            with open(memory_cache_path, 'wb') as f:
                pickle.dump(self._memory_cache, f)
            
            file_index_path = os.path.join(self._cache_dir, "file_index.json")
            with open(file_index_path, 'w', encoding='utf-8') as f:
                json.dump(self._file_cache, f, indent=2)
            
            return True
        except Exception as e:
            warnings.warn(f"キャッシュ永続化エラー: {str(e)}")
            return False
    
    def _check_expirations(self):
        """有効期限切れのキャッシュエントリを削除"""
        now = time.time()
        
        # メモリキャッシュの確認
        expired_keys = []
        for key in self._memory_cache:
            if key in self._cache_metadata["ttl"]:
                expiration = self._cache_metadata["ttl"][key]
                last_access = self._cache_metadata["last_access"].get(key, 0)
                
                if expiration > 0 and now - last_access > expiration:
                    expired_keys.append(key)
        
        # 期限切れエントリの削除
        for key in expired_keys:
            del self._memory_cache[key]
            if key in self._cache_metadata["ttl"]:
                del self._cache_metadata["ttl"][key]
            if key in self._cache_metadata["last_access"]:
                del self._cache_metadata["last_access"][key]
            self._stats["expired"] += 1
        
        # ファイルキャッシュの確認も同様に実施
        expired_file_keys = []
        for key in self._file_cache:
            if key in self._cache_metadata["ttl"]:
                expiration = self._cache_metadata["ttl"][key]
                last_access = self._cache_metadata["last_access"].get(key, 0)
                
                if expiration > 0 and now - last_access > expiration:
                    expired_file_keys.append(key)
        
        # 期限切れファイルの削除
        for key in expired_file_keys:
            file_path = self._file_cache[key]
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    warnings.warn(f"ファイル削除エラー ({file_path}): {str(e)}")
            
            del self._file_cache[key]
            if key in self._cache_metadata["ttl"]:
                del self._cache_metadata["ttl"][key]
            if key in self._cache_metadata["last_access"]:
                del self._cache_metadata["last_access"][key]
            self._stats["expired"] += 1
    
    def _make_room(self, is_file_cache=False):
        """キャッシュがいっぱいの場合、削除ポリシーに従ってエントリを削除する"""
        if is_file_cache:
            cache_dict = self._file_cache
            max_entries = self._config["max_file_entries"]
        else:
            cache_dict = self._memory_cache
            max_entries = self._config["max_memory_entries"]
        
        if len(cache_dict) <= max_entries:
            return
        
        # 削除するエントリ数
        to_remove = len(cache_dict) - max_entries + 10  # バッファとして10個余分に削除
        
        # アクセス時間でソート
        sorted_keys = sorted(
            cache_dict.keys(),
            key=lambda k: self._cache_metadata["last_access"].get(k, 0)
        )
        
        # 最も古いエントリから削除
        for key in sorted_keys[:to_remove]:
            if is_file_cache:
                file_path = cache_dict[key]
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
            
            del cache_dict[key]
            if key in self._cache_metadata["ttl"]:
                del self._cache_metadata["ttl"][key]
            if key in self._cache_metadata["last_access"]:
                del self._cache_metadata["last_access"][key]
            
            self._stats["evictions"] += 1
    
    def get_json(self, key: str, default: Any = None) -> Any:
        """JSONデータをキャッシュから取得
        
        Args:
            key: キャッシュキー
            default: キーが存在しない場合の戻り値
            
        Returns:
            Any: キャッシュから取得したデータ、存在しない場合はdefault
        """
        # 期限切れチェック
        self._check_expirations()
        
        if key in self._memory_cache:
            # アクセス時間を更新
            self._cache_metadata["last_access"][key] = time.time()
            self._stats["hits"] += 1
            return self._memory_cache[key]
        
        self._stats["misses"] += 1
        return default
    
    def set_json(self, key: str, data: Any, ttl: int = None) -> None:
        """JSONデータをキャッシュに保存
        
        Args:
            key: キャッシュキー
            data: 保存するデータ
            ttl: 有効期限（秒）、Noneの場合はデフォルト値を使用
        """
        # 容量チェック
        self._make_room()
        
        # データを保存
        self._memory_cache[key] = data
        
        # メタデータを更新
        self._cache_metadata["last_access"][key] = time.time()
        if ttl is not None:
            self._cache_metadata["ttl"][key] = ttl
        elif key not in self._cache_metadata["ttl"]:
            self._cache_metadata["ttl"][key] = self._config["default_ttl"]
        
        self._stats["sets"] += 1
        
        # 一定間隔でキャッシュを永続化（任意）
        if self._stats["sets"] % 100 == 0 and self._config["persist_on_exit"]:
            self._persist_cache()
    
    def get_file(self, file_path: str, binary: bool = True) -> Optional[Union[bytes, str]]:
        """ファイルをキャッシュから取得
        
        Args:
            file_path: ファイルパス
            binary: バイナリモードで読み込むかどうか
            
        Returns:
            Union[bytes, str]: ファイルデータ、存在しない場合はNone
        """
        # ファイルパスのハッシュをキーとして使用
        key = hashlib.md5(file_path.encode()).hexdigest()
        
        # 期限切れチェック
        self._check_expirations()
        
        if key in self._file_cache:
            cache_path = self._file_cache[key]
            
            if os.path.exists(cache_path):
                try:
                    mode = 'rb' if binary else 'r'
                    encoding = None if binary else 'utf-8'
                    
                    with open(cache_path, mode, encoding=encoding) as f:
                        data = f.read()
                    
                    # アクセス時間を更新
                    self._cache_metadata["last_access"][key] = time.time()
                    self._stats["hits"] += 1
                    
                    return data
                except Exception as e:
                    warnings.warn(f"ファイル読み込みエラー ({cache_path}): {str(e)}")
        
        # キャッシュにない場合は元のファイルを読み込む
        if os.path.exists(file_path):
            try:
                mode = 'rb' if binary else 'r'
                encoding = None if binary else 'utf-8'
                
                with open(file_path, mode, encoding=encoding) as f:
                    data = f.read()
                
                # キャッシュに保存
                self.set_file(file_path, data, binary=binary)
                
                return data
            except Exception as e:
                warnings.warn(f"ファイル読み込みエラー ({file_path}): {str(e)}")
        
        self._stats["misses"] += 1
        return None
    
    def set_file(self, file_path: str, data: Union[bytes, str], ttl: int = None, binary: bool = True) -> bool:
        """ファイルをキャッシュに保存
        
        Args:
            file_path: 元のファイルパス
            data: ファイルデータ
            ttl: 有効期限（秒）、Noneの場合はデフォルト値を使用
            binary: バイナリモードで書き込むかどうか
            
        Returns:
            bool: 保存が成功したかどうか
        """
        # ファイルパスのハッシュをキーとして使用
        key = hashlib.md5(file_path.encode()).hexdigest()
        
        # 容量チェック
        self._make_room(is_file_cache=True)
        
        # キャッシュディレクトリを確認
        file_cache_dir = os.path.join(self._cache_dir, "files")
        os.makedirs(file_cache_dir, exist_ok=True)
        
        # キャッシュファイルパス
        cache_file_name = f"{key}_{os.path.basename(file_path)}"
        cache_path = os.path.join(file_cache_dir, cache_file_name)
        
        try:
            # データを保存
            mode = 'wb' if binary else 'w'
            encoding = None if binary else 'utf-8'
            
            with open(cache_path, mode, encoding=encoding) as f:
                f.write(data)
            
            # メタデータを更新
            self._file_cache[key] = cache_path
            self._cache_metadata["last_access"][key] = time.time()
            
            if ttl is not None:
                self._cache_metadata["ttl"][key] = ttl
            elif key not in self._cache_metadata["ttl"]:
                self._cache_metadata["ttl"][key] = self._config["default_ttl"]
            
            self._stats["sets"] += 1
            return True
            
        except Exception as e:
            warnings.warn(f"ファイル保存エラー ({cache_path}): {str(e)}")
            return False
    
    def clear(self) -> None:
        """キャッシュを完全にクリア"""
        # メモリキャッシュをクリア
        self._memory_cache.clear()
        
        # ファイルキャッシュをクリア
        for key, file_path in self._file_cache.items():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        
        self._file_cache.clear()
        
        # メタデータをリセット
        self._cache_metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "last_access": {},
            "ttl": {}
        }
        
        print("キャッシュを完全にクリアしました")
    
    def clear_key(self, key: str) -> bool:
        """特定のキーのキャッシュをクリア
        
        Args:
            key: クリアするキャッシュキー
            
        Returns:
            bool: キーが存在してクリアされたかどうか
        """
        result = False
        
        # メモリキャッシュから削除
        if key in self._memory_cache:
            del self._memory_cache[key]
            result = True
        
        # ファイルキャッシュから削除
        file_key = hashlib.md5(key.encode()).hexdigest()
        if file_key in self._file_cache:
            file_path = self._file_cache[file_key]
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            
            del self._file_cache[file_key]
            result = True
        
        # メタデータから削除
        if key in self._cache_metadata["ttl"]:
            del self._cache_metadata["ttl"][key]
        
        if key in self._cache_metadata["last_access"]:
            del self._cache_metadata["last_access"][key]
        
        return result
    
    def clear_pattern(self, pattern: str) -> int:
        """パターンに一致するキーのキャッシュをクリア
        
        Args:
            pattern: 検索パターン（部分一致）
            
        Returns:
            int: クリアされたキャッシュの数
        """
        # メモリキャッシュから検索
        memory_keys = [k for k in self._memory_cache if pattern in k]
        
        # ファイルキャッシュから検索
        file_keys = [k for k in self._file_cache if pattern in k]
        
        # 一致するキーをクリア
        for key in memory_keys:
            self.clear_key(key)
        
        for key in file_keys:
            self.clear_key(key)
        
        return len(memory_keys) + len(file_keys)
    
    def get_size(self) -> Dict[str, int]:
        """キャッシュのサイズ情報を取得
        
        Returns:
            Dict[str, int]: キャッシュサイズ情報
        """
        # ファイルキャッシュの実際のサイズを計算
        file_size = 0
        for _, file_path in self._file_cache.items():
            if os.path.exists(file_path):
                file_size += os.path.getsize(file_path)
        
        return {
            "memory_entries": len(self._memory_cache),
            "file_entries": len(self._file_cache),
            "total_entries": len(self._memory_cache) + len(self._file_cache),
            "file_size_bytes": file_size
        }
    
    def get_keys(self, pattern: str = None) -> List[str]:
        """キャッシュのキー一覧を取得
        
        Args:
            pattern: 検索パターン（部分一致）
            
        Returns:
            List[str]: キャッシュキーのリスト
        """
        memory_keys = list(self._memory_cache.keys())
        
        if pattern:
            return [k for k in memory_keys if pattern in k]
        
        return memory_keys
    
    def get_stats(self) -> Dict[str, int]:
        """キャッシュの統計情報を取得
        
        Returns:
            Dict[str, int]: 統計情報
        """
        size_info = self.get_size()
        return {**self._stats, **size_info}
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """キャッシュの設定を更新
        
        Args:
            config: 設定辞書
        """
        for key, value in config.items():
            if key in self._config:
                self._config[key] = value
    
    def get_config(self) -> Dict[str, Any]:
        """キャッシュの設定を取得
        
        Returns:
            Dict[str, Any]: 現在の設定
        """
        return dict(self._config)
    
    def optimize(self) -> Dict[str, Any]:
        """キャッシュを最適化（期限切れ削除と永続化）
        
        Returns:
            Dict[str, Any]: 最適化の結果
        """
        before_size = self.get_size()
        
        # 期限切れエントリを削除
        self._check_expirations()
        
        # キャッシュを永続化
        persisted = self._persist_cache()
        
        after_size = self.get_size()
        
        return {
            "before": before_size,
            "after": after_size,
            "expired_removed": before_size["total_entries"] - after_size["total_entries"],
            "persisted": persisted
        }
    
    def __del__(self):
        """デストラクタ - インスタンス破棄時にキャッシュを永続化"""
        if hasattr(self, '_config') and self._config.get("persist_on_exit", False):
            self._persist_cache()
    
    def __str__(self) -> str:
        """キャッシュの文字列表現
        
        Returns:
            str: キャッシュの情報
        """
        size_info = self.get_size()
        return f"DataCache: {size_info['total_entries']}エントリ（メモリ: {size_info['memory_entries']}, ファイル: {size_info['file_entries']}）, {size_info['file_size_bytes']/1024/1024:.2f}MB"
    
    def __repr__(self) -> str:
        """キャッシュのプログラム的表現
        
        Returns:
            str: キャッシュの詳細情報
        """
        return f"DataCache(stats={self._stats}, size={self.get_size()})"
