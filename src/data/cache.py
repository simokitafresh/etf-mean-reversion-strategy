"""キャッシュ管理のためのユーティリティ"""
import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta

class DataCache:
    """データキャッシュを管理するクラス"""
    
    def __init__(self, cache_dir="data/cache", max_age_days=7):
        """キャッシュの初期化"""
        self.cache_dir = cache_dir
        self.max_age_days = max_age_days
        
        # キャッシュディレクトリが存在しない場合は作成
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key, extension):
        """キャッシュのファイルパスを取得"""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.{extension}")
    
    def _is_cache_valid(self, cache_path):
        """キャッシュが有効かどうかを確認"""
        if not os.path.exists(cache_path):
            return False
        
        # 最終更新日時を確認
        last_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - last_modified
        
        return age.days < self.max_age_days
    
    def get_json(self, key):
        """JSONキャッシュを取得"""
        cache_path = self._get_cache_path(key, "json")
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return None
    
    def set_json(self, key, data):
        """JSONキャッシュを設定"""
        cache_path = self._get_cache_path(key, "json")
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
