"""キャッシュ管理のためのユーティリティ - Google Colab最適化版"""
import os
import json
import pickle
import hashlib
import time
from datetime import datetime, timedelta

class DataCache:
    """データキャッシュを管理するクラス（Google Colab最適化）"""
    
    def __init__(self, cache_dir="data/cache", max_age_days=7, verbose=True):
        """キャッシュの初期化
        
        Args:
            cache_dir: キャッシュディレクトリのパス
            max_age_days: キャッシュの最大有効期間（日数）
            verbose: 詳細なログを出力するかどうか
        """
        self.cache_dir = cache_dir
        self.max_age_days = max_age_days
        self.verbose = verbose
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'errors': 0
        }
        
        # キャッシュディレクトリが存在しない場合は作成
        os.makedirs(cache_dir, exist_ok=True)
        
        if self.verbose:
            print(f"📦 データキャッシュを初期化しました: {cache_dir}")
    
    def _get_cache_path(self, key, extension):
        """キャッシュのファイルパスを取得
        
        Args:
            key: キャッシュのキー
            extension: ファイル拡張子
            
        Returns:
            str: キャッシュファイルのパス
        """
        # キーの正規化（スペースやスラッシュなどを削除）
        normalized_key = str(key).replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # ハッシュ化してファイル名に使用
        hashed_key = hashlib.md5(normalized_key.encode()).hexdigest()
        
        return os.path.join(self.cache_dir, f"{hashed_key}.{extension}")
    
    def _is_cache_valid(self, cache_path):
        """キャッシュが有効かどうかを確認
        
        Args:
            cache_path: キャッシュファイルのパス
            
        Returns:
            bool: キャッシュが有効かどうか
        """
        if not os.path.exists(cache_path):
            return False
        
        try:
            # 最終更新日時を確認
            last_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
            age = datetime.now() - last_modified
            
            return age.days < self.max_age_days
        except Exception as e:
            if self.verbose:
                print(f"⚠️ キャッシュ有効性チェックエラー: {str(e)}")
            return False
    
    def get_json(self, key, default=None):
        """JSONキャッシュを取得
        
        Args:
            key: キャッシュのキー
            default: キャッシュがない場合のデフォルト値
            
        Returns:
            任意: キャッシュされたデータまたはデフォルト値
        """
        cache_path = self._get_cache_path(key, "json")
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                
                self.stats['hits'] += 1
                if self.verbose:
                    print(f"✅ キャッシュヒット: {key}")
                
                return data
            except Exception as e:
                self.stats['errors'] += 1
                if self.verbose:
                    print(f"⚠️ キャッシュ読み込みエラー ({key}): {str(e)}")
        
        self.stats['misses'] += 1
        if self.verbose:
            print(f"❌ キャッシュミス: {key}")
        
        return default
    
    def set_json(self, key, data):
        """JSONキャッシュを設定
        
        Args:
            key: キャッシュのキー
            data: キャッシュするデータ
            
        Returns:
            bool: 保存が成功したかどうか
        """
        cache_path = self._get_cache_path(key, "json")
        
        try:
            # 親ディレクトリが存在することを確認
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # 一時ファイルに書き込んでから移動（原子的操作）
            temp_path = f"{cache_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
            
            # 一時ファイルを本来のファイルに移動
            os.replace(temp_path, cache_path)
            
            self.stats['saves'] += 1
            if self.verbose:
                print(f"✅ キャッシュ保存: {key}")
            
            return True
        except Exception as e:
            self.stats['errors'] += 1
            if self.verbose:
                print(f"⚠️ キャッシュ保存エラー ({key}): {str(e)}")
            return False
    
    def get_pickle(self, key, default=None):
        """Pickleキャッシュを取得
        
        Args:
            key: キャッシュのキー
            default: キャッシュがない場合のデフォルト値
            
        Returns:
            任意: キャッシュされたデータまたはデフォルト値
        """
        cache_path = self._get_cache_path(key, "pkl")
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.stats['hits'] += 1
                if self.verbose:
                    print(f"✅ キャッシュヒット: {key}")
                
                return data
            except Exception as e:
                self.stats['errors'] += 1
                if self.verbose:
                    print(f"⚠️ キャッシュ読み込みエラー ({key}): {str(e)}")
        
        self.stats['misses'] += 1
        if self.verbose:
            print(f"❌ キャッシュミス: {key}")
        
        return default
    
    def set_pickle(self, key, data):
        """Pickleキャッシュを設定
        
        Args:
            key: キャッシュのキー
            data: キャッシュするデータ
            
        Returns:
            bool: 保存が成功したかどうか
        """
        cache_path = self._get_cache_path(key, "pkl")
        
        try:
            # 親ディレクトリが存在することを確認
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # 一時ファイルに書き込んでから移動（原子的操作）
            temp_path = f"{cache_path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            
            # 一時ファイルを本来のファイルに移動
            os.replace(temp_path, cache_path)
            
            self.stats['saves'] += 1
            if self.verbose:
                print(f"✅ キャッシュ保存: {key}")
            
            return True
        except Exception as e:
            self.stats['errors'] += 1
            if self.verbose:
                print(f"⚠️ キャッシュ保存エラー ({key}): {str(e)}")
            return False
    
    def clear_expired(self):
        """期限切れのキャッシュを削除
        
        Returns:
            int: 削除されたファイルの数
        """
        count = 0
        now = datetime.now()
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json') or filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    
                    # ファイルの最終更新日時を確認
                    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age = now - last_modified
                    
                    if age.days >= self.max_age_days:
                        os.remove(file_path)
                        count += 1
            
            if self.verbose and count > 0:
                print(f"🧹 期限切れのキャッシュを{count}件削除しました")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ キャッシュクリーンアップエラー: {str(e)}")
        
        return count
    
    def get_stats(self):
        """キャッシュの統計情報を取得
        
        Returns:
            dict: 統計情報
        """
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_ratio': self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1),
            'saves': self.stats['saves'],
            'errors': self.stats['errors'],
            'cache_dir': self.cache_dir,
            'max_age_days': self.max_age_days
        }
    
    def print_stats(self):
        """キャッシュの統計情報を表示"""
        stats = self.get_stats()
        print(f"📊 キャッシュ統計:")
        print(f"  - ヒット: {stats['hits']} ({stats['hit_ratio']:.1%})")
        print(f"  - ミス: {stats['misses']}")
        print(f"  - 保存: {stats['saves']}")
        print(f"  - エラー: {stats['errors']}")
        print(f"  - ディレクトリ: {stats['cache_dir']}")
        print(f"  - 最大有効期間: {stats['max_age_days']}日")

# キャッシュのグローバルインスタンス
cache = DataCache()
