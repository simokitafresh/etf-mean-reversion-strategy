# src/utils/monitoring.py

import os
import psutil
import platform
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import sys
import gc
import weakref

class PerformanceMonitor:
    """メモリ使用量とパフォーマンスのモニタリングユーティリティ"""
    
    def __init__(self, log_to_file=False, log_file="data/logs/performance.log"):
        """モニターの初期化
        
        Args:
            log_to_file: ファイルにログを記録するかどうか
            log_file: ログファイルのパス
        """
        self.start_time = time.time()
        self.checkpoints = []
        self.log_to_file = log_to_file
        self.log_file = log_file
        
        # 大きなオブジェクトの弱参照を保持する辞書
        self.large_objects = {}
        
        if log_to_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w') as f:
                f.write(f"Performance Monitor Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"System: {platform.system()} {platform.release()}\n")
                f.write(f"Python: {platform.python_version()}\n")
                f.write("-" * 80 + "\n")
    
    def check_memory(self, checkpoint_name=""):
        """現在のメモリ使用量をチェックし、記録する
        
        Args:
            checkpoint_name: チェックポイントの名前
            
        Returns:
            float: メモリ使用量（GB）
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / (1024 ** 3)  # GB単位に変換
            
            # チェックポイント記録
            elapsed_time = time.time() - self.start_time
            checkpoint = {
                'name': checkpoint_name,
                'time': time.strftime('%H:%M:%S'),
                'elapsed': elapsed_time,
                'memory_gb': memory_usage_gb
            }
            self.checkpoints.append(checkpoint)
            
            # ログファイルに記録
            if self.log_to_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"{checkpoint_name}: {memory_usage_gb:.2f} GB, "
                            f"経過時間: {elapsed_time:.2f}秒\n")
            
            return memory_usage_gb
        
        except Exception as e:
            print(f"メモリ使用量の確認に失敗: {str(e)}")
            return None
    
    def display_current_memory(self):
        """現在のメモリ使用量を表示する（IPythonで使用）"""
        memory_usage_gb = self.check_memory("現在")
        
        if memory_usage_gb is not None:
            # メモリ状態に応じたカラー設定
            color = "green" if memory_usage_gb < 10 else "orange" if memory_usage_gb < 12 else "red"
            status = "正常" if memory_usage_gb < 10 else "警告" if memory_usage_gb < 12 else "危険"
            
            html = f"""
            <div style="margin:10px; padding:10px; border-radius:10px; border:1px solid #ddd;">
                <h3 style="margin-top:0; color:{color};">メモリ使用状況: {status}</h3>
                <p>現在のメモリ使用量: <b>{memory_usage_gb:.2f} GB</b></p>
                <p>経過時間: {time.time() - self.start_time:.2f}秒</p>
                <p>Colabの制限: 約12-13 GB</p>
            </div>
            """
            
            display(HTML(html))
        else:
            print("メモリ使用量を取得できませんでした")
    
    def track_object(self, obj, name):
        """大きなオブジェクトを追跡する
        
        Args:
            obj: 追跡するオブジェクト
            name: オブジェクトの識別名
        """
        self.large_objects[name] = weakref.ref(obj)
        print(f"オブジェクト '{name}' の追跡を開始しました")
    
    def get_summary(self):
        """モニタリング結果のサマリーを表として取得する"""
        if not self.checkpoints:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.checkpoints)
        
        if len(df) > 1:
            # メモリ変化を計算
            df['memory_change'] = df['memory_gb'].diff()
            df.loc[0, 'memory_change'] = df.loc[0, 'memory_gb']
            
            # 経過時間の差分を計算
            df['time_diff'] = df['elapsed'].diff()
            df.loc[0, 'time_diff'] = df.loc[0, 'elapsed']
        
        return df
    
    def plot_memory_usage(self, figsize=(10, 6)):
        """メモリ使用量の推移をプロット"""
        if not self.checkpoints:
            print("チェックポイントがありません")
            return
        
        df = self.get_summary()
        
        plt.figure(figsize=figsize)
        plt.plot(df['elapsed'], df['memory_gb'], 'b-o', linewidth=2)
        plt.axhline(y=12, color='r', linestyle='--', label='Colab制限')
        
        for i, r in df.iterrows():
            plt.annotate(
                r['name'],
                (r['elapsed'], r['memory_gb']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                rotation=45
            )
        
        plt.title('メモリ使用量の推移')
        plt.xlabel('経過時間 (秒)')
        plt.ylabel('メモリ使用量 (GB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # 保存
        plt.savefig("data/results/memory_usage.png", dpi=200)
        plt.show()
    
    def display_summary(self):
        """モニタリング結果のサマリーを表示"""
        df = self.get_summary()
        
        if df.empty:
            print("チェックポイントがありません")
            return
        
        # HTML表示用にフォーマット
        format_dict = {
            'elapsed': '{:.2f}秒',
            'memory_gb': '{:.2f} GB',
            'memory_change': '{:.2f} GB',
            'time_diff': '{:.2f}秒'
        }
        
        # 表示する列を選択し、順序を指定
        display_cols = ['name', 'time', 'elapsed', 'memory_gb']
        if 'memory_change' in df.columns:
            display_cols.extend(['memory_change', 'time_diff'])
        
        styled_df = df[display_cols].style.format(format_dict)
        display(styled_df)
        
        # 総メモリ使用量の増加を表示
        if len(df) > 1:
            total_memory_increase = df['memory_gb'].iloc[-1] - df['memory_gb'].iloc[0]
            print(f"総メモリ使用量の増加: {total_memory_increase:.2f} GB")
            print(f"総実行時間: {df['elapsed'].iloc[-1]:.2f}秒")

    def clear_large_objects(self):
        """追跡されている大きなオブジェクトを解放する"""
        cleared_count = 0
        for name, obj_ref in list(self.large_objects.items()):
            obj = obj_ref()
            if obj is not None:
                print(f"オブジェクト '{name}' を解放します")
                del obj
                cleared_count += 1
            
            # 辞書からも削除
            del self.large_objects[name]
        
        return cleared_count

# シングルトンインスタンス
monitor = PerformanceMonitor()

def display_memory_usage():
    """現在のメモリ使用量を表示する（ノートブック用）"""
    monitor.display_current_memory()

def check_memory(checkpoint_name=""):
    """メモリ使用量をチェックして値を返す"""
    return monitor.check_memory(checkpoint_name)

def show_memory_summary():
    """メモリ使用量のサマリーを表示する"""
    monitor.display_summary()
    monitor.plot_memory_usage()

def track_large_object(obj, name):
    """大きなオブジェクトを追跡する"""
    monitor.track_object(obj, name)

def garbage_collect(aggressive=True):
    """ガベージコレクションを強制的に実行する
    
    Args:
        aggressive: 積極的なメモリ解放を行うかどうか
        
    Returns:
        float: 解放されたメモリ量（GB）
    """
    before = check_memory("GC前")
    
    # 標準GCを実行
    gc.collect()
    
    if aggressive:
        # 未使用のキャッシュをクリア
        try:
            # numpyのキャッシュをクリア
            if hasattr(np, '_clear_caches'):  # 新しいバージョンのnumpy
                np._clear_caches()  
            elif hasattr(np.core, '_internal_caches'):  # 古いバージョンのnumpy
                for cache in np.core._internal_caches():
                    cache.clear()
                
            # pandasのキャッシュをクリア
            if hasattr(pd, '_libs') and hasattr(pd._libs, 'hashtable'):
                if hasattr(pd._libs.hashtable, '_clear_caches'):
                    pd._libs.hashtable._clear_caches()
        except Exception as e:
            print(f"キャッシュクリア中にエラーが発生しました: {str(e)}")
        
        # 大きなオブジェクトを解放
        monitor.clear_large_objects()
        
        # 循環参照を解消するために複数回GCを実行
        for _ in range(3):
            gc.collect()
        
        # IPythonの変数履歴をクリア（IPythonの環境にある場合）
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                ipython.magic("reset -f array_")
                ipython.magic("reset -f df_")
                ipython.magic("reset -f tmp_")
        except:
            pass
        
        # 明示的に未使用のモジュールをアンロード
        unused_modules = []
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('_') or module_name in [
                'numpy', 'pandas', 'matplotlib', 'IPython', 'gc', 'weakref', 
                'psutil', 'platform', 'time', 'HTML'
            ]:
                continue
            
            if module_name.startswith(('tmp_', 'test_', 'cache_')):
                unused_modules.append(module_name)
        
        for module_name in unused_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
    
    # システムに実際のメモリを解放するようリクエスト（LinuxおよびmacOSのみ）
    if platform.system() in ['Linux', 'Darwin']:
        try:
            # madviseを呼び出して不要なページを解放
            import ctypes
            if platform.system() == 'Linux':
                libc = ctypes.CDLL('libc.so.6')
                MADV_DONTNEED = 4  # Linux用の定数
                libc.malloc_trim(0)  # システムにメモリを返す
            elif platform.system() == 'Darwin':
                libc = ctypes.CDLL('libc.dylib')
                MADV_FREE = 5  # macOS用の定数
        except:
            pass
    
    # 最終的なGCを実行
    gc.collect()
    
    after = check_memory("GC後")
    released = max(0, before - after)
    print(f"解放されたメモリ: {released:.2f} GB")
    
    # 解放後もまだメモリ使用量が多い場合は警告
    if after > 10:
        print(f"警告: GC後もメモリ使用量が高いままです ({after:.2f} GB)")
        print("対処法: ")
        print(" - 大きなデータフレームや配列を明示的に削除してください")
        print(" - track_large_object()で明示的に大きなオブジェクトを追跡してください")
        print(" - データの処理を小さなチャンクに分割してください")
    
    return released
