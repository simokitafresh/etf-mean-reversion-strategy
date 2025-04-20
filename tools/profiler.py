# tools/profiler.py
"""パフォーマンスプロファイリングツール"""
import cProfile
import pstats
import io
import os
import sys
import time
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def profile_function(func, *args, **kwargs):
    """関数のパフォーマンスをプロファイリングする
    
    Args:
        func: プロファイリング対象の関数
        *args, **kwargs: 関数に渡す引数
        
    Returns:
        Tuple: (関数の戻り値, プロファイリング結果文字列)
    """
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # 上位30件のみ表示
    
    profiling_result = (
        f"関数: {func.__name__}\n"
        f"引数: {args}, {kwargs}\n"
        f"実行時間: {execution_time:.2f}秒\n\n"
        f"--- プロファイリング結果 ---\n"
        f"{s.getvalue()}"
    )
    
    return result, profiling_result

def profile_module(module_name, output_dir="data/profiling"):
    """指定したモジュールの主要関数をプロファイリングする
    
    Args:
        module_name: プロファイリング対象のモジュール名
        output_dir: 結果出力ディレクトリ
        
    Returns:
        str: レポートファイルパス
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # レポートファイル
    report_file = f"{output_dir}/{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # モジュールをインポート
    module = __import__(f"src.{module_name}", fromlist=["*"])
    
    with open(report_file, 'w') as f:
        f.write(f"パフォーマンスプロファイリングレポート: {module_name}\n")
        f.write(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 関数を探索
        functions = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                functions.append((attr_name, attr))
        
        f.write(f"対象関数数: {len(functions)}\n\n")
        
        for func_name, func in functions:
            f.write(f"関数: {func_name}\n")
            f.write("-"*40 + "\n")
            
            try:
                # プロファイリング（サンプル引数を適宜設定）
                # ここでは関数によって適切な引数を設定する必要があります
                # 実際の使用例に合わせて調整してください
                
                f.write("この関数はプロファイリングされていません（適切な引数が必要）\n\n")
            except Exception as e:
                f.write(f"エラー: {str(e)}\n\n")
    
    return report_file

# 主要モジュールのプロファイリング用ヘルパー関数
def profile_universe_selection(sample_size=5):
    """ETFユニバース選定のパフォーマンスをプロファイリングする"""
    from src.data.fetch import get_base_etf_list
    from src.universe import select_universe
    
    # 基礎ETFリストの取得
    base_etfs, profile_result = profile_function(get_base_etf_list)
    
    with open("data/profiling/universe_base_list.txt", 'w') as f:
        f.write(profile_result)
    
    # サンプルサイズを制限
    base_etfs = base_etfs[:sample_size]
    
    # ユニバース選定
    universe, profile_result = profile_function(select_universe, base_etfs)
    
    with open("data/profiling/universe_selection.txt", 'w') as f:
        f.write(profile_result)
    
    return universe

def profile_signal_generation(universe, param_count=3):
    """シグナル生成のパフォーマンスをプロファイリングする"""
    from src.parameters.grid_search import generate_parameter_grid
    from src.signals import calculate_signals_for_universe
    
    # パラメータグリッド生成
    param_grid, profile_result = profile_function(generate_parameter_grid)
    
    with open("data/profiling/parameter_grid.txt", 'w') as f:
        f.write(profile_result)
    
    # サンプルサイズを制限
    param_grid = param_grid[:param_count]
    
    # シグナル生成
    signals_data, profile_result = profile_function(
        calculate_signals_for_universe,
        universe=universe,
        parameter_sets=param_grid,
        period="1y"
    )
    
    with open("data/profiling/signal_generation.txt", 'w') as f:
        f.write(profile_result)
    
    return signals_data

if __name__ == "__main__":
    # プロファイリング用ディレクトリの作成
    os.makedirs("data/profiling", exist_ok=True)
    
    print("パフォーマンスプロファイリングを開始します...")
    
    # ETFユニバース選定のプロファイリング
    universe = profile_universe_selection(sample_size=3)
    
    # シグナル生成のプロファイリング
    signals_data = profile_signal_generation(universe, param_count=2)
    
    print("パフォーマンスプロファイリングが完了しました")
    print("結果はdata/profilingディレクトリに保存されています")
