# src/parameters/__init__.py
"""パラメータ管理モジュール"""
from .grid_search import generate_parameter_grid, run_grid_search
from .stability import identify_stability_zones
import os
import json

def evaluate_parameter_stability(
    universe: list,
    recalculate: bool = False
) -> dict:
    """パラメータの安定性を評価する統合関数
    
    Args:
        universe: ETFユニバース
        recalculate: グリッドサーチを再計算するかどうか
        
    Returns:
        dict: 安定性評価結果
    """
    # 結果ディレクトリの作成
    os.makedirs("data/results/parameters", exist_ok=True)
    
    # パラメータグリッドの生成
    print("パラメータグリッドを生成中...")
    param_grid = generate_parameter_grid()
    
    # グリッドサーチの実行
    print("グリッドサーチを実行中...")
    grid_results = run_grid_search(universe, param_grid, recalculate=recalculate)
    
    # 買いシグナルの安定帯を特定
    print("買いシグナルの安定帯を特定中...")
    buy_stability = identify_stability_zones(
        grid_results, 
        performance_metric='avg_win_rate',
        stability_metric='std_win_rate',
        direction='buy'
    )
    
    # 売りシグナルの安定帯を特定
    print("売りシグナルの安定帯を特定中...")
    sell_stability = identify_stability_zones(
        grid_results, 
        performance_metric='avg_win_rate',
        stability_metric='std_win_rate',
        direction='sell'
    )
    
    # 結果をまとめる
    stability_results = {
        'buy': buy_stability,
        'sell': sell_stability,
        'grid_search_summary': grid_results['summary']
    }
    
    # 結果をJSONとして保存
    with open("data/results/parameters/stability_results.json", 'w') as f:
        json.dump(stability_results, f, indent=2)
    
    print("安定性評価結果を保存しました: data/results/parameters/stability_results.json")
    
    return stability_results
