"""パラメータ管理モジュール

このモジュールは「統計的トレンド逆張り戦略」のパラメータ管理と安定性評価のための
機能を提供します。パラメータグリッド生成から安定性評価、ヒートマップ生成まで、
パラメータ最適化の全プロセスを管理します。

機能:
- パラメータグリッド生成: 指定された範囲のパラメータ組み合わせを生成
- グリッドサーチ: 各パラメータセットの性能評価を実行
- 安定性評価: 各パラメータセットの安定性とパフォーマンスを分析
- 安定領域の特定: パラメータ空間内の安定した領域を視覚化

使用方法:
```python
from src.parameters import evaluate_parameter_stability

# ETFユニバースに対してパラメータ安定性を評価
stability_results = evaluate_parameter_stability(universe)

# 安定パラメータの取得
buy_stable_params = stability_results['buy']['stable_params']
sell_stable_params = stability_results['sell']['stable_params']
```
"""
from .grid_search import generate_parameter_grid, run_grid_search
from .stability import identify_stability_zones
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

# ロガーの設定
logger = logging.getLogger(__name__)

def evaluate_parameter_stability(
    universe: List[Dict[str, Any]],
    recalculate: bool = False,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """パラメータの安定性を評価する統合関数
    
    この関数は以下のステップを実行します:
    1. パラメータグリッドの生成
    2. グリッドサーチの実行（各パラメータセットの性能評価）
    3. 買いシグナルの安定帯の特定
    4. 売りシグナルの安定帯の特定
    5. 結果の保存とヒートマップの生成
    
    Args:
        universe: ETFユニバース。各ETFは辞書形式で、少なくとも'symbol'キーが必要
        recalculate: グリッドサーチを再計算するかどうか。Trueの場合、キャッシュを無視
        options: 追加オプション辞書
            - 'use_simplified_heatmap': 簡易ヒートマップを使用する（True/False）
            - 'limit_parameters': パラメータ数を制限する（True/False）
            - 'performance_metric': 性能評価指標（デフォルト: 'avg_win_rate'）
            - 'stability_metric': 安定性評価指標（デフォルト: 'std_win_rate'）
            - 'percentile_threshold': パーセンタイル閾値（デフォルト: 0.75）
        
    Returns:
        Dict[str, Any]: 安定性評価結果の辞書
            - 'buy': 買いシグナルの安定性評価結果
                - 'stable_params': 安定パラメータのリスト
                - 'heatmaps': ヒートマップファイルのパス
            - 'sell': 売りシグナルの安定性評価結果
                - 'stable_params': 安定パラメータのリスト
                - 'heatmaps': ヒートマップファイルのパス
            - 'grid_search_summary': グリッドサーチの概要
    
    Raises:
        ValueError: ETFユニバースが空の場合
        IOError: 結果の保存に失敗した場合
    """
    # オプションの初期化
    if options is None:
        options = {}
    
    # 入力チェック
    if not universe:
        logger.error("空のETFユニバースが提供されました")
        raise ValueError("ETFユニバースは少なくとも1つのETFを含む必要があります")
    
    # 結果ディレクトリの作成
    results_dir = "data/results/parameters"
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("パラメータ安定性評価を開始します")
    
    # パラメータグリッドの生成
    logger.info("パラメータグリッドを生成中...")
    
    # パラメータ制限オプションの確認
    if options.get('limit_parameters', False):
        logger.info("パラメータ数を制限します（MVPモード）")
        param_grid = generate_parameter_grid(
            bb_windows=[20],           # ボリンジャーバンド期間
            bb_stds=[2.0],             # ボリンジャーバンド標準偏差
            stoch_ks=[14],             # ストキャスティクスK期間
            stoch_ds=[3],              # ストキャスティクスD期間
            ema_periods=[200],         # EMA期間
            ema_slope_periods=[20],    # EMAスロープ計算期間
            holding_periods=[5, 10]    # 保有期間
        )
        logger.info(f"制限付きパラメータグリッド生成: {len(param_grid)}セット")
    else:
        param_grid = generate_parameter_grid()
        logger.info(f"完全パラメータグリッド生成: {len(param_grid)}セット")
    
    # グリッドサーチの実行
    logger.info("グリッドサーチを実行中...")
    grid_results = run_grid_search(universe, param_grid, recalculate=recalculate)
    
    # パフォーマンス指標と安定性指標の設定
    perf_metric = options.get('performance_metric', 'avg_win_rate')
    stab_metric = options.get('stability_metric', 'std_win_rate')
    percentile = options.get('percentile_threshold', 0.75)
    
    # 買いシグナルの安定帯を特定
    logger.info(f"買いシグナルの安定帯を特定中... (指標: {perf_metric}, {stab_metric})")
    buy_stability = identify_stability_zones(
        grid_results, 
        performance_metric=perf_metric,
        stability_metric=stab_metric,
        direction='buy',
        percentile_threshold=percentile
    )
    
    # 売りシグナルの安定帯を特定
    logger.info(f"売りシグナルの安定帯を特定中... (指標: {perf_metric}, {stab_metric})")
    sell_stability = identify_stability_zones(
        grid_results, 
        performance_metric=perf_metric,
        stability_metric=stab_metric,
        direction='sell',
        percentile_threshold=percentile
    )
    
    # 結果をまとめる
    stability_results = {
        'buy': buy_stability,
        'sell': sell_stability,
        'grid_search_summary': grid_results['summary']
    }
    
    # 結果をJSONとして保存
    try:
        result_file = os.path.join(results_dir, "stability_results.json")
        with open(result_file, 'w') as f:
            json.dump(stability_results, f, indent=2)
        logger.info(f"安定性評価結果を保存しました: {result_file}")
    except IOError as e:
        logger.error(f"結果の保存に失敗しました: {str(e)}")
        raise
    
    # 安定パラメータの概要
    buy_count = len(buy_stability.get('stable_params', []))
    sell_count = len(sell_stability.get('stable_params', []))
    logger.info(f"評価完了: 買いシグナル安定パラメータ {buy_count}セット, 売りシグナル安定パラメータ {sell_count}セット")
    
    return stability_results
