# src/parameters/stability.py
"""パラメータ安定性評価ロジック"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import os

def identify_stability_zones(
    grid_results: Dict,
    performance_metric: str = 'avg_win_rate',
    stability_metric: str = 'std_win_rate',
    direction: str = 'buy',
    percentile_threshold: float = 0.75
) -> Dict[str, Any]:
    """パラメータの安定帯を特定する
    
    Args:
        grid_results: グリッドサーチ結果
        performance_metric: パフォーマンス指標
        stability_metric: 安定性指標
        direction: 取引方向（'buy'または'sell'）
        percentile_threshold: パフォーマンス上位と安定性上位の閾値（パーセンタイル）
        
    Returns:
        Dict: 安定帯の情報
    """
    # 結果ディレクトリの作成
    os.makedirs("data/results/stability", exist_ok=True)
    
    print(f"{direction}方向の安定帯を特定しています...")
    
    # パラメータサマリーを取得
    param_summary = grid_results['summary']['parameter_performance']
    
    # パラメータごとのパフォーマンスと安定性のデータを抽出
    param_data = []
    
    for param_key, summaries in param_summary.items():
        if direction in summaries and summaries[direction]:
            direction_summary = summaries[direction]
            
            # 該当メトリクスのデータを取得
            perf = direction_summary.get(performance_metric)
            stab = direction_summary.get(stability_metric)
            
            # サンプル数が十分なもののみ対象とする
            if perf is not None and stab is not None and direction_summary.get('etf_count', 0) >= 3:
                # パラメータから値を抽出
                bb_window, bb_std, stoch_k, stoch_d, ema, holding = parse_param_key(param_key)
                
                param_data.append({
                    'param_key': param_key,
                    'bb_window': bb_window,
                    'bb_std': bb_std,
                    'stoch_k': stoch_k,
                    'stoch_d': stoch_d,
                    'ema': ema,
                    'holding': holding,
                    'performance': perf,
                    'stability': stab,
                    'etf_count': direction_summary.get('etf_count', 0)
                })
    
    # データフレームに変換
    param_df = pd.DataFrame(param_data)
    
    if len(param_df) == 0:
        print(f"警告: {direction}方向の有効なパラメータデータがありません")
        return {
            'stable_params': [],
            'heatmaps': {}
        }
    
    # パフォーマンス上位と安定性上位の閾値
    perf_threshold = param_df['performance'].quantile(percentile_threshold)
    stab_threshold = param_df['stability'].quantile(1 - percentile_threshold)  # 安定性は低いほど良い
    
    # 安定帯に属するパラメータを特定
    stable_params = param_df[
        (param_df['performance'] >= perf_threshold) & 
        (param_df['stability'] <= stab_threshold)
    ]
    
    print(f"安定帯に属するパラメータ: {len(stable_params)}/{len(param_df)}セット")
    
    # ヒートマップの生成
    heatmaps = generate_stability_heatmaps(param_df, stable_params, direction)
    
    # 安定帯のパラメータリストを保存
    stable_params_list = stable_params.to_dict('records')
    
    # CSVとして保存
    stable_params.to_csv(f"data/results/stability/{direction}_stable_parameters.csv", index=False)
    
    return {
        'stable_params': stable_params_list,
        'heatmaps': heatmaps
    }

def parse_param_key(param_key: str) -> Tuple[int, float, int, int, int, int]:
    """パラメータキーからパラメータ値を抽出する
    
    Args:
        param_key: パラメータキー（例："BB20-2.0_Stoch14-3_EMA200_Hold5"）
        
    Returns:
        Tuple: (bb_window, bb_std, stoch_k, stoch_d, ema_period, holding_period)
    """
    try:
        # BBパラメータ
        bb_part = param_key.split('_')[0]
        bb_values = bb_part.replace('BB', '').split('-')
        bb_window = int(bb_values[0])
        bb_std = float(bb_values[1])
        
        # Stochパラメータ
        stoch_part = param_key.split('_')[1]
        stoch_values = stoch_part.replace('Stoch', '').split('-')
        stoch_k = int(stoch_values[0])
        stoch_d = int(stoch_values[1])
        
        # EMAパラメータ
        ema_part = param_key.split('_')[2]
        ema_period = int(ema_part.replace('EMA', ''))
        
        # 保有期間
        hold_part = param_key.split('_')[3]
        holding_period = int(hold_part.replace('Hold', ''))
        
        return bb_window, bb_std, stoch_k, stoch_d, ema_period, holding_period
    
    except Exception as e:
        print(f"パラメータキーの解析エラー ({param_key}): {str(e)}")
        return 0, 0, 0, 0, 0, 0

def generate_stability_heatmaps(
    param_df: pd.DataFrame,
    stable_params: pd.DataFrame,
    direction: str
) -> Dict[str, str]:
    """安定性のヒートマップを生成する
    
    Args:
        param_df: すべてのパラメータデータ
        stable_params: 安定帯に属するパラメータデータ
        direction: 取引方向（'buy'または'sell'）
        
    Returns:
        Dict: ヒートマップのファイルパス
    """
    heatmap_paths = {}
    
    # 1. BB WindowとBB Stdの関係
    bb_heatmap_path = generate_2d_heatmap(
        param_df, 'bb_window', 'bb_std', 'performance',
        stable_params, f"{direction}_bb_stability_heatmap.png",
        title=f"{direction.capitalize()}シグナル: ボリンジャーバンドパラメータの安定性"
    )
    heatmap_paths['bb'] = bb_heatmap_path
    
    # 2. Stoch KとHolding Periodの関係
    stoch_holding_path = generate_2d_heatmap(
        param_df, 'stoch_k', 'holding', 'performance',
        stable_params, f"{direction}_stoch_holding_heatmap.png",
        title=f"{direction.capitalize()}シグナル: Stochasticと保有期間の安定性"
    )
    heatmap_paths['stoch_holding'] = stoch_holding_path
    
    # 3. BB Windowとストキャスティック期間の関係
    bb_stoch_path = generate_2d_heatmap(
        param_df, 'bb_window', 'stoch_k', 'performance',
        stable_params, f"{direction}_bb_stoch_heatmap.png",
        title=f"{direction.capitalize()}シグナル: BBとStochasticのタイムフレーム関係"
    )
    heatmap_paths['bb_stoch'] = bb_stoch_path
    
    return heatmap_paths

def generate_2d_heatmap(
    param_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    value_param: str,
    stable_params: pd.DataFrame,
    filename: str,
    title: str = ""
) -> str:
    """2次元ヒートマップを生成する
    
    Args:
        param_df: すべてのパラメータデータ
        x_param: X軸パラメータ
        y_param: Y軸パラメータ
        value_param: 値パラメータ
        stable_params: 安定帯に属するパラメータデータ
        filename: 保存するファイル名
        title: グラフのタイトル
        
    Returns:
        str: ヒートマップのファイルパス
    """
    # ピボットテーブルの作成
    pivot = param_df.pivot_table(
        index=y_param, 
        columns=x_param, 
        values=value_param,
        aggfunc='mean'
    )
    
    # 安定帯のマスクを作成
    mask = pd.DataFrame(
        False, 
        index=pivot.index, 
        columns=pivot.columns
    )
    
    for _, row in stable_params.iterrows():
        x_val = row[x_param]
        y_val = row[y_param]
        if x_val in mask.columns and y_val in mask.index:
            mask.loc[y_val, x_val] = True
    
    # プロットの作成
    plt.figure(figsize=(10, 8))
    
    # メインのヒートマップ
    ax = sns.heatmap(
        pivot,
        cmap='viridis',
        annot=True,
        fmt=".3f",
        linewidths=0.5
    )
    
    # 安定帯のハイライト
    for i, idx in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            if mask.loc[idx, col]:
                ax.add_patch(plt.Rectangle(
                    (j, i),
                    1, 1,
                    fill=False,
                    edgecolor='red',
                    lw=2,
                    alpha=0.7
                ))
    
    plt.title(title)
    plt.tight_layout()
    
    # 保存パス
    filepath = f"data/results/stability/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ヒートマップを保存しました: {filepath}")
    
    return filepath
