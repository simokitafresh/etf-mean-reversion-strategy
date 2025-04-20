# src/parameters/grid_search.py
"""パラメータグリッドサーチの実装"""
import pandas as pd
import numpy as np
import itertools
from typing import List, Dict, Any, Tuple
import os
import time
from ..data.cache import DataCache

# キャッシュのインスタンス化
cache = DataCache()

def generate_parameter_grid(
    bb_windows: List[int] = [15, 20, 25],
    bb_stds: List[float] = [1.8, 2.0, 2.2],
    stoch_ks: List[int] = [10, 14, 20],
    stoch_ds: List[int] = [3],
    ema_periods: List[int] = [200],
    ema_slope_periods: List[int] = [20],
    holding_periods: List[int] = [3, 5, 10]
) -> List[Dict[str, Any]]:
    """パラメータグリッドを生成する
    
    Args:
        bb_windows: ボリンジャーバンドの期間候補
        bb_stds: ボリンジャーバンドの標準偏差倍率候補
        stoch_ks: ストキャスティクスの%K期間候補
        stoch_ds: ストキャスティクスの%D期間候補
        ema_periods: EMAの期間候補
        ema_slope_periods: EMA傾きの計算期間候補
        holding_periods: ポジション保有期間候補
        
    Returns:
        List[Dict]: パラメータセットのリスト
    """
    # キャッシュから取得を試みる
    param_count = (
        len(bb_windows) * len(bb_stds) * len(stoch_ks) * 
        len(stoch_ds) * len(ema_periods) * len(ema_slope_periods) * 
        len(holding_periods)
    )
    cache_key = f"parameter_grid_{param_count}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print(f"キャッシュからパラメータグリッド({param_count}セット)を取得しました")
        return cached_data
    
    # 全パラメータの組み合わせを生成
    param_grid = []
    
    combinations = itertools.product(
        bb_windows, bb_stds, stoch_ks, stoch_ds, 
        ema_periods, ema_slope_periods, holding_periods
    )
    
    for bb_window, bb_std, stoch_k, stoch_d, ema_period, ema_slope_period, holding_period in combinations:
        param_set = {
            'bb_window': bb_window,
            'bb_std': bb_std,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'ema_period': ema_period,
            'ema_slope_period': ema_slope_period,
            'holding_period': holding_period,
            # パラメータを一意に識別するためのキー
            'param_key': (
                f"BB{bb_window}-{bb_std}_"
                f"Stoch{stoch_k}-{stoch_d}_"
                f"EMA{ema_period}_"
                f"Hold{holding_period}"
            )
        }
        param_grid.append(param_set)
    
    print(f"パラメータグリッドを生成しました: {len(param_grid)}セット")
    
    # キャッシュに保存
    cache.set_json(cache_key, param_grid)
    
    return param_grid

def run_grid_search(
    universe: List[Dict],
    param_grid: List[Dict],
    signals_data: Dict = None,
    recalculate: bool = False
) -> Dict[str, Any]:
    """パラメータグリッドサーチを実行する
    
    Args:
        universe: ETFユニバース
        param_grid: パラメータグリッド
        signals_data: 既存のシグナルデータ（なければ新規計算）
        recalculate: 既存の結果を再計算するかどうか
        
    Returns:
        Dict: グリッドサーチ結果
    """
    from ..signals import calculate_signals_for_universe
    
    # 結果ディレクトリの作成
    os.makedirs("data/results/grid_search", exist_ok=True)
    
    # キャッシュから取得を試みる
    cache_key = f"grid_search_{len(universe)}_{len(param_grid)}"
    if not recalculate:
        cached_data = cache.get_json(cache_key)
        if cached_data:
            print(f"キャッシュからグリッドサーチ結果を取得しました")
            return cached_data
    
    # シグナルデータが提供されていない場合は計算
    if signals_data is None:
        signals_data = calculate_signals_for_universe(universe, param_grid)
    
    print(f"{len(universe)}銘柄 × {len(param_grid)}パラメータセットのグリッドサーチを実行します...")
    
    # グリッドサーチ結果
    results = {
        'by_etf': {},        # ETF別の結果
        'by_parameter': {},  # パラメータセット別の結果
        'all_combinations': {},  # ETF×パラメータの全組み合わせ
        'summary': {}        # 全体サマリー
    }
    
    # ETF×パラメータの各組み合わせを評価
    for etf in universe:
        symbol = etf['symbol']
        print(f"\n{symbol}の評価中...")
        
        etf_results = {}
        
        if symbol not in signals_data:
            print(f"  警告: {symbol}のシグナルデータがありません")
            continue
        
        for param_set in param_grid:
            param_key = param_set['param_key']
            
            if param_key not in signals_data[symbol]:
                print(f"  警告: {symbol}のパラメータセット{param_key}のデータがありません")
                continue
            
            # CSVからシグナルデータを読み込む
            signal_stats = signals_data[symbol][param_key]
            csv_path = signal_stats.get('csv_path')
            
            if not os.path.exists(csv_path):
                print(f"  警告: {csv_path}が見つかりません")
                continue
            
            try:
                signal_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                
                # 各シグナルの評価
                # 買いシグナルの評価
                buy_results = evaluate_signals(
                    signal_df, 
                    'Buy_Signal', 
                    param_set['holding_period']
                )
                
                # 売りシグナルの評価
                sell_results = evaluate_signals(
                    signal_df, 
                    'Sell_Signal', 
                    param_set['holding_period']
                )
                
                # 結果を保存
                combination_key = f"{symbol}_{param_key}"
                results['all_combinations'][combination_key] = {
                    'symbol': symbol,
                    'param_key': param_key,
                    'buy': buy_results,
                    'sell': sell_results,
                    'holding_period': param_set['holding_period']
                }
                
                # ETF別の結果に追加
                if symbol not in results['by_etf']:
                    results['by_etf'][symbol] = {}
                results['by_etf'][symbol][param_key] = {
                    'buy': buy_results,
                    'sell': sell_results
                }
                
                # パラメータセット別の結果に追加
                if param_key not in results['by_parameter']:
                    results['by_parameter'][param_key] = {}
                results['by_parameter'][param_key][symbol] = {
                    'buy': buy_results,
                    'sell': sell_results
                }
                
                print(f"  評価完了: {param_key} - 買い(勝率: {buy_results['win_rate']:.1%}, PF: {buy_results['profit_factor']:.2f}), "
                      f"売り(勝率: {sell_results['win_rate']:.1%}, PF: {sell_results['profit_factor']:.2f})")
            
            except Exception as e:
                print(f"  エラー - {symbol}/{param_key}の評価: {str(e)}")
    
    # 全体サマリーの計算
    print("\n全体サマリーを計算中...")
    
    # パラメータセットごとのパフォーマンス集計
    param_summary = {}
    
    for param_key, etf_results in results['by_parameter'].items():
        buy_win_rates = []
        buy_pfs = []
        buy_sharpes = []
        sell_win_rates = []
        sell_pfs = []
        sell_sharpes = []
        
        for symbol, direction_results in etf_results.items():
            if direction_results['buy']['sample_count'] >= 30:
                buy_win_rates.append(direction_results['buy']['win_rate'])
                buy_pfs.append(direction_results['buy']['profit_factor'])
                buy_sharpes.append(direction_results['buy']['sharpe_ratio'])
            
            if direction_results['sell']['sample_count'] >= 30:
                sell_win_rates.append(direction_results['sell']['win_rate'])
                sell_pfs.append(direction_results['sell']['profit_factor'])
                sell_sharpes.append(direction_results['sell']['sharpe_ratio'])
        
        # 買いシグナルのサマリー
        buy_summary = {}
        if buy_win_rates:
            buy_summary = {
                'avg_win_rate': np.mean(buy_win_rates),
                'std_win_rate': np.std(buy_win_rates),
                'avg_pf': np.mean(buy_pfs),
                'std_pf': np.std(buy_pfs),
                'avg_sharpe': np.mean(buy_sharpes),
                'std_sharpe': np.std(buy_sharpes),
                'etf_count': len(buy_win_rates)
            }
        
        # 売りシグナルのサマリー
        sell_summary = {}
        if sell_win_rates:
            sell_summary = {
                'avg_win_rate': np.mean(sell_win_rates),
                'std_win_rate': np.std(sell_win_rates),
                'avg_pf': np.mean(sell_pfs),
                'std_pf': np.std(sell_pfs),
                'avg_sharpe': np.mean(sell_sharpes),
                'std_sharpe': np.std(sell_sharpes),
                'etf_count': len(sell_win_rates)
            }
        
        # 両方のサマリーを保存
        param_summary[param_key] = {
            'buy': buy_summary,
            'sell': sell_summary,
            # パラメータの安定性スコア（標準偏差が小さいほど安定）
            'stability_score': calculate_stability_score(buy_summary, sell_summary)
        }
    
    # 全体サマリーに追加
    results['summary']['parameter_performance'] = param_summary
    
    # 安定性スコアでパラメータをランク付け
    param_ranking = sorted(
        param_summary.items(),
        key=lambda x: x[1]['stability_score'],
        reverse=True  # 高いスコアが上位
    )
    
    results['summary']['parameter_ranking'] = [
        {'param_key': p[0], 'stability_score': p[1]['stability_score']}
        for p in param_ranking
    ]
    
    print("グリッドサーチが完了しました")
    
    # 結果をJSONとして保存
    import json
    with open("data/results/grid_search/grid_search_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("結果を保存しました: data/results/grid_search/grid_search_results.json")
    
    # キャッシュに保存
    cache.set_json(cache_key, results)
    
    return results

def evaluate_signals(
    df: pd.DataFrame,
    signal_column: str,
    holding_period: int
) -> Dict[str, float]:
    """シグナルのパフォーマンスを評価する
    
    Args:
        df: シグナルを含むデータフレーム
        signal_column: シグナル列名（'Buy_Signal'または'Sell_Signal'）
        holding_period: ポジション保有期間
        
    Returns:
        Dict: 評価結果
    """
    # シグナル日の特定
    signal_days = df[df[signal_column]].index
    
    if len(signal_days) == 0:
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'std_return': 0,
            'sharpe_ratio': 0,
            'sample_count': 0,
            'wilson_lower': 0
        }
    
    # リターンの計算
    returns = []
    is_win = []
    
    for day in signal_days:
        # 現在の価格
        try:
            current_price = df.loc[day, 'Adj Close']
            
            # 保有期間後の価格（取引日ベース）
            # 保有期間後の日付がデータフレームの範囲を超える場合はスキップ
            future_idx = df.index.get_loc(day) + holding_period
            if future_idx >= len(df):
                continue
                
            future_date = df.index[future_idx]
            future_price = df.loc[future_date, 'Adj Close']
            
            # リターンの計算
            if signal_column == 'Buy_Signal':
                ret = (future_price / current_price) - 1
            else:  # 'Sell_Signal'
                ret = (current_price / future_price) - 1
            
            returns.append(ret)
            is_win.append(ret > 0)
        
        except Exception as e:
            print(f"  評価エラー - {day}: {str(e)}")
    
    # 有効なリターンがない場合
    if not returns:
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'std_return': 0,
            'sharpe_ratio': 0,
            'sample_count': 0,
            'wilson_lower': 0
        }
    
    # 統計値の計算
    returns = np.array(returns)
    is_win = np.array(is_win)
    
    win_rate = np.mean(is_win)
    
    # 勝ちトレードと負けトレードの合計
    winning_trades = returns[returns > 0]
    losing_trades = np.abs(returns[returns < 0])
    
    total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
    total_losses = losing_trades.sum() if len(losing_trades) > 0 else 0
    
    # Profit Factor
    profit_factor = (total_wins / total_losses) if total_losses > 0 else (
        float('inf') if total_wins > 0 else 0
    )
    
    # リターンの平均と標準偏差
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    # シャープレシオ（リスクフリーレート0%と仮定）
    sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
    
    # Wilson信頼区間（95%）
    from scipy.stats import norm
    
    z = norm.ppf(0.975)  # 95%信頼区間
    n = len(returns)
    
    if n > 0:
        wilson_lower = (win_rate + z*z/(2*n) - z * np.sqrt((win_rate*(1-win_rate) + z*z/(4*n))/n)) / (1 + z*z/n)
    else:
        wilson_lower = 0
    
    return {
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'avg_return': float(avg_return),
        'std_return': float(std_return),
        'sharpe_ratio': float(sharpe_ratio),
        'sample_count': int(len(returns)),
        'wilson_lower': float(wilson_lower)
    }

def calculate_stability_score(buy_summary: Dict, sell_summary: Dict) -> float:
    """パラメータセットの安定性スコアを計算する
    
    Args:
        buy_summary: 買いシグナルのサマリー
        sell_summary: 売りシグナルのサマリー
        
    Returns:
        float: 安定性スコア
    """
    # どちらもデータがない場合は0
    if not buy_summary and not sell_summary:
        return 0
    
    score = 0
    count = 0
    
    # 買いシグナルのスコア
    if buy_summary and buy_summary.get('etf_count', 0) >= 3:
        # パフォーマンススコア（高いほど良い）
        perf_score = (
            0.4 * buy_summary.get('avg_win_rate', 0) + 
            0.3 * min(buy_summary.get('avg_pf', 0), 3) / 3 + 
            0.3 * min(buy_summary.get('avg_sharpe', 0), 2) / 2
        )
        
        # 安定性スコア（低いほど良い）
        stability = (
            0.4 * (1 - min(buy_summary.get('std_win_rate', 1), 0.2) / 0.2) + 
            0.3 * (1 - min(buy_summary.get('std_pf', 3), 1) / 1) + 
            0.3 * (1 - min(buy_summary.get('std_sharpe', 2), 0.5) / 0.5)
        )
        
        # 総合スコア
        buy_score = 0.6 * perf_score + 0.4 * stability
        score += buy_score
        count += 1
    
    # 売りシグナルのスコア
    if sell_summary and sell_summary.get('etf_count', 0) >= 3:
        # パフォーマンススコア（高いほど良い）
        perf_score = (
            0.4 * sell_summary.get('avg_win_rate', 0) + 
            0.3 * min(sell_summary.get('avg_pf', 0), 3) / 3 + 
            0.3 * min(sell_summary.get('avg_sharpe', 0), 2) / 2
        )
        
        # 安定性スコア（低いほど良い）
        stability = (
            0.4 * (1 - min(sell_summary.get('std_win_rate', 1), 0.2) / 0.2) + 
            0.3 * (1 - min(sell_summary.get('std_pf', 3), 1) / 1) + 
            0.3 * (1 - min(sell_summary.get('std_sharpe', 2), 0.5) / 0.5)
        )
        
        # 総合スコア
        sell_score = 0.6 * perf_score + 0.4 * stability
        score += sell_score
        count += 1
    
    # 平均スコアを返す
    return score / count if count > 0 else 0
