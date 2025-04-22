# src/parameters/grid_search.py
"""パラメータグリッドサーチの実装"""
import pandas as pd
import numpy as np
import itertools
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import time
from ..data.cache import DataCache

# ロガーの設定
logger = logging.getLogger(__name__)

# キャッシュのインスタンス化
try:
    cache = DataCache()
    cache_available = True
except Exception as e:
    logger.warning(f"キャッシュ初期化エラー: {str(e)}")
    logger.info("キャッシュなしで続行します")
    cache_available = False

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
    # B1: パラメータの基本検証
    # 各リストが空でないか、また要素が有効かをチェック
    for name, param_list, min_val in [
        ('bb_windows', bb_windows, 2),
        ('bb_stds', bb_stds, 0.1),
        ('stoch_ks', stoch_ks, 2),
        ('stoch_ds', stoch_ds, 1),
        ('ema_periods', ema_periods, 2),
        ('ema_slope_periods', ema_slope_periods, 1),
        ('holding_periods', holding_periods, 1)
    ]:
        if not param_list:
            logger.warning(f"{name}が空です。デフォルト値を使用します。")
            if name == 'bb_windows':
                bb_windows = [20]
            elif name == 'bb_stds':
                bb_stds = [2.0]
            elif name == 'stoch_ks':
                stoch_ks = [14]
            elif name == 'stoch_ds':
                stoch_ds = [3]
            elif name == 'ema_periods':
                ema_periods = [200]
            elif name == 'ema_slope_periods':
                ema_slope_periods = [20]
            elif name == 'holding_periods':
                holding_periods = [5]
        else:
            # 無効な値をフィルタリング
            filtered_list = [x for x in param_list if x >= min_val]
            if len(filtered_list) < len(param_list):
                logger.warning(f"{name}に無効な値が含まれています。{min_val}未満の値は除外します。")
                if name == 'bb_windows':
                    bb_windows = filtered_list
                elif name == 'bb_stds':
                    bb_stds = filtered_list
                elif name == 'stoch_ks':
                    stoch_ks = filtered_list
                elif name == 'stoch_ds':
                    stoch_ds = filtered_list
                elif name == 'ema_periods':
                    ema_periods = filtered_list
                elif name == 'ema_slope_periods':
                    ema_slope_periods = filtered_list
                elif name == 'holding_periods':
                    holding_periods = filtered_list

    # キャッシュから取得を試みる
    param_count = (
        len(bb_windows) * len(bb_stds) * len(stoch_ks) * 
        len(stoch_ds) * len(ema_periods) * len(ema_slope_periods) * 
        len(holding_periods)
    )
    
    # B1: パラメータグリッドのサイズチェック
    if param_count > 1000:
        logger.warning(f"パラメータグリッドのサイズが非常に大きいです: {param_count}セット")
        logger.warning("これはメモリ使用量が増加し、計算時間が長くなる可能性があります。")
    
    cache_key = f"parameter_grid_{param_count}"
    if cache_available:
        cached_data = cache.get_json(cache_key)
        if cached_data:
            logger.info(f"キャッシュからパラメータグリッド({param_count}セット)を取得しました")
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
    
    logger.info(f"パラメータグリッドを生成しました: {len(param_grid)}セット")
    
    # キャッシュに保存
    if cache_available:
        try:
            cache.set_json(cache_key, param_grid)
        except Exception as e:
            logger.warning(f"キャッシュへの保存に失敗しました: {str(e)}")
    
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
    
    # B1: 入力パラメータの検証
    if not universe:
        logger.error("空のユニバースが提供されました。グリッドサーチを中止します。")
        return {
            'by_etf': {},
            'by_parameter': {},
            'all_combinations': {},
            'summary': {'error': '空のユニバース'}
        }
    
    if not param_grid:
        logger.error("空のパラメータグリッドが提供されました。グリッドサーチを中止します。")
        return {
            'by_etf': {},
            'by_parameter': {},
            'all_combinations': {},
            'summary': {'error': '空のパラメータグリッド'}
        }
    
    # キャッシュから取得を試みる
    cache_key = f"grid_search_{len(universe)}_{len(param_grid)}"
    if not recalculate and cache_available:
        cached_data = cache.get_json(cache_key)
        if cached_data:
            logger.info(f"キャッシュからグリッドサーチ結果を取得しました")
            return cached_data
    
    # シグナルデータが提供されていない場合は計算
    if signals_data is None:
        signals_data = calculate_signals_for_universe(universe, param_grid)
    
    # B1: シグナルデータのチェック
    if not signals_data:
        logger.warning("シグナルデータが空です。グリッドサーチを続行できません。")
        return {
            'by_etf': {},
            'by_parameter': {},
            'all_combinations': {},
            'summary': {'error': 'シグナルデータ取得失敗'}
        }
    
    logger.info(f"{len(universe)}銘柄 × {len(param_grid)}パラメータセットのグリッドサーチを実行します...")
    
    # グリッドサーチ結果
    results = {
        'by_etf': {},        # ETF別の結果
        'by_parameter': {},  # パラメータセット別の結果
        'all_combinations': {},  # ETF×パラメータの全組み合わせ
        'summary': {}        # 全体サマリー
    }
    
    # 進捗表示のための準備
    total_combinations = len(universe) * len(param_grid)
    processed = 0
    start_time = time.time()
    last_update_time = start_time
    
    # ETF×パラメータの各組み合わせを評価
    for etf in universe:
        symbol = etf.get('symbol')
        if not symbol:
            logger.warning(f"シンボルが見つかりません: {etf}")
            continue
            
        logger.info(f"\n{symbol}の評価中...")
        
        etf_results = {}
        
        if symbol not in signals_data:
            logger.warning(f"  警告: {symbol}のシグナルデータがありません")
            continue
        
        for param_set in param_grid:
            param_key = param_set.get('param_key')
            if not param_key:
                logger.warning(f"パラメータキーが見つかりません: {param_set}")
                continue
                
            if param_key not in signals_data[symbol]:
                logger.warning(f"  警告: {symbol}のパラメータセット{param_key}のデータがありません")
                continue
            
            # CSVからシグナルデータを読み込む
            signal_stats = signals_data[symbol][param_key]
            csv_path = signal_stats.get('csv_path')
            
            if not csv_path or not os.path.exists(csv_path):
                logger.warning(f"  警告: {csv_path}が見つかりません")
                continue
            
            try:
                signal_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                
                # B1: データフレームの基本検証
                if signal_df.empty:
                    logger.warning(f"  警告: {csv_path}が空です")
                    continue
                
                required_columns = ['Close', 'Buy_Signal', 'Sell_Signal']
                missing_columns = [col for col in required_columns if col not in signal_df.columns]
                if missing_columns:
                    logger.warning(f"  警告: {csv_path}に必要なカラム{missing_columns}がありません")
                    continue
                
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
                
                # 進捗更新
                processed += 1
                current_time = time.time()
                if current_time - last_update_time >= 5:  # 5秒ごとに進捗表示
                    elapsed = current_time - start_time
                    progress = processed / total_combinations
                    remaining = elapsed / progress - elapsed if progress > 0 else 0
                    logger.info(f"  進捗: {processed}/{total_combinations} ({progress:.1%}) - "
                               f"残り約{int(remaining/60)}分{int(remaining)%60}秒")
                    last_update_time = current_time
                
                logger.info(f"  評価完了: {param_key} - 買い(勝率: {buy_results['win_rate']:.1%}, PF: {buy_results['profit_factor']:.2f}), "
                           f"売り(勝率: {sell_results['win_rate']:.1%}, PF: {sell_results['profit_factor']:.2f})")
            
            except Exception as e:
                logger.error(f"  エラー - {symbol}/{param_key}の評価: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
    
    # 全体サマリーの計算
    logger.info("\n全体サマリーを計算中...")
    
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
            # B1: 有効なサンプル数のみを対象とする
            if direction_results['buy']['sample_count'] >= 30:
                # B1: NaN/Infチェックを追加
                win_rate = direction_results['buy']['win_rate']
                pf = direction_results['buy']['profit_factor']
                sharpe = direction_results['buy']['sharpe_ratio']
                
                if is_valid_metric(win_rate):
                    buy_win_rates.append(win_rate)
                if is_valid_metric(pf) and pf != float('inf'):
                    buy_pfs.append(pf)
                if is_valid_metric(sharpe):
                    buy_sharpes.append(sharpe)
            
            if direction_results['sell']['sample_count'] >= 30:
                # B1: NaN/Infチェックを追加
                win_rate = direction_results['sell']['win_rate']
                pf = direction_results['sell']['profit_factor']
                sharpe = direction_results['sell']['sharpe_ratio']
                
                if is_valid_metric(win_rate):
                    sell_win_rates.append(win_rate)
                if is_valid_metric(pf) and pf != float('inf'):
                    sell_pfs.append(pf)
                if is_valid_metric(sharpe):
                    sell_sharpes.append(sharpe)
        
        # 買いシグナルのサマリー
        buy_summary = {}
        if buy_win_rates:
            buy_summary = {
                'avg_win_rate': safe_mean(buy_win_rates),
                'std_win_rate': safe_std(buy_win_rates),
                'avg_pf': safe_mean(buy_pfs),
                'std_pf': safe_std(buy_pfs),
                'avg_sharpe': safe_mean(buy_sharpes),
                'std_sharpe': safe_std(buy_sharpes),
                'etf_count': len(buy_win_rates)
            }
        
        # 売りシグナルのサマリー
        sell_summary = {}
        if sell_win_rates:
            sell_summary = {
                'avg_win_rate': safe_mean(sell_win_rates),
                'std_win_rate': safe_std(sell_win_rates),
                'avg_pf': safe_mean(sell_pfs),
                'std_pf': safe_std(sell_pfs),
                'avg_sharpe': safe_mean(sell_sharpes),
                'std_sharpe': safe_std(sell_sharpes),
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
    
    # カバレッジ統計の追加
    results['summary']['coverage'] = {
        'total_combinations': total_combinations,
        'processed_combinations': processed,
        'coverage_rate': processed / total_combinations if total_combinations > 0 else 0
    }
    
    logger.info("グリッドサーチが完了しました")
    logger.info(f"カバレッジ: {processed}/{total_combinations} ({processed/total_combinations:.1%})")
    
    # 結果をJSONとして保存
    try:
        import json
        with open("data/results/grid_search/grid_search_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("結果を保存しました: data/results/grid_search/grid_search_results.json")
    except Exception as e:
        logger.error(f"結果のJSON保存に失敗しました: {str(e)}")
    
    # キャッシュに保存
    if cache_available:
        try:
            cache.set_json(cache_key, results)
        except Exception as e:
            logger.warning(f"キャッシュへの保存に失敗しました: {str(e)}")
    
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
    # B1: 入力データの検証
    if df is None or df.empty:
        logger.warning(f"空のデータフレームが渡されました")
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'std_return': 0,
            'sharpe_ratio': 0,
            'sample_count': 0,
            'wilson_lower': 0
        }
    
    if signal_column not in df.columns:
        logger.warning(f"シグナル列 '{signal_column}' がデータフレームにありません")
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'std_return': 0,
            'sharpe_ratio': 0,
            'sample_count': 0,
            'wilson_lower': 0
        }
    
    if 'Close' not in df.columns:
        logger.warning("'Close'列がデータフレームにありません")
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'std_return': 0,
            'sharpe_ratio': 0,
            'sample_count': 0,
            'wilson_lower': 0
        }
    
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
            # B1: 安全なデータアクセス
            if day not in df.index:
                logger.debug(f"シグナル日 {day} がデータフレームにありません")
                continue
                
            current_price = df.loc[day, 'Close']
            if current_price <= 0:
                logger.debug(f"無効な現在価格: {current_price}")
                continue
                
            # 保有期間後の価格（取引日ベース）
            # 保有期間後の日付がデータフレームの範囲を超える場合はスキップ
            future_idx = df.index.get_loc(day) + holding_period
            if future_idx >= len(df):
                continue
                
            future_date = df.index[future_idx]
            future_price = df.loc[future_date, 'Close']
            
            if future_price <= 0:
                logger.debug(f"無効な将来価格: {future_price}")
                continue
            
            # リターンの計算
            if signal_column == 'Buy_Signal':
                ret = (future_price / current_price) - 1
            else:  # 'Sell_Signal'
                ret = (current_price / future_price) - 1
            
            # B1: NaN/Infinityのチェック
            if np.isnan(ret) or np.isinf(ret):
                logger.debug(f"無効なリターン値: {ret}")
                continue
                
            # B1: 極端なリターン値をクリップ
            if abs(ret) > 5.0:  # 500%を超えるリターンは異常値とみなし、クリップ
                logger.debug(f"極端なリターン値をクリップ: {ret} -> {np.sign(ret) * 5.0}")
                ret = np.sign(ret) * 5.0
            
            returns.append(ret)
            is_win.append(ret > 0)
        
        except Exception as e:
            logger.debug(f"リターン計算エラー - {day}: {str(e)}")
            continue
    
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
    
    # B1: 安全なProfit Factor計算
    if total_losses > 0:
        profit_factor = total_wins / total_losses
    elif total_wins > 0:
        profit_factor = float('inf')  # 負けなしの場合
    else:
        profit_factor = 0  # 勝ちもなしの場合
    
    # リターンの平均と標準偏差
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    # B1: 安全なシャープレシオ計算
    sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
    
    # Wilson信頼区間（95%）
    from scipy.stats import norm
    
    z = norm.ppf(0.975)  # 95%信頼区間
    n = len(returns)
    
    # B1: 安全なWilson区間計算
    wilson_lower = 0
    if n > 0:
        try:
            wilson_denominator = 1 + z*z/n
            if wilson_denominator != 0:
                wilson_numerator = win_rate + z*z/(2*n) - z * np.sqrt((win_rate*(1-win_rate) + z*z/(4*n))/n)
                wilson_lower = wilson_numerator / wilson_denominator
            else:
                wilson_lower = 0
        except Exception as e:
            logger.debug(f"Wilson信頼区間計算エラー: {str(e)}")
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
        # B1: 安全なスコア計算
        # パフォーマンススコア（高いほど良い）
        win_rate = buy_summary.get('avg_win_rate', 0)
        pf = min(buy_summary.get('avg_pf', 0), 3)  # 極端な値を制限
        sharpe = min(buy_summary.get('avg_sharpe', 0), 2)  # 極端な値を制限
        
        perf_score = (
            0.4 * safe_get(win_rate, 0) + 
            0.3 * safe_get(pf, 0) / 3 + 
            0.3 * safe_get(sharpe, 0) / 2
        )
        
        # 安定性スコア（低いほど良い）
        std_win_rate = min(buy_summary.get('std_win_rate', 1), 0.2)
        std_pf = min(buy_summary.get('std_pf', 3), 1)
        std_sharpe = min(buy_summary.get('std_sharpe', 2), 0.5)
        
        stability = (
            0.4 * (1 - safe_get(std_win_rate, 0.2) / 0.2) + 
            0.3 * (1 - safe_get(std_pf, 1) / 1) + 
            0.3 * (1 - safe_get(std_sharpe, 0.5) / 0.5)
        )
        
        # 総合スコア
        buy_score = 0.6 * perf_score + 0.4 * stability
        score += buy_score
        count += 1
    
    # 売りシグナルのスコア
    if sell_summary and sell_summary.get('etf_count', 0) >= 3:
        # B1: 安全なスコア計算
        # パフォーマンススコア（高いほど良い）
        win_rate = sell_summary.get('avg_win_rate', 0)
        pf = min(sell_summary.get('avg_pf', 0), 3)  # 極端な値を制限
        sharpe = min(sell_summary.get('avg_sharpe', 0), 2)  # 極端な値を制限
        
        perf_score = (
            0.4 * safe_get(win_rate, 0) + 
            0.3 * safe_get(pf, 0) / 3 + 
            0.3 * safe_get(sharpe, 0) / 2
        )
        
        # 安定性スコア（低いほど良い）
        std_win_rate = min(sell_summary.get('std_win_rate', 1), 0.2)
        std_pf = min(sell_summary.get('std_pf', 3), 1)
        std_sharpe = min(sell_summary.get('std_sharpe', 2), 0.5)
        
        stability = (
            0.4 * (1 - safe_get(std_win_rate, 0.2) / 0.2) + 
            0.3 * (1 - safe_get(std_pf, 1) / 1) + 
            0.3 * (1 - safe_get(std_sharpe, 0.5) / 0.5)
        )
        
        # 総合スコア
        sell_score = 0.6 * perf_score + 0.4 * stability
        score += sell_score
        count += 1
    
    # 平均スコアを返す
    return score / count if count > 0 else 0

# B1: ヘルパー関数の追加
def is_valid_metric(value):
    """メトリックが有効かどうか確認する"""
    return not (np.isnan(value) or np.isinf(value))

def safe_mean(values, default=0):
    """安全な平均計算（NaN/Infを除外）"""
    if not values:
        return default
    valid_values = [v for v in values if is_valid_metric(v)]
    return np.mean(valid_values) if valid_values else default

def safe_std(values, default=0):
    """安全な標準偏差計算（NaN/Infを除外）"""
    if not values:
        return default
    valid_values = [v for v in values if is_valid_metric(v)]
    return np.std(valid_values) if len(valid_values) > 1 else default

def safe_get(value, default=0):
    """安全な値取得（NaN/Infの場合はデフォルト値を使用）"""
    if np.isnan(value) or np.isinf(value):
        return default
    return value
