# src/validation/cpcv.py
"""Combinatorial Purged Cross-Validation (CPCV) の実装"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Callable
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_cpcv_folds(
    data: pd.DataFrame,
    n_splits: int = 5,
    purge_window: int = 5,
    embargo_window: int = 5
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    """CPCVのトレーニング/テストフォールドを生成する
    
    Args:
        data: 対象のデータフレーム
        n_splits: 分割数
        purge_window: パージウィンドウ（日数）
        embargo_window: エンバーゴウィンドウ（日数）
        
    Returns:
        List[Tuple]: (トレーニング期間リスト, テスト期間リスト)のタプルのリスト
    """
    try:
        # データの日付インデックスを取得
        if not isinstance(data.index, pd.DatetimeIndex):
            print("警告: データのインデックスが日付型ではありません。ソートのみ行います。")
        
        # A2: データチェックの強化 - 空のデータフレーム対応
        if data.empty:
            print("エラー: 空のデータフレームが渡されました。有効なウィンドウを生成できません。")
            return []
        
        dates = data.index.sort_values().tolist()
        
        # A2: データチェックの強化 - ウィンドウサイズの検証
        if len(dates) < n_splits * 2:
            print(f"警告: データポイントが少なすぎます（{len(dates)}ポイント）。最低でも{n_splits * 2}ポイント必要です。")
            # データポイント数に基づいて分割数を調整
            n_splits = max(2, len(dates) // 2)
            print(f"分割数を{n_splits}に調整します。")
        
        # 各バケットに含まれる日付のインデックス
        indices = np.arange(len(dates))
        
        # フォールドのサイズを計算
        fold_size = len(dates) // n_splits
        
        # A2: フォールドサイズが0になる場合の対応
        if fold_size == 0:
            print("エラー: フォールドサイズが0になります。データポイント数が足りません。")
            return []
        
        # 各フォールドの開始インデックス
        fold_start_indices = [i * fold_size for i in range(n_splits)]
        
        # 最後のフォールドサイズを調整
        fold_start_indices.append(len(dates))
        
        # 各フォールドの期間を取得
        fold_periods = []
        for i in range(n_splits):
            start_idx = fold_start_indices[i]
            end_idx = fold_start_indices[i + 1]
            fold_periods.append((dates[start_idx], dates[end_idx - 1] if end_idx < len(dates) else dates[-1]))
        
        # CPCVの組み合わせを生成
        cpcv_folds = []
        
        # k=n_splits、r=n_splits-1 の組み合わせを生成
        # つまり、1つのフォールドをテストに、残りをトレーニングに使用
        fold_indices = list(range(n_splits))
        
        for test_fold_idx in range(n_splits):
            train_fold_indices = [i for i in fold_indices if i != test_fold_idx]
            
            # テスト期間を決定
            test_start, test_end = fold_periods[test_fold_idx]
            
            # A2: インデックスエラー回避のための確認
            try:
                test_start_idx = dates.index(test_start)
                test_end_idx = dates.index(test_end)
            except ValueError as e:
                print(f"警告: テスト期間の日付が見つかりません: {e}")
                continue
            
            # A2: パージウィンドウとエンバーゴウィンドウのサイズチェック
            max_window = len(dates) // (n_splits * 2)
            safe_purge_window = min(purge_window, max_window)
            safe_embargo_window = min(embargo_window, max_window)
            
            if safe_purge_window != purge_window or safe_embargo_window != embargo_window:
                print(f"警告: ウィンドウサイズを調整しました。パージ: {purge_window}->{safe_purge_window}, エンバーゴ: {embargo_window}->{safe_embargo_window}")
            
            # トレーニング期間の日付リストを作成
            train_dates = []
            
            for train_fold_idx in train_fold_indices:
                train_start, train_end = fold_periods[train_fold_idx]
                
                try:
                    train_start_idx = dates.index(train_start)
                    train_end_idx = dates.index(train_end)
                except ValueError as e:
                    print(f"警告: トレーニング期間の日付が見つかりません: {e}")
                    continue
                
                # パージウィンドウ内の日付を除外
                purge_start_idx = max(0, test_start_idx - safe_purge_window)
                purge_end_idx = min(len(dates) - 1, test_end_idx + safe_purge_window)
                
                # エンバーゴウィンドウ内の日付を除外
                embargo_start_idx = max(0, purge_start_idx - safe_embargo_window)
                embargo_end_idx = min(len(dates) - 1, purge_end_idx + safe_embargo_window)
                
                # パージ・エンバーゴを適用したトレーニング期間
                if embargo_start_idx <= train_start_idx and embargo_end_idx >= train_end_idx:
                    # 完全に除外すべき場合
                    continue
                elif embargo_start_idx > train_end_idx or embargo_end_idx < train_start_idx:
                    # 完全に独立している場合
                    train_period_dates = dates[train_start_idx:train_end_idx+1]
                    train_dates.extend(train_period_dates)
                else:
                    # 部分的に重複している場合
                    if train_start_idx < embargo_start_idx:
                        train_period_dates = dates[train_start_idx:embargo_start_idx]
                        train_dates.extend(train_period_dates)
                    
                    if train_end_idx > embargo_end_idx:
                        train_period_dates = dates[embargo_end_idx+1:train_end_idx+1]
                        train_dates.extend(train_period_dates)
            
            # テスト期間の日付リスト
            test_dates = dates[test_start_idx:test_end_idx+1]
            
            # 重複を排除
            train_dates = sorted(list(set(train_dates)))
            test_dates = sorted(list(set(test_dates)))
            
            # A2: データ数の確認を追加
            if len(train_dates) < n_splits or len(test_dates) < 2:
                print(f"警告: フォールド{test_fold_idx}のデータ数が不足しています (トレーニング: {len(train_dates)}, テスト: {len(test_dates)})")
            
            # 有効なフォールドのみ追加（空のトレーニングやテストフォールドは回避）
            if train_dates and test_dates:
                cpcv_folds.append((train_dates, test_dates))
            else:
                print(f"警告: フォールド{test_fold_idx}が空のため、除外されました")
        
        # A2: 少なくとも1つのフォールドが有効かチェック
        if not cpcv_folds:
            print("エラー: 有効なCPCVフォールドが生成できませんでした。")
        
        return cpcv_folds
    
    except Exception as e:
        print(f"CPCVフォールド生成エラー: {str(e)}")
        return []

def run_cpcv_analysis(
    symbol: str,
    data: pd.DataFrame,
    signal_column: str,
    holding_period: int,
    n_splits: int = 5,
    purge_window: int = None,
    embargo_window: int = None
) -> Dict[str, Any]:
    """CPCVを使用してシグナルのパフォーマンスを評価する
    
    Args:
        symbol: ETFのシンボル
        data: テクニカル指標を含むデータフレーム
        signal_column: シグナル列名（'Buy_Signal'または'Sell_Signal'）
        holding_period: ポジション保有期間
        n_splits: 分割数
        purge_window: パージウィンドウ（指定なしの場合はholding_period + 1）
        embargo_window: エンバーゴウィンドウ（指定なしの場合はholding_period + 1）
        
    Returns:
        Dict: CPCV分析結果
    """
    try:
        # 入力チェック
        if not isinstance(data, pd.DataFrame):
            print(f"エラー: データがDataFrameではありません: {type(data)}")
            return {
                'symbol': symbol,
                'signal_type': signal_column,
                'holding_period': holding_period,
                'folds': [],
                'overall': {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'wilson_lower': 0
                },
                'error': 'データ型エラー'
            }
        
        if signal_column not in data.columns:
            print(f"エラー: データに{signal_column}カラムが含まれていません")
            return {
                'symbol': symbol,
                'signal_type': signal_column,
                'holding_period': holding_period,
                'folds': [],
                'overall': {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'wilson_lower': 0
                },
                'error': f'{signal_column}カラムが見つかりません'
            }
        
        if 'Close' not in data.columns:
            print("エラー: データにCloseカラムが含まれていません")
            return {
                'symbol': symbol,
                'signal_type': signal_column,
                'holding_period': holding_period,
                'folds': [],
                'overall': {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'wilson_lower': 0
                },
                'error': 'Closeカラムが見つかりません'
            }
        
        # A2: データサイズの確認を追加
        if len(data) < 2 * n_splits:
            print(f"警告: データポイント数({len(data)})が少なすぎます。フォールド数を調整します。")
            n_splits = max(2, len(data) // 2)
            
        # デフォルト値の設定
        if purge_window is None:
            purge_window = holding_period + 1
        
        if embargo_window is None:
            embargo_window = holding_period + 1
        
        print(f"{symbol}の{signal_column}に対してCPCV分析を実行中...")
        
        # CPCVフォールドの生成
        cpcv_folds = generate_cpcv_folds(
            data, 
            n_splits=n_splits,
            purge_window=purge_window,
            embargo_window=embargo_window
        )
        
        if not cpcv_folds:
            print("警告: 有効なCPCVフォールドが生成できませんでした")
            return {
                'symbol': symbol,
                'signal_type': signal_column,
                'holding_period': holding_period,
                'folds': [],
                'overall': {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'wilson_lower': 0
                },
                'warning': 'フォールド生成失敗'
            }
        
        # 各フォールドのパフォーマンスを評価
        fold_results = []
        
        for i, (train_dates, test_dates) in enumerate(cpcv_folds):
            try:
                # トレーニングデータとテストデータの分割
                train_data = data.loc[train_dates]
                test_data = data.loc[test_dates]
                
                # テストデータ内のシグナル日
                test_signal_days = test_data[test_data[signal_column]].index
                
                if len(test_signal_days) == 0:
                    print(f"  フォールド{i+1}: テストデータにシグナルが含まれていません")
                    continue
                
                # リターンの計算
                returns = []
                is_win = []
                
                for day in test_signal_days:
                    # 現在の価格
                    try:
                        current_price = test_data.loc[day, 'Close']
                        
                        # 保有期間後の価格（取引日ベース）
                        # 保有期間後の日付がデータフレームの範囲を超える場合はスキップ
                        try:
                            # A2: インデックスエラー回避の処理
                            if day not in data.index:
                                continue
                                
                            day_idx = data.index.get_loc(day)
                            future_idx = day_idx + holding_period
                            
                            # インデックス範囲チェック
                            if future_idx >= len(data):
                                continue
                                
                            future_date = data.index[future_idx]
                            
                            # future_dateがデータに存在するか確認
                            if future_date not in data.index:
                                continue
                                
                            future_price = data.loc[future_date, 'Close']
                            
                            # リターンの計算
                            if signal_column == 'Buy_Signal':
                                ret = (future_price / current_price) - 1
                            else:  # 'Sell_Signal'
                                ret = (current_price / future_price) - 1
                            
                            returns.append(ret)
                            is_win.append(ret > 0)
                        except (KeyError, IndexError) as e:
                            print(f"  インデックスエラー - {day}: {str(e)}")
                            continue
                    
                    except Exception as e:
                        print(f"  評価エラー - {day}: {str(e)}")
                        continue
                
                # 有効なリターンがない場合
                if not returns:
                    print(f"  フォールド{i+1}: 有効なリターンが計算できませんでした")
                    continue
                
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
                
                fold_result = {
                    'fold': i + 1,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'signal_count': len(test_signal_days),
                    'valid_trades': len(returns),
                    'win_rate': float(win_rate),
                    'profit_factor': float(profit_factor),
                    'avg_return': float(avg_return),
                    'std_return': float(std_return),
                    'sharpe_ratio': float(sharpe_ratio),
                    'wilson_lower': float(wilson_lower),
                    'training_period': f"{train_dates[0]} to {train_dates[-1]}",
                    'test_period': f"{test_dates[0]} to {test_dates[-1]}"
                }
                
                fold_results.append(fold_result)
                
                print(f"  フォールド{i+1}: 勝率 = {win_rate:.1%}, PF = {profit_factor:.2f}, 信頼下限 = {wilson_lower:.1%}")
            
            except Exception as e:
                print(f"  フォールド{i+1}の評価エラー: {str(e)}")
                continue
        
        # 全体の統計値
        if fold_results:
            overall_win_rate = np.mean([r['win_rate'] for r in fold_results])
            # 無限値を除外してPF平均を計算
            finite_pfs = [r['profit_factor'] for r in fold_results if r['profit_factor'] != float('inf')]
            overall_pf = np.mean(finite_pfs) if finite_pfs else 0
            overall_sharpe = np.mean([r['sharpe_ratio'] for r in fold_results])
            overall_wilson_lower = np.mean([r['wilson_lower'] for r in fold_results])
            
            print(f"  全体: 勝率 = {overall_win_rate:.1%}, PF = {overall_pf:.2f}, 信頼下限 = {overall_wilson_lower:.1%}")
        else:
            print(f"  警告: 有効なフォールド結果がありません")
            
            overall_win_rate = 0
            overall_pf = 0
            overall_sharpe = 0
            overall_wilson_lower = 0
        
        # 結果をまとめる
        result = {
            'symbol': symbol,
            'signal_type': signal_column,
            'holding_period': holding_period,
            'folds': fold_results,
            'overall': {
                'win_rate': float(overall_win_rate),
                'profit_factor': float(overall_pf),
                'sharpe_ratio': float(overall_sharpe),
                'wilson_lower': float(overall_wilson_lower)
            }
        }
        
        return result
    
    except Exception as e:
        print(f"CPCV分析全体エラー: {str(e)}")
        return {
            'symbol': symbol,
            'signal_type': signal_column,
            'holding_period': holding_period,
            'folds': [],
            'overall': {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'wilson_lower': 0
            },
            'error': str(e)
        }

def visualize_cpcv_results(
    results: Dict[str, Any],
    output_dir: str = "data/results/cpcv"
) -> Dict[str, str]:
    """CPCV結果を可視化する
    
    Args:
        results: CPCV分析結果
        output_dir: 出力ディレクトリ
        
    Returns:
        Dict: 可視化ファイルのパス
    """
    try:
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        symbol = results.get('symbol', 'unknown')
        signal_type = results.get('signal_type', 'unknown')
        
        # 可視化ファイルのパス
        visualization_paths = {}
        
        # フォールド結果がない場合
        if not results.get('folds', []):
            print(f"警告: {symbol}の{signal_type}には有効なフォールド結果がありません")
            return visualization_paths
        
        # 結果をデータフレームに変換
        fold_df = pd.DataFrame(results['folds'])
        
        # 1. 勝率の分布
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=fold_df['win_rate'])
        plt.axhline(0.5, color='red', linestyle='--')
        plt.title(f"{symbol} - {signal_type}の勝率分布")
        plt.ylabel('勝率')
        plt.grid(True, alpha=0.3)
        
        win_rate_path = f"{output_dir}/{symbol}_{signal_type}_win_rate_distribution.png"
        plt.savefig(win_rate_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['win_rate'] = win_rate_path
        
        # 2. リターン分布
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=fold_df['avg_return'])
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f"{symbol} - {signal_type}のリターン分布")
        plt.ylabel('平均リターン')
        plt.grid(True, alpha=0.3)
        
        return_path = f"{output_dir}/{symbol}_{signal_type}_return_distribution.png"
        plt.savefig(return_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['return'] = return_path
        
        # 3. Profit Factorの分布（無限値を除外）
        finite_pf = fold_df[fold_df['profit_factor'] != float('inf')]['profit_factor']
        
        if not finite_pf.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=finite_pf)
            plt.axhline(1, color='red', linestyle='--')
            plt.title(f"{symbol} - {signal_type}のProfit Factor分布")
            plt.ylabel('Profit Factor')
            plt.grid(True, alpha=0.3)
            
            pf_path = f"{output_dir}/{symbol}_{signal_type}_profit_factor_distribution.png"
            plt.savefig(pf_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths['profit_factor'] = pf_path
        
        # 4. 各フォールドのパフォーマンスサマリー
        plt.figure(figsize=(12, 8))
        
        metrics = ['win_rate', 'wilson_lower', 'sharpe_ratio']
        metric_labels = ['勝率', 'Wilson下限', 'シャープレシオ']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            plt.subplot(len(metrics), 1, i+1)
            bars = plt.bar(fold_df['fold'], fold_df[metric])
            
            # 基準線
            if metric in ['win_rate', 'wilson_lower']:
                plt.axhline(0.5, color='red', linestyle='--')
            elif metric == 'sharpe_ratio':
                plt.axhline(0, color='red', linestyle='--')
            
            plt.title(f"{label}")
            plt.xlabel('フォールド')
            plt.ylabel(label)
            plt.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.2f}',
                    ha='center', va='bottom'
                )
        
        plt.tight_layout()
        
        summary_path = f"{output_dir}/{symbol}_{signal_type}_fold_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['summary'] = summary_path
        
        return visualization_paths
    
    except Exception as e:
        print(f"CPCV可視化エラー: {str(e)}")
        return {}
