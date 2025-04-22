# src/validation/walk_forward.py
"""Walk-Forward分析モジュール"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Callable
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import logging
from datetime import datetime, timedelta

# ロガー設定
logger = logging.getLogger(__name__)

def generate_walk_forward_windows(
    data: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
    step_months: int = 12
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    """Walk-Forwardウィンドウを生成する
    
    Args:
        data: 対象のデータフレーム
        train_years: トレーニング期間（年）
        test_years: テスト期間（年）
        step_months: ステップサイズ（月）
        
    Returns:
        List[Tuple]: (トレーニング期間リスト, テスト期間リスト)のタプルのリスト
    """
    # A4: 入力データのエラーチェック強化
    if data is None or data.empty:
        logger.error("空のデータフレームが渡されました。ウィンドウを生成できません。")
        return []
    
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("データのインデックスが日付型ではありません。日付処理を正しく行えない可能性があります。")
        try:
            # インデックスを日付型に変換を試みる
            data.index = pd.to_datetime(data.index)
        except:
            logger.error("インデックスを日付型に変換できません。ウィンドウを生成できません。")
            return []
    
    # データの日付インデックスを取得
    try:
        dates = data.index.sort_values().tolist()
    except Exception as e:
        logger.error(f"日付のソートに失敗しました: {str(e)}")
        return []
    
    # データが空の場合はエラー
    if not dates:
        logger.error("有効な日付が見つかりません。ウィンドウを生成できません。")
        return []
    
    # データの開始日と終了日
    start_date = dates[0]
    end_date = dates[-1]
    
    # A4: 日付範囲の検証
    if end_date <= start_date:
        logger.error("開始日が終了日より後か同じです。ウィンドウを生成できません。")
        return []
    
    # データ期間の長さ（年）
    data_years = (end_date - start_date).days / 365.25
    
    # 最低限必要なデータ期間
    min_years_required = train_years + test_years
    
    if data_years < min_years_required:
        logger.warning(f"データ期間({data_years:.1f}年)が最低要件({min_years_required}年)を満たしていません")
        
        # データが十分にない場合、より短い期間で調整
        if data_years >= 2:  # 最低2年はあると仮定
            # A4: 期間を動的に調整
            old_train_years, old_test_years = train_years, test_years
            train_years = max(1, int(data_years * 0.8))  # 全期間の80%をトレーニングに
            test_years = max(1, int(data_years * 0.2))   # 全期間の20%をテストに
            step_months = max(3, int(data_years * 4))    # 4ヶ月/年のステップ
            
            logger.info(f"期間を調整: トレーニング={old_train_years}→{train_years}年, "
                       f"テスト={old_test_years}→{test_years}年, ステップ={step_months}ヶ月")
        else:
            logger.error(f"データ期間が短すぎます({data_years:.1f}年)。最低2年必要です。")
            return []
    
    # ウィンドウ生成
    windows = []
    
    # 最初のウィンドウの開始日
    current_start = start_date
    
    # A4: 無限ループ防止のためのセーフガード
    max_iterations = 100  # 最大イテレーション数
    iteration_count = 0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        
        # トレーニング期間の終了日
        train_end_date = current_start + timedelta(days=int(train_years * 365.25))
        
        # テスト期間の終了日
        test_end_date = train_end_date + timedelta(days=int(test_years * 365.25))
        
        # A4: 日付が範囲外にならないよう確認
        if train_end_date >= end_date:
            logger.warning("トレーニング期間の終了日がデータ範囲を超えています。ウィンドウ生成を終了します。")
            break
        
        # 終了条件: テスト期間の終了日がデータ範囲を超える場合
        if test_end_date > end_date:
            # A4: 最後のウィンドウを調整
            test_end_date = end_date
            logger.info(f"最後のウィンドウのテスト期間を調整: {train_end_date.strftime('%Y-%m-%d')} → {test_end_date.strftime('%Y-%m-%d')}")
            
            # トレーニング期間とテスト期間に含まれる日付を取得
            train_dates = [d for d in dates if current_start <= d <= train_end_date]
            test_dates = [d for d in dates if train_end_date < d <= test_end_date]
            
            # 有効なウィンドウの場合のみ追加
            if len(train_dates) > 0 and len(test_dates) > 0:
                windows.append((train_dates, test_dates))
                logger.info(f"最終ウィンドウを追加: トレーニング {len(train_dates)}日, テスト {len(test_dates)}日")
            
            break
        
        # トレーニング期間とテスト期間に含まれる日付を取得
        train_dates = [d for d in dates if current_start <= d <= train_end_date]
        test_dates = [d for d in dates if train_end_date < d <= test_end_date]
        
        # A4: 各期間の日数が最低限あるかチェック
        min_train_days = 30  # 最低30日のトレーニングデータ
        min_test_days = 10   # 最低10日のテストデータ
        
        if len(train_dates) < min_train_days:
            logger.warning(f"トレーニングデータが不足しています ({len(train_dates)}日 < {min_train_days}日)。このウィンドウをスキップします。")
        elif len(test_dates) < min_test_days:
            logger.warning(f"テストデータが不足しています ({len(test_dates)}日 < {min_test_days}日)。このウィンドウをスキップします。")
        else:
            # 有効なウィンドウの場合のみ追加
            windows.append((train_dates, test_dates))
            logger.debug(f"ウィンドウを追加: トレーニング {current_start.strftime('%Y-%m-%d')} → {train_end_date.strftime('%Y-%m-%d')} ({len(train_dates)}日), "
                       f"テスト {train_end_date.strftime('%Y-%m-%d')} → {test_end_date.strftime('%Y-%m-%d')} ({len(test_dates)}日)")
        
        # 次のウィンドウの開始日
        current_start = current_start + timedelta(days=int(step_months * 30.44))
    
    # A4: 無限ループ検出
    if iteration_count >= max_iterations:
        logger.error(f"最大イテレーション数 ({max_iterations}) に達しました。ウィンドウ生成に問題がある可能性があります。")
    
    # A4: 結果の検証
    if not windows:
        logger.warning("有効なウィンドウが生成されませんでした。")
    else:
        logger.info(f"{len(windows)}個のWalk-Forwardウィンドウを生成しました。")
    
    return windows

def run_walk_forward_analysis(
    symbol: str,
    data: pd.DataFrame,
    signal_column: str,
    holding_period: int,
    train_years: int = 5,
    test_years: int = 1,
    step_months: int = 12
) -> Dict[str, Any]:
    """Walk-Forward分析を実行する
    
    Args:
        symbol: ETFのシンボル
        data: シグナルを含むデータフレーム
        signal_column: シグナル列名（'Buy_Signal'または'Sell_Signal'）
        holding_period: ポジション保有期間
        train_years: トレーニング期間（年）
        test_years: テスト期間（年）
        step_months: ステップサイズ（月）
        
    Returns:
        Dict: Walk-Forward分析結果
    """
    logger.info(f"{symbol}の{signal_column}に対してWalk-Forward分析を実行中...")
    
    # A4: 入力データのバリデーション
    if data is None or data.empty:
        logger.error("空のデータフレームが渡されました。分析を中止します。")
        return {
            'symbol': symbol,
            'signal_type': signal_column,
            'holding_period': holding_period,
            'windows': [],
            'overall': {},
            'equity_curve': []
        }
    
    if signal_column not in data.columns:
        logger.error(f"指定されたシグナル列 '{signal_column}' がデータフレームに存在しません。")
        return {
            'symbol': symbol,
            'signal_type': signal_column,
            'holding_period': holding_period,
            'windows': [],
            'overall': {},
            'equity_curve': [],
            'error': 'シグナル列が見つかりません'
        }
    
    if 'Close' not in data.columns:
        logger.error("'Close'列がデータフレームに存在しません。")
        return {
            'symbol': symbol,
            'signal_type': signal_column,
            'holding_period': holding_period,
            'windows': [],
            'overall': {},
            'equity_curve': [],
            'error': 'Close列が見つかりません'
        }
    
    # A4: ホールディング期間の検証
    if holding_period <= 0:
        logger.warning(f"無効なホールディング期間 ({holding_period})。デフォルト値(5)を使用します。")
        holding_period = 5
    
    # Walk-Forwardウィンドウの生成
    windows = generate_walk_forward_windows(
        data,
        train_years=train_years,
        test_years=test_years,
        step_months=step_months
    )
    
    if not windows:
        logger.error("有効なWalk-Forwardウィンドウが生成できませんでした")
        return {
            'symbol': symbol,
            'signal_type': signal_column,
            'holding_period': holding_period,
            'windows': [],
            'overall': {},
            'equity_curve': [],
            'error': 'ウィンドウ生成失敗'
        }
    
    # 各ウィンドウのパフォーマンスを評価
    window_results = []
    equity_curve = []
    initial_equity = 1.0  # 初期資本
    
    for i, (train_dates, test_dates) in enumerate(windows):
        # トレーニングデータとテストデータの分割
        try:
            train_data = data.loc[train_dates]
            test_data = data.loc[test_dates]
            
            # A4: データ欠損の検証
            if train_data.empty:
                logger.warning(f"ウィンドウ{i+1}: トレーニングデータが空です")
                continue
                
            if test_data.empty:
                logger.warning(f"ウィンドウ{i+1}: テストデータが空です")
                continue
            
            # テストデータ内のシグナル日
            test_signal_days = test_data[test_data[signal_column]].index
            
            if len(test_signal_days) == 0:
                logger.info(f"ウィンドウ{i+1}: テストデータにシグナルが含まれていません")
                
                # 資本曲線用に前回のエクイティを維持
                if equity_curve:
                    last_equity = equity_curve[-1]['equity']
                    equity_curve.append({
                        'window': i + 1,
                        'date': test_dates[-1],
                        'equity': last_equity
                    })
                
                continue
            
            # トレードリターンの計算
            trade_returns = []
            trade_dates = []
            
            for day in test_signal_days:
                # 現在の価格
                try:
                    if 'Close' not in test_data.columns:
                        logger.error(f"エラー: 'Close'カラムがデータに存在しません")
                        continue
                        
                    current_price = test_data.loc[day, 'Close']
                    
                    # 保有期間後の価格（取引日ベース）
                    # 保有期間後の日付がデータフレームの範囲を超える場合はスキップ
                    try:
                        # A4: インデックスのセーフティチェック
                        if day not in data.index:
                            logger.warning(f"シグナル日 {day} がデータフレームのインデックスに見つかりません")
                            continue
                            
                        day_idx = data.index.get_loc(day)
                        future_idx = day_idx + holding_period
                        
                        # インデックス範囲チェック
                        if future_idx >= len(data):
                            logger.debug(f"将来の日付がデータ範囲外です: {day} + {holding_period}日")
                            continue
                            
                        future_date = data.index[future_idx]
                        
                        # 将来の日付が実際にデータに存在するか確認
                        if future_date not in data.index:
                            logger.warning(f"将来の日付 {future_date} がデータフレームに見つかりません")
                            continue
                            
                        future_price = data.loc[future_date, 'Close']
                        
                        # ゼロディバイダー対策
                        if current_price <= 0 or future_price <= 0:
                            logger.warning(f"無効な価格: 現在={current_price}, 将来={future_price}")
                            continue
                        
                        # リターンの計算
                        if signal_column == 'Buy_Signal':
                            ret = (future_price / current_price) - 1
                        else:  # 'Sell_Signal'
                            ret = (current_price / future_price) - 1
                        
                        # NaNやInf値のチェック
                        if np.isnan(ret) or np.isinf(ret):
                            logger.warning(f"無効なリターン値: {ret}")
                            continue
                        
                        trade_returns.append(ret)
                        trade_dates.append(day)
                    except KeyError as e:
                        logger.warning(f"警告: インデックスエラー ({day}) - {str(e)}")
                        continue
                    except IndexError as e:
                        logger.warning(f"警告: インデックス範囲エラー ({day}) - {str(e)}")
                        continue
                    
                except Exception as e:
                    logger.warning(f"評価エラー - {day}: {str(e)}")
                    continue
            
            # 有効なリターンがない場合
            if not trade_returns:
                logger.warning(f"ウィンドウ{i+1}: 有効なリターンが計算できませんでした")
                
                # 資本曲線用に前回のエクイティを維持
                if equity_curve:
                    last_equity = equity_curve[-1]['equity']
                    equity_curve.append({
                        'window': i + 1,
                        'date': test_dates[-1],
                        'equity': last_equity
                    })
                
                continue
            
            # 統計値の計算
            trade_returns = np.array(trade_returns)
            is_win = trade_returns > 0
            
            win_rate = np.mean(is_win)
            
            # 勝ちトレードと負けトレードの合計
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = np.abs(trade_returns[trade_returns < 0])
            
            total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
            total_losses = losing_trades.sum() if len(losing_trades) > 0 else 0
            
            # Profit Factor
            profit_factor = (total_wins / total_losses) if total_losses > 0 else (
                float('inf') if total_wins > 0 else 0
            )
            
            # リターンの平均と標準偏差
            avg_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            
            # シャープレシオ（リスクフリーレート0%と仮定）
            sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
            
            # Wilson信頼区間（95%）
            from scipy.stats import norm
            
            z = norm.ppf(0.975)  # 95%信頼区間
            n = len(trade_returns)
            
            wilson_lower = 0
            if n > 0:
                # A4: ZeroDivisionError対策
                try:
                    wilson_denominator = 1 + z*z/n
                    if wilson_denominator != 0:
                        wilson_numerator = win_rate + z*z/(2*n) - z * np.sqrt((win_rate*(1-win_rate) + z*z/(4*n))/n)
                        wilson_lower = wilson_numerator / wilson_denominator
                    else:
                        wilson_lower = 0
                except Exception as e:
                    logger.warning(f"Wilson信頼区間計算エラー: {str(e)}")
                    wilson_lower = 0
            
            # 資本曲線の計算
            if not equity_curve:
                current_equity = initial_equity
            else:
                current_equity = equity_curve[-1]['equity']
            
            # 複利と仮定：(1 + r1) * (1 + r2) * ... * (1 + rn)
            # 各トレードでポートフォリオの一定割合（例えば10%）を投資すると仮定
            portfolio_fraction = 0.1  # 各トレードでポートフォリオの10%を投資
            
            # トレード結果を時系列順に処理
            for trade_date, trade_return in zip(trade_dates, trade_returns):
                # A4: 極端なリターン値をクリップ
                safe_return = np.clip(trade_return, -0.9, 10.0)  # -90%〜1000%に制限
                
                # ポートフォリオへの影響を計算（複利）
                current_equity *= (1 + (safe_return * portfolio_fraction))
                
                # 資本曲線に追加
                equity_curve.append({
                    'window': i + 1,
                    'date': trade_date,
                    'equity': current_equity
                })
            
            # 最後のエクイティ値を使用して最終日を追加
            equity_curve.append({
                'window': i + 1,
                'date': test_dates[-1],
                'equity': current_equity
            })
            
            window_result = {
                'window': i + 1,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'signal_count': len(test_signal_days),
                'valid_trades': len(trade_returns),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'avg_return': float(avg_return),
                'std_return': float(std_return),
                'sharpe_ratio': float(sharpe_ratio),
                'wilson_lower': float(wilson_lower),
                'training_period': f"{train_dates[0]} to {train_dates[-1]}",
                'test_period': f"{test_dates[0]} to {test_dates[-1]}",
                'window_return': float(current_equity / initial_equity - 1)
            }
            
            window_results.append(window_result)
            
            logger.info(f"ウィンドウ{i+1}: 勝率 = {win_rate:.1%}, PF = {profit_factor:.2f}, 信頼下限 = {wilson_lower:.1%}, リターン = {window_result['window_return']:.1%}")
        
        except Exception as e:
            logger.error(f"ウィンドウ{i+1}の処理中にエラーが発生しました: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # エラーが発生しても処理を継続
            continue
    
    # 全体の統計値
    if window_results:
        # A4: NaNやInf値を除外した安全な平均計算
        def safe_mean(values, default=0):
            finite_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
            return np.mean(finite_values) if finite_values else default
        
        overall_win_rate = safe_mean([r['win_rate'] for r in window_results])
        # 無限値を除外してPF平均を計算
        overall_pf = safe_mean([r['profit_factor'] for r in window_results if r['profit_factor'] != float('inf')])
        overall_sharpe = safe_mean([r['sharpe_ratio'] for r in window_results])
        overall_wilson_lower = safe_mean([r['wilson_lower'] for r in window_results])
        
        # 累積リターン
        if equity_curve:
            overall_return = equity_curve[-1]['equity'] / initial_equity - 1
        else:
            overall_return = 0
        
        logger.info(f"全体: 勝率 = {overall_win_rate:.1%}, PF = {overall_pf:.2f}, 信頼下限 = {overall_wilson_lower:.1%}, 累積リターン = {overall_return:.1%}")
    else:
        logger.warning(f"警告: 有効なウィンドウ結果がありません")
        
        overall_win_rate = 0
        overall_pf = 0
        overall_sharpe = 0
        overall_wilson_lower = 0
        overall_return = 0
    
    # 結果をまとめる
    result = {
        'symbol': symbol,
        'signal_type': signal_column,
        'holding_period': holding_period,
        'windows': window_results,
        'overall': {
            'win_rate': float(overall_win_rate),
            'profit_factor': float(overall_pf),
            'sharpe_ratio': float(overall_sharpe),
            'wilson_lower': float(overall_wilson_lower),
            'return': float(overall_return)
        },
        'equity_curve': equity_curve
    }
    
    return result

def visualize_walk_forward_results(
    results: Dict[str, Any],
    output_dir: str = "data/results/walk_forward"
) -> Dict[str, str]:
    """Walk-Forward分析結果を可視化する
    
    Args:
        results: Walk-Forward分析結果
        output_dir: 出力ディレクトリ
        
    Returns:
        Dict: 可視化ファイルのパス
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    symbol = results.get('symbol', 'unknown')
    signal_type = results.get('signal_type', 'unknown')
    
    # 可視化ファイルのパス
    visualization_paths = {}
    
    # ウィンドウ結果がない場合
    if not results.get('windows', []):
        logger.warning(f"{symbol}の{signal_type}には有効なウィンドウ結果がありません")
        return visualization_paths
    
    # 結果をデータフレームに変換
    window_df = pd.DataFrame(results['windows'])
    
    # 1. 各ウィンドウのパフォーマンスサマリー
    plt.figure(figsize=(12, 8))
    
    metrics = ['win_rate', 'wilson_lower', 'window_return']
    metric_labels = ['勝率', 'Wilson下限', 'ウィンドウリターン']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        plt.subplot(len(metrics), 1, i+1)
        bars = plt.bar(window_df['window'], window_df[metric])
        
        # 基準線
        if metric in ['win_rate', 'wilson_lower']:
            plt.axhline(0.5, color='red', linestyle='--')
        elif metric == 'window_return':
            plt.axhline(0, color='red', linestyle='--')
        
        plt.title(f"{label}")
        plt.xlabel('ウィンドウ')
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
    
    summary_path = f"{output_dir}/{symbol}_{signal_type}_window_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualization_paths['summary'] = summary_path
    
    # 2. 資本曲線（エクイティカーブ）
    if results.get('equity_curve', []):
        try:
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df = equity_df.sort_values('date')
            
            plt.figure(figsize=(14, 7))
            plt.plot(equity_df['date'], equity_df['equity'], 'b-', linewidth=1.5)
            
            # ウィンドウごとに色を変えて描画
            for window in equity_df['window'].unique():
                window_data = equity_df[equity_df['window'] == window]
                plt.plot(window_data['date'], window_data['equity'], marker='o', markersize=3, linestyle='-', label=f'ウィンドウ{window}')
            
            plt.title(f"{symbol} - {signal_type}のエクイティカーブ")
            plt.ylabel('資本')
            plt.xlabel('日付')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 日付フォーマットの設定
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            equity_path = f"{output_dir}/{symbol}_{signal_type}_equity_curve.png"
            plt.savefig(equity_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths['equity_curve'] = equity_path
        except Exception as e:
            logger.error(f"エクイティカーブの可視化中にエラーが発生しました: {str(e)}")
    
    return visualization_paths
