# src/validation/walk_forward.py
"""Walk-Forward分析モジュール"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Callable
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import datetime, timedelta

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
    # データの日付インデックスを取得
    dates = data.index.sort_values().tolist()
    
    # データの開始日と終了日
    start_date = dates[0]
    end_date = dates[-1]
    
    # データ期間の長さ（年）
    data_years = (end_date - start_date).days / 365.25
    
    # 最低限必要なデータ期間
    min_years_required = train_years + test_years
    
    if data_years < min_years_required:
        print(f"警告: データ期間({data_years:.1f}年)が最低要件({min_years_required}年)を満たしていません")
        
        # データが十分にない場合、より短い期間で調整
        if data_years >= 2:  # 最低2年はあると仮定
            train_years = max(1, int(data_years * 0.8))  # 全期間の80%をトレーニングに
            test_years = max(1, int(data_years * 0.2))   # 全期間の20%をテストに
            step_months = max(3, int(data_years * 4))    # 4ヶ月/年のステップ
            
            print(f"期間を調整: トレーニング={train_years}年, テスト={test_years}年, ステップ={step_months}ヶ月")
        else:
            print(f"エラー: 有効なWalk-Forwardウィンドウを生成できません")
            return []
    
    # ウィンドウ生成
    windows = []
    
    # 最初のウィンドウの開始日
    current_start = start_date
    
    while True:
        # トレーニング期間の終了日
        train_end_date = current_start + timedelta(days=int(train_years * 365.25))
        
        # テスト期間の終了日
        test_end_date = train_end_date + timedelta(days=int(test_years * 365.25))
        
        # 終了条件: テスト期間の終了日がデータ範囲を超える場合
        if test_end_date > end_date:
            break
        
        # トレーニング期間とテスト期間に含まれる日付を取得
        train_dates = [d for d in dates if current_start <= d <= train_end_date]
        test_dates = [d for d in dates if train_end_date < d <= test_end_date]
        
        # 有効なウィンドウの場合のみ追加
        if train_dates and test_dates:
            windows.append((train_dates, test_dates))
        
        # 次のウィンドウの開始日
        current_start = current_start + timedelta(days=int(step_months * 30.44))
    
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
    print(f"{symbol}の{signal_column}に対してWalk-Forward分析を実行中...")
    
    # Walk-Forwardウィンドウの生成
    windows = generate_walk_forward_windows(
        data,
        train_years=train_years,
        test_years=test_years,
        step_months=step_months
    )
    
    if not windows:
        print(f"  エラー: 有効なWalk-Forwardウィンドウが生成できませんでした")
        return {
            'symbol': symbol,
            'signal_type': signal_column,
            'holding_period': holding_period,
            'windows': [],
            'overall': {},
            'equity_curve': []
        }
    
    # 各ウィンドウのパフォーマンスを評価
    window_results = []
    equity_curve = []
    initial_equity = 1.0  # 初期資本
    
    for i, (train_dates, test_dates) in enumerate(windows):
        # トレーニングデータとテストデータの分割
        train_data = data.loc[train_dates]
        test_data = data.loc[test_dates]
        
        # テストデータ内のシグナル日
        test_signal_days = test_data[test_data[signal_column]].index
        
        if len(test_signal_days) == 0:
            print(f"  ウィンドウ{i+1}: テストデータにシグナルが含まれていません")
            
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
                current_price = test_data.loc[day, 'Adj Close']
                
                # 保有期間後の価格（取引日ベース）
                # 保有期間後の日付がデータフレームの範囲を超える場合はスキップ
                future_idx = data.index.get_loc(day) + holding_period
                if future_idx >= len(data):
                    continue
                    
                future_date = data.index[future_idx]
                future_price = data.loc[future_date, 'Adj Close']
                
                # リターンの計算
                if signal_column == 'Buy_Signal':
                    ret = (future_price / current_price) - 1
                else:  # 'Sell_Signal'
                    ret = (current_price / future_price) - 1
                
                trade_returns.append(ret)
                trade_dates.append(day)
            
            except Exception as e:
                print(f"  評価エラー - {day}: {str(e)}")
        
        # 有効なリターンがない場合
        if not trade_returns:
            print(f"  ウィンドウ{i+1}: 有効なリターンが計算できませんでした")
            
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
        
        if n > 0:
            wilson_lower = (win_rate + z*z/(2*n) - z * np.sqrt((win_rate*(1-win_rate) + z*z/(4*n))/n)) / (1 + z*z/n)
        else:
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
            # ポートフォリオへの影響を計算（複利）
            current_equity *= (1 + (trade_return * portfolio_fraction))
            
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
        
        print(f"  ウィンドウ{i+1}: 勝率 = {win_rate:.1%}, PF = {profit_factor:.2f}, 信頼下限 = {wilson_lower:.1%}, リターン = {window_result['window_return']:.1%}")
    
    # 全体の統計値
    if window_results:
        overall_win_rate = np.mean([r['win_rate'] for r in window_results])
        overall_pf = np.mean([r['profit_factor'] for r in window_results if r['profit_factor'] != float('inf')])
        overall_sharpe = np.mean([r['sharpe_ratio'] for r in window_results])
        overall_wilson_lower = np.mean([r['wilson_lower'] for r in window_results])
        
        # 累積リターン
        if equity_curve:
            overall_return = equity_curve[-1]['equity'] / initial_equity - 1
        else:
            overall_return = 0
        
        print(f"  全体: 勝率 = {overall_win_rate:.1%}, PF = {overall_pf:.2f}, 信頼下限 = {overall_wilson_lower:.1%}, 累積リターン = {overall_return:.1%}")
    else:
        print(f"  警告: 有効なウィンドウ結果がありません")
        
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
    
    symbol = results['symbol']
    signal_type = results['signal_type']
    
    # 可視化ファイルのパス
    visualization_paths = {}
    
    # ウィンドウ結果がない場合
    if not results['windows']:
        print(f"警告: {symbol}の{signal_type}には有効なウィンドウ結果がありません")
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
    if results['equity_curve']:
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
    
    return visualization_paths
