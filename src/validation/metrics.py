# src/validation/metrics.py
"""パフォーマンス評価指標モジュール"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats

def calculate_basic_metrics(
    returns: np.ndarray
) -> Dict[str, float]:
    """基本的なトレードメトリクスを計算する
    
    Args:
        returns: トレードリターンの配列
        
    Returns:
        Dict: 各メトリクスの値
    """
    if len(returns) == 0:
        return {
            'win_rate': 0,
            'avg_return': 0,
            'std_return': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'recovery_factor': 0,
            'expectancy': 0
        }
    
    try:
        # 勝ちトレードと負けトレード
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        # 勝率
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        
        # 平均リターンと標準偏差
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Profit Factor
        total_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        total_loss = np.abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
        
        # シャープレシオ（リスクフリーレート0%と仮定）
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # 最大ドローダウン（複利ベース）
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = np.min(drawdowns)
        
        # リカバリーファクター
        total_return = cumulative_returns[-1] - 1
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown < 0 else np.inf
        
        # 期待値 (Expectancy)
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'std_return': float(std_return),
            'profit_factor': float(profit_factor),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'recovery_factor': float(recovery_factor),
            'expectancy': float(expectancy)
        }
    except Exception as e:
        print(f"メトリクス計算エラー: {str(e)}")
        return {
            'win_rate': 0,
            'avg_return': 0,
            'std_return': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'recovery_factor': 0,
            'expectancy': 0,
            'error': str(e)
        }

def calculate_statistical_significance(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    min_sample_size: int = 10
) -> Dict[str, Any]:
    """リターンの統計的有意性を評価する
    
    Args:
        returns: トレードリターンの配列
        confidence_level: 信頼水準
        min_sample_size: 最低サンプルサイズ
        
    Returns:
        Dict: 統計的検定の結果
    """
    # 最低サンプルサイズの確認
    if len(returns) < min_sample_size:
        return {
            'is_significant': False,
            'p_value': 1.0,
            'is_normal': False,
            't_stat': 0,
            'wilson_lower': 0,
            'mean_ci_lower': 0,
            'mean_ci_upper': 0,
            'sample_size': len(returns),
            'warning': f"サンプルサイズ不足 ({len(returns)} < {min_sample_size})"
        }
    
    try:
        # t検定（平均リターンが0より大きいかどうか）
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # 正規性検定（Shapiro-Wilk）
        # 注意: サンプルサイズが3以上5000以下である必要がある
        is_normal = False
        shapiro_p = 1.0
        
        if 3 <= len(returns) <= 5000:
            _, shapiro_p = stats.shapiro(returns)
            is_normal = shapiro_p > 0.05  # p > 0.05で正規分布と見なせる
        else:
            shapiro_p = None
            is_normal = None
        
        # Wilson信頼区間（勝率の信頼区間）
        win_rate = np.mean(returns > 0)
        z = stats.norm.ppf((1 + confidence_level) / 2)
        n = len(returns)
        
        wilson_lower = (win_rate + z*z/(2*n) - z * np.sqrt((win_rate*(1-win_rate) + z*z/(4*n))/n)) / (1 + z*z/n)
        
        # 平均リターンの信頼区間
        mean = np.mean(returns)
        se = stats.sem(returns)
        mean_ci = stats.t.interval(confidence_level, len(returns)-1, mean, se)
        
        return {
            'is_significant': p_value < (1 - confidence_level) and t_stat > 0,
            'p_value': float(p_value),
            'is_normal': bool(is_normal) if is_normal is not None else None,
            'shapiro_p': float(shapiro_p) if shapiro_p is not None else None,
            't_stat': float(t_stat),
            'wilson_lower': float(wilson_lower),
            'mean_ci_lower': float(mean_ci[0]),
            'mean_ci_upper': float(mean_ci[1]),
            'sample_size': n
        }
    except Exception as e:
        print(f"統計的有意性評価エラー: {str(e)}")
        return {
            'is_significant': False,
            'p_value': 1.0,
            'is_normal': False,
            't_stat': 0,
            'wilson_lower': 0,
            'mean_ci_lower': 0,
            'mean_ci_upper': 0,
            'sample_size': len(returns) if isinstance(returns, np.ndarray) else 0,
            'error': str(e)
        }

def calculate_market_regime_performance(
    returns: np.ndarray,
    market_returns: np.ndarray,
    dates: List[pd.Timestamp]
) -> Dict[str, Dict[str, float]]:
    """市場レジーム別のパフォーマンスを計算する
    
    Args:
        returns: トレードリターンの配列
        market_returns: 同期間の市場リターンの配列
        dates: トレード日付のリスト
        
    Returns:
        Dict: レジーム別のパフォーマンス指標
    """
    if len(returns) == 0:
        return {}
    
    try:
        # 入力データの整合性チェック
        if len(market_returns) == 0:
            return {'warning': '市場リターンデータが空です'}
            
        if len(returns) != len(dates):
            return {'warning': 'リターンと日付の長さが一致しません'}
        
        # 市場リターンをデータフレームに変換
        try:
            market_df = pd.DataFrame({'return': market_returns})
            market_df.index = pd.DatetimeIndex(dates)
        except Exception as e:
            return {'error': f'市場データフレーム変換エラー: {str(e)}'}
        
        # 月次リターンに変換
        try:
            monthly_returns = market_df.resample('M').mean()
        except Exception as e:
            return {'error': f'月次リターン計算エラー: {str(e)}'}
        
        # 市場レジームの定義
        bull_months = monthly_returns[monthly_returns['return'] > 0.05].index
        bear_months = monthly_returns[monthly_returns['return'] < -0.05].index
        sideways_months = monthly_returns[
            (monthly_returns['return'] >= -0.05) & 
            (monthly_returns['return'] <= 0.05)
        ].index
        
        # 各トレードの月を取得
        trade_months = pd.DatetimeIndex([pd.Timestamp(d).to_period('M').to_timestamp() for d in dates])
        
        # レジーム別のトレードを分類
        bull_mask = np.array([m in bull_months for m in trade_months])
        bear_mask = np.array([m in bear_months for m in trade_months])
        sideways_mask = np.array([m in sideways_months for m in trade_months])
        
        # レジーム別のリターン
        bull_returns = returns[bull_mask] if any(bull_mask) else np.array([])
        bear_returns = returns[bear_mask] if any(bear_mask) else np.array([])
        sideways_returns = returns[sideways_mask] if any(sideways_mask) else np.array([])
        
        # 各レジームのパフォーマンス指標
        regime_performance = {
            'counts': {
                'bull': int(np.sum(bull_mask)),
                'bear': int(np.sum(bear_mask)),
                'sideways': int(np.sum(sideways_mask))
            }
        }
        
        # 上昇相場
        if len(bull_returns) > 0:
            regime_performance['bull'] = calculate_basic_metrics(bull_returns)
            regime_performance['bull']['trade_count'] = len(bull_returns)
        
        # 下落相場
        if len(bear_returns) > 0:
            regime_performance['bear'] = calculate_basic_metrics(bear_returns)
            regime_performance['bear']['trade_count'] = len(bear_returns)
        
        # 横ばい相場
        if len(sideways_returns) > 0:
            regime_performance['sideways'] = calculate_basic_metrics(sideways_returns)
            regime_performance['sideways']['trade_count'] = len(sideways_returns)
        
        return regime_performance
    except Exception as e:
        print(f"市場レジーム別パフォーマンス計算エラー: {str(e)}")
        return {'error': str(e)}

def analyze_holding_period(
    prices: pd.DataFrame,
    signal_dates: List[pd.Timestamp],
    signal_type: str,
    max_holding_days: int = 10
) -> Dict[str, Any]:
    """最適保有期間を分析する
    
    Args:
        prices: 価格データフレーム
        signal_dates: シグナル発生日のリスト
        signal_type: シグナルタイプ（'Buy_Signal'または'Sell_Signal'）
        max_holding_days: 分析する最大保有日数
        
    Returns:
        Dict: 保有期間分析結果
    """
    if len(signal_dates) == 0:
        return {
            'optimal_holding': 0,
            'holding_returns': {},
            'warning': 'シグナル日が0件です'
        }
    
    try:
        # 「Close」カラムの存在確認
        if 'Close' not in prices.columns:
            return {
                'optimal_holding': 0,
                'holding_returns': {},
                'error': '価格データに「Close」カラムがありません'
            }
        
        # 各保有期間のリターンを計算
        holding_returns = {}
        
        for holding_days in range(1, max_holding_days + 1):
            returns = []
            
            for day in signal_dates:
                try:
                    # 現在の価格
                    current_price = prices.loc[day, 'Close']
                    
                    # インデックスの位置を取得
                    day_idx = prices.index.get_loc(day)
                    
                    # 保有期間後の価格
                    future_idx = day_idx + holding_days
                    if future_idx >= len(prices):
                        continue
                        
                    future_date = prices.index[future_idx]
                    future_price = prices.loc[future_date, 'Close']
                    
                    # リターンの計算
                    if signal_type == 'Buy_Signal':
                        ret = (future_price / current_price) - 1
                    else:  # 'Sell_Signal'
                        ret = (current_price / future_price) - 1
                    
                    returns.append(ret)
                
                except KeyError:
                    # インデックスが見つからない場合はスキップ
                    continue
                except Exception as e:
                    print(f"保有期間分析エラー (日付: {day}, 期間: {holding_days}): {str(e)}")
                    continue
            
            if returns:
                # 基本メトリクスを計算
                metrics = calculate_basic_metrics(np.array(returns))
                
                # シャープベースの最適化指標（標準偏差がゼロでないことを確認）
                if metrics['std_return'] > 0:
                    optimization_score = metrics['expectancy'] / metrics['std_return']
                else:
                    # 標準偏差がゼロの場合、リターンの符号に基づく代替スコア
                    if metrics['avg_return'] > 0:
                        optimization_score = float('inf')  # 正のリターンでばらつきなし
                    elif metrics['avg_return'] < 0:
                        optimization_score = float('-inf')  # 負のリターンでばらつきなし
                    else:
                        optimization_score = 0  # ゼロリターン
                
                holding_returns[holding_days] = {
                    'avg_return': metrics['avg_return'],
                    'win_rate': metrics['win_rate'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'expectancy': metrics['expectancy'],
                    'trade_count': len(returns),
                    'optimization_score': optimization_score
                }
        
        # 最適保有期間を特定
        if holding_returns:
            # ソートされた最適化スコアの最大値を持つ保有期間を選択
            optimal_holding = max(
                holding_returns.items(),
                key=lambda x: x[1]['optimization_score']
            )[0]
        else:
            optimal_holding = 0
        
        return {
            'optimal_holding': optimal_holding,
            'holding_returns': holding_returns
        }
    
    except Exception as e:
        print(f"保有期間分析全体エラー: {str(e)}")
        return {
            'optimal_holding': 0,
            'holding_returns': {},
            'error': str(e)
        }
