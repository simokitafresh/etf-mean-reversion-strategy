# src/validation/metrics.py
"""パフォーマンス評価指標モジュール"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats

# ロガーの設定
logger = logging.getLogger(__name__)

def calculate_basic_metrics(
    returns: np.ndarray
) -> Dict[str, float]:
    """基本的なトレードメトリクスを計算する
    
    Args:
        returns: トレードリターンの配列
        
    Returns:
        Dict: 各メトリクスの値
    """
    # B3: 入力検証の強化
    if not isinstance(returns, np.ndarray):
        logger.warning(f"入力が numpy.ndarray ではありません: {type(returns)}")
        try:
            returns = np.array(returns)
        except:
            logger.error("入力を numpy.ndarray に変換できません")
            return empty_metrics_result()
    
    if len(returns) == 0:
        logger.warning("空のリターン配列が渡されました")
        return empty_metrics_result()
    
    # B3: 無効な値の検出と除外
    # NaNや無限大を持つインデックスを特定
    invalid_mask = np.isnan(returns) | np.isinf(returns)
    invalid_count = np.sum(invalid_mask)
    
    if invalid_count > 0:
        logger.warning(f"{invalid_count}/{len(returns)} ({invalid_count/len(returns):.1%})の無効な値を検出しました")
        # 有効な値のみを抽出
        valid_returns = returns[~invalid_mask]
        
        if len(valid_returns) == 0:
            logger.error("有効なリターン値がありません")
            return empty_metrics_result()
        
        returns = valid_returns
    
    # B3: 極端な値のクリッピング
    if np.max(np.abs(returns)) > 5.0:  # 500%を超えるリターンは異常値と考えられる
        extreme_count = np.sum(np.abs(returns) > 5.0)
        logger.warning(f"{extreme_count}件の極端なリターン値を検出しました。値をクリップします。")
        returns = np.clip(returns, -5.0, 5.0)
    
    try:
        # 勝ちトレードと負けトレード
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        # B3: サンプルサイズが小さい場合の対応
        if len(returns) < 10:
            logger.warning(f"サンプルサイズが非常に小さいです ({len(returns)}件)。結果の信頼性に注意してください。")
            
            # 信頼区間の追加
            confidence = calculate_confidence_level(len(returns))
            logger.info(f"小サンプルデータの信頼水準: {confidence:.1%}")
        
        # 勝率
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        
        # B3: 勝率の信頼区間（小サンプル用）
        win_rate_ci = calculate_win_rate_confidence_interval(win_rate, len(returns))
        
        # 平均リターンと標準偏差
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0
        
        # B3: 標準誤差の計算（小サンプル用）
        se = std_return / np.sqrt(len(returns)) if len(returns) > 1 else 0
        
        # Profit Factor
        total_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        total_loss = np.abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
        
        # B3: 安全なシャープレシオ計算
        # 小サンプルの場合は補正係数を使用
        if len(returns) < 30:
            # 小サンプル用の補正係数
            correction = 1 - (1 / (4 * len(returns)))
            sharpe_ratio = (avg_return / std_return) * correction if std_return > 0 else 0
        else:
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
        
        # B3: 小サンプル向けの追加メトリクス
        small_sample_metrics = {}
        if len(returns) < 30:
            small_sample_metrics.update({
                'win_rate_lower_ci': float(win_rate_ci[0]),
                'win_rate_upper_ci': float(win_rate_ci[1]),
                'return_se': float(se),
                'confidence_level': float(calculate_confidence_level(len(returns))),
                'is_small_sample': True
            })
        
        # 結果を統合
        result = {
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'std_return': float(std_return),
            'profit_factor': float(profit_factor),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'recovery_factor': float(recovery_factor),
            'expectancy': float(expectancy),
            'sample_size': len(returns)
        }
        
        # 小サンプル用メトリクスを追加
        result.update(small_sample_metrics)
        
        return result
    
    except Exception as e:
        logger.error(f"メトリクス計算エラー: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
        # エラー時にはデフォルト値を返す
        error_result = empty_metrics_result()
        error_result['error'] = str(e)
        return error_result

def empty_metrics_result() -> Dict[str, float]:
    """空のメトリクス結果を生成"""
    return {
        'win_rate': 0,
        'avg_return': 0,
        'std_return': 0,
        'profit_factor': 0,
        'sharpe_ratio': 0,
        'max_drawdown': 0,
        'recovery_factor': 0,
        'expectancy': 0,
        'sample_size': 0
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
    # B3: 入力検証の強化
    if not isinstance(returns, np.ndarray):
        logger.warning(f"入力が numpy.ndarray ではありません: {type(returns)}")
        try:
            returns = np.array(returns)
        except:
            logger.error("入力を numpy.ndarray に変換できません")
            return invalid_significance_result(f"入力タイプエラー: {type(returns)}")
    
    # 無効な値のチェックと除外
    invalid_mask = np.isnan(returns) | np.isinf(returns)
    invalid_count = np.sum(invalid_mask)
    
    if invalid_count > 0:
        logger.warning(f"{invalid_count}/{len(returns)} の無効な値を検出しました")
        valid_returns = returns[~invalid_mask]
        
        if len(valid_returns) == 0:
            logger.error("有効なリターン値がありません")
            return invalid_significance_result("有効なデータなし")
        
        returns = valid_returns
        
    # 最低サンプルサイズの確認
    if len(returns) < min_sample_size:
        warning_msg = f"サンプルサイズ不足 ({len(returns)} < {min_sample_size})"
        logger.warning(warning_msg)
        
        # B3: 小サンプルの場合でも可能な限り計算を実行
        is_small_sample = True
        
        # 信頼水準を調整
        adjusted_confidence = calculate_confidence_level(len(returns))
        logger.info(f"小サンプルのため信頼水準を調整: {confidence_level:.2f} -> {adjusted_confidence:.2f}")
        confidence_level = adjusted_confidence
    else:
        is_small_sample = False
    
    try:
        # B3: 小サンプル用の特殊処理
        if is_small_sample:
            # t検定（平均リターンが0より大きいかどうか）- Welchのt検定
            t_stat, p_value = stats.ttest_1samp(returns, 0, alternative='greater')
            
            # 効果量（Cohen's d）を計算
            effect_size = calculate_cohens_d(returns)
            
            # 検出力分析
            power = calculate_power(len(returns), effect_size, alpha=1-confidence_level)
            
            # 正規性検定（Shapiro-Wilk）
            # サンプルサイズが小さいのでShapiro-Wilkが適している
            is_normal = None
            shapiro_p = None
            
            if 3 <= len(returns) <= 5000:
                _, shapiro_p = stats.shapiro(returns)
                is_normal = shapiro_p > 0.05  # p > 0.05で正規分布と見なせる
        else:
            # t検定（平均リターンが0より大きいかどうか）
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            
            # 効果量（Cohen's d）
            effect_size = calculate_cohens_d(returns)
            
            # 検出力は大サンプルでは通常1に近いので計算不要
            power = 1.0
            
            # 正規性検定
            # サンプルサイズが大きい場合はKS検定が適している
            if len(returns) <= 5000:
                _, shapiro_p = stats.shapiro(returns)
                is_normal = shapiro_p > 0.05
            else:
                # サンプルサイズが大きい場合はKolmogorov-Smirnovテスト
                _, ks_p = stats.kstest(returns, 'norm', args=(np.mean(returns), np.std(returns)))
                shapiro_p = ks_p
                is_normal = ks_p > 0.05
        
        # Wilson信頼区間（勝率の信頼区間）
        win_rate = np.mean(returns > 0)
        wilson_ci = calculate_win_rate_confidence_interval(win_rate, len(returns), confidence_level)
        wilson_lower, wilson_upper = wilson_ci
        
        # 平均リターンの信頼区間
        mean = np.mean(returns)
        se = np.std(returns, ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else 0
        
        # B3: 小サンプル用のt分布を使用
        if is_small_sample:
            df = len(returns) - 1  # 自由度
            t_crit = stats.t.ppf((1 + confidence_level) / 2, df)  # t分布の臨界値
            mean_ci = (mean - t_crit * se, mean + t_crit * se)
        else:
            mean_ci = stats.t.interval(confidence_level, len(returns)-1, mean, se)
        
        # B3: 結果の安全性チェック
        if not all(isinstance(x, (int, float)) for x in [t_stat, p_value, wilson_lower, wilson_upper, 
                                                         mean_ci[0], mean_ci[1]]):
            logger.warning("計算結果に非数値が含まれています")
            
            # 無効な値を修正
            t_stat = float(t_stat) if isinstance(t_stat, (int, float)) else 0
            p_value = float(p_value) if isinstance(p_value, (int, float)) else 1.0
            wilson_lower = float(wilson_lower) if isinstance(wilson_lower, (int, float)) else 0
            wilson_upper = float(wilson_upper) if isinstance(wilson_upper, (int, float)) else 0
            mean_ci = (
                float(mean_ci[0]) if isinstance(mean_ci[0], (int, float)) else 0,
                float(mean_ci[1]) if isinstance(mean_ci[1], (int, float)) else 0
            )
        
        return {
            'is_significant': p_value < (1 - confidence_level) and t_stat > 0,
            'p_value': float(p_value),
            'is_normal': bool(is_normal) if is_normal is not None else None,
            'shapiro_p': float(shapiro_p) if shapiro_p is not None else None,
            't_stat': float(t_stat),
            'wilson_lower': float(wilson_lower),
            'wilson_upper': float(wilson_upper),
            'mean_ci_lower': float(mean_ci[0]),
            'mean_ci_upper': float(mean_ci[1]),
            'sample_size': len(returns),
            'effect_size': float(effect_size),
            'power': float(power),
            'is_small_sample': is_small_sample,
            'confidence_level': float(confidence_level)
        }
    except Exception as e:
        logger.error(f"統計的有意性評価エラー: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return invalid_significance_result(str(e))

def invalid_significance_result(error_msg: str) -> Dict[str, Any]:
    """無効な統計的有意性の結果を生成"""
    return {
        'is_significant': False,
        'p_value': 1.0,
        'is_normal': False,
        't_stat': 0,
        'wilson_lower': 0,
        'wilson_upper': 0,
        'mean_ci_lower': 0,
        'mean_ci_upper': 0,
        'sample_size': 0,
        'effect_size': 0,
        'power': 0,
        'is_small_sample': True,
        'confidence_level': 0,
        'error': error_msg
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
    # B3: 入力検証の強化
    if len(returns) == 0:
        logger.warning("空のリターン配列が渡されました")
        return {}
    
    # 入力データの検証
    try:
        # 配列に変換
        returns = np.array(returns)
        
        # 市場リターンが空の場合
        if len(market_returns) == 0:
            logger.warning("市場リターンデータが空です")
            return {'warning': '市場リターンデータが空です'}
            
        market_returns = np.array(market_returns)
        
        # 配列の長さが一致するか確認
        if len(returns) != len(dates):
            logger.warning(f"リターンと日付の長さが一致しません: returns={len(returns)}, dates={len(dates)}")
            return {'warning': 'リターンと日付の長さが一致しません'}
        
        # NaN/Infを除外
        valid_indices = ~(np.isnan(returns) | np.isinf(returns))
        if not np.all(valid_indices):
            logger.warning(f"{np.sum(~valid_indices)}/{len(returns)}の無効な値を検出しました")
            returns = returns[valid_indices]
            dates = [dates[i] for i, valid in enumerate(valid_indices) if valid]
            
            if len(returns) == 0:
                logger.error("有効なリターン値がありません")
                return {'error': '有効なリターン値がありません'}
        
        # 市場データフレームに変換
        try:
            market_df = pd.DataFrame({'return': market_returns})
            if not isinstance(dates[0], pd.Timestamp):
                dates = pd.to_datetime(dates)
            market_df.index = pd.DatetimeIndex(dates)
        except Exception as e:
            logger.error(f"市場データフレーム変換エラー: {str(e)}")
            return {'error': f'市場データフレーム変換エラー: {str(e)}'}
        
        # B3: 小さすぎるサンプルをチェック
        if len(market_df) < 5:
            logger.warning(f"市場データが少なすぎます ({len(market_df)}件)")
            if len(market_df) < 3:
                return {'warning': f'市場データが不足しています ({len(market_df)}件)'}
        
        # 月次リターンに変換
        try:
            # B3: 安全なリサンプリング
            # 日付インデックスのソートと重複の削除
            market_df = market_df.sort_index()
            market_df = market_df[~market_df.index.duplicated(keep='first')]
            
            # 月次へのリサンプリング
            monthly_returns = market_df.resample('M').mean()
            
            # 少なくとも1つの有効な値があることを確認
            if monthly_returns.empty or monthly_returns['return'].isna().all():
                logger.warning("月次リターンの計算結果が空またはすべて欠損値です")
                return {'warning': '月次リターンの計算に失敗しました'}
        except Exception as e:
            logger.error(f"月次リターン計算エラー: {str(e)}")
            return {'error': f'月次リターン計算エラー: {str(e)}'}
        
        # 市場レジームの定義
        # B3: より堅牢なレジーム分類（外れ値の影響を軽減）
        bull_threshold = monthly_returns['return'].quantile(0.7)  # 上位30%を強気相場とする
        bear_threshold = monthly_returns['return'].quantile(0.3)  # 下位30%を弱気相場とする
        
        bull_months = monthly_returns[monthly_returns['return'] > bull_threshold].index
        bear_months = monthly_returns[monthly_returns['return'] < bear_threshold].index
        sideways_months = monthly_returns[
            (monthly_returns['return'] >= bear_threshold) & 
            (monthly_returns['return'] <= bull_threshold)
        ].index
        
        logger.info(f"市場レジーム分類: 強気={len(bull_months)}ヶ月, 弱気={len(bear_months)}ヶ月, 横ばい={len(sideways_months)}ヶ月")
        
        # 各トレードの月を取得
        # B3: 日付変換の安全性向上
        trade_months = []
        for d in dates:
            try:
                if not isinstance(d, pd.Timestamp):
                    d = pd.Timestamp(d)
                trade_months.append(d.to_period('M').to_timestamp())
            except Exception as e:
                logger.warning(f"日付変換エラー ({d}): {str(e)}")
                # エラー時はNoneを追加（後で除外）
                trade_months.append(None)
        
        # Noneを除外
        valid_trades = [(r, tm) for r, tm in zip(returns, trade_months) if tm is not None]
        if len(valid_trades) < len(returns):
            logger.warning(f"{len(returns) - len(valid_trades)}件のトレードが日付変換エラーにより除外されました")
            if not valid_trades:
                return {'error': 'すべてのトレードが日付変換エラーにより除外されました'}
            
            returns = np.array([vt[0] for vt in valid_trades])
            trade_months = [vt[1] for vt in valid_trades]
        
        # レジーム別のトレードを分類
        bull_mask = np.array([m in bull_months for m in trade_months])
        bear_mask = np.array([m in bear_months for m in trade_months])
        sideways_mask = np.array([m in sideways_months for m in trade_months])
        
        # レジーム別のリターン
        bull_returns = returns[bull_mask] if any(bull_mask) else np.array([])
        bear_returns = returns[bear_mask] if any(bear_mask) else np.array([])
        sideways_returns = returns[sideways_mask] if any(sideways_mask) else np.array([])
        
        # B3: 小サンプルチェック
        if len(bull_returns) < 5:
            logger.warning(f"強気相場のサンプル数が少なすぎます ({len(bull_returns)}件)")
        if len(bear_returns) < 5:
            logger.warning(f"弱気相場のサンプル数が少なすぎます ({len(bear_returns)}件)")
        if len(sideways_returns) < 5:
            logger.warning(f"横ばい相場のサンプル数が少なすぎます ({len(sideways_returns)}件)")
        
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
        logger.error(f"市場レジーム別パフォーマンス計算エラー: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
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
    # B3: 入力検証
    if not isinstance(prices, pd.DataFrame):
        logger.error(f"価格データが DataFrame ではありません: {type(prices)}")
        return {
            'optimal_holding': 0,
            'holding_returns': {},
            'error': '価格データが DataFrame ではありません'
        }
    
    if prices.empty:
        logger.warning("価格データが空です")
        return {
            'optimal_holding': 0,
            'holding_returns': {},
            'warning': '価格データが空です'
        }
    
    if len(signal_dates) == 0:
        logger.warning("シグナル日が0件です")
        return {
            'optimal_holding': 0,
            'holding_returns': {},
            'warning': 'シグナル日が0件です'
        }
    
    # B3: 小サンプルチェック
    if len(signal_dates) < 10:
        logger.warning(f"シグナル日のサンプル数が少なすぎます ({len(signal_dates)}件)。結果の信頼性に注意してください。")
    
    try:
        # 「Close」カラムの存在確認
        if 'Close' not in prices.columns:
            logger.error('価格データに「Close」カラムがありません')
            return {
                'optimal_holding': 0,
                'holding_returns': {},
                'error': '価格データに「Close」カラムがありません'
            }
        
        # B3: 入力データの準備
        # 日付インデックスの確認と変換
        if not isinstance(prices.index, pd.DatetimeIndex):
            logger.warning("価格データのインデックスが DatetimeIndex ではありません。変換を試みます。")
            try:
                prices.index = pd.to_datetime(prices.index)
            except Exception as e:
                logger.error(f"インデックスの日付変換に失敗しました: {str(e)}")
                return {
                    'optimal_holding': 0,
                    'holding_returns': {},
                    'error': 'インデックスの日付変換に失敗しました'
                }
        
        # シグナル日の日付型変換確認
        signal_dates_ts = []
        for date in signal_dates:
            if not isinstance(date, pd.Timestamp):
                try:
                    date = pd.Timestamp(date)
                except:
                    logger.warning(f"シグナル日 {date} を Timestamp に変換できません。スキップします。")
                    continue
            signal_dates_ts.append(date)
        
        if len(signal_dates_ts) < len(signal_dates):
            logger.warning(f"{len(signal_dates) - len(signal_dates_ts)}件のシグナル日が変換できず除外されました")
        
        if not signal_dates_ts:
            logger.error("有効なシグナル日がありません")
            return {
                'optimal_holding': 0,
                'holding_returns': {},
                'error': '有効なシグナル日がありません'
            }
        
        signal_dates = signal_dates_ts
        
        # 各保有期間のリターンを計算
        holding_returns = {}
        min_trades_for_significance = 5  # 統計的有意性に必要な最小トレード数
        
        for holding_days in range(1, max_holding_days + 1):
            returns = []
            dates = []
            
            for day in signal_dates:
                try:
                    # 現在の価格
                    if day not in prices.index:
                        # 警告は最初の数回だけ表示（大量の警告を避ける）
                        if len(returns) < 3:
                            logger.debug(f"シグナル日 {day} が価格データに見つかりません")
                        continue
                        
                    current_price = prices.loc[day, 'Close']
                    if np.isnan(current_price) or current_price <= 0:
                        if len(returns) < 3:
                            logger.debug(f"無効な現在価格: {current_price}")
                        continue
                    
                    # インデックスの位置を取得
                    day_idx = prices.index.get_loc(day)
                    
                    # 保有期間後の価格
                    future_idx = day_idx + holding_days
                    if future_idx >= len(prices):
                        continue
                        
                    future_date = prices.index[future_idx]
                    future_price = prices.loc[future_date, 'Close']
                    
                    if np.isnan(future_price) or future_price <= 0:
                        if len(returns) < 3:
                            logger.debug(f"無効な将来価格: {future_price}")
                        continue
                    
                    # リターンの計算
                    if signal_type == 'Buy_Signal':
                        ret = (future_price / current_price) - 1
                    else:  # 'Sell_Signal'
                        ret = (current_price / future_price) - 1
                    
                    # NaNやInf値のチェック
                    if np.isnan(ret) or np.isinf(ret):
                        if len(returns) < 3:
                            logger.debug(f"無効なリターン値: {ret}")
                        continue
                    
                    # 極端な値のクリッピング
                    if abs(ret) > 5.0:  # 500%を超えるリターンは異常値
                        ret = np.sign(ret) * 5.0
                        
                    returns.append(ret)
                    dates.append(day)
                
                except Exception as e:
                    if len(returns) < 3:
                        logger.debug(f"リターン計算エラー ({day}, 期間: {holding_days}): {str(e)}")
                    continue
            
            if returns:
                # 基本メトリクスを計算
                metrics = calculate_basic_metrics(np.array(returns))
                
                # B3: 小サンプル時の処理
                if len(returns) < min_trades_for_significance:
                    logger.warning(f"保有期間 {holding_days}日のサンプル数が少なすぎます ({len(returns)}件)")
                    metrics['is_reliable'] = False
                else:
                    metrics['is_reliable'] = True
                
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
                
                # B3: 最適化スコアの安全性チェック
                if np.isnan(optimization_score) or np.isinf(optimization_score):
                    logger.warning(f"無効な最適化スコア: {optimization_score}。0に置き換えます。")
                    optimization_score = 0
                
                # 結果を格納
                holding_returns[holding_days] = {
                    'avg_return': metrics['avg_return'],
                    'win_rate': metrics['win_rate'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'expectancy': metrics['expectancy'],
                    'trade_count': len(returns),
                    'optimization_score': float(optimization_score),
                    'is_reliable': metrics['is_reliable']
                }
        
        # 最適保有期間を特定
        optimal_holding = 0
        
        if holding_returns:
            # B3: 信頼性のある期間のみから選択
            reliable_periods = {k: v for k, v in holding_returns.items() if v.get('is_reliable', False)}
            
            if reliable_periods:
                # 信頼性のある期間から最適なものを選択
                optimal_holding = max(
                    reliable_periods.items(),
                    key=lambda x: x[1]['optimization_score']
                )[0]
                logger.info(f"信頼性のある期間から最適保有期間を選択: {optimal_holding}日")
            else:
                # 信頼性のある期間がない場合、すべての期間から選択
                optimal_holding = max(
                    holding_returns.items(),
                    key=lambda x: x[1]['optimization_score']
                )[0]
                logger.warning(f"信頼性のある期間がないため、すべての期間から最適保有期間を選択: {optimal_holding}日")
        
        return {
            'optimal_holding': optimal_holding,
            'holding_returns': holding_returns
        }
    
    except Exception as e:
        logger.error(f"保有期間分析全体エラー: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            'optimal_holding': 0,
            'holding_returns': {},
            'error': str(e)
        }

# B3: 追加のヘルパー関数
def calculate_win_rate_confidence_interval(win_rate, n, confidence=0.95):
    """勝率の信頼区間を計算する（Wilson score interval）"""
    if n <= 0:
        return (0, 0)
    
    try:
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2/n
        
        if denominator == 0:  # ゼロ除算防止
            return (0, 0)
            
        center = (win_rate + z**2/(2*n)) / denominator
        
        # 平方根の中身が負にならないように保護
        radicand = win_rate * (1 - win_rate) / n + z**2 / (4 * n**2)
        if radicand < 0:
            radicand = 0
        
        half_width = z * np.sqrt(radicand) / denominator
        
        lower = max(0, center - half_width)
        upper = min(1, center + half_width)
        
        return (lower, upper)
    except Exception as e:
        logger.warning(f"勝率信頼区間計算エラー: {str(e)}")
        return (0, 0)

def calculate_confidence_level(sample_size):
    """サンプルサイズに基づいて調整された信頼水準を計算する"""
    # 小サンプルの場合は信頼水準を下げる
    if sample_size < 5:
        return 0.5  # 非常に小さいサンプルは信頼度50%
    elif sample_size < 10:
        return 0.7  # 小さいサンプルは信頼度70%
    elif sample_size < 20:
        return 0.8  # 中程度のサンプルは信頼度80%
    elif sample_size < 30:
        return 0.9  # やや大きいサンプルは信頼度90%
    else:
        return 0.95  # 十分なサンプルは信頼度95%

def calculate_cohens_d(returns):
    """Cohen's d効果量を計算する"""
    if len(returns) <= 1:
        return 0
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    
    if std == 0:
        return 0
        
    # 0との比較（リターンが0より大きいかどうか）
    d = mean / std
    
    return d

def calculate_power(sample_size, effect_size, alpha=0.05):
    """検出力を計算する"""
    if sample_size <= 1 or effect_size == 0:
        return 0
    
    # 非心t分布を使用して検出力を計算
    df = sample_size - 1
    nc = effect_size * np.sqrt(sample_size)
    t_crit = stats.t.ppf(1 - alpha, df)
    
    # 検出力 = 1 - β = P(t > t_crit | H1)
    try:
        power = 1 - stats.nct.cdf(t_crit, df, nc)
        return power
    except:
        # 計算エラー時は近似値を返す
        if effect_size > 0.8:
            return 0.9  # 大きい効果量
        elif effect_size > 0.5:
            return 0.7  # 中程度の効果量
        elif effect_size > 0.2:
            return 0.4  # 小さい効果量
        else:
            return 0.1  # とても小さい効果量
