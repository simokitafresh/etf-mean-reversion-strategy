# src/validation/__init__.py
"""統計的検証モジュール"""
from .cpcv import run_cpcv_analysis, visualize_cpcv_results
from .walk_forward import run_walk_forward_analysis, visualize_walk_forward_results
from .metrics import calculate_basic_metrics, calculate_statistical_significance, calculate_market_regime_performance, analyze_holding_period
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Any
import yfinance as yf
from ..utils.param_utils import parse_param_key  # 共通ユーティリティからインポート

def run_statistical_validation(
    symbol: str,
    param_key: str,
    signal_data_path: str,
    output_dir: str = "data/results/validation",
    market_symbol: str = "SPY"
) -> Dict[str, Any]:
    """シグナルの統計的検証を実行する統合関数
    
    Args:
        symbol: ETFのシンボル
        param_key: パラメータキー
        signal_data_path: シグナルデータのファイルパス
        output_dir: 出力ディレクトリ
        market_symbol: 市場指標のシンボル
        
    Returns:
        Dict: 統計的検証結果
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{symbol}のシグナルに対する統計的検証を開始します...")
    
    # シグナルデータの読み込み
    try:
        data = pd.read_csv(signal_data_path, index_col=0, parse_dates=True)
        print(f"シグナルデータを読み込みました: {signal_data_path}")
    except Exception as e:
        print(f"エラー: シグナルデータの読み込みに失敗しました - {str(e)}")
        return {}
    
    # パラメータ情報の抽出
    try:
        bb_window, bb_std, stoch_k, stoch_d, ema_period, holding_period = parse_param_key(param_key)
    except:
        print(f"エラー: パラメータキーの解析に失敗しました - {param_key}")
        return {}
    
    # 市場データの取得
    try:
        market_data = yf.download(market_symbol, start=data.index[0], end=data.index[-1], progress=False)
        market_returns = market_data['Close'].pct_change().dropna()
    except Exception as e:
        print(f"警告: 市場データの取得に失敗しました - {str(e)}")
        market_returns = pd.Series(index=data.index)
    
    # 買いシグナルと売りシグナルの統計的検証
    results = {}
    
    for signal_type in ['Buy_Signal', 'Sell_Signal']:
        print(f"\n{signal_type}の検証中...")
        
        # シグナル発生日の取得
        signal_days = data[data[signal_type]].index
        
        if len(signal_days) < 10:
            print(f"  警告: {signal_type}のサンプル数({len(signal_days)})が不足しています")
            continue
        
        # シグナル検証結果
        signal_results = {
            'symbol': symbol,
            'param_key': param_key,
            'signal_type': signal_type,
            'holding_period': holding_period,
            'sample_count': len(signal_days)
        }
        
        # 1. CPCV分析
        print("  CPCV分析を実行中...")
        cpcv_results = run_cpcv_analysis(
            symbol=symbol,
            data=data,
            signal_column=signal_type,
            holding_period=holding_period
        )
        
        signal_results['cpcv'] = cpcv_results
        
        # CPCVの可視化
        cpcv_viz = visualize_cpcv_results(
            cpcv_results,
            output_dir=f"{output_dir}/cpcv"
        )
        
        signal_results['cpcv_visualizations'] = cpcv_viz
        
        # 2. Walk-Forward分析
        print("  Walk-Forward分析を実行中...")
        wf_results = run_walk_forward_analysis(
            symbol=symbol,
            data=data,
            signal_column=signal_type,
            holding_period=holding_period
        )
        
        signal_results['walk_forward'] = wf_results
        
        # Walk-Forwardの可視化
        wf_viz = visualize_walk_forward_results(
            wf_results,
            output_dir=f"{output_dir}/walk_forward"
        )
        
        signal_results['wf_visualizations'] = wf_viz
        
        # 3. トレードパフォーマンスの計算
        trade_returns = []
        trade_dates = []
        
        for day in signal_days:
            # 現在の価格
            try:
                current_price = data.loc[day, 'Close']
                
                # 保有期間後の価格
                future_idx = data.index.get_loc(day) + holding_period
                if future_idx >= len(data):
                    continue
                    
                future_date = data.index[future_idx]
                future_price = data.loc[future_date, 'Close']
                
                # リターンの計算
                if signal_type == 'Buy_Signal':
                    ret = (future_price / current_price) - 1
                else:  # 'Sell_Signal'
                    ret = (current_price / future_price) - 1
                
                trade_returns.append(ret)
                trade_dates.append(day)
            
            except Exception as e:
                continue
        
        if trade_returns:
            # 基本メトリクスを計算
            trade_metrics = calculate_basic_metrics(np.array(trade_returns))
            
            # 統計的有意性を評価
            significance = calculate_statistical_significance(np.array(trade_returns))
            
            # 市場レジーム別パフォーマンス
            regime_performance = calculate_market_regime_performance(
                np.array(trade_returns),
                market_returns.loc[trade_dates].values if not market_returns.empty else np.array([]),
                trade_dates
            )
            
            signal_results['trade_metrics'] = trade_metrics
            signal_results['statistical_significance'] = significance
            signal_results['regime_performance'] = regime_performance
        
        # 4. 最適保有期間の分析
        holding_analysis = analyze_holding_period(
            data,
            signal_days,
            signal_type,
            max_holding_days=10
        )
        
        signal_results['holding_analysis'] = holding_analysis
        
        # 結果を保存
        results[signal_type] = signal_results
    
    # 結果を統合
    validation_results = {
        'symbol': symbol,
        'param_key': param_key,
        'signals': results,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 結果をJSONとして保存
    result_path = f"{output_dir}/{symbol}_{param_key}_validation.json"
    with open(result_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n統計的検証結果を保存しました: {result_path}")
    
    return validation_results
