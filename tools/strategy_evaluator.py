# tools/strategy_evaluator.py
"""戦略評価ツール - 包括的な戦略分析とレポート生成"""
import os
import sys
import json
import pandas as pd
import numpy as np
import time
import argparse
from datetime import datetime
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# プロジェクトモジュールのインポート
from src.data.fetch import get_base_etf_list
from src.universe import select_universe
from src.parameters.grid_search import generate_parameter_grid
from src.signals import calculate_signals_for_universe
from src.parameters import evaluate_parameter_stability
from src.validation import run_statistical_validation
from src.utils.plotting import create_summary_dashboard

def evaluate_strategy(
    etfs_json_path: str = "data/results/final_etf_universe.json",
    params_json_path: str = "data/results/parameters/stability_results.json",
    signals_dir: str = "data/results/signals",
    output_dir: str = "data/results/evaluation",
    max_etfs: int = 5,
    max_params: int = 3
):
    """戦略を総合的に評価し、レポートを生成する
    
    Args:
        etfs_json_path: ETFユニバースのJSONパス
        params_json_path: パラメータ安定性結果のJSONパス
        signals_dir: シグナルデータのディレクトリ
        output_dir: 出力ディレクトリ
        max_etfs: 評価するETFの最大数
        max_params: 評価するパラメータの最大数
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    dashboard_dir = os.path.join(output_dir, "dashboards")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    print("="*80)
    print(f"戦略評価ツール - 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 評価レポートファイル
    report_path = os.path.join(output_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    with open(report_path, 'w') as report:
        report.write(f"# ETF版「統計的トレンド逆張り」戦略評価レポート\n\n")
        report.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. ETFユニバースの読み込み
        print("\n1. ETFユニバースの読み込み")
        report.write("## 1. ETFユニバース\n\n")
        
        try:
            with open(etfs_json_path, 'r') as f:
                universe = json.load(f)
                
            print(f"  ETFユニバース: {len(universe)}銘柄")
            report.write(f"合計: {len(universe)}銘柄\n\n")
            
            # 評価用に上位ETFを選択
            selected_etfs = universe[:max_etfs]
            
            report.write("| シンボル | 名称 | 平均出来高 | AUM | スプレッド | クラスタ |\n")
            report.write("|---------|------|-----------|-----|------------|----------|\n")
            
            for etf in selected_etfs:
                report.write(
                    f"| {etf.get('symbol', 'N/A')} | "
                    f"{etf.get('name', 'N/A')} | "
                    f"{etf.get('avg_volume', 0):,.0f} | "
                    f"${etf.get('aum', 0)/1e9:.1f}B | "
                    f"{etf.get('estimated_spread', 0):.4f} | "
                    f"{etf.get('cluster', 'N/A')} |\n"
                )
            
            report.write(f"\n選択ETF: {[etf['symbol'] for etf in selected_etfs]}\n\n")
            
        except Exception as e:
            print(f"  エラー: ETFユニバースの読み込みに失敗しました - {str(e)}")
            report.write(f"ETFユニバースの読み込みに失敗: {str(e)}\n\n")
            selected_etfs = []
        
        # 2. 安定パラメータの読み込み
        print("\n2. 安定パラメータの読み込み")
        report.write("## 2. 安定パラメータ\n\n")
        
        try:
            with open(params_json_path, 'r') as f:
                stability_results = json.load(f)
                
            buy_stable_params = stability_results['buy']['stable_params']
            sell_stable_params = stability_results['sell']['stable_params']
            
            print(f"  買いシグナルの安定パラメータ: {len(buy_stable_params)}セット")
            print(f"  売りシグナルの安定パラメータ: {len(sell_stable_params)}セット")
            
            report.write("### 買いシグナルの安定パラメータ\n\n")
            report.write(f"合計: {len(buy_stable_params)}セット\n\n")
            
            if buy_stable_params:
                report.write("| パラメータキー | BB期間 | BB幅 | Stoch K | Stoch D | 保有期間 | 性能スコア |\n")
                report.write("|--------------|--------|------|---------|---------|--------|-----------|\n")
                
                for i, param in enumerate(buy_stable_params[:max_params]):
                    report.write(
                        f"| {param.get('param_key', 'N/A')} | "
                        f"{param.get('bb_window', 'N/A')} | "
                        f"{param.get('bb_std', 'N/A')} | "
                        f"{param.get('stoch_k', 'N/A')} | "
                        f"{param.get('stoch_d', 'N/A')} | "
                        f"{param.get('holding', 'N/A')} | "
                        f"{param.get('performance', 0):.4f} |\n"
                    )
            
            report.write("\n### 売りシグナルの安定パラメータ\n\n")
            report.write(f"合計: {len(sell_stable_params)}セット\n\n")
            
            if sell_stable_params:
                report.write("| パラメータキー | BB期間 | BB幅 | Stoch K | Stoch D | 保有期間 | 性能スコア |\n")
                report.write("|--------------|--------|------|---------|---------|--------|-----------|\n")
                
                for i, param in enumerate(sell_stable_params[:max_params]):
                    report.write(
                        f"| {param.get('param_key', 'N/A')} | "
                        f"{param.get('bb_window', 'N/A')} | "
                        f"{param.get('bb_std', 'N/A')} | "
                        f"{param.get('stoch_k', 'N/A')} | "
                        f"{param.get('stoch_d', 'N/A')} | "
                        f"{param.get('holding', 'N/A')} | "
                        f"{param.get('performance', 0):.4f} |\n"
                    )
            
            # 評価用に上位パラメータを選択
            selected_buy_params = [param['param_key'] for param in buy_stable_params[:max_params]] if buy_stable_params else []
            selected_sell_params = [param['param_key'] for param in sell_stable_params[:max_params]] if sell_stable_params else []
            
            report.write(f"\n選択買いパラメータ: {selected_buy_params}\n\n")
            report.write(f"\n選択売りパラメータ: {selected_sell_params}\n\n")
            
        except Exception as e:
            print(f"  エラー: 安定パラメータの読み込みに失敗しました - {str(e)}")
            report.write(f"安定パラメータの読み込みに失敗: {str(e)}\n\n")
            selected_buy_params = []
            selected_sell_params = []
        
        # 3. 統計的検証
        print("\n3. 統計的検証")
        report.write("## 3. 統計的検証\n\n")
        
        validation_results = []
        dashboards = []
        
        # 各ETFと安定パラメータの組み合わせを検証
        for etf in selected_etfs:
            symbol = etf['symbol']
            
            # 買いパラメータの検証
            for param_key in selected_buy_params:
                signal_data_path = os.path.join(signals_dir, f"{symbol}_{param_key}.csv")
                
                if os.path.exists(signal_data_path):
                    print(f"  {symbol}の{param_key}（買い）を検証中...")
                    
                    try:
                        # シグナルデータの読み込み
                        signal_data = pd.read_csv(signal_data_path, index_col=0, parse_dates=True)
                        
                        # 統計的検証を実行
                        validation_result = run_statistical_validation(
                            symbol=symbol,
                            param_key=param_key,
                            signal_data_path=signal_data_path
                        )
                        
                        if validation_result and 'signals' in validation_result and 'Buy_Signal' in validation_result['signals']:
                            buy_result = validation_result['signals']['Buy_Signal']
                            
                            # ダッシュボードの生成
                            dashboard_path = create_summary_dashboard(
                                data=signal_data,
                                validation_results=buy_result,
                                etf_info=etf,
                                output_dir=dashboard_dir,
                                symbol=f"{symbol}_buy_{param_key}"
                            )
                            
                            dashboards.append({
                                'symbol': symbol,
                                'param_key': param_key,
                                'signal_type': 'Buy_Signal',
                                'dashboard_path': dashboard_path
                            })
                            
                            validation_results.append({
                                'symbol': symbol,
                                'param_key': param_key,
                                'signal_type': 'Buy_Signal',
                                'cpcv_win_rate': buy_result.get('cpcv', {}).get('overall', {}).get('win_rate', 0),
                                'cpcv_pf': buy_result.get('cpcv', {}).get('overall', {}).get('profit_factor', 0),
                                'wf_win_rate': buy_result.get('walk_forward', {}).get('overall', {}).get('win_rate', 0),
                                'wf_return': buy_result.get('walk_forward', {}).get('overall', {}).get('return', 0),
                                'is_significant': buy_result.get('statistical_significance', {}).get('is_significant', False),
                                'sample_count': buy_result.get('sample_count', 0)
                            })
                    except Exception as e:
                        print(f"  エラー: {symbol}の{param_key}（買い）の検証に失敗しました - {str(e)}")
                else:
                    print(f"  警告: {signal_data_path}が見つかりません")
            
            # 売りパラメータの検証
            for param_key in selected_sell_params:
                signal_data_path = os.path.join(signals_dir, f"{symbol}_{param_key}.csv")
                
                if os.path.exists(signal_data_path):
                    print(f"  {symbol}の{param_key}（売り）を検証中...")
                    
                    try:
                        # シグナルデータの読み込み
                        signal_data = pd.read_csv(signal_data_path, index_col=0, parse_dates=True)
                        
                        # 統計的検証を実行
                        validation_result = run_statistical_validation(
                            symbol=symbol,
                            param_key=param_key,
                            signal_data_path=signal_data_path
                        )
                        
                        if validation_result and 'signals' in validation_result and 'Sell_Signal' in validation_result['signals']:
                            sell_result = validation_result['signals']['Sell_Signal']
                            
                            # ダッシュボードの生成
                            dashboard_path = create_summary_dashboard(
                                data=signal_data,
                                validation_results=sell_result,
                                etf_info=etf,
                                output_dir=dashboard_dir,
                                symbol=f"{symbol}_sell_{param_key}"
                            )
                            
                            dashboards.append({
                                'symbol': symbol,
                                'param_key': param_key,
                                'signal_type': 'Sell_Signal',
                                'dashboard_path': dashboard_path
                            })
                            
                            validation_results.append({
                                'symbol': symbol,
                                'param_key': param_key,
                                'signal_type': 'Sell_Signal',
                                'cpcv_win_rate': sell_result.get('cpcv', {}).get('overall', {}).get('win_rate', 0),
                                'cpcv_pf': sell_result.get('cpcv', {}).get('overall', {}).get('profit_factor', 0),
                                'wf_win_rate': sell_result.get('walk_forward', {}).get('overall', {}).get('win_rate', 0),
                                'wf_return': sell_result.get('walk_forward', {}).get('overall', {}).get('return', 0),
                                'is_significant': sell_result.get('statistical_significance', {}).get('is_significant', False),
                                'sample_count': sell_result.get('sample_count', 0)
                            })
                    except Exception as e:
                        print(f"  エラー: {symbol}の{param_key}（売り）の検証に失敗しました - {str(e)}")
                else:
                    print(f"  警告: {signal_data_path}が見つかりません")
        
        # 検証結果のレポート
        if validation_results:
            report.write("### 検証結果サマリー\n\n")
            report.write("| シンボル | パラメータ | シグナル | CPCV勝率 | CPCV PF | WF勝率 | WFリターン | 有意性 | サンプル数 |\n")
            report.write("|---------|------------|---------|----------|---------|---------|------------|--------|------------|\n")
            
            for result in sorted(validation_results, key=lambda x: x.get('wf_return', 0), reverse=True):
                report.write(
                    f"| {result['symbol']} | "
                    f"{result['param_key']} | "
                    f"{result['signal_type']} | "
                    f"{result['cpcv_win_rate']:.1%} | "
                    f"{result['cpcv_pf']:.2f} | "
                    f"{result['wf_win_rate']:.1%} | "
                    f"{result['wf_return']:.1%} | "
                    f"{'Yes' if result['is_significant'] else 'No'} | "
                    f"{result['sample_count']} |\n"
                )
            
            # 理論的エッジの判定
            edge_confirmed = any(
                result['cpcv_win_rate'] > 0.55 and 
                result['wf_win_rate'] > 0.55 and 
                result['wf_return'] > 0.05 and 
                result['is_significant'] and
                result['sample_count'] >= 30
                for result in validation_results
            )
            
            report.write("\n### 理論的エッジの判定\n\n")
            
            if edge_confirmed:
                report.write("✅ **理論的エッジが確認されました**\n\n")
                report.write("以下の基準を満たすシグナルが見つかりました：\n")
                report.write("- CPCV勝率 > 55%\n")
                report.write("- Walk-Forward勝率 > 55%\n")
                report.write("- Walk-Forwardリターン > 5%\n")
                report.write("- 統計的有意性あり\n")
                report.write("- サンプル数 ≥ 30\n\n")
                report.write("フェーズ2（実用性検証）に進むことが推奨されます。\n\n")
            else:
                report.write("❌ **理論的エッジが確認できませんでした**\n\n")
                report.write("すべての基準を満たすシグナルが見つかりませんでした。戦略の見直しが必要です。\n\n")
            
            # ダッシュボードへのリンク
            report.write("### ダッシュボード\n\n")
            
            for dashboard in dashboards:
                dashboard_filename = os.path.basename(dashboard['dashboard_path'])
                report.write(f"- [{dashboard['symbol']} {dashboard['signal_type']} ({dashboard['param_key']})](dashboards/{dashboard_filename})\n")
        else:
            report.write("検証結果がありません。\n\n")
        
        # 4. 結論と次のステップ
        report.write("\n## 4. 結論と次のステップ\n\n")
        
        if validation_results:
            if edge_confirmed:
                report.write("### 推奨される次のステップ\n\n")
                report.write("1. フェーズ2（実用性検証）の実装\n")
                report.write("   - コスト構造の組み込み\n")
                report.write("   - リアルな執行モデルの実装\n")
                report.write("   - シグナルフィルタの追加\n")
                report.write("   - ポジションサイジング戦略の開発\n\n")
                report.write("2. 最も有望なシグナルを特定\n")
                report.write("   - パフォーマンスと安定性のバランスを考慮\n")
                report.write("   - 異なる市場環境での一貫性を確認\n\n")
                report.write("3. 小規模な実運用テスト\n")
                report.write("   - リスク制限付きで実際のトレードを開始\n")
                report.write("   - パフォーマンスを緊密にモニタリング\n")
            else:
                report.write("### 推奨される次のステップ\n\n")
                report.write("1. 戦略の見直し\n")
                report.write("   - シグナル生成ロジックの再検討\n")
                report.write("   - より広いパラメータ範囲の検証\n")
                report.write("   - 異なるETFユニバースの検討\n\n")
                report.write("2. 代替アプローチの検討\n")
                report.write("   - 他のテクニカル指標の組み合わせ\n")
                report.write("   - 機械学習アプローチの検討\n")
                report.write("   - 異なる時間枠の分析\n")
        else:
            report.write("検証結果がないため、次のステップを提案できません。戦略の実装から見直してください。\n")
    
    print("\n評価レポートが生成されました:")
    print(f"  {report_path}")
    
    if dashboards:
        print(f"\nダッシュボードが生成されました:")
        for dashboard in dashboards:
            print(f"  {dashboard['dashboard_path']}")
    
    print("\n"+"="*80)
    print(f"戦略評価完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return report_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF版「統計的トレンド逆張り」戦略評価ツール")
    parser.add_argument("--etfs", type=str, default="data/results/final_etf_universe.json", help="ETFユニバースのJSONパス")
    parser.add_argument("--params", type=str, default="data/results/parameters/stability_results.json", help="パラメータ安定性結果のJSONパス")
    parser.add_argument("--signals", type=str, default="data/results/signals", help="シグナルデータのディレクトリ")
    parser.add_argument("--output", type=str, default="data/results/evaluation", help="出力ディレクトリ")
    parser.add_argument("--max-etfs", type=int, default=5, help="評価するETFの最大数")
    parser.add_argument("--max-params", type=int, default=3, help="評価するパラメータの最大数")
    
    args = parser.parse_args()
    
    evaluate_strategy(
        etfs_json_path=args.etfs,
        params_json_path=args.params,
        signals_dir=args.signals,
        output_dir=args.output,
        max_etfs=args.max_etfs,
        max_params=args.max_params
    )
