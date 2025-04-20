# tests/integration_test.py
"""統合テスト - ETF版「統計的トレンド逆張り」戦略フレームワーク"""
import os
import sys
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# モジュールのインポート
from src.data.fetch import get_base_etf_list
from src.universe import select_universe
from src.parameters.grid_search import generate_parameter_grid
from src.signals import calculate_signals_for_universe
from src.parameters import evaluate_parameter_stability
from src.validation import run_statistical_validation

def run_integration_test(
    sample_size: int = 3,
    parameter_count: int = 5,
    use_cached_data: bool = True
):
    """統合テストを実行する
    
    Args:
        sample_size: テスト用ETFサンプル数
        parameter_count: テスト用パラメータ数
        use_cached_data: キャッシュデータを使用するかどうか
    """
    print("="*80)
    print("ETF版「統計的トレンド逆張り」戦略フレームワーク - 統合テスト")
    print("="*80)
    
    start_time = time.time()
    
    # 結果ディレクトリの作成
    os.makedirs("data/test_results", exist_ok=True)
    
    # テスト結果のログファイル
    log_file = f"data/test_results/integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log(message):
        """ログを出力する"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
    
    # テスト開始
    log(f"統合テスト開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # ステップ1: ETFユニバース選定
        log("\nステップ1: ETFユニバース選定")
        
        if use_cached_data and os.path.exists("data/results/final_etf_universe.json"):
            with open("data/results/final_etf_universe.json", 'r') as f:
                universe = json.load(f)
            log(f"  キャッシュからETFユニバースを読み込みました: {len(universe)}銘柄")
            
            # テスト用にサンプルサイズを制限
            universe = universe[:sample_size]
            log(f"  テスト用にサンプルサイズを制限: {len(universe)}銘柄")
        else:
            log("  ETFユニバースを新規に選定します")
            base_etfs = get_base_etf_list()
            log(f"  基礎候補リスト: {len(base_etfs)}銘柄")
            
            universe = select_universe(base_etfs)
            log(f"  選定されたETFユニバース: {len(universe)}銘柄")
            
            # テスト用にサンプルサイズを制限
            universe = universe[:sample_size]
            log(f"  テスト用にサンプルサイズを制限: {len(universe)}銘柄")
        
        # ステップ2: パラメータグリッド生成
        log("\nステップ2: パラメータグリッド生成")
        
        # テスト用に小さなパラメータグリッドを生成
        param_grid = generate_parameter_grid(
            bb_windows=[20],
            bb_stds=[2.0],
            stoch_ks=[14],
            stoch_ds=[3],
            ema_periods=[200],
            ema_slope_periods=[20],
            holding_periods=[5]
        )
        
        # さらにテスト用にサンプルサイズを制限
        param_grid = param_grid[:parameter_count]
        log(f"  テスト用パラメータグリッド: {len(param_grid)}セット")
        
        # ステップ3: シグナル生成
        log("\nステップ3: シグナル生成")
        signals_data = calculate_signals_for_universe(
            universe=universe,
            parameter_sets=param_grid,
            period="1y",  # テスト用に短い期間
            min_samples=10  # テスト用に少ないサンプル数
        )
        
        log(f"  シグナル生成完了: {len(signals_data)}銘柄")
        
        # ステップ4: パラメータ安定性評価
        log("\nステップ4: パラメータ安定性評価")
        stability_results = evaluate_parameter_stability(universe)
        
        # 安定パラメータの取得
        buy_stable_params = stability_results.get('buy', {}).get('stable_params', [])
        sell_stable_params = stability_results.get('sell', {}).get('stable_params', [])
        
        log(f"  買いシグナルの安定パラメータ: {len(buy_stable_params)}セット")
        log(f"  売りシグナルの安定パラメータ: {len(sell_stable_params)}セット")
        
        # ステップ5: 統計的検証
        log("\nステップ5: 統計的検証")
        
        validation_results = []
        
        # 各ETFと安定パラメータの組み合わせを検証
        for symbol in [etf['symbol'] for etf in universe]:
            # パラメータを選択（安定パラメータまたはテストパラメータ）
            test_params = []
            
            if buy_stable_params:
                test_params.extend([param['param_key'] for param in buy_stable_params[:1]])
            elif param_grid:
                test_params.append(param_grid[0]['param_key'])
            
            for param_key in test_params:
                # シグナルデータのパスを構築
                signal_data_path = f"data/results/signals/{symbol}_{param_key}.csv"
                
                if os.path.exists(signal_data_path):
                    log(f"  {symbol}の{param_key}を検証中...")
                    
                    # 統計的検証を実行
                    validation_result = run_statistical_validation(
                        symbol=symbol,
                        param_key=param_key,
                        signal_data_path=signal_data_path
                    )
                    
                    if validation_result:
                        validation_results.append(validation_result)
                else:
                    log(f"  警告: {signal_data_path}が見つかりません")
        
        log(f"  統計的検証完了: {len(validation_results)}検証")
        
        # テスト結果のサマリー
        log("\n統合テスト結果サマリー:")
        log(f"  テスト対象ETF: {len(universe)}銘柄")
        log(f"  テスト対象パラメータ: {len(param_grid)}セット")
        log(f"  シグナル生成: {'成功' if signals_data else '失敗'}")
        log(f"  パラメータ安定性評価: {'成功' if stability_results else '失敗'}")
        log(f"  統計的検証: {'成功' if validation_results else '失敗'}")
        
        # 成功
        log("\n統合テスト成功!")
        
    except Exception as e:
        log(f"\nエラー: {str(e)}")
        import traceback
        log(traceback.format_exc())
        log("\n統合テスト失敗!")
    
    # 完了時間
    end_time = time.time()
    execution_time = end_time - start_time
    log(f"\n処理時間: {execution_time:.2f}秒 ({execution_time/60:.2f}分)")
    log(f"統合テスト完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return log_file

if __name__ == "__main__":
    log_file = run_integration_test(
        sample_size=2,  # テスト用に少ない銘柄数
        parameter_count=2,  # テスト用に少ないパラメータ数
        use_cached_data=True  # キャッシュデータを使用
    )
    
    print(f"\nテストログファイル: {log_file}")
