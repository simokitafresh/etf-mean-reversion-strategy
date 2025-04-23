# src/signals/_implementation.py

"""シグナル計算の実装モジュール（内部使用）"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple

# 内部モジュールの参照にはパッケージの相対インポートを使用
from .indicators import calculate_all_indicators
from .rules import generate_signals

# 別パッケージのインポートには絶対インポートを使用
from src.data.cache_manager import CacheManager

# キャッシュマネージャーの取得（シングルトンアクセス）
cache_manager = CacheManager.get_instance()

def calculate_signals_for_etf(
    symbol: str,
    bb_window: int = 20,
    bb_std: float = 2.0,
    stoch_k: int = 14,
    stoch_d: int = 3,
    ema_period: int = 200,
    ema_slope_period: int = 20,
    period: str = "5y",
    min_samples: int = 30,
    price_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """指定されたETFのシグナルを計算する
    
    Args:
        symbol: ETFのティッカーシンボル
        bb_window: ボリンジャーバンドの期間
        bb_std: ボリンジャーバンドの標準偏差倍率
        stoch_k: ストキャスティクスの%K期間
        stoch_d: ストキャスティクスの%D期間
        ema_period: EMAの期間
        ema_slope_period: EMA傾きの計算期間
        period: データ取得期間
        min_samples: 有効なシグナルとみなす最小のサンプル数
        price_data: 既存の価格データ（Noneの場合はダウンロード）
        
    Returns:
        pd.DataFrame: シグナルを含むデータフレーム
    """
    # キャッシュキーの生成
    param_key = f"BB{bb_window}-{bb_std}_Stoch{stoch_k}-{stoch_d}_EMA{ema_period}_Hold{0}"
    cache_key = f"signals_{symbol}_{param_key}"
    
    # キャッシュから取得を試みる
    cached_data = cache_manager.get_json(cache_key)
    if cached_data is not None:
        print(f"{symbol}のシグナルデータをキャッシュから取得しました")
        return pd.read_json(cached_data, orient='split')
    
    # 価格データの取得
    if price_data is None:
        print(f"{symbol}の価格データを取得しています...")
        try:
            price_data = yf.download(symbol, period=period, progress=False)
            if price_data.empty:
                print(f"警告: {symbol}の価格データが取得できませんでした")
                return pd.DataFrame()
        except Exception as e:
            print(f"エラー: {symbol}の価格データ取得に失敗しました - {str(e)}")
            return pd.DataFrame()
    
    # 必須カラムの存在確認
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in price_data.columns]
    
    if missing_columns:
        print(f"警告: {symbol}の価格データに必要なカラム {missing_columns} が含まれていません")
        return pd.DataFrame()
    
    # インデックスがDatetimeIndexでない場合は変換
    if not isinstance(price_data.index, pd.DatetimeIndex):
        try:
            price_data.index = pd.to_datetime(price_data.index)
        except:
            print(f"警告: {symbol}のインデックスをDatetimeIndexに変換できませんでした")
            return pd.DataFrame()
    
    # テクニカル指標の計算
    print(f"{symbol}のテクニカル指標を計算しています...")
    indicators_data = calculate_all_indicators(
        price_data,
        bb_window=bb_window,
        bb_std=bb_std,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        ema_period=ema_period,
        ema_slope_period=ema_slope_period
    )
    
    # シグナル生成
    print(f"{symbol}のシグナルを生成しています...")
    signals_data = generate_signals(indicators_data, min_samples=min_samples)
    
    # シンボル情報を追加
    signals_data['Symbol'] = symbol
    
    # 結果ディレクトリの確認
    os.makedirs("data/results/signals", exist_ok=True)
    
    # CSVとして保存
    csv_path = f"data/results/signals/{symbol}_{param_key}.csv"
    signals_data.to_csv(csv_path)
    print(f"{symbol}のシグナルデータを保存しました: {csv_path}")
    
    # キャッシュに保存
    cache_manager.set_json(cache_key, signals_data.to_json(orient='split'))
    
    return signals_data

def calculate_signals_for_universe(
    universe: List[Dict[str, Any]],
    parameter_sets: List[Dict[str, Any]],
    period: str = "5y",
    min_samples: int = 30
) -> Dict[str, Dict[str, Any]]:
    """ETFユニバース全体のシグナルを計算する
    
    Args:
        universe: ETFユニバースのリスト
        parameter_sets: パラメータセットのリスト
        period: データ取得期間
        min_samples: 有効なシグナルとみなす最小のサンプル数
        
    Returns:
        Dict: シンボルとパラメータセットごとのシグナル結果
    """
    # 結果ディレクトリの確認
    os.makedirs("data/results/signals", exist_ok=True)
    
    # ETFユニバースのシンボルを取得
    symbols = [etf['symbol'] for etf in universe]
    results = {}
    
    print(f"{len(symbols)}銘柄 × {len(parameter_sets)}パラメータセットの計算を開始...")
    
    # 価格データを一度に取得し、再利用
    price_data = {}
    for symbol in symbols:
        try:
            data = yf.download(symbol, period=period, progress=False)
            if not data.empty:
                price_data[symbol] = data
            time.sleep(0.5)  # APIレート制限対策
        except Exception as e:
            print(f"警告: {symbol}の価格データ取得エラー - {str(e)}")
    
    print(f"価格データを取得しました: {len(price_data)}/{len(symbols)}銘柄")
    
    # ETFごとにパラメータセットを適用
    for symbol in symbols:
        if symbol not in price_data:
            print(f"スキップ: {symbol}の価格データがありません")
            continue
            
        print(f"\n{symbol}のシグナル計算を開始...")
        symbol_results = {}
        
        for params in parameter_sets:
            param_key = params['param_key']
            print(f"  パラメータセット: {param_key}")
            
            # キャッシュキーの生成
            cache_key = f"signals_{symbol}_{param_key}"
            
            # キャッシュから取得を試みる
            cached_result = cache_manager.get_json(cache_key)
            if cached_result is not None:
                df = pd.read_json(cached_result, orient='split')
                
                # パラメータ情報の抽出
                csv_path = f"data/results/signals/{symbol}_{param_key}.csv"
                signal_stats = {
                    'param_key': param_key,
                    'buy_signals': int(df['Buy_Signal'].sum()),
                    'sell_signals': int(df['Sell_Signal'].sum()),
                    'date_range': f"{df.index[0]} to {df.index[-1]}",
                    'data_points': len(df),
                    'csv_path': csv_path
                }
                
                symbol_results[param_key] = signal_stats
                print(f"    キャッシュから読み込み: 買い={signal_stats['buy_signals']}, 売り={signal_stats['sell_signals']}")
                continue
            
            # シグナル計算
            signals_df = calculate_signals_for_etf(
                symbol=symbol,
                bb_window=params['bb_window'],
                bb_std=params['bb_std'],
                stoch_k=params['stoch_k'],
                stoch_d=params['stoch_d'],
                ema_period=params['ema_period'],
                ema_slope_period=params.get('ema_slope_period', 20),
                period=period,
                min_samples=min_samples,
                price_data=price_data[symbol]
            )
            
            if signals_df.empty:
                print(f"    エラー: シグナル計算に失敗しました")
                continue
            
            # CSVとして保存
            csv_path = f"data/results/signals/{symbol}_{param_key}.csv"
            signals_df.to_csv(csv_path)
            
            # シグナル統計情報
            signal_stats = {
                'param_key': param_key,
                'buy_signals': int(signals_df['Buy_Signal'].sum()),
                'sell_signals': int(signals_df['Sell_Signal'].sum()),
                'date_range': f"{signals_df.index[0]} to {signals_df.index[-1]}",
                'data_points': len(signals_df),
                'csv_path': csv_path
            }
            
            symbol_results[param_key] = signal_stats
            print(f"    計算完了: 買い={signal_stats['buy_signals']}, 売り={signal_stats['sell_signals']}")
        
        if symbol_results:
            results[symbol] = symbol_results
    
    print(f"\n計算完了: {len(results)}/{len(symbols)}銘柄")
    return results
