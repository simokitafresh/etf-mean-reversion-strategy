# src/signals/__init__.py
"""シグナル生成モジュール"""
from .indicators import calculate_all_indicators
from .rules import generate_signals
import pandas as pd
import yfinance as yf
import time
import os
from ..data.cache import DataCache

# キャッシュのインスタンス化
cache = DataCache()

def calculate_signals_for_etf(
    symbol: str,
    bb_window: int = 20,
    bb_std: float = 2.0,
    stoch_k: int = 14,
    stoch_d: int = 3,
    ema_period: int = 200,
    ema_slope_period: int = 20,
    period: str = "5y",
    min_samples: int = 30
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
        
    Returns:
        pd.DataFrame: シグナルを含むデータフレーム
    """
    # キャッシュから取得を試みる
    cache_key = f"signals_{symbol}_{bb_window}_{bb_std}_{stoch_k}_{stoch_d}_{ema_period}_{period}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print(f"キャッシュから{symbol}のシグナルを取得しました")
        return pd.read_json(cached_data, orient='split')
    
    try:
        print(f"{symbol}の価格データを取得中...")
        # データ取得
        data = yf.download(symbol, period=period, progress=False)
        
        if len(data) == 0:
            print(f"警告: {symbol}のデータが取得できません")
            return pd.DataFrame()
        
        print(f"{symbol}のテクニカル指標を計算中...")
        # テクニカル指標の計算
        data_with_indicators = calculate_all_indicators(
            data,
            bb_window=bb_window,
            bb_std=bb_std,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            ema_period=ema_period,
            ema_slope_period=ema_slope_period
        )
        
        print(f"{symbol}のシグナルを生成中...")
        # シグナル生成
        result = generate_signals(
            data_with_indicators,
            min_samples=min_samples
        )
        
        # シグナル概要の表示
        buy_signals = result['Buy_Signal'].sum()
        sell_signals = result['Sell_Signal'].sum()
        
        print(f"{symbol}のシグナル計算完了 - 買い: {buy_signals}件, 売り: {sell_signals}件")
        
        # キャッシュに保存
        cache.set_json(cache_key, result.to_json(orient='split'))
        
        return result
    
    except Exception as e:
        print(f"エラー - {symbol}のシグナル計算: {str(e)}")
        return pd.DataFrame()

def calculate_signals_for_universe(
    universe: list,
    parameter_sets: list,
    period: str = "5y",
    min_samples: int = 30
) -> dict:
    """ETFユニバース全体のシグナルを計算する
    
    Args:
        universe: ETFユニバースのリスト
        parameter_sets: パラメータセットのリスト
        period: データ取得期間
        min_samples: 有効なシグナルとみなす最小のサンプル数
        
    Returns:
        dict: シンボルとパラメータセットごとのシグナル結果を格納した辞書
    """
    # 結果ディレクトリの作成
    os.makedirs("data/results/signals", exist_ok=True)
    
    # キャッシュから取得を試みる
    cache_key = f"universe_signals_{len(universe)}_{len(parameter_sets)}_{period}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print(f"キャッシュからユニバース全体のシグナルを取得しました")
        # 結果は辞書として返される
        return cached_data
    
    print(f"{len(universe)}銘柄 × {len(parameter_sets)}パラメータセットのシグナルを計算します...")
    
    all_signals = {}
    
    for etf in universe:
        symbol = etf['symbol']
        print(f"\n{symbol} ({etf.get('name', '')})のシグナル計算中...")
        
        symbol_signals = {}
        
        for params in parameter_sets:
            param_key = (
                f"BB{params['bb_window']}-{params['bb_std']}_"
                f"Stoch{params['stoch_k']}-{params['stoch_d']}_"
                f"EMA{params['ema_period']}"
            )
            
            print(f"  パラメータセット: {param_key}")
            
            # シグナル計算
            signals = calculate_signals_for_etf(
                symbol=symbol,
                bb_window=params['bb_window'],
                bb_std=params['bb_std'],
                stoch_k=params['stoch_k'],
                stoch_d=params['stoch_d'],
                ema_period=params['ema_period'],
                ema_slope_period=params['ema_slope_period'],
                period=period,
                min_samples=min_samples
            )
            
            # 結果が空でなければ保存
            if not signals.empty:
                # CSVとして保存
                csv_path = f"data/results/signals/{symbol}_{param_key}.csv"
                signals.to_csv(csv_path)
                
                # メモリ効率のため、実際のデータフレームではなく統計値だけを保存
                stats = {
                    'buy_signals': int(signals['Buy_Signal'].sum()),
                    'sell_signals': int(signals['Sell_Signal'].sum()),
                    'data_length': len(signals),
                    'avg_buy_strength': float(signals.loc[signals['Buy_Signal'], 'Buy_Signal_Strength'].mean()) if signals['Buy_Signal'].any() else 0,
                    'avg_sell_strength': float(signals.loc[signals['Sell_Signal'], 'Sell_Signal_Strength'].mean()) if signals['Sell_Signal'].any() else 0,
                    'csv_path': csv_path
                }
                
                symbol_signals[param_key] = stats
            
            # API制限対策
            time.sleep(0.5)
        
        all_signals[symbol] = symbol_signals
    
    # 結果をJSON形式で保存
    import json
    with open("data/results/all_signals.json", 'w') as f:
        json.dump(all_signals, f, indent=2)
    
    print(f"すべてのシグナル計算が完了しました。結果を保存しました: data/results/all_signals.json")
    
    # キャッシュに保存
    cache.set_json(cache_key, all_signals)
    
    return all_signals
