"""流動性に基づくETFスクリーニング"""
import yfinance as yf
import time
import pandas as pd
import numpy as np
import os
import warnings
import logging
from typing import List, Dict, Any, Optional, Union
from src.data.cache_manager import CacheManager

# ロガーの設定
logger = logging.getLogger(__name__)

# キャッシュマネージャーのシングルトンインスタンスを取得
cache_manager = CacheManager.get_instance()

def screen_liquidity(etf_base_list: List[Dict[str, Any]], 
                     min_volume: int = 100000,
                     min_aum: int = 1000000000,  # 10億USD
                     max_spread: float = 0.001,  # 0.10%
                     min_age_years: float = 1.0) -> List[Dict[str, Any]]:
    """流動性条件に基づいてETFをスクリーニングする
    
    Args:
        etf_base_list: 基本ETFリスト
        min_volume: 最小平均出来高
        min_aum: 最小AUM（運用資産総額、USD）
        max_spread: 最大スプレッド
        min_age_years: 最小運用年数
        
    Returns:
        List[Dict[str, Any]]: 流動性条件を満たすETFのリスト
    """
    # 入力チェック
    if not etf_base_list:
        logger.warning("流動性スクリーニングの入力が空です")
        return []
    
    # キャッシュから取得を試みる
    cache_key = f"liquidity_screened_etfs_{len(etf_base_list)}_{min_volume}_{min_aum}"
    cached_data = cache_manager.get_json(cache_key)
    if cached_data:
        logger.info("キャッシュから流動性スクリーニング結果を取得しました")
        return cached_data
    
    # 結果ディレクトリの確認
    os.makedirs("data/results", exist_ok=True)
    
    qualified_etfs = []
    error_etfs = []
    
    logger.info(f"{len(etf_base_list)}銘柄の流動性スクリーニングを開始します...")
    
    # バッチ処理でAPI制限に対応
    batch_size = 5
    for i in range(0, len(etf_base_list), batch_size):
        batch = etf_base_list[i:i+batch_size]
        symbols = [etf['symbol'] for etf in batch]
        
        try:
            logger.info(f"バッチ取得中: {symbols}")
            
            # 複数銘柄のデータを一度に取得
            data = yf.download(
                symbols, 
                period="3mo", 
                group_by="ticker", 
                progress=False,
                threads=True  # 並列ダウンロードを使用
            )
            
            for etf in batch:
                symbol = etf['symbol']
                
                try:
                    # 該当銘柄のデータを抽出
                    etf_data = None
                    
                    if len(symbols) > 1:
                        if symbol in data:
                            etf_data = data[symbol]
                        else:
                            logger.info(f"データがありません: {symbol}")
                            error_etfs.append({'symbol': symbol, 'error': 'データなし'})
                            continue
                    else:
                        etf_data = data
                    
                    # データが空かチェック
                    if etf_data.empty:
                        logger.info(f"空のデータ: {symbol}")
                        error_etfs.append({'symbol': symbol, 'error': '空のデータ'})
                        continue
                    
                    # 必要なカラムが存在するか確認
                    required_columns = ['Volume', 'High', 'Low', 'Close']
                    missing_columns = [col for col in required_columns if col not in etf_data.columns]
                    
                    if missing_columns:
                        logger.info(f"カラム不足 ({symbol}): {missing_columns}")
                        error_etfs.append({'symbol': symbol, 'error': f'カラム不足: {missing_columns}'})
                        continue
                    
                    # データのクリーニング - 欠損値を前方補完してから後方補完
                    etf_data = etf_data.fillna(method='ffill').fillna(method='bfill')
                    
                    # NaN値がまだある場合は処理をスキップ
                    if etf_data[required_columns].isna().any().any():
                        logger.info(f"欠損値が残っています: {symbol}")
                        error_etfs.append({'symbol': symbol, 'error': '欠損値が残っている'})
                        continue
                    
                    # 平均出来高の計算
                    avg_volume = etf_data['Volume'].mean()
                    
                    # AUMの取得（直接取得が難しいため個別にAPIを使用）
                    try:
                        ticker_obj = yf.Ticker(symbol)
                        ticker_info = ticker_obj.info or {}
                        
                        # AUMの取得（キー名が変わっている可能性に対応）
                        aum = ticker_info.get('totalAssets') or ticker_info.get('fundAssets') or 0
                        
                        # AUMが文字列の場合は数値に変換
                        if isinstance(aum, str):
                            try:
                                # 単位を処理（例: "1.2B" → 1200000000）
                                aum = aum.upper()
                                if 'B' in aum:
                                    aum = float(aum.replace('B', '')) * 1e9
                                elif 'M' in aum:
                                    aum = float(aum.replace('M', '')) * 1e6
                                elif 'K' in aum:
                                    aum = float(aum.replace('K', '')) * 1e3
                                else:
                                    aum = float(aum)
                            except:
                                aum = 0
                    except Exception as e:
                        logger.info(f"AUM取得エラー ({symbol}): {str(e)}")
                        aum = 0
                    
                    # Bid-Askスプレッドの推定（直接取得できない場合）
                    # 日中のボラティリティをプロキシとして使用
                    volatility = (etf_data['High'] - etf_data['Low']).mean() / etf_data['Close'].mean()
                    estimated_spread = volatility * 0.1  # 簡易推定
                    
                    # データ長から運用年数を推定
                    history_length = len(etf_data)
                    age_in_years = history_length / 252 if history_length > 0 else 0
                    
                    # 条件判定
                    if (avg_volume >= min_volume and 
                        aum >= min_aum and
                        estimated_spread <= max_spread and
                        age_in_years >= min_age_years):
                        
                        # 合格したETF情報を更新して保存
                        qualified_etf = etf.copy()  # 元のデータを変更しない
                        qualified_etf.update({
                            'avg_volume': float(avg_volume),
                            'aum': float(aum),
                            'estimated_spread': float(estimated_spread),
                            'age_in_years': float(age_in_years)
                        })
                        
                        qualified_etfs.append(qualified_etf)
                        
                        logger.info(f"適格: {symbol} (出来高: {avg_volume:.0f}, AUM: ${aum/1e9:.1f}B)")
                    else:
                        # 条件を満たさない理由をログ
                        reasons = []
                        if avg_volume < min_volume:
                            reasons.append(f"出来高不足 ({avg_volume:.0f} < {min_volume})")
                        if aum < min_aum:
                            reasons.append(f"AUM不足 (${aum/1e9:.1f}B < ${min_aum/1e9:.1f}B)")
                        if estimated_spread > max_spread:
                            reasons.append(f"スプレッド超過 ({estimated_spread:.2%} > {max_spread:.2%})")
                        if age_in_years < min_age_years:
                            reasons.append(f"運用年数不足 ({age_in_years:.1f}年 < {min_age_years}年)")
                        
                        logger.info(f"不適格: {symbol} - {', '.join(reasons)}")
                
                except Exception as e:
                    logger.error(f"ETF処理エラー ({symbol}): {str(e)}")
                    error_etfs.append({'symbol': symbol, 'error': str(e)})
            
            # API制限対策
            time.sleep(1)
        
        except Exception as e:
            logger.error(f"バッチダウンロードエラー {symbols}: {str(e)}")
            for symbol in symbols:
                error_etfs.append({'symbol': symbol, 'error': f'バッチダウンロードエラー: {str(e)}'})
            
            # エラー時は長めに待機
            time.sleep(5)
    
    # エラーの集計
    if error_etfs:
        logger.info(f"\n処理中にエラーが発生した銘柄: {len(error_etfs)}/{len(etf_base_list)}")
        
        # エラーログをCSVとして保存
        error_df = pd.DataFrame(error_etfs)
        error_log_path = "data/results/liquidity_screening_errors.csv"
        error_df.to_csv(error_log_path, index=False)
        logger.info(f"エラーログを保存しました: {error_log_path}")
    
    # 結果の集計
    logger.info(f"\n流動性スクリーニング結果: {len(qualified_etfs)}/{len(etf_base_list)} 銘柄が適格")
    
    # 結果をCSVとして保存
    if qualified_etfs:
        qualified_df = pd.DataFrame(qualified_etfs)
        qualified_path = "data/results/qualified_etfs.csv"
        qualified_df.to_csv(qualified_path, index=False)
        logger.info(f"適格ETFリストを保存しました: {qualified_path}")
    
    # キャッシュに保存
    cache_manager.set_json(cache_key, qualified_etfs)
    
    return qualified_etfs
