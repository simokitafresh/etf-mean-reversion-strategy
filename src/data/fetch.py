"""データ取得のためのユーティリティ"""
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import logging
from typing import List, Dict, Any, Optional, Union
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
from .cache import DataCache

# ロガーの設定
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# キャッシュのインスタンス化
cache = DataCache()

# リトライ設定
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒

def get_base_etf_list() -> List[Dict[str, Any]]:
    """基本的なETFリストを取得する
    
    Returns:
        List[Dict[str, Any]]: ETF情報のリスト
        
    Raises:
        ValueError: ETFリストの取得に失敗した場合
    """
    # キャッシュから取得を試みる
    cached_data = cache.get_json("base_etf_list")
    if cached_data:
        logger.info("キャッシュからETFリストを取得しました")
        return cached_data
    
    logger.info("ETFリストを取得しています...")
    
    # 主要なETFティッカーのリスト（実際にはもっと多くなります）
    base_etfs = [
        "SPY", "IVV", "VOO",  # S&P 500
        "QQQ", "VGT", "XLK",  # テクノロジー
        "DIA", "IWM", "VTI",  # ダウ、ラッセル2000、トータルマーケット
        "XLF", "VFH", "KBE",  # 金融
        "XLE", "VDE", "OIH",  # エネルギー
        "XLV", "VHT", "IBB",  # ヘルスケア
        "XLY", "VCR", "XRT",  # 一般消費財
        "XLP", "VDC", "PBJ",  # 生活必需品
        "XLI", "VIS", "IYT",  # 工業
        "XLU", "VPU", "IDU",  # 公共事業
        "XLB", "VAW", "GDX",  # 素材
        "XLRE", "VNQ", "IYR", # 不動産
        "GLD", "SLV", "IAU",  # 貴金属
        "TLT", "IEF", "SHY",  # 債券
        "VEA", "EFA", "VWO",  # 先進国・新興国
        "VXX", "UVXY"         # ボラティリティ
    ]
    
    etf_info = []
    failed_symbols = []
    
    # 10銘柄ずつバッチ処理
    for i in range(0, len(base_etfs), 10):
        batch = base_etfs[i:i+10]
        logger.info(f"バッチ処理中: {batch}")
        
        try:
            for symbol in batch:
                retry_count = 0
                success = False
                
                while retry_count < MAX_RETRIES and not success:
                    try:
                        etf = yf.Ticker(symbol)
                        info = etf.info
                        
                        # 空のデータチェック
                        if not info or len(info) <= 1:
                            raise ValueError(f"空またはデータが不足: {symbol}")
                        
                        etf_data = {
                            'symbol': symbol,
                            'name': info.get('shortName', info.get('longName', symbol)),
                            'category': info.get('category', ''),
                            'description': info.get('description', '')[:100] + '...' if info.get('description') else ''
                        }
                        
                        # 追加情報の取得
                        etf_data['exchange'] = info.get('exchange', '')
                        etf_data['quoteType'] = info.get('quoteType', '')
                        
                        etf_info.append(etf_data)
                        logger.info(f"取得成功: {symbol}")
                        success = True
                    
                    except HTTPError as e:
                        retry_count += 1
                        if retry_count < MAX_RETRIES:
                            logger.warning(f"HTTP エラー ({symbol}): {str(e)}. リトライ {retry_count}/{MAX_RETRIES}...")
                            time.sleep(RETRY_DELAY * retry_count)  # 指数バックオフ
                        else:
                            logger.error(f"HTTP エラー ({symbol}): {str(e)}. リトライ回数超過.")
                            failed_symbols.append({"symbol": symbol, "error": f"HTTP エラー: {str(e)}"})
                    
                    except ConnectionError as e:
                        retry_count += 1
                        if retry_count < MAX_RETRIES:
                            logger.warning(f"接続エラー ({symbol}): {str(e)}. リトライ {retry_count}/{MAX_RETRIES}...")
                            time.sleep(RETRY_DELAY * retry_count)
                        else:
                            logger.error(f"接続エラー ({symbol}): {str(e)}. リトライ回数超過.")
                            failed_symbols.append({"symbol": symbol, "error": f"接続エラー: {str(e)}"})
                    
                    except Timeout as e:
                        retry_count += 1
                        if retry_count < MAX_RETRIES:
                            logger.warning(f"タイムアウト ({symbol}): {str(e)}. リトライ {retry_count}/{MAX_RETRIES}...")
                            time.sleep(RETRY_DELAY * retry_count)
                        else:
                            logger.error(f"タイムアウト ({symbol}): {str(e)}. リトライ回数超過.")
                            failed_symbols.append({"symbol": symbol, "error": f"タイムアウト: {str(e)}"})
                    
                    except ValueError as e:
                        # データ不足の場合はリトライ
                        retry_count += 1
                        if retry_count < MAX_RETRIES:
                            logger.warning(f"データ不足 ({symbol}): {str(e)}. リトライ {retry_count}/{MAX_RETRIES}...")
                            time.sleep(RETRY_DELAY * retry_count)
                        else:
                            logger.error(f"データ不足 ({symbol}): {str(e)}. リトライ回数超過.")
                            failed_symbols.append({"symbol": symbol, "error": f"データ不足: {str(e)}"})
                    
                    except Exception as e:
                        retry_count += 1
                        if retry_count < MAX_RETRIES:
                            logger.warning(f"未知のエラー ({symbol}): {str(e)}. リトライ {retry_count}/{MAX_RETRIES}...")
                            time.sleep(RETRY_DELAY * retry_count)
                        else:
                            logger.error(f"未知のエラー ({symbol}): {str(e)}. リトライ回数超過.")
                            failed_symbols.append({"symbol": symbol, "error": f"未知のエラー: {str(e)}"})
            
            # API制限を避けるため少し待機
            time.sleep(1)
        
        except Exception as e:
            logger.error(f"バッチ処理エラー: {str(e)}")
            # バッチ全体が失敗した場合、長めの待機後に次のバッチへ
            time.sleep(5)
    
    # 失敗した銘柄の情報をログに記録
    if failed_symbols:
        logger.warning(f"{len(failed_symbols)}/{len(base_etfs)} 銘柄の取得に失敗しました")
        # 失敗情報をファイルに記録（オプション）
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        failed_df = pd.DataFrame(failed_symbols)
        failed_df.to_csv(f"{log_dir}/failed_etfs.csv", index=False)
        logger.info(f"失敗情報を {log_dir}/failed_etfs.csv に保存しました")
    
    # 結果が空の場合はエラーを発生させる
    if not etf_info:
        raise ValueError("ETFリストの取得に完全に失敗しました")
    
    # キャッシュに保存
    cache.set_json("base_etf_list", etf_info)
    
    logger.info(f"ETFリストを取得しました: {len(etf_info)}/{len(base_etfs)} 銘柄")
    return etf_info

def get_etf_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    retry_count: int = MAX_RETRIES
) -> Optional[pd.DataFrame]:
    """ETFの価格データを取得する
    
    Args:
        symbol: ETFのティッカーシンボル
        period: データ期間 (例: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        interval: データ間隔 (例: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
        retry_count: リトライ回数
        
    Returns:
        pd.DataFrame: 価格データ、取得失敗時はNone
    """
    # キャッシュキーの生成
    cache_key = f"etf_data_{symbol}_{period}_{interval}"
    
    # キャッシュから取得を試みる
    cached_data = cache.get_json(cache_key)
    if cached_data:
        logger.info(f"{symbol} のデータをキャッシュから取得しました")
        return pd.read_json(cached_data, orient='split')
    
    logger.info(f"{symbol} のデータを取得しています (期間: {period}, 間隔: {interval})...")
    
    for attempt in range(retry_count):
        try:
            # データ取得
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                show_errors=False
            )
            
            # データチェック
            if data.empty:
                logger.warning(f"{symbol} のデータが空です")
                if attempt < retry_count - 1:
                    delay = RETRY_DELAY * (attempt + 1)
                    logger.info(f"リトライ中... ({attempt + 1}/{retry_count}) {delay}秒後")
                    time.sleep(delay)
                    continue
                return None
            
            # キャッシュに保存
            cache.set_json(cache_key, data.to_json(orient='split'))
            
            logger.info(f"{symbol} のデータを取得しました: {len(data)}行")
            return data
        
        except HTTPError as e:
            if "429" in str(e):  # Too Many Requests
                delay = RETRY_DELAY * (attempt + 1) * 2  # API制限の場合は長めに待機
                logger.warning(f"API制限エラー ({symbol}): {str(e)}. {delay}秒後にリトライします...")
                time.sleep(delay)
            else:
                logger.warning(f"HTTP エラー ({symbol}): {str(e)}")
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        except (ConnectionError, Timeout) as e:
            logger.warning(f"ネットワークエラー ({symbol}): {str(e)}")
            time.sleep(RETRY_DELAY * (attempt + 1))
        
        except Exception as e:
            logger.warning(f"データ取得エラー ({symbol}): {str(e)}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    logger.error(f"{symbol} のデータ取得に失敗しました")
    return None

def get_multiple_etf_data(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1d",
    batch_size: int = 5
) -> Dict[str, pd.DataFrame]:
    """複数のETFデータを一度に取得する
    
    Args:
        symbols: ETFのティッカーシンボルのリスト
        period: データ期間
        interval: データ間隔
        batch_size: バッチサイズ
        
    Returns:
        Dict[str, pd.DataFrame]: シンボルごとのデータフレーム
    """
    logger.info(f"{len(symbols)} 銘柄のデータを取得します...")
    
    result = {}
    failed_symbols = []
    
    # バッチに分割して処理
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        logger.info(f"バッチ処理中: {batch}")
        
        try:
            # バッチでダウンロード
            if len(batch) > 1:
                data = yf.download(
                    batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    show_errors=False
                )
                
                # TickerからDataFrameへの変換
                for symbol in batch:
                    if symbol in data:
                        symbol_data = data[symbol]
                        if not symbol_data.empty:
                            result[symbol] = symbol_data
                        else:
                            logger.warning(f"{symbol} のデータが空です")
                            failed_symbols.append({"symbol": symbol, "error": "空のデータ"})
                    else:
                        # 個別に再取得を試みる
                        logger.warning(f"{symbol} のデータがバッチ結果に含まれていません。個別に取得を試みます...")
                        single_data = get_etf_data(symbol, period, interval)
                        if single_data is not None and not single_data.empty:
                            result[symbol] = single_data
                        else:
                            failed_symbols.append({"symbol": symbol, "error": "データ取得失敗"})
            else:
                # 単一銘柄の場合
                symbol = batch[0]
                single_data = get_etf_data(symbol, period, interval)
                if single_data is not None and not single_data.empty:
                    result[symbol] = single_data
                else:
                    failed_symbols.append({"symbol": symbol, "error": "データ取得失敗"})
            
            # API制限対策
            time.sleep(1)
        
        except Exception as e:
            logger.error(f"バッチ処理エラー {batch}: {str(e)}")
            for symbol in batch:
                failed_symbols.append({"symbol": symbol, "error": f"バッチエラー: {str(e)}"})
            time.sleep(5)  # エラー時は長めに待機
    
    # 結果の確認
    logger.info(f"{len(result)}/{len(symbols)} 銘柄のデータを取得しました")
    
    if failed_symbols:
        logger.warning(f"{len(failed_symbols)} 銘柄の取得に失敗しました")
        # 失敗情報をファイルに記録（オプション）
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        failed_df = pd.DataFrame(failed_symbols)
        failed_df.to_csv(f"{log_dir}/failed_data_fetch.csv", index=False)
        logger.info(f"失敗情報を {log_dir}/failed_data_fetch.csv に保存しました")
    
    return result
