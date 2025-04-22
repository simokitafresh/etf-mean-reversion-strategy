"""データ取得のためのユーティリティ"""
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import logging
import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException, ReadTimeout, TooManyRedirects
from urllib3.exceptions import MaxRetryError, NewConnectionError
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
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # 秒
MAX_RETRY_DELAY = 120  # 最大バックオフ：2分
RATE_LIMIT_HEADERS = ['Retry-After', 'X-RateLimit-Reset', 'X-Rate-Limit-Remaining']

# APiレート制限追跡
rate_limit_info = {
    'last_limit_hit': None,
    'cool_off_until': None,
    'remaining_calls': None
}

# ユーザーエージェントのリスト - APIブロックの防止に役立つ
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]

def get_random_user_agent() -> str:
    """ランダムなユーザーエージェントを取得"""
    return random.choice(USER_AGENTS)

def check_rate_limit_status() -> bool:
    """APIレート制限状態をチェックする
    
    Returns:
        bool: リクエストを続行可能かどうか
    """
    now = datetime.now()
    
    # クールオフ期間中なら待機
    if rate_limit_info['cool_off_until'] and now < rate_limit_info['cool_off_until']:
        wait_time = (rate_limit_info['cool_off_until'] - now).total_seconds()
        logger.warning(f"APIレート制限のクールオフ期間中です。{wait_time:.1f}秒後に再試行します。")
        time.sleep(wait_time + 1)  # 余裕を持って待機
    
    # 残りの呼び出し回数が少ない場合は遅延を入れる
    if rate_limit_info['remaining_calls'] is not None and rate_limit_info['remaining_calls'] < 10:
        delay = 5  # APIレート制限に近づいている場合は遅延を入れる
        logger.info(f"APIレート制限に近づいています（残り{rate_limit_info['remaining_calls']}回）。{delay}秒間遅延します。")
        time.sleep(delay)
    
    return True

def handle_rate_limit_response(response: requests.Response) -> int:
    """レート制限レスポンスを処理し、待機すべき時間を返す
    
    Args:
        response: APIレスポンス
        
    Returns:
        int: 待機すべき秒数
    """
    now = datetime.now()
    rate_limit_info['last_limit_hit'] = now
    
    # レスポンスヘッダーからRetry-Afterを抽出
    retry_after = None
    for header in RATE_LIMIT_HEADERS:
        if header in response.headers:
            try:
                retry_after = int(response.headers[header])
                break
            except (ValueError, TypeError):
                pass
    
    # デフォルトの待機時間と実際の待機時間
    default_wait = 60  # デフォルトの待機時間（1分）
    wait_time = retry_after if retry_after else default_wait
    
    # 過去のレート制限ヒットに基づいて待機時間を調整
    if rate_limit_info['last_limit_hit']:
        time_since_last_limit = (now - rate_limit_info['last_limit_hit']).total_seconds()
        if time_since_last_limit < 300:  # 5分以内に再度ヒットした場合
            wait_time = wait_time * 2  # 待機時間を2倍に
    
    # 最大待機時間を設定
    wait_time = min(wait_time, 3600)  # 最大1時間まで
    
    # クールオフ期間を設定
    rate_limit_info['cool_off_until'] = now + timedelta(seconds=wait_time)
    
    logger.warning(f"APIレート制限に達しました。{wait_time}秒間待機します。")
    return wait_time

def exponential_backoff_delay(attempt: int) -> float:
    """指数バックオフによる遅延時間を計算する
    
    Args:
        attempt: 試行回数
        
    Returns:
        float: 待機すべき秒数
    """
    delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** attempt))
    # ジッターを追加して同時リトライによる負荷を分散
    jitter = random.uniform(0, 0.1 * delay)
    return delay + jitter

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
    recovery_mode = False
    recovery_limit = len(base_etfs) // 3  # 3分の1以上失敗したら回復モードに
    
    # リトライカウンターを設定（全体の最大リトライ回数）
    global_retry_count = 0
    max_global_retries = 3
    
    while global_retry_count < max_global_retries:
        try:
            # レート制限状態をチェック
            check_rate_limit_status()
            
            # 10銘柄ずつバッチ処理
            batch_size = 5 if not recovery_mode else 3  # 回復モードでは小さいバッチサイズ
            for i in range(0, len(base_etfs), batch_size):
                batch_symbols = base_etfs[i:i+batch_size]
                batch_symbols = [s for s in batch_symbols if s not in [etf.get('symbol') for etf in etf_info]]
                
                if not batch_symbols:
                    continue  # すでに取得済みの銘柄はスキップ
                
                logger.info(f"バッチ取得中: {batch_symbols}")
                
                try:
                    for symbol in batch_symbols:
                        retry_count = 0
                        success = False
                        last_error = None
                        
                        while retry_count < MAX_RETRIES and not success:
                            try:
                                # レート制限状態をチェック
                                check_rate_limit_status()
                                
                                # yfinance設定を調整してAPIエラーを回避
                                session = requests.Session()
                                session.headers.update({'User-Agent': get_random_user_agent()})
                                
                                etf = yf.Ticker(symbol, session=session)
                                # タイムアウト設定を追加
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
                                
                                # 流動性指標を取得
                                etf_data['avg_volume'] = info.get('averageVolume', 0)
                                etf_data['aum'] = info.get('totalAssets', 0)
                                
                                # 主要財務指標を追加
                                etf_data['expense_ratio'] = info.get('annualReportExpenseRatio', None)
                                etf_data['beta'] = info.get('beta', None)
                                etf_data['asset_class'] = info.get('assetClass', '')
                                
                                etf_info.append(etf_data)
                                logger.info(f"取得成功: {symbol}")
                                success = True
                            
                            # 各例外タイプに固有の処理を追加
                            except HTTPError as e:
                                # HTTPエラー（404, 403, 500など）
                                last_error = e
                                retry_count += 1
                                status_code = getattr(e.response, 'status_code', 0)
                                
                                if status_code == 429:  # レート制限
                                    wait_time = handle_rate_limit_response(e.response)
                                    logger.warning(f"APIレート制限に達しました ({symbol}): {str(e)}. {wait_time}秒間待機します。")
                                    time.sleep(wait_time)
                                elif status_code >= 500:  # サーバーエラー
                                    delay = exponential_backoff_delay(retry_count)
                                    logger.warning(f"サーバーエラー ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({retry_count}/{MAX_RETRIES}).")
                                    time.sleep(delay)
                                elif status_code == 404:  # 見つからない場合はスキップ
                                    logger.error(f"ETFが見つかりません ({symbol}): {str(e)}.")
                                    failed_symbols.append({"symbol": symbol, "error": f"404エラー: {str(e)}"})
                                    break  # リトライしない
                                else:  # その他のHTTPエラー
                                    delay = exponential_backoff_delay(retry_count)
                                    logger.warning(f"HTTP エラー ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({retry_count}/{MAX_RETRIES}).")
                                    time.sleep(delay)
                            
                            except ConnectionError as e:
                                # ネットワーク接続エラー
                                last_error = e
                                retry_count += 1
                                delay = exponential_backoff_delay(retry_count)
                                logger.warning(f"接続エラー ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({retry_count}/{MAX_RETRIES}).")
                                time.sleep(delay)
                            
                            except Timeout as e:
                                # リクエストタイムアウト
                                last_error = e
                                retry_count += 1
                                delay = exponential_backoff_delay(retry_count)
                                logger.warning(f"タイムアウト ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({retry_count}/{MAX_RETRIES}).")
                                time.sleep(delay)
                            
                            except ReadTimeout as e:
                                # 読み取りタイムアウト
                                last_error = e
                                retry_count += 1
                                delay = exponential_backoff_delay(retry_count)
                                logger.warning(f"読み取りタイムアウト ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({retry_count}/{MAX_RETRIES}).")
                                time.sleep(delay)
                            
                            except (MaxRetryError, NewConnectionError) as e:
                                # リトライ上限到達、または新規接続エラー
                                last_error = e
                                retry_count += 1
                                delay = exponential_backoff_delay(retry_count)
                                logger.warning(f"接続リトライエラー ({symbol}): {str(e)}. {delay:.1f}秒後に再試行します ({retry_count}/{MAX_RETRIES}).")
                                time.sleep(delay)
                            
                            except TooManyRedirects as e:
                                # リダイレクト上限到達
                                last_error = e
                                retry_count += 1
                                logger.error(f"リダイレクトエラー ({symbol}): {str(e)}.")
                                failed_symbols.append({"symbol": symbol, "error": f"リダイレクトエラー: {str(e)}"})
                                break  # この銘柄はスキップ
                            
                            except ValueError as e:
                                # データ不足などの値エラー
                                last_error = e
                                retry_count += 1
                                if retry_count < MAX_RETRIES:
                                    delay = exponential_backoff_delay(retry_count)
                                    logger.warning(f"データ不足 ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({retry_count}/{MAX_RETRIES}).")
                                    time.sleep(delay)
                                else:
                                    logger.error(f"データ不足 ({symbol}): {str(e)}. リトライ回数超過.")
                                    failed_symbols.append({"symbol": symbol, "error": f"データ不足: {str(e)}"})
                            
                            except json.JSONDecodeError as e:
                                # JSONデコードエラー（無効なレスポンス）
                                last_error = e
                                retry_count += 1
                                delay = exponential_backoff_delay(retry_count)
                                logger.warning(f"JSON解析エラー ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({retry_count}/{MAX_RETRIES}).")
                                time.sleep(delay)
                            
                            except Exception as e:
                                # その他の未知のエラー
                                last_error = e
                                retry_count += 1
                                delay = exponential_backoff_delay(retry_count)
                                logger.warning(f"未知のエラー ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({retry_count}/{MAX_RETRIES}).")
                                time.sleep(delay)
                        
                        # 最大リトライ後も失敗した場合
                        if not success:
                            logger.error(f"最大リトライ回数に達しました ({symbol}): {str(last_error)}")
                            failed_symbols.append({"symbol": symbol, "error": f"最大リトライ回数に達しました: {str(last_error)}"})
                    
                    # API制限を避けるため少し待機
                    time.sleep(2)  # バッチ間の待機時間
                
                except Exception as e:
                    logger.error(f"バッチ処理エラー: {str(e)}")
                    # バッチ全体が失敗した場合、長めの待機後に次のバッチへ
                    for symbol in batch_symbols:
                        if symbol not in [fs["symbol"] for fs in failed_symbols] and symbol not in [etf['symbol'] for etf in etf_info]:
                            failed_symbols.append({"symbol": symbol, "error": f"バッチエラー: {str(e)}"})
                    time.sleep(10)  # バッチエラー後の長い待機時間
            
            # 失敗した銘柄の処理
            if failed_symbols:
                failure_rate = len(failed_symbols) / len(base_etfs)
                if failure_rate > 0.33 and not recovery_mode:  # 33%以上の失敗で回復モードへ
                    recovery_mode = True
                    logger.warning(f"失敗率が高いため（{failure_rate:.1%}）、回復モードに切り替えます")
                    # もう一度同じ処理を実行するための準備
                    base_etfs = [fs["symbol"] for fs in failed_symbols]
                    failed_symbols = []
                    global_retry_count += 1
                    
                    # 回復モード用の長い遅延
                    delay = 30 + random.uniform(0, 30)  # 30-60秒のランダム遅延
                    logger.info(f"回復モードの前に{delay:.1f}秒間待機します")
                    time.sleep(delay)
                    continue  # while文の次のイテレーションへ
                else:
                    # 通常モードまたは回復モード失敗時はループを抜ける
                    break
            else:
                # 失敗がなければループを抜ける
                break
                
        except Exception as e:
            logger.error(f"ETFリスト取得の全体処理エラー: {str(e)}")
            global_retry_count += 1
            time.sleep(30)  # 大きな障害からの回復のための長い待機時間
    
    # 失敗した銘柄の情報をログに記録
    if failed_symbols:
        logger.warning(f"{len(failed_symbols)}/{len(base_etfs)} 銘柄の取得に失敗しました")
        # 失敗情報をファイルに記録
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_df = pd.DataFrame(failed_symbols)
        failed_df.to_csv(f"{log_dir}/failed_etfs_{timestamp}.csv", index=False)
        logger.info(f"失敗情報を {log_dir}/failed_etfs_{timestamp}.csv に保存しました")
    
    # 結果が空の場合はエラーを発生させる
    if not etf_info:
        error_msg = "ETFリストの取得に完全に失敗しました。ネットワーク接続と権限を確認してください。"
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif len(etf_info) < len(base_etfs) * 0.5:  # 50%未満しか取得できなかった場合
        logger.warning(f"ETFリストの取得が部分的に失敗しました（{len(etf_info)}/{len(base_etfs)}銘柄のみ取得）")
    
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
    
    # レート制限状態をチェック
    check_rate_limit_status()
    
    last_error = None
    for attempt in range(retry_count):
        try:
            # セッションを設定してユーザーエージェントをカスタマイズ
            session = requests.Session()
            session.headers.update({'User-Agent': get_random_user_agent()})
            
            # データ取得 - プログレスバーを無効化し、エラーを表示
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                show_errors=True,
                session=session,
                timeout=(10, 30)  # 接続タイムアウトと読み取りタイムアウトを設定
            )
            
            # データチェック
            if data.empty:
                logger.warning(f"{symbol} のデータが空です")
                if attempt < retry_count - 1:
                    delay = exponential_backoff_delay(attempt)
                    logger.info(f"リトライ中... ({attempt + 1}/{retry_count}) {delay:.1f}秒後")
                    time.sleep(delay)
                    continue
                return None
            
            # NaNをチェックし、多すぎる場合は警告
            nan_percentage = data.isna().mean().mean() * 100
            if nan_percentage > 20:  # NaNが20%を超える場合
                logger.warning(f"{symbol} のデータにNaNが多すぎます ({nan_percentage:.1f}%)。データの信頼性に問題がある可能性があります。")
            
            # キャッシュに保存
            cache.set_json(cache_key, data.to_json(orient='split'))
            
            logger.info(f"{symbol} のデータを取得しました: {len(data)}行")
            return data
        
        except HTTPError as e:
            last_error = e
            status_code = getattr(e.response, 'status_code', 0)
            
            if status_code == 429:  # レート制限
                wait_time = handle_rate_limit_response(e.response)
                logger.warning(f"APIレート制限に達しました ({symbol}): {str(e)}. {wait_time}秒間待機します。")
                time.sleep(wait_time)
            elif status_code >= 500:  # サーバーエラー
                delay = exponential_backoff_delay(attempt)
                logger.warning(f"サーバーエラー ({symbol}): {str(e)}. {delay:.1f}秒後にリトライします ({attempt + 1}/{retry_count}).")
                time.sleep(delay)
            elif status_code == 404:  # 見つからない
                logger.error(f"データが見つかりません ({symbol}): {str(e)}.")
                return None  # 404の場合はリトライしない
            else:
                logger.warning(f"HTTP エラー ({symbol}): {str(e)}")
                delay = exponential_backoff_delay(attempt)
                time.sleep(delay)
        
        except (ConnectionError, Timeout, ReadTimeout) as e:
            last_error = e
            logger.warning(f"ネットワークエラー ({symbol}): {str(e)}")
            delay = exponential_backoff_delay(attempt)
            time.sleep(delay)
        
        except (MaxRetryError, NewConnectionError) as e:
            last_error = e
            logger.warning(f"接続リトライエラー ({symbol}): {str(e)}.")
            delay = exponential_backoff_delay(attempt)
            time.sleep(delay)
        
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"JSON解析エラー ({symbol}): {str(e)}.")
            delay = exponential_backoff_delay(attempt)
            time.sleep(delay)
        
        except Exception as e:
            last_error = e
            logger.warning(f"データ取得エラー ({symbol}): {str(e)}")
            delay = exponential_backoff_delay(attempt)
            time.sleep(delay)
    
    # 全リトライ失敗後のエラーログ
    error_msg = f"{symbol} のデータ取得に失敗しました: {str(last_error)}"
    logger.error(error_msg)
    
    # エラーログをファイルに保存
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    error_log = {
        'symbol': symbol,
        'period': period,
        'interval': interval,
        'timestamp': timestamp,
        'error': str(last_error),
        'error_type': type(last_error).__name__
    }
    
    error_log_file = f"{log_dir}/data_fetch_error_{timestamp}.json"
    with open(error_log_file, 'w') as f:
        json.dump(error_log, f, indent=4)
    
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
    
    # 一時ダウンロード障害エリアを追跡
    failed_areas = []
    current_area = []
    success_status = {}  # どのシンボルが成功したかを追跡
    
    # 前回の試行でのエラー状態を追跡
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    result = {}
    failed_symbols = []
    
    # レート制限状態をチェック
    check_rate_limit_status()
    
    # 適応的なバッチサイズの初期設定
    adaptive_batch_size = batch_size
    min_batch_size = 1
    max_batch_size = 10
    
    # バッチに分割して処理
    for i in range(0, len(symbols), adaptive_batch_size):
        # 未処理シンボルだけをバッチ化
        unprocessed_symbols = [s for s in symbols[i:i+adaptive_batch_size] if s not in result]
        
        if not unprocessed_symbols:
            continue  # すべて処理済み
        
        current_area = unprocessed_symbols
        logger.info(f"バッチ処理中 ({adaptive_batch_size}銘柄): {current_area}")
        
        try:
            # バッチでダウンロード
            if len(unprocessed_symbols) > 1:
                # セッション設定
                session = requests.Session()
                session.headers.update({'User-Agent': get_random_user_agent()})
                
                data = yf.download(
                    unprocessed_symbols,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    show_errors=True,
                    session=session,
                    timeout=(10, 30)  # 接続タイムアウトと読み取りタイムアウト
                )
                
                # データ処理
                success_in_batch = False
                
                # TickerからDataFrameへの変換
                for symbol in unprocessed_symbols:
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        # 1つの銘柄だけだった場合は構造が異なる
                        if len(unprocessed_symbols) == 1 or ('Adj Close' in data.columns and not isinstance(data.columns, pd.MultiIndex)):
                            symbol_data = data
                            if not symbol_data.empty:
                                result[symbol] = symbol_data
                                success_status[symbol] = True
                                success_in_batch = True
                            else:
                                logger.warning(f"{symbol} のデータが空です")
                                failed_symbols.append({"symbol": symbol, "error": "空のデータ"})
                                success_status[symbol] = False
                            break  # 1銘柄の場合はループ不要
                        
                        # 複数銘柄の場合
                        elif symbol in data:
                            symbol_data = data[symbol]
                            if not symbol_data.empty and not symbol_data['Close'].isna().all():
                                result[symbol] = symbol_data
                                success_status[symbol] = True
                                success_in_batch = True
                            else:
                                logger.warning(f"{symbol} のデータが空です")
                                failed_symbols.append({"symbol": symbol, "error": "空のデータ"})
                                success_status[symbol] = False
                        else:
                            # 個別に再取得を試みる
                            logger.warning(f"{symbol} のデータがバッチ結果に含まれていません。個別に取得を試みます...")
                            individual_retry(symbol, period, interval, result, failed_symbols, success_status)
                    else:
                        # データが空またはNoneの場合、個別に再取得
                        logger.warning(f"バッチ全体が空です。各銘柄を個別に取得します。")
                        for s in unprocessed_symbols:
                            individual_retry(s, period, interval, result, failed_symbols, success_status)
                            time.sleep(1)  # 個別取得間の待機
                
                # バッチの結果に基づいて処理状態を更新
                if success_in_batch:
                    consecutive_errors = 0
                    
                    # 成功したので、バッチサイズを少し増やす（上限あり）
                    adaptive_batch_size = min(max_batch_size, adaptive_batch_size + 1)
                else:
                    consecutive_errors += 1
                    
                    # 失敗したので、バッチサイズを大幅に減らす
                    adaptive_batch_size = max(min_batch_size, adaptive_batch_size // 2)
                    
                    # 失敗したエリアを記録
                    failed_areas.append(current_area)
            else:
                # 単一銘柄の場合
                symbol = unprocessed_symbols[0]
                individual_retry(symbol, period, interval, result, failed_symbols, success_status)
            
            # 連続エラーが多すぎる場合はより長く待機
            if consecutive_errors >= max_consecutive_errors:
                logger.warning(f"連続エラーが多すぎます ({consecutive_errors})。長めに待機します...")
                time.sleep(30 + random.uniform(0, 30))  # 30-60秒のランダム待機
                consecutive_errors = 0  # リセット
            else:
                # API制限対策
                time.sleep(random.uniform(1, 3))  # 1-3秒のランダム待機
        
        except Exception as e:
            logger.error(f"バッチ処理エラー {current_area}: {str(e)}")
            consecutive_errors += 1
            
            # 失敗したエリアを記録
            failed_areas.append(current_area)
            
            # 各シンボルを失敗リストに追加（まだ処理されていなければ）
            for symbol in current_area:
                if symbol not in result and symbol not in [fs["symbol"] for fs in failed_symbols]:
                    failed_symbols.append({"symbol": symbol, "error": f"バッチエラー: {str(e)}"})
                    success_status[symbol] = False
            
            # バッチサイズを減らす
            adaptive_batch_size = max(min_batch_size, adaptive_batch_size // 2)
            
            # エラー時は長めに待機
            time.sleep(5 + random.uniform(0, 5))  # 5-10秒のランダム待機
    
    # 失敗エリアの再試行（特に重要な場合）
    if failed_areas and len(result) < len(symbols) * 0.7:  # 70%未満しか成功していない場合
        logger.info("失敗エリアを再試行します...")
        
        # 連続する失敗エリアのマージ（最大3つまで）
        merged_areas = []
        current_merged = []
        
        for area in failed_areas:
            if not current_merged:
                current_merged = area
            elif len(current_merged) + len(area) <= 3:  # 最大3銘柄までマージ
                current_merged.extend(area)
            else:
                merged_areas.append(current_merged)
                current_merged = area
        
        if current_merged:  # 最後のマージエリアを追加
            merged_areas.append(current_merged)
        
        # 再試行
        for area in merged_areas:
            for symbol in area:
                if symbol not in result:  # まだ取得できていない銘柄のみ
                    logger.info(f"失敗エリアから再試行: {symbol}")
                    individual_retry(symbol, period, interval, result, failed_symbols, success_status, retry_count=2)
                    time.sleep(random.uniform(1, 3))  # 1-3秒のランダム待機
    
    # 結果の確認
    logger.info(f"{len(result)}/{len(symbols)} 銘柄のデータを取得しました")
    
    if failed_symbols:
        logger.warning(f"{len(failed_symbols)} 銘柄の取得に失敗しました")
        # 失敗情報をファイルに記録
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_df = pd.DataFrame(failed_symbols)
        failed_df.to_csv(f"{log_dir}/failed_data_fetch_{timestamp}.csv", index=False)
        logger.info(f"失敗情報を {log_dir}/failed_data_fetch_{timestamp}.csv に保存しました")
    
    return result

def individual_retry(
    symbol: str, 
    period: str, 
    interval: str, 
    result: Dict[str, pd.DataFrame], 
    failed_symbols: List[Dict[str, str]], 
    success_status: Dict[str, bool],
    retry_count: int = 1
) -> None:
    """個別銘柄のデータ取得を再試行する
    
    Args:
        symbol: ETFシンボル
        period: データ期間
        interval: データ間隔
        result: 結果ディクショナリ（更新される）
        failed_symbols: 失敗シンボルリスト（更新される）
        success_status: 成功状態ディクショナリ（更新される）
        retry_count: リトライ回数
    """
    # 既に成功している場合はスキップ
    if symbol in result:
        return
    
    logger.info(f"個別取得: {symbol}")
    single_data = get_etf_data(symbol, period, interval, retry_count=retry_count)
    
    if single_data is not None and not single_data.empty:
        result[symbol] = single_data
        success_status[symbol] = True
    else:
        if symbol not in [fs["symbol"] for fs in failed_symbols]:
            failed_symbols.append({"symbol": symbol, "error": "データ取得失敗"})
        success_status[symbol] = False
