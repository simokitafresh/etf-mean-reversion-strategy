"""エラーハンドリングと統合ロギングユーティリティ

このモジュールはプロジェクト全体で一貫したエラーハンドリングとロギングを提供します。
エラーの詳細な情報と適切なフォーマットを提供し、効率的なトラブルシューティングを支援します。
"""
import functools
import time
import traceback
import logging
import os
import sys
import json
import uuid
from datetime import datetime
from typing import Callable, Dict, Any, List, Tuple, Optional, Union, Type

# ロギングディレクトリの設定
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)

# メインロガーの設定
main_logger = logging.getLogger('etf_strategy')
main_logger.setLevel(logging.INFO)

# ログファイル名の設定
log_file_path = os.path.join(LOG_DIR, f"error_log_{datetime.now().strftime('%Y%m%d')}.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# ストリームハンドラ（コンソール出力）
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# フォーマッタの設定
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# ハンドラをロガーに追加
main_logger.addHandler(file_handler)
main_logger.addHandler(stream_handler)

# ロガー取得用関数
def get_logger(name: str) -> logging.Logger:
    """指定した名前のロガーを取得する
    
    Args:
        name: ロガー名（通常はモジュール名を使用）
        
    Returns:
        logging.Logger: 設定済みのロガーインスタンス
    """
    logger = logging.getLogger(f'etf_strategy.{name}')
    
    # ハンドラが既に設定済みの場合はスキップ
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger

# このモジュール用のロガー
logger = get_logger('error_handling')

def retry(
    max_attempts: int = 3, 
    backoff_factor: float = 2.0, 
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_name: Optional[str] = None
):
    """関数の実行を最大試行回数まで再試行するデコレータ
    
    Args:
        max_attempts: 最大試行回数
        backoff_factor: 再試行間隔の増加係数
        retry_exceptions: 再試行対象の例外タイプのタプル
        logger_name: 使用するロガー名（Noneの場合は関数名を使用）
        
    Returns:
        装飾された関数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__name__)
            attempt = 0
            last_exception = None
            error_id = str(uuid.uuid4())[:8]  # エラー追跡用の一意のID
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    attempt += 1
                    last_exception = e
                    
                    if attempt < max_attempts:
                        wait_time = backoff_factor ** attempt
                        func_logger.warning(
                            f"Error ID {error_id} in {func.__name__}: {str(e)}. "
                            f"Retrying in {wait_time:.1f} seconds... "
                            f"(Attempt {attempt}/{max_attempts})"
                        )
                        time.sleep(wait_time)
                    else:
                        # 最大試行回数に達した場合、エラースタックトレースとともに詳細をログに記録
                        func_logger.error(
                            f"Error ID {error_id} in {func.__name__}: {str(e)}. "
                            f"Max attempts reached ({max_attempts}).\n"
                            f"Stack trace: {traceback.format_exc()}"
                        )
                        
                        # エラー詳細情報をJSONファイルとして保存
                        error_detail = {
                            'error_id': error_id,
                            'function': func.__name__,
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'timestamp': datetime.now().isoformat(),
                            'args': str(args),
                            'kwargs': str(kwargs),
                            'traceback': traceback.format_exc()
                        }
                        
                        error_file = os.path.join(LOG_DIR, f"error_{error_id}.json")
                        with open(error_file, 'w') as f:
                            json.dump(error_detail, f, indent=2)
                        
                        func_logger.error(f"詳細なエラー情報を保存しました: {error_file}")
            
            # 最大試行回数を超えた場合
            raise last_exception
        
        return wrapper
    
    return decorator

def exception_handler(
    log_level: int = logging.ERROR, 
    return_default: Any = None,
    logger_name: Optional[str] = None,
    error_message: Optional[str] = None
):
    """例外を処理し、デフォルト値を返すデコレータ
    
    Args:
        log_level: ログレベル
        return_default: エラー時に返すデフォルト値
        logger_name: 使用するロガー名（Noneの場合は関数名を使用）
        error_message: カスタムエラーメッセージ（Noneの場合は例外メッセージを使用）
        
    Returns:
        装飾された関数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__name__)
            error_id = str(uuid.uuid4())[:8]  # エラー追跡用の一意のID
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                message = error_message or f"エラー発生: {str(e)}"
                func_logger.log(
                    log_level,
                    f"Error ID {error_id} in {func.__name__}: {message}\n"
                    f"Stack trace: {traceback.format_exc()}"
                )
                
                # エラー詳細情報をJSONファイルとして保存（重大なエラーのみ）
                if log_level >= logging.ERROR:
                    error_detail = {
                        'error_id': error_id,
                        'function': func.__name__,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'args': str(args),
                        'kwargs': str(kwargs),
                        'traceback': traceback.format_exc()
                    }
                    
                    error_file = os.path.join(LOG_DIR, f"error_{error_id}.json")
                    with open(error_file, 'w') as f:
                        json.dump(error_detail, f, indent=2)
                    
                    func_logger.error(f"詳細なエラー情報を保存しました: {error_file}")
                
                return return_default
        
        return wrapper
    
    return decorator

def validate_input(schema: Dict[str, Any], logger_name: Optional[str] = None):
    """入力パラメータのバリデーションを行うデコレータ
    
    Args:
        schema: バリデーションスキーマ
        logger_name: 使用するロガー名（Noneの場合は関数名を使用）
        
    Returns:
        装飾された関数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__name__)
            
            # 引数をチェック
            for param_name, param_schema in schema.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    
                    # 型チェック
                    expected_type = param_schema.get('type')
                    if expected_type and not isinstance(value, expected_type):
                        error_message = (
                            f"パラメータ '{param_name}' は {expected_type.__name__} 型であるべきですが、"
                            f"{type(value).__name__} 型が指定されました"
                        )
                        func_logger.error(error_message)
                        raise TypeError(error_message)
                    
                    # 範囲チェック
                    if 'min' in param_schema and value < param_schema['min']:
                        error_message = (
                            f"パラメータ '{param_name}' は {param_schema['min']} 以上であるべきですが、"
                            f"{value} が指定されました"
                        )
                        func_logger.error(error_message)
                        raise ValueError(error_message)
                    
                    if 'max' in param_schema and value > param_schema['max']:
                        error_message = (
                            f"パラメータ '{param_name}' は {param_schema['max']} 以下であるべきですが、"
                            f"{value} が指定されました"
                        )
                        func_logger.error(error_message)
                        raise ValueError(error_message)
                    
                    # 列挙値チェック
                    if 'enum' in param_schema and value not in param_schema['enum']:
                        error_message = (
                            f"パラメータ '{param_name}' は {param_schema['enum']} のいずれかであるべきですが、"
                            f"{value} が指定されました"
                        )
                        func_logger.error(error_message)
                        raise ValueError(error_message)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def log_execution_time(func=None, *, logger_name: Optional[str] = None):
    """関数の実行時間をログに記録するデコレータ
    
    Args:
        func: 装飾対象の関数
        logger_name: 使用するロガー名（Noneの場合は関数名を使用）
        
    Returns:
        装飾された関数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__name__)
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 実行時間に応じてログレベルを調整
            if execution_time < 0.1:
                log_level = logging.DEBUG
            elif execution_time < 1.0:
                log_level = logging.INFO
            elif execution_time < 10.0:
                log_level = logging.INFO
            else:
                log_level = logging.WARNING
            
            func_logger.log(
                log_level, 
                f"{func.__name__} の実行時間: {execution_time:.3f} 秒"
            )
            
            return result
        return wrapper
    
    # デコレータを引数なしで使用した場合
    if func is not None:
        return decorator(func)
    
    # デコレータを引数付きで使用した場合
    return decorator

def handle_missing_data(data_frame, fill_method: str = 'ffill', max_gap: int = 3):
    """データフレームの欠損値を処理する
    
    Args:
        data_frame: 処理対象のデータフレーム
        fill_method: 補完方法（'ffill', 'bfill', 'interpolate', 'zero'）
        max_gap: 許容する連続欠損値の最大数
        
    Returns:
        処理済みのデータフレーム
    """
    if data_frame.empty:
        return data_frame
    
    # 連続欠損値の長さを計算
    gap_lengths = data_frame.isnull().astype(int).groupby(
        data_frame.notnull().astype(int).cumsum()
    ).sum()
    
    # 長いギャップを特定
    long_gaps = gap_lengths[gap_lengths > max_gap]
    
    if not long_gaps.empty:
        logger.warning(
            f"データに {len(long_gaps)} 個のギャップ (長さ > {max_gap}) があります。"
            f"最長のギャップ: {long_gaps.max()} ポイント"
        )
    
    # 欠損値を補完
    if fill_method == 'ffill':
        result = data_frame.fillna(method='ffill')
    elif fill_method == 'bfill':
        result = data_frame.fillna(method='bfill')
    elif fill_method == 'interpolate':
        result = data_frame.interpolate(method='linear')
    elif fill_method == 'zero':
        result = data_frame.fillna(0)
    else:
        raise ValueError(f"不明な補完方法: {fill_method}")
    
    # 残りの欠損値をチェック
    remaining_nulls = result.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"データに {remaining_nulls} 個の欠損値が残っています (補完方法: {fill_method})")
    
    return result

def log_important_info(message: str, logger_name: Optional[str] = None) -> None:
    """重要な情報をログに記録する
    
    Args:
        message: 記録するメッセージ
        logger_name: 使用するロガー名（Noneの場合はメインロガーを使用）
    """
    log_logger = get_logger(logger_name or 'main')
    log_logger.info(f"[重要] {message}")

def log_warning(message: str, logger_name: Optional[str] = None) -> None:
    """警告をログに記録する
    
    Args:
        message: 記録するメッセージ
        logger_name: 使用するロガー名（Noneの場合はメインロガーを使用）
    """
    log_logger = get_logger(logger_name or 'main')
    log_logger.warning(message)

def log_error(message: str, error: Optional[Exception] = None, logger_name: Optional[str] = None) -> str:
    """エラーをログに記録し、エラーIDを返す
    
    Args:
        message: 記録するメッセージ
        error: 例外オブジェクト（あれば）
        logger_name: 使用するロガー名（Noneの場合はメインロガーを使用）
        
    Returns:
        str: エラー追跡用のID
    """
    log_logger = get_logger(logger_name or 'main')
    error_id = str(uuid.uuid4())[:8]
    
    if error:
        log_logger.error(
            f"Error ID {error_id}: {message} - {type(error).__name__}: {str(error)}"
        )
        
        # エラー詳細情報をJSONファイルとして保存
        error_detail = {
            'error_id': error_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'traceback': traceback.format_exc()
        }
        
        error_file = os.path.join(LOG_DIR, f"error_{error_id}.json")
        with open(error_file, 'w') as f:
            json.dump(error_detail, f, indent=2)
    else:
        log_logger.error(f"Error ID {error_id}: {message}")
    
    return error_id

def setup_logger_for_module(module_name: str) -> logging.Logger:
    """モジュール用のロガーをセットアップする
    
    Args:
        module_name: モジュール名
        
    Returns:
        logging.Logger: 設定済みのロガーインスタンス
    """
    logger = get_logger(module_name)
    logger.info(f"{module_name} モジュールのロガーを初期化しました")
    return logger
