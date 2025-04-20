# src/utils/error_handling.py
"""エラーハンドリングユーティリティ"""
import functools
import time
import traceback
import logging
import os
import json
from typing import Callable, Dict, Any, List, Tuple, Optional

# ロガーの設定
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/error.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def retry(max_attempts: int = 3, backoff_factor: float = 2.0):
    """関数の実行を最大試行回数まで再試行するデコレータ
    
    Args:
        max_attempts: 最大試行回数
        backoff_factor: 再試行間隔の増加係数
        
    Returns:
        装飾された関数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    last_exception = e
                    
                    if attempt < max_attempts:
                        wait_time = backoff_factor ** attempt
                        logger.warning(
                            f"Error in {func.__name__}: {str(e)}. "
                            f"Retrying in {wait_time:.1f} seconds... "
                            f"(Attempt {attempt}/{max_attempts})"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Error in {func.__name__}: {str(e)}. "
                            f"Max attempts reached ({max_attempts})."
                        )
            
            # 最大試行回数を超えた場合
            raise last_exception
        
        return wrapper
    
    return decorator

def exception_handler(log_level: int = logging.ERROR, return_default: Any = None):
    """例外を処理し、デフォルト値を返すデコレータ
    
    Args:
        log_level: ログレベル
        return_default: エラー時に返すデフォルト値
        
    Returns:
        装飾された関数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                return return_default
        
        return wrapper
    
    return decorator

def validate_input(schema: Dict[str, Any]):
    """入力パラメータのバリデーションを行うデコレータ
    
    Args:
        schema: バリデーションスキーマ
        
    Returns:
        装飾された関数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 引数を関数の仮引数名にマッピング
            # 実際の実装には関数のシグネチャを調べる必要があります
            
            # 簡易的なバリデーション
            for param_name, param_schema in schema.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    
                    # 型チェック
                    expected_type = param_schema.get('type')
                    if expected_type and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' should be of type {expected_type.__name__}, "
                            f"but got {type(value).__name__}"
                        )
                    
                    # 範囲チェック
                    if 'min' in param_schema and value < param_schema['min']:
                        raise ValueError(
                            f"Parameter '{param_name}' should be >= {param_schema['min']}, "
                            f"but got {value}"
                        )
                    
                    if 'max' in param_schema and value > param_schema['max']:
                        raise ValueError(
                            f"Parameter '{param_name}' should be <= {param_schema['max']}, "
                            f"but got {value}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def log_execution_time(func):
    """関数の実行時間をログに記録するデコレータ
    
    Args:
        func: 装飾対象の関数
        
    Returns:
        装飾された関数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper

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
            f"Found {len(long_gaps)} gaps longer than {max_gap} in the data. "
            f"Longest gap: {long_gaps.max()} points"
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
        raise ValueError(f"Unknown fill method: {fill_method}")
    
    # 残りの欠損値をチェック
    remaining_nulls = result.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"{remaining_nulls} null values remain in the data after {fill_method}")
    
    return result
