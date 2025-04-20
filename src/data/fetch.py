"""データ取得のためのユーティリティ"""
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from .cache import DataCache

# キャッシュのインスタンス化
cache = DataCache()

def get_base_etf_list():
    """基本的なETFリストを取得する"""
    # キャッシュから取得を試みる
    cached_data = cache.get_json("base_etf_list")
    if cached_data:
        print("キャッシュからETFリストを取得しました")
        return cached_data
    
    print("ETFリストを取得しています...")
    
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
    
    # 10銘柄ずつバッチ処理
    for i in range(0, len(base_etfs), 10):
        batch = base_etfs[i:i+10]
        try:
            for symbol in batch:
                try:
                    etf = yf.Ticker(symbol)
                    info = etf.info
                    
                    etf_data = {
                        'symbol': symbol,
                        'name': info.get('shortName', info.get('longName', symbol)),
                        'category': info.get('category', ''),
                        'description': info.get('description', '')[:100] + '...'  # 説明は短く
                    }
                    etf_info.append(etf_data)
                    print(f"取得: {symbol}")
                except Exception as e:
                    print(f"エラー ({symbol}): {str(e)}")
            
            # API制限を避けるため少し待機
            time.sleep(1)
        except Exception as e:
            print(f"バッチ処理エラー: {str(e)}")
    
    # キャッシュに保存
    cache.set_json("base_etf_list", etf_info)
    
    return etf_info
