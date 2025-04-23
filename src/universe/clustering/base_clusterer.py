"""クラスタリング基底クラス"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import yfinance as yf
import warnings
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

# 絶対インポートの使用（プロジェクト内の別パッケージへの参照）
from src.data.cache_manager import CacheManager

# ロガーの設定
logger = logging.getLogger(__name__)

# 再現性のためのランダムシード
RANDOM_SEED = 42

# キャッシュマネージャーのインスタンスを取得
cache_manager = CacheManager.get_instance()

class BaseClusterer(ABC):
    """クラスタリング基底クラス"""
    
    def __init__(self, name="base", **kwargs):
        """初期化
        
        Args:
            name: クラスタリング手法の名前
            **kwargs: 追加パラメータ
        """
        self.name = name
        self.params = kwargs
        
        # 結果ディレクトリの確認
        self.results_dir = kwargs.get("results_dir", "data/results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def cluster(self, etfs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ETFのクラスタリングを実行
        
        Args:
            etfs: ETF情報のリスト
            
        Returns:
            List[Dict[str, Any]]: 選択されたETFのリスト
        """
        if not etfs:
            logger.warning("ETFリストが空です")
            return []
        
        # キャッシュキーの生成
        cache_key = self._generate_cache_key(etfs)
        cached_data = cache_manager.get_json(cache_key)
        if cached_data:
            logger.info(f"キャッシュから{self.name}クラスタリング結果を取得しました")
            return cached_data
        
        # シンボルリストの取得
        symbols = [etf.get('symbol', '') for etf in etfs]
        symbols = [s for s in symbols if s]  # 空文字を除外
        
        # リターンデータの取得
        returns_df = self.get_returns_data(symbols)
        
        if returns_df.empty:
            logger.warning("リターンデータが取得できませんでした")
            return etfs
        
        # 実際のクラスタリングを実行（サブクラスで実装）
        selected_etfs = self.perform_clustering(etfs, returns_df)
        
        # 結果をキャッシュに保存
        cache_manager.set_json(cache_key, selected_etfs)
        
        return selected_etfs
    
    @abstractmethod
    def perform_clustering(self, etfs: List[Dict[str, Any]], returns_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """実際のクラスタリングを実行（サブクラスで実装）
        
        Args:
            etfs: ETF情報のリスト
            returns_df: リターンデータのデータフレーム
            
        Returns:
            List[Dict[str, Any]]: クラスタリング結果（選択されたETF）
        """
        pass
    
    def get_returns_data(self, symbols: List[str], period="1y") -> pd.DataFrame:
        """銘柄リストから日次リターンデータを取得
        
        Args:
            symbols: ETFシンボルのリスト
            period: データ取得期間（例: "1y", "2y", "max"）
            
        Returns:
            pd.DataFrame: 日次対数リターンデータを含むデータフレーム
        """
        # シンボルリストの検証
        if not symbols:
            logger.warning("シンボルリストが空です")
            return pd.DataFrame()
            
        # キャッシュから取得を試みる
        cache_key = f"returns_data_{'-'.join(sorted(symbols[:5]))}_{len(symbols)}_{period}"
        cached_data = cache_manager.get_json(cache_key)
        if cached_data is not None:
            logger.info("キャッシュからリターンデータを取得しました")
            return pd.read_json(cached_data, orient='split')
        
        logger.info(f"{len(symbols)}銘柄の価格データを取得中...")
        
        prices_df = pd.DataFrame()
        batch_size = 5  # APIレート制限対策
        
        # バッチ処理
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            logger.info(f"バッチ取得中: {batch_symbols}")
            
            try:
                # YFinanceで価格データを取得
                batch_data = yf.download(
                    batch_symbols, 
                    period=period, 
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    threads=True  # 並列ダウンロードを使用
                )
                
                # 単一銘柄の場合は構造が異なる
                if len(batch_symbols) == 1:
                    symbol = batch_symbols[0]
                    if 'Close' in batch_data.columns:
                        prices_df[symbol] = batch_data['Close']
                else:
                    # 複数銘柄の場合
                    for symbol in batch_symbols:
                        if symbol in batch_data and 'Close' in batch_data[symbol]:
                            prices_df[symbol] = batch_data[symbol]['Close']
                
                # APIレート制限対策で待機
                time.sleep(1)
            
            except Exception as e:
                logger.warning(f"警告: バッチ{i}のデータ取得エラー: {str(e)}")
                time.sleep(5)  # エラー時は長めに待機
        
        # シンボルが1つも取得できなかった場合
        if prices_df.empty:
            logger.error("エラー: 有効な価格データが取得できませんでした")
            return pd.DataFrame()
        
        # 欠損値処理
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        
        # 欠損値が多すぎる列を削除（80%以上のデータがある列のみを保持）
        min_data_points = 0.8 * len(prices_df)
        prices_df = prices_df.loc[:, prices_df.count() >= min_data_points]
        
        if prices_df.empty:
            logger.error("エラー: 欠損値処理後にデータがなくなりました")
            return pd.DataFrame()
        
        # 日次対数リターンの計算
        log_returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        
        # キャッシュに保存
        cache_manager.set_json(cache_key, log_returns_df.to_json(orient='split'))
        
        logger.info(f"リターンデータ取得完了: {log_returns_df.shape[0]}日 x {log_returns_df.shape[1]}銘柄")
        return log_returns_df
    
    def select_representative_etfs(self, etf_clusters: List[Dict[str, Any]], 
                                   returns_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """各クラスタから代表的なETFを選択
        
        Args:
            etf_clusters: クラスタIDが付与されたETF情報のリスト
            returns_data: リターンデータのデータフレーム
            
        Returns:
            List[Dict[str, Any]]: 選択されたETF情報のリスト
        """
        # クラスタID別にETFをグループ化
        clusters = {}
        for etf in etf_clusters:
            cluster_id = etf.get('cluster', -1)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(etf)
        
        # 選択されたETF
        selected_etfs = []
        
        # クラスタごとに代表ETFを選択
        for cluster_id, cluster_etfs in clusters.items():
            # ノイズクラスタはスキップ
            if cluster_id == -1:
                continue
            
            # クラスタ内のETFシンボル
            cluster_symbols = [etf['symbol'] for etf in cluster_etfs]
            
            # クラスタのETFが1つしかない場合はそれを選択
            if len(cluster_symbols) == 1:
                selected_etfs.append(cluster_etfs[0])
                continue
            
            try:
                # リターンデータからクラスタ内のETFを取得
                valid_symbols = [s for s in cluster_symbols if s in returns_data.columns]
                
                if not valid_symbols:
                    raise ValueError("有効なETFシンボルがありません")
                
                cluster_returns = returns_data[valid_symbols]
                
                # クラスタの平均リターンを計算
                cluster_mean = cluster_returns.mean(axis=1)
                
                # 各ETFとクラスタ平均の相関を計算
                correlations = {}
                for symbol in valid_symbols:
                    correlations[symbol] = cluster_returns[symbol].corr(cluster_mean)
                
                # 最も平均的な振る舞いをするETF（中心的なETF）を選択
                if correlations:
                    central_symbol = max(correlations, key=correlations.get)
                    # 対応するETF情報を検索
                    central_etf = next((etf for etf in cluster_etfs if etf['symbol'] == central_symbol), None)
                    
                    if central_etf:
                        selected_etfs.append(central_etf)
                        continue
            except Exception as e:
                logger.error(f"クラスタ{cluster_id}の代表ETF選択エラー: {str(e)}")
            
            # エラーまたはデータが不十分な場合、出来高が最も多いETFを選択
            fallback_etf = max(cluster_etfs, key=lambda x: x.get('avg_volume', 0))
            selected_etfs.append(fallback_etf)
        
        return selected_etfs
    
    def visualize_clusters(self, etfs: List[Dict[str, Any]], 
                          cluster_labels: np.ndarray, 
                          embedding: np.ndarray,
                          title: str = None) -> str:
        """クラスタリング結果を可視化
        
        Args:
            etfs: ETF情報のリスト
            cluster_labels: クラスタリング結果のラベル
            embedding: 2次元平面に埋め込まれたデータ
            title: グラフのタイトル
            
        Returns:
            str: 保存されたファイルのパス
        """
        try:
            plt.figure(figsize=(12, 10))
            
            # クラスタごとに色分け
            unique_labels = np.unique(cluster_labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            # 各クラスタを描画
            for k, col in zip(unique_labels, colors):
                if k == -1:  # ノイズ
                    col = 'lightgray'
                    label = 'ノイズ'
                else:
                    label = f'クラスタ {k}'
                
                class_mask = cluster_labels == k
                plt.scatter(
                    embedding[class_mask, 0],
                    embedding[class_mask, 1],
                    c=[col],
                    label=label,
                    s=60,
                    alpha=0.8
                )
            
            # ETFシンボルのアノテーション
            symbols = [etf.get('symbol', '') for etf in etfs]
            max_annotations = min(50, len(symbols))  # 最大表示数
            
            for i, symbol in enumerate(symbols[:max_annotations]):
                if i < len(embedding):
                    plt.annotate(
                        symbol,
                        (embedding[i, 0], embedding[i, 1]),
                        fontsize=8,
                        alpha=0.7
                    )
            
            # タイトルの設定
            if title:
                plt.title(title)
            else:
                plt.title(f'ETFクラスタマップ ({self.name})')
            
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            
            # 可視化結果を保存
            visualization_path = os.path.join(self.results_dir, f"etf_clusters_{self.name}.png")
            plt.savefig(visualization_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            logger.info(f"クラスタマップを保存しました: {visualization_path}")
            return visualization_path
            
        except Exception as e:
            logger.error(f"可視化エラー: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return ""
    
    def _generate_cache_key(self, etfs: List[Dict[str, Any]]) -> str:
        """キャッシュキーを生成
        
        Args:
            etfs: ETF情報のリスト
            
        Returns:
            str: キャッシュキー
        """
        # シンボルだけを抽出、ソートし、先頭5つだけを使用（キーが長すぎるのを防ぐため）
        symbols = sorted([etf.get('symbol', '') for etf in etfs if etf.get('symbol')])
        symbols_hash = '-'.join(symbols[:5]) + f"_{len(symbols)}"
        
        # パラメータのハッシュを追加
        params_str = '_'.join([f"{k}={v}" for k, v in self.params.items() if k != 'results_dir'])
        if params_str:
            return f"{self.name}_clustering_{symbols_hash}_{params_str}"
        else:
            return f"{self.name}_clustering_{symbols_hash}"
