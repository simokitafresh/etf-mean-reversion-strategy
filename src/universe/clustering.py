# src/universe/clustering.py の改善版

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA  # TSNEの代わりにPCAも提供
from sklearn.manifold import TSNE
import os
import time
import yfinance as yf
from ..data.cache import DataCache

# 再現性のためのランダムシード
RANDOM_SEED = 42

# キャッシュのインスタンス化
cache = DataCache()

def get_returns_data(symbols, period="1y"):
    """銘柄リストから日次リターンデータを取得"""
    # （既存コードと同じ）
    return log_returns_df

def calculate_risk_metrics(returns_df):
    """リターンデータからリスク指標を計算"""
    # （既存コードと同じ）
    return metrics

def perform_clustering(etfs, method="optics", max_samples=100, perplexity=30, 
                      n_iter=1000, min_samples=2, xi=0.05, min_cluster_size=2):
    """改善版クラスタリング関数
    
    Args:
        etfs: ETF情報のリスト
        method: クラスタリング手法 ("optics", "tsne_optics", "pca_optics")
        max_samples: 処理する最大サンプル数（大きなデータセット対策）
        perplexity: TSNEのパーブレキシティパラメータ
        n_iter: TSNEの反復回数
        min_samples: OPTICSの最小サンプル数
        xi: OPTICSのクラスタ境界パラメータ
        min_cluster_size: 最小クラスタサイズ
        
    Returns:
        list: 選択されたETFのリスト (各クラスタから代表銘柄)
    """
    # キャッシュからの取得処理（既存コード）
    symbols = [etf['symbol'] for etf in etfs]
    cache_key = f"clustering_{method}_{len(symbols)}_{min_samples}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュからクラスタリング結果を取得しました")
        return cached_data
    
    print(f"{method}アルゴリズムによるETF分類を開始します...")
    
    # リターンデータの取得
    returns_df = get_returns_data(symbols)
    
    # パフォーマンス向上のため、データ量が多い場合は制限する
    if len(returns_df.columns) > max_samples:
        print(f"パフォーマンス向上のため、ETF数を{max_samples}に制限します")
        # シャープレシオでソートして上位を選択
        risk_metrics = calculate_risk_metrics(returns_df)
        top_symbols = risk_metrics.sort_values('sharpe_ratio', ascending=False).index[:max_samples].tolist()
        returns_df = returns_df[top_symbols]
        # symbolsも更新
        symbols = top_symbols
        etfs = [etf for etf in etfs if etf['symbol'] in symbols]
    
    # リスク指標の計算
    risk_metrics = calculate_risk_metrics(returns_df)
    
    # 処理するデータがあるか確認
    if returns_df.empty or returns_df.shape[1] < 2:
        print("エラー: 有効なリターンデータが不足しています")
        return etfs  # 処理できないため元のETFリストを返す
    
    # 1. 相関行列の計算
    corr_matrix = returns_df.corr()
    
    # 2. 次元削減とクラスタリング（選択した方法に基づく）
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_df.T)
    
    # 選択した次元削減・クラスタリング手法を適用
    if method == "optics":
        # OPTICSのみ使用（次元削減なし）
        # パラメータ調整: min_samplesは全データの2%以上かつ最低2
        min_samples_value = max(min_samples, min(2, int(len(symbols) * 0.02)))
        
        optics = OPTICS(
            min_samples=min_samples_value,
            xi=xi,
            min_cluster_size=min_cluster_size,
            n_jobs=-1  # 並列処理
        )
        cluster_labels = optics.fit_predict(scaled_returns)
        reduced_data = scaled_returns  # 簡易的な2次元表示用にPCAを適用
        if reduced_data.shape[1] > 2:
            pca = PCA(n_components=2, random_state=RANDOM_SEED)
            reduced_data = pca.fit_transform(scaled_returns)
        
    elif method == "pca_optics":
        # PCA + OPTICS（TSNEより計算効率が良い）
        pca = PCA(n_components=min(10, len(symbols)-1), random_state=RANDOM_SEED)
        pca_results = pca.fit_transform(scaled_returns)
        
        # OPTICSクラスタリング
        min_samples_value = max(min_samples, min(2, int(len(symbols) * 0.02)))
        optics = OPTICS(
            min_samples=min_samples_value,
            xi=xi,
            min_cluster_size=min_cluster_size,
            n_jobs=-1
        )
        cluster_labels = optics.fit_predict(pca_results)
        
        # 可視化用に2次元に縮小
        reduced_data = pca_results[:, :2] if pca_results.shape[1] >= 2 else pca.fit_transform(scaled_returns)[:, :2]
        
    else:  # "tsne_optics" (デフォルト)
        # TSNE + OPTICS（計算コストが高いが可視化に優れる）
        try:
            tsne = TSNE(
                n_components=2,
                random_state=RANDOM_SEED,
                perplexity=min(perplexity, len(symbols) - 1),
                n_iter=n_iter,
                n_jobs=-1,  # 並列処理
                init='pca'  # 初期化にPCAを使用（計算効率向上）
            )
            reduced_data = tsne.fit_transform(scaled_returns)
            
            # OPTICSクラスタリング
            min_samples_value = max(min_samples, min(2, int(len(symbols) * 0.02)))
            optics = OPTICS(
                min_samples=min_samples_value,
                xi=xi,
                min_cluster_size=min_cluster_size,
                n_jobs=-1
            )
            cluster_labels = optics.fit_predict(reduced_data)
        except Exception as e:
            print(f"TSNE実行エラー: {str(e)}、PCAにフォールバックします")
            # TSNEが失敗した場合はPCAにフォールバック
            pca = PCA(n_components=2, random_state=RANDOM_SEED)
            reduced_data = pca.fit_transform(scaled_returns)
            
            optics = OPTICS(
                min_samples=min_samples,
                xi=xi,
                min_cluster_size=min_cluster_size,
                n_jobs=-1
            )
            cluster_labels = optics.fit_predict(reduced_data)
    
    # 結果ディレクトリの確認
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # クラスタリング結果の可視化（簡素化バージョン）
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
                reduced_data[class_mask, 0],
                reduced_data[class_mask, 1],
                c=[col],
                label=label,
                s=60,
                alpha=0.8
            )
        
        # シンボルのアノテーション（ラベル表示）- レンダリング負荷軽減のため上限設定
        max_annotations = min(50, len(symbols))  # 最大50銘柄までラベル表示
        for i, symbol in enumerate(symbols[:max_annotations]):
            plt.annotate(
                symbol,
                (reduced_data[i, 0], reduced_data[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        plt.title(f'ETFクラスタマップ ({method})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        # 可視化結果を保存
        visualization_path = os.path.join(results_dir, f"etf_clusters_{method}.png")
        plt.savefig(visualization_path, dpi=200, bbox_inches='tight')  # 解像度を下げて保存速度向上
        plt.close()
        print(f"クラスタマップを保存しました: {visualization_path}")
    except Exception as e:
        print(f"可視化エラー: {str(e)}")
    
    # 各クラスタから代表銘柄を選出（既存と同様）
    # （既存コードを再利用 - クラスタ毎に代表銘柄を選ぶ処理）
    # ...

    # キャッシュに保存
    cache.set_json(cache_key, selected_etfs)
    
    return selected_etfs

# 後方互換性のための関数
def tda_clustering(etfs, *args, **kwargs):
    """後方互換性のための関数 - 実際にはOPTICSを使用"""
    print("注: tda_clusteringはOPTICSベースのクラスタリングにリダイレクトされました")
    return perform_clustering(etfs, method="pca_optics", *args, **kwargs)
