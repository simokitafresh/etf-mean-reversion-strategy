# src/universe/stable_clustering.py
"""より安定したクラスタリングアルゴリズムによるETF分類"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, KMeans
from sklearn.metrics import silhouette_score
import os
import time
import yfinance as yf
from ..data.cache import DataCache

# キャッシュのインスタンス化
cache = DataCache()

def get_returns_data(symbols, period="1y"):
    """銘柄リストから日次リターンデータを取得"""
    log_returns_df = pd.DataFrame()
    
    # データ取得（バッチ処理）
    for i in range(0, len(symbols), 10):
        batch_symbols = symbols[i:i+10]
        try:
            print(f"リターンデータ取得中: {batch_symbols}")
            batch_data = yf.download(
                batch_symbols, 
                period=period, 
                interval="1d",
                group_by="ticker",
                progress=False
            )['Adj Close']
            
            # バッチデータ処理
            if len(batch_symbols) == 1:
                symbol = batch_symbols[0]
                prices = batch_data
                log_returns = np.log(prices / prices.shift(1)).dropna()
                log_returns_df[symbol] = log_returns
            else:
                for symbol in batch_symbols:
                    if symbol in batch_data:
                        prices = batch_data[symbol]
                        log_returns = np.log(prices / prices.shift(1)).dropna()
                        log_returns_df[symbol] = log_returns
            
            time.sleep(1)
        except Exception as e:
            print(f"エラー - クラスタリングデータバッチ {i}: {str(e)}")
    
    # 欠損値処理
    log_returns_df.fillna(method='ffill', inplace=True)
    log_returns_df.fillna(0, inplace=True)  # 残りの欠損値を0で埋める
    
    return log_returns_df

def calculate_risk_metrics(returns_df):
    """リターンデータからリスク指標を計算"""
    metrics = pd.DataFrame(index=returns_df.columns)
    
    # 平均リターン
    metrics['mean_return'] = returns_df.mean()
    
    # ボラティリティ（標準偏差）
    metrics['volatility'] = returns_df.std()
    
    # シャープレシオ（簡易版：リスクフリーレート0%）
    metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['volatility']
    
    # 最大ドローダウン
    cum_returns = (1 + returns_df).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns / rolling_max) - 1
    metrics['max_drawdown'] = drawdowns.min()
    
    return metrics

def perform_clustering(etfs, n_clusters_range=range(3, 11)):
    """安定したクラスタリングアルゴリズムでETFを分類"""
    # キャッシュから取得を試みる
    symbols = [etf['symbol'] for etf in etfs]
    cache_key = f"stable_clustering_{len(symbols)}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュからクラスタリング結果を取得しました")
        return cached_data
    
    print("安定したクラスタリングアルゴリズムによるETF分類を開始します...")
    
    # リターンデータの取得
    returns_df = get_returns_data(symbols)
    
    # リスク指標の計算
    risk_metrics = calculate_risk_metrics(returns_df)
    
    # 相関行列の計算
    corr_matrix = returns_df.corr()
    
    # 1. 標準化
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(risk_metrics[['mean_return', 'volatility', 'sharpe_ratio', 'max_drawdown']])
    
    # 2. PCAで次元削減（2次元に）
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_metrics)
    
    # 3. 最適なクラスタ数を探索
    silhouette_scores = []
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_result)
        
        # 少なくとも2つのクラスターがあり、各クラスターに少なくとも1つの要素がある場合
        if len(np.unique(cluster_labels)) > 1:
            score = silhouette_score(pca_result, cluster_labels)
            silhouette_scores.append((n_clusters, score))
    
    # 最適なクラスタ数を選択
    optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0] if silhouette_scores else 5
    
    # 4. KMeansでクラスタリング
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_result)
    
    # 結果ディレクトリの確認
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # クラスタリング結果を可視化
    plt.figure(figsize=(12, 10))
    
    # クラスタごとに色分け
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        class_mask = cluster_labels == k
        plt.scatter(
            pca_result[class_mask, 0],
            pca_result[class_mask, 1],
            c=[col],
            label=f'クラスタ {k}',
            s=60,
            alpha=0.8
        )
    
    # 銘柄名のアノテーション
    for i, symbol in enumerate(symbols):
        plt.annotate(
            symbol,
            (pca_result[i, 0], pca_result[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title('ETFクラスタマップ (PCA + KMeans)')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.colorbar(label='クラスタ')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 可視化結果を保存
    visualization_path = os.path.join(results_dir, "etf_clusters_stable.png")
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"クラスタマップを保存しました: {visualization_path}")
    
    # 各クラスタから代表銘柄を選出
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        
        symbol = symbols[i]
        etf_info = next((e for e in etfs if e['symbol'] == symbol), None)
        
        if etf_info:
            # シャープレシオをスコアとして使用
            etf_info['sharpe_ratio'] = float(risk_metrics.loc[symbol, 'sharpe_ratio'])
            etf_info['cluster'] = int(label)
            clusters[label].append(etf_info)
    
    # 各クラスタから最もシャープレシオの高い銘柄を選出
    selected_etfs = []
    
    print("\nクラスタ分析結果:")
    
    for label, cluster_etfs in clusters.items():
        if cluster_etfs:
            print(f"クラスタ {label}: {len(cluster_etfs)}銘柄")
            
            # シャープレシオで並べ替え
            sorted_etfs = sorted(cluster_etfs, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
            selected_etf = sorted_etfs[0]
            selected_etfs.append(selected_etf)
            
            print(f"  代表銘柄: {selected_etf['symbol']} ({selected_etf['name']})")
            print(f"  同クラスタ: {[e['symbol'] for e in sorted_etfs[1:]]}")
    
    # 結果の詳細をCSVに保存
    all_etfs_with_clusters = []
    for label, cluster_etfs in clusters.items():
        all_etfs_with_clusters.extend(cluster_etfs)
    
    clusters_df = pd.DataFrame(all_etfs_with_clusters)
    clusters_csv_path = os.path.join(results_dir, "etf_clusters_stable_details.csv")
    clusters_df.to_csv(clusters_csv_path, index=False)
    print(f"クラスタ詳細を保存しました: {clusters_csv_path}")
    
    # キャッシュに保存
    cache.set_json(cache_key, selected_etfs)
    
    return selected_etfs
