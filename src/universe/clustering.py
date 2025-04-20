# src/universe/clustering.py
"""トポロジカルデータ分析（TDA）アプローチによるETFクラスタリング"""
import numpy as np
import pandas as pd
import yfinance as yf
import time
import os
import matplotlib.pyplot as plt
from ..data.cache import DataCache

# キャッシュのインスタンス化
cache = DataCache()

def tda_clustering(filtered_etfs, min_cluster_size=3):
    """UMAPとHDBSCANを用いたTDAアプローチによるETFクラスタリング
    
    Args:
        filtered_etfs (list): 相関フィルタリングを通過したETFのリスト
        min_cluster_size (int): クラスタリングの最小クラスタサイズ
        
    Returns:
        list: クラスタリング後の代表ETFリスト
    """
    # 必要なライブラリをインポート（必要に応じてインストール）
    try:
        import umap
        import hdbscan
    except ImportError:
        print("必要なライブラリをインストールしています...")
        import subprocess
        subprocess.check_call(["pip", "install", "umap-learn==0.5.5", "hdbscan==0.8.33"])
        import umap
        import hdbscan
    
    # キャッシュから取得を試みる
    cache_key = f"tda_clustered_etfs_{len(filtered_etfs)}_{min_cluster_size}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュからクラスタリング結果を取得しました")
        return cached_data
    
    print("TDAアプローチによるクラスタリングを開始します...")
    
    symbols = [etf['symbol'] for etf in filtered_etfs]
    
    # 1年間の日次対数リターン取得
    log_returns_df = pd.DataFrame()
    
    # データ取得（バッチ処理）
    for i in range(0, len(symbols), 10):
        batch_symbols = symbols[i:i+10]
        try:
            print(f"リターンデータ取得中: {batch_symbols}")
            batch_data = yf.download(
                batch_symbols, 
                period="1y", 
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
    
    print(f"リターンデータ取得完了: {log_returns_df.shape} (行, 列)")
    
    # 相関行列から距離行列を計算
    print("相関行列と距離行列を計算中...")
    corr_matrix = log_returns_df.corr()
    distance_matrix = 1 - corr_matrix.abs()  # 絶対値の相関を使用
    
    # UMAP次元削減
    print("UMAP次元削減を実行中...")
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        n_components=2,
        metric='precomputed',
        random_state=42
    )
    
    embedding = reducer.fit_transform(distance_matrix)
    
    # HDBSCANクラスタリング
    print("HDBSCANクラスタリングを実行中...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric='euclidean',
        gen_min_span_tree=True
    )
    
    cluster_labels = clusterer.fit_predict(embedding)
    
    # 結果ディレクトリの確認
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # クラスタリング結果を可視化
    print("クラスタリング結果を可視化中...")
    plt.figure(figsize=(12, 10))
    
    # クラスタごとに色分け
    unique_labels = set(cluster_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:  # ノイズポイント（クラスタに属さない点）
            col = [0, 0, 0, 1]  # 黒
        
        class_mask = cluster_labels == k
        plt.scatter(
            embedding[class_mask, 0],
            embedding[class_mask, 1],
            c=[col],
            label=f'クラスタ {k}' if k != -1 else 'ノイズ',
            s=60,
            alpha=0.8
        )
    
    # 銘柄名のアノテーション
    for i, symbol in enumerate(symbols):
        plt.annotate(
            symbol,
            (embedding[i, 0], embedding[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title('ETFクラスタマップ (UMAP + HDBSCAN)')
    plt.colorbar(label='クラスタ')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 可視化結果を保存
    visualization_path = os.path.join(results_dir, "etf_clusters.png")
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"クラスタマップを保存しました: {visualization_path}")
    
    # 各クラスタから代表銘柄を選出
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        
        symbol = symbols[i]
        etf_info = next((e for e in filtered_etfs if e['symbol'] == symbol), None)
        
        if etf_info:
            # 流動性スコア = 出来高 × AUM
            etf_info['liquidity_score'] = etf_info.get('avg_volume', 0) * etf_info.get('aum', 0)
            etf_info['cluster'] = int(label)
            clusters[label].append(etf_info)
    
    # 各クラスタから最も流動性の高い銘柄を選出
    selected_etfs = []
    
    print("\nクラスタ分析結果:")
    
    for label, cluster_etfs in clusters.items():
        if cluster_etfs:
            if label == -1:
                print(f"ノイズポイント: {len(cluster_etfs)}銘柄")
                # ノイズポイントは追加しない、または別の条件で選別
                continue
            
            print(f"クラスタ {label}: {len(cluster_etfs)}銘柄")
            
            # 流動性スコアで並べ替え
            sorted_etfs = sorted(cluster_etfs, key=lambda x: x.get('liquidity_score', 0), reverse=True)
            selected_etf = sorted_etfs[0]
            selected_etfs.append(selected_etf)
            
            print(f"  代表銘柄: {selected_etf['symbol']} ({selected_etf['name']})")
            print(f"  同クラスタ: {[e['symbol'] for e in sorted_etfs[1:]]}")
    
    # 結果の詳細をCSVに保存
    all_etfs_with_clusters = []
    for label, cluster_etfs in clusters.items():
        all_etfs_with_clusters.extend(cluster_etfs)
    
    clusters_df = pd.DataFrame(all_etfs_with_clusters)
    clusters_csv_path = os.path.join(results_dir, "etf_clusters_details.csv")
    clusters_df.to_csv(clusters_csv_path, index=False)
    print(f"クラスタ詳細を保存しました: {clusters_csv_path}")
    
    # データの保存と共有
    embedding_df = pd.DataFrame({
        'symbol': symbols,
        'cluster': cluster_labels,
        'x': embedding[:, 0],
        'y': embedding[:, 1]
    })
    embedding_csv_path = os.path.join(results_dir, "etf_embedding.csv")
    embedding_df.to_csv(embedding_csv_path, index=False)
    print(f"埋め込み座標を保存しました: {embedding_csv_path}")
    
    # 最終的に選出された代表銘柄リストを保存
    selected_df = pd.DataFrame(selected_etfs)
    selected_csv_path = os.path.join(results_dir, "selected_representative_etfs.csv")
    selected_df.to_csv(selected_csv_path, index=False)
    print(f"代表銘柄リストを保存しました: {selected_csv_path}")
    
    # キャッシュに保存
    cache.set_json(cache_key, selected_etfs)
    
    return selected_etfs
