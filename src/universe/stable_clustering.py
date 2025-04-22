"""より安定したクラスタリングアルゴリズム(OPTICS)によるETF分類"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
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

def perform_clustering(etfs, min_samples=2, xi=0.05, min_cluster_size=0.1):
    """OPTICSアルゴリズムでETFを分類"""
    # キャッシュから取得を試みる
    symbols = [etf['symbol'] for etf in etfs]
    cache_key = f"optics_clustering_{len(symbols)}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュからクラスタリング結果を取得しました")
        return cached_data
    
    print("OPTICSアルゴリズムによるETF分類を開始します...")
    
    # リターンデータの取得
    returns_df = get_returns_data(symbols)
    
    # リスク指標の計算
    risk_metrics = calculate_risk_metrics(returns_df)
    
    # 相関行列の計算
    corr_matrix = returns_df.corr()
    
    # 1. リスク特性の標準化
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(risk_metrics[['mean_return', 'volatility', 'sharpe_ratio', 'max_drawdown']])
    
    # 2. PCAで可視化のために次元削減（2次元に）- クラスタリング自体は元のデータで行う
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_metrics)
    
    # サンプル数に基づいたパラメータ調整
    if min_cluster_size < 1:
        min_cluster_size = max(2, int(len(symbols) * min_cluster_size))
    
    # 3. OPTICSによるクラスタリング
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    cluster_labels = optics.fit_predict(scaled_metrics)  # スケーリングされたリスク指標に基づくクラスタリング
    
    # 結果ディレクトリの確認
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # クラスタリング結果を可視化
    plt.figure(figsize=(12, 10))
    
    # クラスタごとに色分け
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # OPTICSの場合は特にノイズポイント(-1)を考慮する
    for k, col in zip(unique_labels, colors):
        if k == -1:  # ノイズポイントは別途処理
            continue
        
        class_mask = cluster_labels == k
        plt.scatter(
            pca_result[class_mask, 0],
            pca_result[class_mask, 1],
            c=[col],
            label=f'クラスタ {k}',
            s=60,
            alpha=0.8
        )
    
    # ノイズポイント(クラスタ = -1)をグレーで表示
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        plt.scatter(
            pca_result[noise_mask, 0],
            pca_result[noise_mask, 1],
            c='lightgray',
            label='ノイズ',
            s=60,
            alpha=0.6
        )
    
    # 銘柄名のアノテーション
    for i, symbol in enumerate(symbols):
        plt.annotate(
            symbol,
            (pca_result[i, 0], pca_result[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title('ETFクラスタマップ (OPTICS)')
    plt.xlabel('主成分1 (PCA)')
    plt.ylabel('主成分2 (PCA)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 可視化結果を保存
    visualization_path = os.path.join(results_dir, "etf_clusters_optics.png")
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
            # ノイズ(-1)の場合は特別に表示
            cluster_name = "ノイズ" if label == -1 else f"クラスタ {label}"
            print(f"{cluster_name}: {len(cluster_etfs)}銘柄")
            
            # シャープレシオで並べ替え
            sorted_etfs = sorted(cluster_etfs, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
            selected_etf = sorted_etfs[0]
            selected_etfs.append(selected_etf)
            
            print(f"  代表銘柄: {selected_etf['symbol']} ({selected_etf['name']})")
            print(f"  同グループ: {[e['symbol'] for e in sorted_etfs[1:]]}")
    
    # 結果の詳細をCSVに保存
    all_etfs_with_clusters = []
    for label, cluster_etfs in clusters.items():
        all_etfs_with_clusters.extend(cluster_etfs)
    
    clusters_df = pd.DataFrame(all_etfs_with_clusters)
    clusters_csv_path = os.path.join(results_dir, "etf_clusters_optics_details.csv")
    clusters_df.to_csv(clusters_csv_path, index=False)
    print(f"クラスタ詳細を保存しました: {clusters_csv_path}")
    
    # 分析サマリーを表示
    print("\nOPTICSクラスタリング結果サマリー:")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for label, count in cluster_counts.items():
        cluster_name = "ノイズ" if label == -1 else f"クラスタ {label}"
        print(f"- {cluster_name}: {count}銘柄")
    
    # キャッシュに保存
    cache.set_json(cache_key, selected_etfs)
    
    return selected_etfs
