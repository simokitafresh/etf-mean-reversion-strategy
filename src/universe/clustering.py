"""ETFクラスタリングモジュール - TSNE + OPTICS アプローチ"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
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
    """銘柄リストから日次リターンデータを取得
    
    Args:
        symbols: ETFシンボルのリスト
        period: データ取得期間 (デフォルト: "1y")
        
    Returns:
        pd.DataFrame: 日次リターンのデータフレーム
    """
    log_returns_df = pd.DataFrame()
    
    # データ取得（バッチ処理でAPIレート制限に対応）
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
            )['Adj Close']  # 調整済み終値を使用
            
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
            
            time.sleep(1)  # APIレート制限対策
        except Exception as e:
            print(f"エラー - クラスタリングデータバッチ {i}: {str(e)}")
    
    # 欠損値処理
    log_returns_df.fillna(method='ffill', inplace=True)
    log_returns_df.fillna(0, inplace=True)  # 残りの欠損値を0で埋める
    
    return log_returns_df

def calculate_risk_metrics(returns_df):
    """リターンデータからリスク指標を計算
    
    Args:
        returns_df: 日次リターンのデータフレーム
        
    Returns:
        pd.DataFrame: 各ETFのリスク指標のデータフレーム
    """
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

def perform_clustering(etfs, perplexity=30, n_iter=1000, min_samples=2, xi=0.05, min_cluster_size=2):
    """TSNEとOPTICSを組み合わせたETFクラスタリング
    
    Args:
        etfs: ETF情報のリスト
        perplexity: TSNEのパーブレキシティパラメータ
        n_iter: TSNEの反復回数
        min_samples: OPTICSの最小サンプル数
        xi: OPTICSのクラスタ境界パラメータ
        min_cluster_size: 最小クラスタサイズ
        
    Returns:
        list: 選択されたETFのリスト (各クラスタから代表銘柄)
    """
    # キャッシュから取得を試みる
    symbols = [etf['symbol'] for etf in etfs]
    cache_key = f"tsne_optics_clustering_{len(symbols)}_{perplexity}_{min_samples}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュからクラスタリング結果を取得しました")
        return cached_data
    
    print("TSNE + OPTICSアルゴリズムによるETF分類を開始します...")
    
    # リターンデータの取得
    returns_df = get_returns_data(symbols)
    
    # リスク指標の計算
    risk_metrics = calculate_risk_metrics(returns_df)
    
    # 処理するデータがあるか確認
    if returns_df.empty or returns_df.shape[1] < 2:
        print("エラー: 有効なリターンデータが不足しています")
        return etfs  # 処理できないため元のETFリストを返す
    
    # 1. 相関行列の計算
    corr_matrix = returns_df.corr()
    
    # 2. 次元削減のためのTSNE
    # まずリターンデータを標準化
    scaler = StandardScaler()
    # リターンデータの転置（ETF間の距離を計算するため）
    scaled_returns = scaler.fit_transform(returns_df.T)
    
    print("TSNEによる次元削減を実行中...")
    # TSNEで2次元に圧縮
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        perplexity=min(perplexity, len(symbols) - 1),  # perplexityはデータ点数より小さくする必要がある
        n_iter=n_iter,
        n_jobs=-1,  # 並列処理
        init='pca'  # 初期化にPCAを使用（計算効率向上）
    )
    tsne_results = tsne.fit_transform(scaled_returns)
    
    # 3. OPTICSによるクラスタリング
    print("OPTICSによるクラスタリングを実行中...")
    
    # パラメータ調整: min_samples は全データの10%程度を目安とするが、少なくとも2
    min_samples_value = max(min_samples, min(2, len(symbols) // 10))
    
    # min_cluster_sizeが小数の場合は割合として扱う
    if isinstance(min_cluster_size, float) and min_cluster_size < 1:
        min_cluster_size = max(2, int(len(symbols) * min_cluster_size))
    
    # OPTICSクラスタリングの実行
    optics = OPTICS(
        min_samples=min_samples_value,
        xi=xi,
        min_cluster_size=min_cluster_size,
        n_jobs=-1  # 並列処理
    )
    cluster_labels = optics.fit_predict(tsne_results)
    
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
            tsne_results[class_mask, 0],
            tsne_results[class_mask, 1],
            c=[col],
            label=f'クラスタ {k}',
            s=60,
            alpha=0.8
        )
    
    # ノイズポイント(クラスタ = -1)をグレーで表示
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        plt.scatter(
            tsne_results[noise_mask, 0],
            tsne_results[noise_mask, 1],
            c='lightgray',
            label='ノイズ',
            s=60,
            alpha=0.6
        )
    
    # 銘柄名のアノテーション
    for i, symbol in enumerate(symbols):
        plt.annotate(
            symbol,
            (tsne_results[i, 0], tsne_results[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title('ETFクラスタマップ (TSNE + OPTICS)')
    plt.xlabel('TSNE次元1')
    plt.ylabel('TSNE次元2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
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
        etf_info = next((e for e in etfs if e['symbol'] == symbol), None)
        
        if etf_info:
            # ETFの情報を拡張
            etf_info['cluster'] = int(label)
            etf_info['tsne_x'] = float(tsne_results[i, 0])
            etf_info['tsne_y'] = float(tsne_results[i, 1])
            
            # リスク指標を追加
            if symbol in risk_metrics.index:
                etf_info['sharpe_ratio'] = float(risk_metrics.loc[symbol, 'sharpe_ratio'])
                etf_info['volatility'] = float(risk_metrics.loc[symbol, 'volatility'])
                etf_info['max_drawdown'] = float(risk_metrics.loc[symbol, 'max_drawdown'])
            
            clusters[label].append(etf_info)
    
    # 各クラスタから代表ETFを選出（クラスタ中心に最も近いETF）
    selected_etfs = []
    
    print("\nクラスタ分析結果:")
    
    for label, cluster_etfs in clusters.items():
        if cluster_etfs:
            # ノイズ(-1)の場合は特別に表示
            cluster_name = "ノイズ" if label == -1 else f"クラスタ {label}"
            print(f"{cluster_name}: {len(cluster_etfs)}銘柄")
            
            if label == -1:
                # ノイズクラスタからはシャープレシオが最も高いETFを選出
                sorted_etfs = sorted(cluster_etfs, 
                                     key=lambda x: x.get('sharpe_ratio', 0), 
                                     reverse=True)
            else:
                # 通常クラスタではクラスタの中心に最も近いETFを選出
                
                # クラスタの中心を計算
                center_x = np.mean([e['tsne_x'] for e in cluster_etfs])
                center_y = np.mean([e['tsne_y'] for e in cluster_etfs])
                
                # 中心からの距離を計算
                for etf in cluster_etfs:
                    dx = etf['tsne_x'] - center_x
                    dy = etf['tsne_y'] - center_y
                    etf['center_distance'] = np.sqrt(dx*dx + dy*dy)
                
                # 中心に最も近いETFを選出（ただし流動性も考慮）
                # 距離が近い上位3つからシャープレシオと流動性のバランスで選択
                sorted_by_distance = sorted(cluster_etfs, key=lambda x: x.get('center_distance', float('inf')))
                top_candidates = sorted_by_distance[:min(3, len(sorted_by_distance))]
                
                # 流動性スコアとシャープレシオのバランスで最終選択
                def selection_score(etf):
                    volume = etf.get('avg_volume', 0)
                    aum = etf.get('aum', 0)
                    sharpe = etf.get('sharpe_ratio', 0)
                    
                    # 流動性スコア
                    liquidity = volume * aum if volume and aum else 0
                    log_liquidity = np.log1p(liquidity) if liquidity > 0 else 0
                    
                    # 総合スコア（流動性70%、シャープレシオ30%）
                    return 0.7 * log_liquidity + 0.3 * (sharpe if sharpe > 0 else 0)
                
                sorted_etfs = sorted(top_candidates, key=selection_score, reverse=True)
            
            # 最終的な代表ETF選出
            if sorted_etfs:
                selected_etf = sorted_etfs[0]
                selected_etfs.append(selected_etf)
                
                print(f"  代表銘柄: {selected_etf['symbol']} ({selected_etf['name']})")
                print(f"  同グループ: {[e['symbol'] for e in sorted_etfs[1:min(len(sorted_etfs), 6)]]}" + 
                      ("..." if len(sorted_etfs) > 6 else ""))
    
    # 結果の詳細をCSVに保存
    all_etfs_with_clusters = []
    for label, cluster_etfs in clusters.items():
        all_etfs_with_clusters.extend(cluster_etfs)
    
    clusters_df = pd.DataFrame(all_etfs_with_clusters)
    clusters_csv_path = os.path.join(results_dir, "etf_clusters_details.csv")
    clusters_df.to_csv(clusters_csv_path, index=False)
    print(f"クラスタ詳細を保存しました: {clusters_csv_path}")
    
    # 分析サマリーを表示
    print("\nクラスタリング結果サマリー:")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for label, count in cluster_counts.items():
        cluster_name = "ノイズ" if label == -1 else f"クラスタ {label}"
        print(f"- {cluster_name}: {count}銘柄")
    
    # キャッシュに保存
    cache.set_json(cache_key, selected_etfs)
    
    return selected_etfs

def tda_clustering(etfs, *args, **kwargs):
    """後方互換性のための関数 - 実際にはTSNE+OPTICSを使用
    
    Args:
        etfs: ETF情報のリスト
        *args, **kwargs: perform_clustering関数に渡す追加パラメータ
        
    Returns:
        list: 選択されたETFのリスト
    """
    print("注意: tda_clustering関数は後方互換性のために維持されていますが、内部ではTSNE+OPTICSを使用しています")
    return perform_clustering(etfs, *args, **kwargs)
