# src/universe/clustering.py
"""ETFクラスタリングモジュール - TDAとOPTICSを使用"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
import os
import time
import yfinance as yf
from ..data.cache import DataCache
import warnings

# 再現性のためのランダムシード
RANDOM_SEED = 42

# キャッシュのインスタンス化
cache = DataCache()

# インストール済みのScikit-TDAライブラリを使用
try:
    from kmapper import KeplerMapper
    from ripser import Rips
    from persim import plot_diagrams
    SKLEARN_TDA_AVAILABLE = True
except ImportError:
    SKLEARN_TDA_AVAILABLE = False
    warnings.warn("Scikit-TDAがインストールされていません。従来のクラスタリング方法を使用します。")


def get_returns_data(symbols, period="1y"):
    """銘柄リストから日次リターンデータを取得
    
    Args:
        symbols: ETFシンボルのリスト
        period: データ取得期間（例: "1y", "2y", "max"）
        
    Returns:
        pd.DataFrame: 日次対数リターンデータを含むデータフレーム
    """
    # キャッシュから取得を試みる
    cache_key = f"returns_data_{'-'.join(symbols)}_{period}"
    cached_data = cache.get_json(cache_key)
    if cached_data is not None:
        print("キャッシュからリターンデータを取得しました")
        return pd.read_json(cached_data, orient='split')
    
    print(f"{len(symbols)}銘柄の価格データを取得中...")
    
    prices_df = pd.DataFrame()
    batch_size = 5  # APIレート制限対策
    
    # バッチ処理
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        print(f"バッチ取得中: {batch_symbols}")
        
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
            print(f"警告: バッチ{i}のデータ取得エラー: {str(e)}")
            time.sleep(5)  # エラー時は長めに待機
    
    # シンボルが1つも取得できなかった場合
    if prices_df.empty:
        print("エラー: 有効な価格データが取得できませんでした")
        return pd.DataFrame()
    
    # 欠損値処理
    prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
    
    # 欠損値が多すぎる列を削除（80%以上のデータがある列のみを保持）
    min_data_points = 0.8 * len(prices_df)
    prices_df = prices_df.loc[:, prices_df.count() >= min_data_points]
    
    if prices_df.empty:
        print("エラー: 欠損値処理後にデータがなくなりました")
        return pd.DataFrame()
    
    # 日次対数リターンの計算
    log_returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    
    # キャッシュに保存
    cache.set_json(cache_key, log_returns_df.to_json(orient='split'))
    
    print(f"リターンデータ取得完了: {log_returns_df.shape[0]}日 x {log_returns_df.shape[1]}銘柄")
    return log_returns_df


def calculate_risk_metrics(returns_df):
    """リターンデータからリスク指標を計算
    
    Args:
        returns_df: リターンデータのデータフレーム
        
    Returns:
        pd.DataFrame: リスク指標のデータフレーム
    """
    metrics = pd.DataFrame(index=returns_df.columns)
    
    # 平均リターン
    metrics['mean_return'] = returns_df.mean()
    
    # ボラティリティ
    metrics['volatility'] = returns_df.std()
    
    # シャープレシオ（リスクフリーレート0と仮定）
    metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['volatility']
    
    # 歪度
    metrics['skewness'] = returns_df.skew()
    
    # 尖度
    metrics['kurtosis'] = returns_df.kurtosis()
    
    # 最大ドローダウン
    cum_returns = (1 + returns_df).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    metrics['max_drawdown'] = drawdown.min()
    
    return metrics


def perform_tda_clustering(returns_data, etfs, max_samples=100, resolution=10, 
                           overlap=0.5, clusterer_params=None):
    """TDAを使用したクラスタリングを実行
    
    Args:
        returns_data: リターンデータの配列
        etfs: ETF情報のリスト
        max_samples: 最大サンプル数
        resolution: Mapper解像度（区間数）
        overlap: 区間の重複率
        clusterer_params: OPTICSのパラメータ辞書
        
    Returns:
        list: クラスタリング結果と選択されたETF
    """
    if not SKLEARN_TDA_AVAILABLE:
        print("Scikit-TDAが利用できないため、従来のクラスタリング手法を使用します")
        return perform_clustering(etfs)
    
    # パラメータのデフォルト値を設定
    if clusterer_params is None:
        clusterer_params = {
            'min_samples': 2,
            'xi': 0.05,
            'min_cluster_size': 2,
            'n_jobs': -1
        }
    
    print("TDA (Mapper) + OPTICSクラスタリングを実行中...")
    
    # データの標準化
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_data)
    
    # KeplerMapper初期化
    mapper = KeplerMapper()
    
    # 投影（lens）関数を適用
    lens = mapper.fit_transform(scaled_returns, projection="knn_distance_10")
    
    # Mapper グラフを構築
    graph = mapper.map(
        lens, 
        scaled_returns,
        cover=[[0, 1, resolution, overlap]],  # [min, max, 区間数, 重複率]
        clusterer=OPTICS(**clusterer_params)
    )
    
    print(f"Mapperグラフを構築しました: {len(graph['nodes'])}ノード, {len(graph['links'])}リンク")
    
    # ETFシンボルのリスト
    symbols = [etf['symbol'] for etf in etfs]
    
    # クラスタの抽出（接続成分をクラスタとして使用）
    clusters = extract_clusters_from_graph(graph, symbols)
    print(f"グラフから{len(clusters)}個のクラスタを抽出しました")
    
    # クラスタとETFの関連付け
    etf_clusters = []
    
    for i, etf in enumerate(etfs):
        symbol = etf['symbol']
        # このETFが属するクラスタを検索
        cluster_id = -1  # デフォルトはノイズ
        
        for cluster_idx, cluster in enumerate(clusters):
            if symbol in cluster:
                cluster_id = cluster_idx
                break
        
        # ETF情報にクラスタIDを追加
        etf_info = etf.copy()
        etf_info['cluster'] = cluster_id
        etf_clusters.append(etf_info)
    
    # クラスタごとに代表ETFを選出
    selected_etfs = select_representative_etfs(etf_clusters, returns_data, symbols)
    print(f"各クラスタから代表ETFを選出しました: 合計{len(selected_etfs)}銘柄")
    
    # 結果の可視化
    try:
        visualize_tda_results(graph, etfs, symbols, selected_etfs)
    except Exception as e:
        print(f"可視化エラー: {str(e)}")
    
    return selected_etfs


def extract_clusters_from_graph(graph, symbols):
    """Mapperグラフからクラスタを抽出（接続成分を使用）
    
    Args:
        graph: KeplerMapperグラフ
        symbols: ETFシンボルのリスト
        
    Returns:
        list: クラスタのリスト（各クラスタはシンボルのリスト）
    """
    # ノードIDとETFインデックスのマッピング
    node_to_etfs = {}
    for node_id, etf_indices in graph['nodes'].items():
        node_to_etfs[node_id] = [symbols[idx] for idx in etf_indices if idx < len(symbols)]
    
    # 接続グラフを構築
    import networkx as nx
    G = nx.Graph()
    
    # ノードを追加
    for node_id in graph['nodes'].keys():
        G.add_node(node_id)
    
    # エッジを追加
    for link in graph['links']:
        source, target = link
        G.add_edge(source, target)
    
    # 接続成分（クラスタ）を取得
    connected_components = list(nx.connected_components(G))
    
    # クラスタ（ETFシンボルのリスト）を構築
    clusters = []
    for component in connected_components:
        cluster_etfs = set()
        for node_id in component:
            cluster_etfs.update(node_to_etfs.get(node_id, []))
        
        if cluster_etfs:  # 空でない場合のみ追加
            clusters.append(list(cluster_etfs))
    
    return clusters


def select_representative_etfs(etf_clusters, returns_data, symbols):
    """各クラスタから代表的なETFを選択
    
    Args:
        etf_clusters: クラスタIDが付与されたETF情報のリスト
        returns_data: リターンデータ
        symbols: ETFシンボルのリスト
        
    Returns:
        list: 選択されたETF情報のリスト
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
            cluster_returns = returns_data[cluster_symbols]
            
            # クラスタの平均リターンを計算
            cluster_mean = cluster_returns.mean()
            
            # 各ETFとクラスタ平均の相関を計算
            correlations = {}
            for symbol in cluster_symbols:
                if symbol in cluster_returns:
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
            print(f"クラスタ{cluster_id}の代表ETF選択エラー: {str(e)}")
        
        # エラーまたはデータが不十分な場合、出来高が最も多いETFを選択
        fallback_etf = max(cluster_etfs, key=lambda x: x.get('avg_volume', 0))
        selected_etfs.append(fallback_etf)
    
    return selected_etfs


def visualize_tda_results(graph, etfs, symbols, selected_etfs, output_dir="data/results"):
    """TDAクラスタリング結果を可視化
    
    Args:
        graph: KeplerMapperグラフ
        etfs: ETF情報のリスト
        symbols: ETFシンボルのリスト
        selected_etfs: 選択されたETF情報のリスト
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 選択されたETFのシンボル
    selected_symbols = [etf['symbol'] for etf in selected_etfs]
    
    # 選択されたフラグを作成（可視化で使用）
    color_function = []
    for symbol in symbols:
        if symbol in selected_symbols:
            color_function.append(1)  # 選択されたETF
        else:
            color_function.append(0)  # 選択されていないETF
    
    # Mapper可視化をファイルに保存
    mapper_path = os.path.join(output_dir, "etf_mapper_graph.html")
    
    from kmapper import visualization
    visualization.save_html(
        graph, 
        mapper_path, 
        custom_title="ETF Mapper Graph",
        custom_tooltips=symbols,
        color_function=color_function
    )
    print(f"Mapper可視化を保存しました: {mapper_path}")
    
    # 2D投影での可視化
    try:
        from sklearn.manifold import TSNE
        
        # TSNEを使って2次元に投影
        mapper_projection = []
        for node_id, members in graph['nodes'].items():
            for member_idx in members:
                if member_idx < len(symbols):
                    mapper_projection.append({
                        'node_id': node_id,
                        'symbol': symbols[member_idx],
                        'selected': symbols[member_idx] in selected_symbols
                    })
        
        # 可視化用データフレームを作成
        proj_df = pd.DataFrame(mapper_projection)
        
        # ユニークなノードID
        unique_nodes = proj_df['node_id'].unique()
        
        # ノードの中心座標を作成（TSNE使用）
        node_centers = {}
        for node_id in unique_nodes:
            node_members = proj_df[proj_df['node_id'] == node_id]['symbol'].tolist()
            node_centers[node_id] = {'x': np.random.rand(), 'y': np.random.rand(), 'members': node_members}
        
        # ノード中心の座標を用いて2D投影を作成
        plt.figure(figsize=(12, 10))
        
        # ノード（クラスタ）を描画
        for node_id, center in node_centers.items():
            # ノードメンバーに選択されたETFが含まれるかチェック
            has_selected = any(symbol in selected_symbols for symbol in center['members'])
            
            # ノードの色
            color = 'green' if has_selected else 'gray'
            alpha = 0.8 if has_selected else 0.4
            
            # ノードを描画
            plt.scatter(center['x'], center['y'], s=100, color=color, alpha=alpha, edgecolors='black')
            
            # ノードのメンバー数を表示
            plt.annotate(str(len(center['members'])), 
                         (center['x'], center['y']),
                         fontsize=8, ha='center', va='center')
        
        # リンク（エッジ）を描画
        for link in graph['links']:
            source, target = link
            if source in node_centers and target in node_centers:
                source_x = node_centers[source]['x']
                source_y = node_centers[source]['y']
                target_x = node_centers[target]['x']
                target_y = node_centers[target]['y']
                
                plt.plot([source_x, target_x], [source_y, target_y], 'gray', alpha=0.3)
        
        # 選択されたETFを強調表示
        for symbol in selected_symbols:
            # このETFを含むノードを検索
            for node_id, center in node_centers.items():
                if symbol in center['members']:
                    plt.annotate(symbol, 
                                 (center['x'], center['y']),
                                 xytext=(10, 5),
                                 textcoords='offset points',
                                 fontsize=9, color='blue', fontweight='bold')
                    break
        
        plt.title('TDA Mapper Graph of ETFs')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # 可視化を保存
        plt.savefig(os.path.join(output_dir, "etf_tda_clusters.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"TDAクラスタリング可視化を保存しました: {os.path.join(output_dir, 'etf_tda_clusters.png')}")
        
    except Exception as e:
        print(f"2D可視化エラー: {str(e)}")


def perform_clustering(etfs, method="optics", max_samples=100, perplexity=30, 
                      n_iter=1000, min_samples=2, xi=0.05, min_cluster_size=2):
    """従来のクラスタリング手法を使用するフォールバック関数
    
    Args:
        etfs: ETF情報のリスト
        method: クラスタリング手法 ("optics", "tsne_optics", "pca_optics")
        max_samples: 処理する最大サンプル数
        perplexity: TSNEのパーブレキシティパラメータ
        n_iter: TSNEの反復回数
        min_samples: OPTICSの最小サンプル数
        xi: OPTICSのクラスタ境界パラメータ
        min_cluster_size: 最小クラスタサイズ
        
    Returns:
        list: 選択されたETFのリスト
    """
    # キャッシュからの取得処理
    symbols = [etf['symbol'] for etf in etfs]
    cache_key = f"clustering_{method}_{len(symbols)}_{min_samples}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュからクラスタリング結果を取得しました")
        return cached_data
    
    print(f"{method}アルゴリズムによるETF分類を開始します...")
    
    # リターンデータの取得
    returns_df = get_returns_data(symbols)
    
    if returns_df.empty:
        print("リターンデータが取得できないため、元のETFリストを返します")
        return etfs
    
    # パフォーマンス向上のため、データ量が多い場合は制限する
    if len(returns_df.columns) > max_samples:
        print(f"パフォーマンス向上のため、ETF数を{max_samples}に制限します")
        # シャープレシオでソートして上位を選択
        risk_metrics = calculate_risk_metrics(returns_df)
        top_symbols = risk_metrics.sort_values('sharpe_ratio', ascending=False).index[:max_samples].tolist()
        returns_df = returns_df[top_symbols]
        # symbolsも更新
        symbols = top_symbols
        # etfsも更新
        etfs = [etf for etf in etfs if etf['symbol'] in symbols]
    
    # リスク指標の計算
    risk_metrics = calculate_risk_metrics(returns_df)
    
    # 処理するデータがあるか確認
    if returns_df.empty or returns_df.shape[1] < 2:
        print("エラー: 有効なリターンデータが不足しています")
        return etfs  # 処理できないため元のETFリストを返す
    
    # 1. 相関行列の計算
    corr_matrix = returns_df.corr()
    
    # 2. 次元削減とクラスタリング
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_df.T)
    
    # 選択した次元削減・クラスタリング手法を適用
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
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
    
    # クラスタリング結果の可視化
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
    
    # 各クラスタから代表銘柄を選出
    selected_etfs = []
    
    # クラスタごとに代表ETFを選択
    for cluster_id in np.unique(cluster_labels):
        # ノイズクラスタ(-1)も含める
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # このクラスタのETF情報
        cluster_etfs = [etfs[i] for i in cluster_indices if i < len(etfs)]
        
        if not cluster_etfs:
            continue
        
        # クラスタ内のETFシンボル
        cluster_symbols = [etf['symbol'] for etf in cluster_etfs]
        
        # クラスタ情報をETFに追加
        for etf in cluster_etfs:
            etf['cluster'] = int(cluster_id)
        
        # 代表ETFの選択基準
        if cluster_id == -1:  # ノイズクラスタの場合
            # 流動性（出来高）が最も高いETFを選択
            best_etf = max(cluster_etfs, key=lambda x: x.get('avg_volume', 0))
            selected_etfs.append(best_etf)
        else:
            try:
                # クラスタの中心に最も近いETFを選択
                cluster_points = reduced_data[cluster_indices]
                centroid = cluster_points.mean(axis=0)
                
                # 各点と中心の距離を計算
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                
                # 代表ETFを追加
                if closest_idx < len(etfs):
                    selected_etfs.append(etfs[closest_idx])
            except Exception as e:
                print(f"クラスタ{cluster_id}の代表ETF選択エラー: {str(e)}")
                # エラー時は出来高最大のETFをフォールバックとして選択
                if cluster_etfs:
                    best_etf = max(cluster_etfs, key=lambda x: x.get('avg_volume', 0))
                    selected_etfs.append(best_etf)
    
    print(f"クラスタリング完了: {len(np.unique(cluster_labels))}クラスタから{len(selected_etfs)}銘柄を選択")
    
    # キャッシュに保存
    cache.set_json(cache_key, selected_etfs)
    
    return selected_etfs


def tda_clustering(etfs, *args, **kwargs):
    """TDAクラスタリングを実行する関数
    
    Args:
        etfs: ETF情報のリスト
        *args, **kwargs: 追加パラメータ
        
    Returns:
        list: 選択されたETFのリスト
    """
    # キャッシュからの取得
    symbols = [etf['symbol'] for etf in etfs]
    cache_key = f"tda_clustering_{len(symbols)}"
    cached_data = cache.get_json(cache_key)
    if cached_data:
        print("キャッシュからTDAクラスタリング結果を取得しました")
        return cached_data
    
    print("TDAクラスタリングを開始します...")
    
    # リターンデータの取得
    returns_df = get_returns_data(symbols)
    
    if returns_df.empty:
        print("リターンデータが取得できないため、元のETFリストを返します")
        return etfs
    
    # Scikit-TDAが利用可能ならTDAクラスタリングを実行
    if SKLEARN_TDA_AVAILABLE:
        # KeplerMapperを使用したTDAクラスタリング
        selected_etfs = perform_tda_clustering(returns_df, etfs)
    else:
        # フォールバックとして従来のクラスタリングを使用
        print("Scikit-TDAがインストールされていないため、従来のクラスタリング手法にフォールバックします")
        selected_etfs = perform_clustering(etfs, method="pca_optics")
    
    # キャッシュに保存
    cache.set_json(cache_key, selected_etfs)
    
    return selected_etfs
