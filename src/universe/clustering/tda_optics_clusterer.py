"""TDA+OPTICSクラスタリング実装"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import os
import warnings
from typing import List, Dict, Any, Tuple

# 同一パッケージ内の基底クラスを相対インポートで参照
from .base_clusterer import BaseClusterer

# 再現性のためのランダムシード
RANDOM_SEED = 42

# Scikit-TDAパッケージチェック
try:
    from sklearn_tda import Mapper
    from sklearn_tda.preprocessing import Scaler
    from sklearn_tda.plot import plot_diagram
    SKLEARN_TDA_AVAILABLE = True
except ImportError:
    try:
        # 代替インポート（scikit-tdaの最新バージョン）
        from gtda.mapper import Mapper
        from gtda.mapper import plot_static_mapper_graph
        SKLEARN_TDA_AVAILABLE = True
    except ImportError:
        # fallback
        SKLEARN_TDA_AVAILABLE = False
        warnings.warn("Scikit-TDAがインストールされていません。従来のクラスタリング方法を使用します。")

class TDAOPTICSClusterer(BaseClusterer):
    """TDAとOPTICSを組み合わせたクラスタリング"""
    
    def __init__(self, **kwargs):
        """初期化"""
        super().__init__(name="tda_optics", **kwargs)
        
        # TDA用パラメータ
        self.tda_params = {
            "resolution": kwargs.get("resolution", 10),
            "overlap": kwargs.get("overlap", 0.5),
            "lens_type": kwargs.get("lens_type", "knn_distance_10")
        }
        
        # OPTICS用パラメータ
        self.optics_params = {
            "min_samples": kwargs.get("min_samples", 2),
            "xi": kwargs.get("xi", 0.05),
            "min_cluster_size": kwargs.get("min_cluster_size", 2),
            "n_jobs": kwargs.get("n_jobs", -1)
        }
    
    def perform_clustering(self, etfs: List[Dict[str, Any]], returns_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """TDAとOPTICSを組み合わせたクラスタリングを実行
        
        Args:
            etfs: ETF情報のリスト
            returns_df: リターンデータのデータフレーム
            
        Returns:
            List[Dict[str, Any]]: 選択されたETFのリスト
        """
        if not SKLEARN_TDA_AVAILABLE:
            warnings.warn("Scikit-TDAが利用できないため、OPTICSのみを使用します")
            # OPTICSのみを使用するフォールバック
            return self._fallback_clustering(etfs, returns_df)
        
        print("TDA+OPTICSハイブリッドクラスタリングを実行中...")
        
        # 対象シンボルとリターンデータの整合性確認
        symbols = [etf['symbol'] for etf in etfs]
        valid_symbols = [s for s in symbols if s in returns_df.columns]
        
        if len(valid_symbols) < len(symbols):
            print(f"警告: {len(symbols) - len(valid_symbols)}銘柄のデータが欠落しています")
        
        if len(valid_symbols) < 2:
            print("エラー: クラスタリングに十分なデータがありません")
            return etfs
        
        # 有効なシンボルのリターンデータを取得
        valid_returns = returns_df[valid_symbols]
        
        # 1. TDAクラスタリング - マクロ構造の把握
        tda_clusters, tda_graph = self._perform_tda_clustering(valid_returns.values, valid_symbols)
        
        # 2. TDA結果の可視化（オプション）
        try:
            self._visualize_tda_results(tda_graph, etfs, valid_symbols)
        except Exception as e:
            print(f"TDA可視化エラー: {str(e)}")
        
        # 3. 各TDAクラスタ内でOPTICSによる細分化
        refined_clusters = []
        cluster_labels = np.full(len(valid_symbols), -1)  # デフォルトはノイズ
        
        next_cluster_id = 0
        
        for cluster_idx, tda_cluster in enumerate(tda_clusters):
            # クラスタ内のインデックスを取得
            cluster_indices = [valid_symbols.index(symbol) for symbol in tda_cluster 
                              if symbol in valid_symbols]
            
            if len(cluster_indices) < 2:
                # 1つしかない場合は別クラスタとして追加
                for idx in cluster_indices:
                    cluster_labels[idx] = next_cluster_id
                    next_cluster_id += 1
                continue
            
            # クラスタのリターンデータを取得
            cluster_returns = valid_returns.iloc[:, cluster_indices]
            
            # OPTICSクラスタリングを適用
            optics_labels = self._perform_optics_clustering(cluster_returns.values.T)
            
            # クラスタラベルを更新
            unique_optics_labels = np.unique(optics_labels)
            for optics_label in unique_optics_labels:
                if optics_label == -1:  # ノイズは-1のまま
                    continue
                
                # 新しいクラスタIDを割り当て
                sub_indices = [cluster_indices[i] for i, label in enumerate(optics_labels) 
                              if label == optics_label]
                
                for idx in sub_indices:
                    cluster_labels[idx] = next_cluster_id
                
                # サブクラスタの情報を追加
                sub_cluster = [valid_symbols[idx] for idx in sub_indices]
                refined_clusters.append(sub_cluster)
                
                next_cluster_id += 1
        
        # 4. 結果の2D埋め込みと可視化
        embedding = self._create_2d_embedding(valid_returns.values.T)
        self.visualize_clusters(
            [etf for etf in etfs if etf['symbol'] in valid_symbols],
            cluster_labels,
            embedding,
            title=f'TDA+OPTICS ハイブリッドクラスタリング (クラスタ数: {next_cluster_id})'
        )
        
        # 5. クラスタラベルをETF情報に追加
        etf_clusters = []
        for i, etf in enumerate(etfs):
            symbol = etf['symbol']
            if symbol in valid_symbols:
                idx = valid_symbols.index(symbol)
                etf_info = etf.copy()
                etf_info['cluster'] = int(cluster_labels[idx])
                etf_clusters.append(etf_info)
            else:
                # データのないETFはノイズクラスタに
                etf_info = etf.copy()
                etf_info['cluster'] = -1
                etf_clusters.append(etf_info)
        
        # 6. 各クラスタから代表ETFを選択
        selected_etfs = self.select_representative_etfs(etf_clusters, valid_returns)
        
        print(f"TDA+OPTICSクラスタリング完了: {len(selected_etfs)}銘柄を選択")
        return selected_etfs
    
    def _perform_tda_clustering(self, returns_data: np.ndarray, valid_symbols: List[str]) -> Tuple[List[List[str]], Dict]:
        """TDAクラスタリングを実行
        
        Args:
            returns_data: リターンデータの配列
            valid_symbols: 有効なETFシンボルのリスト
            
        Returns:
            Tuple[List[List[str]], Dict]: クラスタのリストとMapperグラフ
        """
        print("TDA (Mapper)クラスタリングを実行中...")
        
        # データの標準化
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_data)
        
        try:
            # scikit-tda方式でのマッパー作成
            if 'sklearn_tda' in globals():
                # Mapperオブジェクトの作成
                mapper = Mapper(
                    verbose=0,
                    coverer="uniform", 
                    n_cubes=self.tda_params["resolution"],
                    overlap=self.tda_params["overlap"]
                )
                
                # フィルター（レンズ）関数の適用
                # KNN距離と同等の機能を使用
                if self.tda_params["lens_type"] == "knn_distance_10":
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=10)
                    nn.fit(scaled_returns)
                    lens = nn.kneighbors(scaled_returns, return_distance=True)[0].mean(axis=1).reshape(-1, 1)
                else:
                    # デフォルトはデータそのものを使用
                    lens = scaled_returns
                
                # Mapperグラフを構築
                graph = mapper.fit_transform(
                    lens,
                    scaled_returns,
                    clusterer=OPTICS(**self.optics_params)
                )
            else:
                # gtda方式でのマッパー作成
                from gtda.mapper import CubicalCover, make_mapper_pipeline
                from gtda.mapper import Projection, Eccentricity
                
                # カバーとクラスタリングの設定
                cover = CubicalCover(
                    n_intervals=self.tda_params["resolution"],
                    overlap_frac=self.tda_params["overlap"]
                )
                
                # レンズ関数の設定
                if self.tda_params["lens_type"] == "knn_distance_10":
                    lens = Eccentricity(n_neighbors=10)
                else:
                    lens = Projection(columns=[0, 1])
                
                # マッパーパイプラインの構築
                mapper = make_mapper_pipeline(
                    lens=lens,
                    cover=cover,
                    clusterer=OPTICS(**self.optics_params),
                    verbose=True
                )
                
                # マッパーグラフの構築
                graph = mapper.fit_transform(scaled_returns)
            
            print(f"Mapperグラフを構築しました: {len(graph['nodes'])}ノード, {len(graph['links'])}リンク")
        except Exception as e:
            print(f"TDAマッパー構築エラー: {str(e)}。代替アプローチを使用します。")
            # 代替アプローチ：scikit-learn のみで簡易TDAを実装
            from sklearn.neighbors import kneighbors_graph
            import networkx as nx
            
            # k-近傍グラフを構築
            k = min(10, len(scaled_returns) - 1)  # データポイント数以下のkを選択
            connectivity = kneighbors_graph(scaled_returns, n_neighbors=k, mode='distance')
            
            # NetworkXグラフに変換
            G = nx.from_scipy_sparse_matrix(connectivity)
            
            # ノードをクラスタリング
            optics = OPTICS(**self.optics_params)
            labels = optics.fit_predict(scaled_returns)
            
            # グラフ構造を作成
            graph = {
                'nodes': {},
                'links': []
            }
            
            # ノード情報を追加
            for i, label in enumerate(labels):
                cluster_id = int(label) if label != -1 else -1
                if cluster_id not in graph['nodes']:
                    graph['nodes'][cluster_id] = []
                graph['nodes'][cluster_id].append(i)
            
            # エッジ情報を追加
            for edge in G.edges():
                source, target = edge
                source_cluster = int(labels[source]) if labels[source] != -1 else -1
                target_cluster = int(labels[target]) if labels[target] != -1 else -1
                
                if source_cluster != -1 and target_cluster != -1 and source_cluster != target_cluster:
                    graph['links'].append((source_cluster, target_cluster))
            
            print(f"代替マッパーグラフを構築しました: {len(graph['nodes'])}ノード, {len(graph['links'])}リンク")
        
        # クラスタの抽出（接続成分をクラスタとして使用）
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
            cluster_indices = set()
            for node_id in component:
                cluster_indices.update(graph['nodes'][node_id])
            
            cluster_symbols = [valid_symbols[idx] for idx in cluster_indices 
                              if idx < len(valid_symbols)]
            
            if cluster_symbols:  # 空でない場合のみ追加
                clusters.append(cluster_symbols)
        
        return clusters, graph
    
    def _perform_optics_clustering(self, data: np.ndarray) -> np.ndarray:
        """OPTICSクラスタリングを実行
        
        Args:
            data: クラスタリング対象データ
            
        Returns:
            np.ndarray: クラスタラベル
        """
        # データの標準化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # OPTICSクラスタリング
        optics = OPTICS(**self.optics_params)
        labels = optics.fit_predict(scaled_data)
        
        return labels
    
    def _create_2d_embedding(self, data: np.ndarray) -> np.ndarray:
        """2次元埋め込みを作成
        
        Args:
            data: 埋め込み対象データ
            
        Returns:
            np.ndarray: 2次元埋め込み
        """
        from sklearn.decomposition import PCA
        
        # 2次元に削減
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        embedding = pca.fit_transform(data)
        
        return embedding
    
    def _visualize_tda_results(self, graph, etfs, symbols):
        """TDAクラスタリング結果の可視化
        
        Args:
            graph: KeplerMapperグラフ
            etfs: ETF情報のリスト
            symbols: ETFシンボルのリスト
        """
        try:
            # 可視化方法はscikit-tdaの実装によって異なる
            if 'sklearn_tda' in globals():
                from sklearn_tda.visualization import plot_mapper_graph
                
                # Mapper可視化をファイルに保存
                mapper_path = os.path.join(self.results_dir, "etf_tda_mapper_graph.png")
                
                plt.figure(figsize=(12, 10))
                plot_mapper_graph(graph, node_size=50)
                plt.title("ETF TDA Mapper Graph")
                plt.savefig(mapper_path, dpi=300, bbox_inches='tight')
                plt.close()
            elif 'gtda' in globals():
                from gtda.plotting import plot_static_mapper_graph
                
                # Mapper可視化をファイルに保存
                mapper_path = os.path.join(self.results_dir, "etf_tda_mapper_graph.png")
                
                plt.figure(figsize=(12, 10))
                plot_static_mapper_graph(graph)
                plt.title("ETF TDA Mapper Graph")
                plt.savefig(mapper_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # NetworkXを使った基本的な可視化
                import networkx as nx
                
                # グラフ構造をNetworkXに変換
                G = nx.Graph()
                
                # ノードを追加
                for node_id, indices in graph['nodes'].items():
                    G.add_node(node_id, size=len(indices))
                
                # エッジを追加
                for source, target in graph['links']:
                    G.add_edge(source, target)
                
                # 可視化
                mapper_path = os.path.join(self.results_dir, "etf_tda_mapper_graph.png")
                
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(G, seed=RANDOM_SEED)
                node_sizes = [G.nodes[n].get('size', 10) * 20 for n in G.nodes()]
                nx.draw(G, pos, with_labels=True, node_size=node_sizes, 
                        node_color='skyblue', font_size=8)
                plt.title("ETF TDA Mapper Graph")
                plt.savefig(mapper_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Mapper可視化を保存しました: {mapper_path}")
        except Exception as e:
            print(f"TDA可視化エラー: {str(e)}")
    
    def _fallback_clustering(self, etfs, returns_df):
        """Scikit-TDAが利用できない場合のフォールバック
        
        Args:
            etfs: ETF情報のリスト
            returns_df: リターンデータのデータフレーム
            
        Returns:
            List[Dict[str, Any]]: 選択されたETFのリスト
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import OPTICS
        
        print("フォールバック: OPTICSクラスタリングを実行中...")
        
        # 対象シンボルとリターンデータの整合性確認
        symbols = [etf['symbol'] for etf in etfs]
        valid_symbols = [s for s in symbols if s in returns_df.columns]
        
        if len(valid_symbols) < 2:
            print("エラー: クラスタリングに十分なデータがありません")
            return etfs
        
        # 有効なシンボルのリターンデータを取得
        valid_returns = returns_df[valid_symbols]
        
        # データの標準化
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(valid_returns.values.T)
        
        # OPTICSクラスタリング
        optics = OPTICS(**self.optics_params)
        cluster_labels = optics.fit_predict(scaled_returns)
        
        # 2次元埋め込み
        embedding = self._create_2d_embedding(scaled_returns)
        
        # 結果の可視化
        self.visualize_clusters(
            [etf for etf in etfs if etf['symbol'] in valid_symbols],
            cluster_labels,
            embedding,
            title='OPTICSクラスタリング (TDAフォールバック)'
        )
        
        # クラスタラベルをETF情報に追加
        etf_clusters = []
        for i, etf in enumerate(etfs):
            symbol = etf['symbol']
            if symbol in valid_symbols:
                idx = valid_symbols.index(symbol)
                etf_info = etf.copy()
                etf_info['cluster'] = int(cluster_labels[idx])
                etf_clusters.append(etf_info)
            else:
                # データのないETFはノイズクラスタに
                etf_info = etf.copy()
                etf_info['cluster'] = -1
                etf_clusters.append(etf_info)
        
        # 各クラスタから代表ETFを選択
        selected_etfs = self.select_representative_etfs(etf_clusters, valid_returns)
        
        print(f"フォールバックOPTICSクラスタリング完了: {len(selected_etfs)}銘柄を選択")
        return selected_etfs
