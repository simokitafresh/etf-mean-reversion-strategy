"""TDA+OPTICSクラスタリング実装"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import os
import warnings
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

# 同一パッケージ内の基底クラスを相対インポートで参照
from .base_clusterer import BaseClusterer

# ロガー設定
logger = logging.getLogger(__name__)

# 再現性のためのランダムシード
RANDOM_SEED = 42

# TDAライブラリの情報を格納するための変数
tda_package_info = {
    'available': False,     # TDAライブラリが利用可能かどうか
    'package_name': None,   # 使用しているパッケージ名
    'version': None,        # パッケージのバージョン
    'mapper_class': None,   # 使用するMapperクラス
    'import_error': None    # インポートエラーがあれば保存
}

def setup_tda_environment():
    """TDAライブラリをインポートし、環境をセットアップする
    
    この関数は、利用可能なTDAライブラリをインポートし、必要なクラスや関数を設定します。
    優先順位は sklearn_tda > gtda > なし です。
    
    Returns:
        bool: TDAライブラリが正常にセットアップされたかどうか
    """
    global tda_package_info
    
    # 1. sklearn_tda をインポート試行（最優先）
    try:
        from sklearn_tda import Mapper
        from sklearn_tda.preprocessing import Scaler
        from sklearn_tda.plot import plot_diagram
        
        # バージョン情報の取得
        try:
            import sklearn_tda
            version = getattr(sklearn_tda, '__version__', 'unknown')
        except:
            version = 'unknown'
        
        # パッケージ情報を設定
        tda_package_info.update({
            'available': True,
            'package_name': 'sklearn_tda',
            'version': version,
            'mapper_class': Mapper
        })
        
        logger.info(f"sklearn_tda (バージョン: {version}) を使用します。")
        return True
    except ImportError as e:
        # エラー情報を保存
        tda_package_info['import_error'] = str(e)
        logger.debug(f"sklearn_tda のインポートに失敗しました: {str(e)}")
    
    # 2. gtda をインポート試行（次の優先順位）
    try:
        from gtda.mapper import Mapper
        from gtda.mapper import CubicalCover, make_mapper_pipeline
        
        # バージョン情報の取得
        try:
            import gtda
            version = getattr(gtda, '__version__', 'unknown')
        except:
            version = 'unknown'
        
        # パッケージ情報を設定
        tda_package_info.update({
            'available': True,
            'package_name': 'gtda',
            'version': version,
            'mapper_class': Mapper
        })
        
        logger.info(f"gtda (バージョン: {version}) を使用します。")
        return True
    except ImportError as e:
        # すでに保存されているエラーに追加
        prev_error = tda_package_info.get('import_error', '')
        tda_package_info['import_error'] = f"{prev_error}; gtda: {str(e)}"
        logger.debug(f"gtda のインポートに失敗しました: {str(e)}")
    
    # どちらのライブラリもインポートできなかった場合
    tda_package_info.update({
        'available': False,
        'package_name': None,
        'version': None,
        'mapper_class': None
    })
    
    warning_msg = "TDAライブラリ (sklearn_tda または gtda) がインストールされていません。"
    warnings.warn(warning_msg)
    logger.warning(f"{warning_msg} フォールバックのクラスタリング方法を使用します。")
    return False

# 起動時にTDA環境をセットアップ
TDA_AVAILABLE = setup_tda_environment()

class TDAOPTICSClusterer(BaseClusterer):
    """TDAとOPTICSを組み合わせたクラスタリング"""
    
    def __init__(self, **kwargs):
        """初期化"""
        super().__init__(name="tda_optics", **kwargs)
        
        # TDA使用可能かチェックし、警告を表示
        if not TDA_AVAILABLE:
            warnings.warn(
                f"TDAライブラリがインストールされていないため、OPTICSのみが使用されます。"
                f"エラー: {tda_package_info.get('import_error')}"
            )
        else:
            logger.info(f"TDA+OPTICSクラスタリングを使用します。TDAパッケージ: {tda_package_info['package_name']} "
                      f"(バージョン: {tda_package_info['version']})")
        
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
        if not TDA_AVAILABLE:
            logger.warning("TDAライブラリが利用できないため、OPTICSのみを使用します")
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
            # TDAライブラリに基づいたマッパー処理を実行
            if tda_package_info['package_name'] == 'sklearn_tda':
                return self._perform_sklearn_tda_mapping(scaled_returns, valid_symbols)
            elif tda_package_info['package_name'] == 'gtda':
                return self._perform_gtda_mapping(scaled_returns, valid_symbols)
            else:
                raise ImportError("利用可能なTDAライブラリがありません")
                
        except Exception as e:
            print(f"TDAマッパー構築エラー: {str(e)}。代替アプローチを使用します。")
            return self._perform_alternative_mapping(scaled_returns, valid_symbols)
    
    def _perform_sklearn_tda_mapping(self, scaled_returns: np.ndarray, valid_symbols: List[str]) -> Tuple[List[List[str]], Dict]:
        """sklearn_tdaを使用したマッピング処理
        
        Args:
            scaled_returns: スケーリングされたリターンデータ
            valid_symbols: 有効なETFシンボルのリスト
            
        Returns:
            Tuple[List[List[str]], Dict]: クラスタのリストとMapperグラフ
        """
        from sklearn_tda import Mapper
        
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
        
        print(f"Mapperグラフを構築しました: {len(graph['nodes'])}ノード, {len(graph['links'])}リンク")
        
        # グラフからクラスタを抽出
        return self._extract_clusters_from_graph(graph, valid_symbols)
    
    def _perform_gtda_mapping(self, scaled_returns: np.ndarray, valid_symbols: List[str]) -> Tuple[List[List[str]], Dict]:
        """gtdaを使用したマッピング処理
        
        Args:
            scaled_returns: スケーリングされたリターンデータ
            valid_symbols: 有効なETFシンボルのリスト
            
        Returns:
            Tuple[List[List[str]], Dict]: クラスタのリストとMapperグラフ
        """
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
        
        # グラフからクラスタを抽出
        return self._extract_clusters_from_graph(graph, valid_symbols)
    
    def _perform_alternative_mapping(self, scaled_returns: np.ndarray, valid_symbols: List[str]) -> Tuple[List[List[str]], Dict]:
        """代替のマッピング処理（TDAライブラリが利用できない場合）
        
        Args:
            scaled_returns: スケーリングされたリターンデータ
            valid_symbols: 有効なETFシンボルのリスト
            
        Returns:
            Tuple[List[List[str]], Dict]: クラスタのリストとMapperグラフ
        """
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
        
        # グラフからクラスタを抽出
        return self._extract_clusters_from_graph(graph, valid_symbols)
    
    def _extract_clusters_from_graph(self, graph: Dict, valid_symbols: List[str]) -> Tuple[List[List[str]], Dict]:
        """マッパーグラフからクラスタを抽出する
        
        Args:
            graph: マッパーグラフ
            valid_symbols: 有効なETFシンボルのリスト
            
        Returns:
            Tuple[List[List[str]], Dict]: クラスタのリストとグラフ
        """
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
            # TDAライブラリに基づいた可視化
            if tda_package_info['package_name'] == 'sklearn_tda':
                self._visualize_sklearn_tda_results(graph)
            elif tda_package_info['package_name'] == 'gtda':
                self._visualize_gtda_results(graph)
            else:
                self._visualize_fallback_results(graph)
                
        except Exception as e:
            print(f"TDA可視化エラー: {str(e)}")
            # フォールバックの可視化
            self._visualize_fallback_results(graph)
    
    def _visualize_sklearn_tda_results(self, graph):
        """sklearn_tdaを使用した可視化
        
        Args:
            graph: マッパーグラフ
        """
        try:
            from sklearn_tda.visualization import plot_mapper_graph
            
            # Mapper可視化をファイルに保存
            mapper_path = os.path.join(self.results_dir, "etf_tda_mapper_graph.png")
            
            plt.figure(figsize=(12, 10))
            plot_mapper_graph(graph, node_size=50)
            plt.title("ETF TDA Mapper Graph (sklearn_tda)")
            plt.savefig(mapper_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Mapper可視化を保存しました: {mapper_path}")
        except Exception as e:
            logger.warning(f"sklearn_tda可視化エラー: {str(e)}")
            # フォールバックを使用
            self._visualize_fallback_results(graph)
    
    def _visualize_gtda_results(self, graph):
        """gtdaを使用した可視化
        
        Args:
            graph: マッパーグラフ
        """
        try:
            from gtda.plotting import plot_static_mapper_graph
            
            # Mapper可視化をファイルに保存
            mapper_path = os.path.join(self.results_dir, "etf_tda_mapper_graph.png")
            
            plt.figure(figsize=(12, 10))
            plot_static_mapper_graph(graph)
            plt.title("ETF TDA Mapper Graph (gtda)")
            plt.savefig(mapper_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Mapper可視化を保存しました: {mapper_path}")
        except Exception as e:
            logger.warning(f"gtda可視化エラー: {str(e)}")
            # フォールバックを使用
            self._visualize_fallback_results(graph)
    
    def _visualize_fallback_results(self, graph):
        """フォールバックの可視化
        
        Args:
            graph: マッパーグラフ
        """
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
        plt.title("ETF TDA Mapper Graph (フォールバック)")
        plt.savefig(mapper_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mapper可視化を保存しました: {mapper_path}")
    
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
