"""Google Colabとの連携設定 - 強化版"""
import os
import sys
import subprocess
import platform
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

# ロギング設定
logger = logging.getLogger("setup_colab")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def setup_colab(
    project_name: str = "etf-mean-reversion-strategy", 
    force_clone: bool = False, 
    use_sample_etfs: bool = True,
    log_to_file: bool = True,
    setup_local_env: bool = False
) -> Dict[str, Any]:
    """Colabでの実行環境をセットアップし、安定性を強化する
    
    Args:
        project_name: リポジトリ/プロジェクト名
        force_clone: 既存のプロジェクトがあっても強制的に再クローンするかどうか
        use_sample_etfs: サンプルETFセットを使用するかどうか
        log_to_file: ログをファイルに記録するかどうか
        setup_local_env: ローカル環境のセットアップも行うかどうか
        
    Returns:
        Dict[str, Any]: セットアップ情報の辞書
    """
    start_time = datetime.now()
    logger.info(f"Google Colab環境のセットアップを開始します - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 実行環境の検出
    is_colab = 'google.colab' in sys.modules
    is_jupyter = 'ipykernel' in sys.modules
    env_info = {
        'is_colab': is_colab,
        'is_jupyter': is_jupyter,
        'python_version': platform.python_version(),
        'system': platform.system(),
        'start_time': start_time,
    }
    
    logger.info(f"実行環境: {'Google Colab' if is_colab else 'Jupyter Notebook' if is_jupyter else 'その他'}")
    logger.info(f"Python バージョン: {env_info['python_version']}")
    logger.info(f"OS: {env_info['system']}")
    
    # ロギング設定（ファイル出力）
    if log_to_file:
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"ログを {log_file} に記録します")
        env_info['log_file'] = log_file
    
    # プロジェクトディレクトリのセットアップ
    env_info['project_dir'] = setup_project_directory(
        project_name=project_name,
        is_colab=is_colab,
        force_clone=force_clone
    )
    
    # 作業ディレクトリをプロジェクトパスに変更
    try:
        os.chdir(env_info['project_dir'])
        logger.info(f"作業ディレクトリを変更しました: {os.getcwd()}")
    except Exception as e:
        logger.error(f"作業ディレクトリの変更に失敗しました: {str(e)}")
    
    # プロジェクトパスをPYTHONPATHに追加
    if env_info['project_dir'] not in sys.path:
        sys.path.append(env_info['project_dir'])
        logger.info("PYTHONPATHにプロジェクトディレクトリを追加しました")
    
    # データディレクトリ構造を作成
    create_directory_structure(env_info['project_dir'])
    
    # 必要なライブラリをインストール
    env_info['installed_packages'] = install_dependencies(env_info['project_dir'], is_colab)
    
    # サンプルETFモードの設定
    if use_sample_etfs:
        setup_sample_etfs(env_info['project_dir'])
    
    # メモリ使用量モニタリングのセットアップ
    setup_memory_monitoring(env_info['project_dir'])
    
    # ローカル環境固有のセットアップ
    if setup_local_env and not is_colab:
        setup_local_environment(env_info['project_dir'])
    
    # 完了情報
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"環境セットアップが完了しました - 所要時間: {duration:.2f}秒")
    
    env_info['end_time'] = end_time
    env_info['duration'] = duration
    
    # セットアップのサマリーを表示
    display_setup_summary(env_info)
    
    return env_info

def setup_project_directory(
    project_name: str,
    is_colab: bool,
    force_clone: bool
) -> str:
    """プロジェクトディレクトリをセットアップする
    
    Args:
        project_name: プロジェクト名
        is_colab: Colabで実行しているかどうか
        force_clone: 強制的に再クローンするかどうか
        
    Returns:
        str: プロジェクトディレクトリのパス
    """
    # Googleドライブをマウント（Colabの場合）
    drive_mounted = False
    if is_colab:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Driveのマウント成功")
            drive_mounted = True
        except Exception as e:
            logger.error(f"Google Driveのマウントエラー: {str(e)}")
            logger.warning("メモリの永続化ができないため、キャッシュが保存されません")
    
    # プロジェクトディレクトリのパスを設定
    if is_colab and drive_mounted:
        project_path = f'/content/drive/MyDrive/{project_name}'
    elif is_colab:
        project_path = f'/content/{project_name}'
    else:
        # ローカル環境の場合は現在のディレクトリの親をプロジェクトルートと仮定
        project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        if not os.path.basename(project_path) == project_name:
            # 親ディレクトリが一致しない場合は現在のディレクトリを使用
            project_path = os.getcwd()
    
    # プロジェクトディレクトリが存在しない場合はGitからクローン
    if not os.path.exists(project_path) or force_clone:
        if os.path.exists(project_path) and force_clone:
            logger.info(f"強制再クローン指定のため、既存のディレクトリを削除します: {project_path}")
            try:
                subprocess.run(f"rm -rf {project_path}", shell=True, check=True)
            except Exception as e:
                logger.warning(f"既存ディレクトリの削除エラー: {str(e)}")
        
        logger.info(f"Gitからプロジェクトをクローンしています...")
        
        # リポジトリのURL（実際のURLは適宜変更すること）
        repo_url = f"https://github.com/your-username/{project_name}.git"
        
        try:
            if is_colab and drive_mounted:
                # ドライブ内にクローン
                clone_command = f"cd /content/drive/MyDrive && git clone {repo_url}"
            else:
                # 現在の環境内にクローン
                clone_command = f"cd {os.path.dirname(project_path)} && git clone {repo_url}"
            
            subprocess.run(clone_command, shell=True, check=True)
            logger.info(f"リポジトリのクローン成功: {project_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"リポジトリのクローンエラー: {str(e)}")
            # 基本的なディレクトリだけ作成する
            os.makedirs(project_path, exist_ok=True)
            logger.warning(f"空のプロジェクトディレクトリを作成しました: {project_path}")
    else:
        logger.info(f"既存のプロジェクトディレクトリを使用します: {project_path}")
    
    return project_path

def create_directory_structure(project_path: str) -> None:
    """データディレクトリ構造を作成する
    
    Args:
        project_path: プロジェクトディレクトリのパス
    """
    data_dirs = [
        "data",
        "data/cache", 
        "data/raw", 
        "data/processed", 
        "data/results",
        "data/results/signals",
        "data/results/parameters",
        "data/results/validation",
        "data/logs"
    ]
    
    for data_dir in data_dirs:
        dir_path = os.path.join(project_path, data_dir)
        os.makedirs(dir_path, exist_ok=True)
    
    logger.info("データディレクトリ構造を作成しました")

def install_dependencies(
    project_path: str,
    is_colab: bool
) -> List[str]:
    """必要なライブラリをインストールする
    
    Args:
        project_path: プロジェクトディレクトリのパス
        is_colab: Colabで実行しているかどうか
        
    Returns:
        List[str]: インストールされたパッケージのリスト
    """
    installed = []
    
    logger.info("必要なライブラリをインストールしています...")
    
    # システムパッケージを確認
    missing_system_deps = check_system_dependencies()
    if missing_system_deps and (is_colab or os.geteuid() == 0):  # CoalbかrootユーザーのみSystem依存関係をインストール
        for pkg in missing_system_deps:
            try:
                logger.info(f"システム依存関係をインストール中: {pkg}")
                subprocess.run(f"apt-get update && apt-get install -y {pkg}", shell=True, check=True)
            except Exception as e:
                logger.warning(f"システムパッケージのインストールエラー ({pkg}): {str(e)}")
    
    try:
        requirements_file = os.path.join(project_path, "requirements.txt")
        if os.path.exists(requirements_file):
            logger.info("requirements.txtからライブラリをインストールしています...")
            
            if is_colab:
                # Colabでは静かにインストール
                subprocess.run(f"pip install -q -r {requirements_file}", shell=True, check=True)
            else:
                # ローカルでは進捗を表示
                subprocess.run(f"pip install -r {requirements_file}", shell=True, check=True)
            
            # インストールされたパッケージリストを取得
            try:
                with open(requirements_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            installed.append(line)
            except Exception as e:
                logger.warning(f"インストールパッケージリストの読み取りエラー: {str(e)}")
            
            logger.info("依存ライブラリのインストール成功")
        else:
            logger.warning("requirements.txtが見つかりません。代わりに主要ライブラリを個別にインストールします")
            # 必須ライブラリのみをインストール
            essential_packages = [
                "pandas>=1.3.0,<2.0.0", 
                "numpy>=1.20.0,<2.0.0", 
                "scikit-learn>=1.0.0,<2.0.0", 
                "matplotlib>=3.4.0,<3.8.0", 
                "seaborn>=0.11.2,<0.13.0", 
                "yfinance>=0.1.70,<0.2.0", 
                "ta>=0.10.0,<0.11.0",
                "networkx>=2.5",
                "psutil>=5.8.0",
            ]
            
            install_cmd = "pip install -q " if is_colab else "pip install "
            subprocess.run(f"{install_cmd} {' '.join(essential_packages)}", shell=True, check=True)
            installed = essential_packages
            logger.info("主要ライブラリのインストール成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"ライブラリのインストールエラー: {str(e)}")
    except Exception as e:
        logger.error(f"依存関係のインストール中に予期しないエラーが発生しました: {str(e)}")
    
    return installed

def check_system_dependencies() -> List[str]:
    """システム依存関係をチェックする
    
    Returns:
        List[str]: 不足しているシステムパッケージのリスト
    """
    missing = []
    
    # 必要なシステムパッケージのリスト
    system_packages = ["libgraphviz-dev", "graphviz"]
    
    # 各パッケージがインストールされているか確認
    for pkg in system_packages:
        try:
            result = subprocess.run(f"dpkg -l | grep {pkg}", shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                missing.append(pkg)
        except Exception:
            # チェックに失敗した場合は念のためリストに追加
            missing.append(pkg)
    
    return missing

def setup_sample_etfs(project_path: str) -> None:
    """サンプルETFモードを設定する
    
    Args:
        project_path: プロジェクトディレクトリのパス
    """
    logger.info("サンプルETFモードを設定中...")
    
    # src/universe ディレクトリの存在確認
    universe_dir = os.path.join(project_path, "src", "universe")
    if not os.path.exists(universe_dir):
        os.makedirs(universe_dir, exist_ok=True)
        logger.info(f"ディレクトリを作成しました: {universe_dir}")
    
    # sample_etfs.py ファイルを作成または更新
    sample_etfs_path = os.path.join(universe_dir, "sample_etfs.py")
    with open(sample_etfs_path, 'w') as f:
        f.write('''"""MVPテスト用のサンプルETFモジュール"""
import importlib
import types
import warnings
from typing import List, Dict, Any, Optional, Callable

def get_sample_etfs() -> List[Dict[str, Any]]:
    """MVPテスト用の厳選ETFリスト"""
    return [
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "category": "US Large Cap", "avg_volume": 75000000, "aum": 380000000000, "estimated_spread": 0.0001},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust", "category": "US Tech", "avg_volume": 50000000, "aum": 180000000000, "estimated_spread": 0.0001},
        {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "category": "US Small Cap", "avg_volume": 25000000, "aum": 60000000000, "estimated_spread": 0.0002},
        {"symbol": "EFA", "name": "iShares MSCI EAFE ETF", "category": "Developed Markets", "avg_volume": 18000000, "aum": 50000000000, "estimated_spread": 0.0002},
        {"symbol": "EEM", "name": "iShares MSCI Emerging Markets ETF", "category": "Emerging Markets", "avg_volume": 35000000, "aum": 28000000000, "estimated_spread": 0.0003},
        {"symbol": "GLD", "name": "SPDR Gold Shares", "category": "Commodities", "avg_volume": 8000000, "aum": 55000000000, "estimated_spread": 0.0002},
        {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "category": "US Treasury Bonds", "avg_volume": 12000000, "aum": 25000000000, "estimated_spread": 0.0002},
        {"symbol": "VNQ", "name": "Vanguard Real Estate Index Fund", "category": "Real Estate", "avg_volume": 5000000, "aum": 30000000000, "estimated_spread": 0.0002}
    ]

def override_universe_selection(module_path: str = 'src.universe') -> List[Dict[str, Any]]:
    """ETFユニバース選定関数をオーバーライドして高速サンプルモードにする
    
    Args:
        module_path: src.universeモジュールパス（通常は'src.universe'）
        
    Returns:
        List[Dict[str, Any]]: サンプルETFのリスト
    """
    try:
        # モジュールをインポート
        universe_module = importlib.import_module(module_path)
        
        # オリジナルの関数を保存
        if hasattr(universe_module, 'select_universe'):
            original_select_universe = universe_module.select_universe
        else:
            warnings.warn(f"モジュール {module_path} に select_universe 関数が見つかりません")
            return get_sample_etfs()
        
        # サンプルETFを返す新しい関数
        def sample_select_universe(*args, **kwargs):
            print("🔍 サンプルETFモードを使用します（処理時間短縮のため）")
            sample_etfs = get_sample_etfs()
            
            # clustering_methodパラメータを取得（デフォルトは'tda'）
            clustering_method = kwargs.get('clustering_method', 'tda')
            
            # クラスタ情報を追加（各ETFにクラスタIDを設定）
            for i, etf in enumerate(sample_etfs):
                etf['cluster'] = i % 4  # 0, 1, 2, 3 のクラスタに均等に分配
            
            print(f"📊 選択された{len(sample_etfs)}銘柄のETF (方式: {clustering_method}):")
            for etf in sample_etfs:
                print(f"  • {etf['symbol']}: {etf['name']} ({etf['category']})")
            return sample_etfs
        
        # 関数をオーバーライド
        universe_module.select_universe = sample_select_universe
        
        # 元の関数を保持（復元用）
        if not hasattr(universe_module, 'original_select_universe'):
            universe_module.original_select_universe = original_select_universe
        
        print("✅ ETFユニバース選定関数をサンプルモードに切り替えました")
        print("💡 ヒント: 完全なユニバース選定に戻すには、以下を実行してください:")
        print("   import src.universe; src.universe.select_universe = src.universe.original_select_universe")
        
        return get_sample_etfs()
        
    except Exception as e:
        warnings.warn(f"ユニバース選定関数のオーバーライドに失敗しました: {str(e)}")
        print("サンプルETFリストを代わりに返します")
        return get_sample_etfs()

def restore_universe_selection(module_path: str = 'src.universe') -> bool:
    """ETFユニバース選定関数を元に戻す
    
    Args:
        module_path: src.universeモジュールパス（通常は'src.universe'）
        
    Returns:
        bool: 復元が成功したかどうか
    """
    try:
        # モジュールをインポート
        universe_module = importlib.import_module(module_path)
        
        # 元の関数が保存されているか確認
        if hasattr(universe_module, 'original_select_universe'):
            # 元の関数に戻す
            universe_module.select_universe = universe_module.original_select_universe
            print("✅ ETFユニバース選定関数を元に戻しました")
            return True
        else:
            warnings.warn("元のselect_universe関数が保存されていません")
            return False
            
    except Exception as e:
        warnings.warn(f"ユニバース選定関数の復元に失敗しました: {str(e)}")
        return False
''')
    
    logger.info(f"サンプルETFリストを作成しました: {sample_etfs_path}")
    
    # サンプルETFモードを有効化
    try:
        sys.path.append(project_path)
        from src.universe.sample_etfs import override_universe_selection
        override_universe_selection()
        logger.info("サンプルETFモードを有効化しました")
    except Exception as e:
        logger.warning(f"サンプルETFモードの有効化に失敗しました: {str(e)}")

def setup_memory_monitoring(project_path: str) -> None:
    """メモリ使用量モニタリング機能を設定する
    
    Args:
        project_path: プロジェクトディレクトリのパス
    """
    logger.info("メモリ使用量モニタリング機能を設定しています...")
    
    # モニタリングモジュールが存在するか確認
    utils_dir = os.path.join(project_path, "src", "utils")
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir, exist_ok=True)
    
    monitoring_path = os.path.join(utils_dir, "monitoring.py")
    
    # monitoringモジュールを作成
    with open(monitoring_path, 'w') as f:
        f.write('''"""メモリ使用量モニタリング機能"""
import os
import psutil
import platform
import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

class PerformanceMonitor:
    """メモリ使用量とパフォーマンスのモニタリングユーティリティ"""
    
    def __init__(self, log_to_file=False, log_file="data/logs/performance.log"):
        """モニターの初期化
        
        Args:
            log_to_file: ファイルにログを記録するかどうか
            log_file: ログファイルのパス
        """
        self.start_time = time.time()
        self.checkpoints = []
        self.log_to_file = log_to_file
        self.log_file = log_file
        
        if log_to_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w') as f:
                f.write(f"Performance Monitor Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"System: {platform.system()} {platform.release()}\\n")
                f.write(f"Python: {platform.python_version()}\\n")
                f.write("-" * 80 + "\\n")
    
    def check_memory(self, checkpoint_name=""):
        """現在のメモリ使用量をチェックし、記録する
        
        Args:
            checkpoint_name: チェックポイントの名前
            
        Returns:
            float: メモリ使用量（GB）
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / (1024 ** 3)  # GB単位に変換
            
            # チェックポイント記録
            elapsed_time = time.time() - self.start_time
            checkpoint = {
                'name': checkpoint_name,
                'time': time.strftime('%H:%M:%S'),
                'elapsed': elapsed_time,
                'memory_gb': memory_usage_gb
            }
            self.checkpoints.append(checkpoint)
            
            # ログファイルに記録
            if self.log_to_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"{checkpoint_name}: {memory_usage_gb:.2f} GB, "
                            f"経過時間: {elapsed_time:.2f}秒\\n")
            
            return memory_usage_gb
        
        except Exception as e:
            print(f"メモリ使用量の確認に失敗: {str(e)}")
            return None
    
    def display_current_memory(self):
        """現在のメモリ使用量を表示する（IPythonで使用）"""
        memory_usage_gb = self.check_memory("現在")
        
        if memory_usage_gb is not None:
            # メモリ状態に応じたカラー設定
            color = "green" if memory_usage_gb < 10 else "orange" if memory_usage_gb < 12 else "red"
            status = "正常" if memory_usage_gb < 10 else "警告" if memory_usage_gb < 12 else "危険"
            
            html = f"""
            <div style="margin:10px; padding:10px; border-radius:10px; border:1px solid #ddd;">
                <h3 style="margin-top:0; color:{color};">メモリ使用状況: {status}</h3>
                <p>現在のメモリ使用量: <b>{memory_usage_gb:.2f} GB</b></p>
                <p>経過時間: {time.time() - self.start_time:.2f}秒</p>
                <p>Colabの制限: 約12-13 GB</p>
            </div>
            """
            
            display(HTML(html))
        else:
            print("メモリ使用量を取得できませんでした")
    
    def get_summary(self):
        """モニタリング結果のサマリーを表として取得する"""
        if not self.checkpoints:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.checkpoints)
        
        if len(df) > 1:
            # メモリ変化を計算
            df['memory_change'] = df['memory_gb'].diff()
            df.loc[0, 'memory_change'] = df.loc[0, 'memory_gb']
            
            # 経過時間の差分を計算
            df['time_diff'] = df['elapsed'].diff()
            df.loc[0, 'time_diff'] = df.loc[0, 'elapsed']
        
        return df
    
    def plot_memory_usage(self, figsize=(10, 6)):
        """メモリ使用量の推移をプロット"""
        if not self.checkpoints:
            print("チェックポイントがありません")
            return
        
        df = self.get_summary()
        
        plt.figure(figsize=figsize)
        plt.plot(df['elapsed'], df['memory_gb'], 'b-o', linewidth=2)
        plt.axhline(y=12, color='r', linestyle='--', label='Colab制限')
        
        for i, r in df.iterrows():
            plt.annotate(
                r['name'],
                (r['elapsed'], r['memory_gb']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                rotation=45
            )
        
        plt.title('メモリ使用量の推移')
        plt.xlabel('経過時間 (秒)')
        plt.ylabel('メモリ使用量 (GB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # 保存
        plt.savefig("data/results/memory_usage.png", dpi=200)
        plt.show()
    
    def display_summary(self):
        """モニタリング結果のサマリーを表示"""
        df = self.get_summary()
        
        if df.empty:
            print("チェックポイントがありません")
            return
        
        # HTML表示用にフォーマット
        format_dict = {
            'elapsed': '{:.2f}秒',
            'memory_gb': '{:.2f} GB',
            'memory_change': '{:.2f} GB',
            'time_diff': '{:.2f}秒'
        }
        
        # 表示する列を選択し、順序を指定
        display_cols = ['name', 'time', 'elapsed', 'memory_gb']
        if 'memory_change' in df.columns:
            display_cols.extend(['memory_change', 'time_diff'])
        
        styled_df = df[display_cols].style.format(format_dict)
        display(styled_df)
        
        # 総メモリ使用量の増加を表示
        if len(df) > 1:
            total_memory_increase = df['memory_gb'].iloc[-1] - df['memory_gb'].iloc[0]
            print(f"総メモリ使用量の増加: {total_memory_increase:.2f} GB")
            print(f"総実行時間: {df['elapsed'].iloc[-1]:.2f}秒")

# 使いやすいインターフェース関数
monitor = PerformanceMonitor()

def display_memory_usage():
    """現在のメモリ使用量を表示する（ノートブック用）"""
    monitor.display_current_memory()

def check_memory(checkpoint_name=""):
    """メモリ使用量をチェックして値を返す"""
    return monitor.check_memory(checkpoint_name)

def show_memory_summary():
    """メモリ使用量のサマリーを表示する"""
    monitor.display_summary()
    monitor.plot_memory_usage()

def garbage_collect():
    """ガベージコレクションを強制的に実行する"""
    import gc
    before = check_memory("GC前")
    gc.collect()
    after = check_memory("GC後")
    print(f"解放されたメモリ: {max(0, before - after):.2f} GB")
    return before - after
''')
    
    logger.info("メモリモニタリングモジュールを作成しました")
    
    # グローバルにメモリ使用量モニタリング関数を提供
    try:
        sys.path.append(project_path)
        from src.utils.monitoring import display_memory_usage, check_memory, show_memory_summary, garbage_collect
        
        # グローバル変数として設定
        globals_dict = globals()
        globals_dict['display_memory_usage'] = display_memory_usage
        globals_dict['check_memory'] = check_memory
        globals_dict['show_memory_summary'] = show_memory_summary
        globals_dict['garbage_collect'] = garbage_collect
        
        logger.info("メモリ使用量モニタリング機能をグローバルに設定しました")
        
        # 現在のメモリ使用量を表示
        try:
            display_memory_usage()
        except Exception as e:
            logger.warning(f"メモリ使用量表示エラー: {str(e)}")
    except Exception as e:
        logger.warning(f"メモリモニタリング機能のグローバル設定に失敗しました: {str(e)}")

def setup_local_environment(project_path: str) -> None:
    """ローカル環境特有のセットアップを行う
    
    Args:
        project_path: プロジェクトディレクトリのパス
    """
    logger.info("ローカル環境向けの追加セットアップを実行中...")
    
    # 仮想環境設定（venv）
    venv_path = os.path.join(project_path, "venv")
    if not os.path.exists(venv_path):
        try:
            logger.info(f"仮想環境を作成中: {venv_path}")
            subprocess.run(f"python -m venv {venv_path}", shell=True, check=True)
            logger.info("仮想環境を作成しました")
            
            # 仮想環境の有効化方法を表示
            if platform.system() == "Windows":
                logger.info("仮想環境を有効化するには: venv\\Scripts\\activate.bat")
            else:
                logger.info("仮想環境を有効化するには: source venv/bin/activate")
        except Exception as e:
            logger.warning(f"仮想環境の作成に失敗しました: {str(e)}")
    
    # Jupyter拡張機能のインストール
    try:
        logger.info("Jupyter拡張機能をインストール中...")
        subprocess.run("pip install jupyter_contrib_nbextensions", shell=True, check=True)
        subprocess.run("jupyter contrib nbextension install --user", shell=True, check=True)
        logger.info("Jupyter拡張機能をインストールしました")
    except Exception as e:
        logger.warning(f"Jupyter拡張機能のインストールに失敗しました: {str(e)}")
    
    # VS Code設定ファイル作成
    vscode_dir = os.path.join(project_path, ".vscode")
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir, exist_ok=True)
        
        # settings.json
        settings_path = os.path.join(vscode_dir, "settings.json")
        with open(settings_path, 'w') as f:
            f.write('''{
    "python.pythonPath": "venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "autopep8",
    "editor.formatOnSave": true,
    "python.linting.pylintArgs": [
        "--disable=C0111,C0103"
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ]
}''')
        
        logger.info("VS Code設定ファイルを作成しました")

def display_setup_summary(env_info: Dict[str, Any]) -> None:
    """セットアップ情報のサマリーを表示する
    
    Args:
        env_info: 環境情報の辞書
    """
    try:
        from IPython.display import display, HTML
        
        summary_html = f"""
        <div style="margin:10px; padding:15px; border-radius:10px; border:1px solid #ddd; background-color:#f9f9f9;">
            <h2 style="margin-top:0; color:#3366CC;">環境セットアップ完了 🎉</h2>
            
            <h3>システム情報</h3>
            <ul>
                <li>環境: {'Google Colab' if env_info['is_colab'] else 'Jupyter Notebook' if env_info['is_jupyter'] else 'その他'}</li>
                <li>Python: {env_info['python_version']}</li>
                <li>OS: {env_info['system']}</li>
                <li>セットアップ時間: {env_info['duration']:.2f}秒</li>
            </ul>
            
            <h3>プロジェクト情報</h3>
            <ul>
                <li>作業ディレクトリ: {env_info['project_dir']}</li>
                <li>データディレクトリ: {os.path.join(env_info['project_dir'], 'data')}</li>
                <li>キャッシュディレクトリ: {os.path.join(env_info['project_dir'], 'data/cache')}</li>
            </ul>
            
            <div style="margin-top:15px; padding:10px; border-radius:5px; background-color:#e8f4f8;">
                <h4 style="margin-top:0;">使用可能な便利機能:</h4>
                <ul>
                    <li><code>display_memory_usage()</code> - 現在のメモリ使用量を表示</li>
                    <li><code>check_memory("チェックポイント名")</code> - メモリ使用量をチェック</li>
                    <li><code>show_memory_summary()</code> - メモリ使用推移のサマリー表示</li>
                    <li><code>garbage_collect()</code> - 明示的なメモリ解放</li>
                </ul>
            </div>
        </div>
        """
        
        display(HTML(summary_html))
    except Exception as e:
        # IPythonが使えない場合はテキスト表示
        logger.info("=" * 50)
        logger.info("環境セットアップ完了 🎉")
        logger.info("=" * 50)
        logger.info(f"作業ディレクトリ: {env_info['project_dir']}")
        logger.info(f"キャッシュディレクトリ: {os.path.join(env_info['project_dir'], 'data/cache')}")
        logger.info(f"セットアップ時間: {env_info['duration']:.2f}秒")
        
        logger.info("\n使用可能な便利機能:")
        logger.info("- display_memory_usage() - 現在のメモリ使用量を表示")
        logger.info("- check_memory(\"チェックポイント名\") - メモリ使用量をチェック")
        logger.info("- show_memory_summary() - メモリ使用推移のサマリー表示")
        logger.info("- garbage_collect() - 明示的なメモリ解放")
        logger.info("=" * 50)

# Colabで実行されている場合は自動的にセットアップを実行
if 'google.colab' in sys.modules:
    env_info = setup_colab()
