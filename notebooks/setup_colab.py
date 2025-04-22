"""Google Colabとの連携設定 - 強化版"""
import os
import sys
import subprocess
from datetime import datetime

def setup_colab(project_name="etf-mean-reversion-strategy", force_clone=False, use_sample_etfs=True):
    """Colabでの実行環境をセットアップし、安定性を強化する
    
    Args:
        project_name: リポジトリ/プロジェクト名
        force_clone: 既存のプロジェクトがあっても強制的に再クローンするかどうか
        use_sample_etfs: サンプルETFセットを使用するかどうか
        
    Returns:
        str: プロジェクトのパス
    """
    print(f"🚀 Google Colab環境のセットアップを開始します - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Googleドライブをマウント
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Driveのマウント成功")
    except Exception as e:
        print(f"❌ Google Driveのマウントエラー: {str(e)}")
        print("  メモリの永続化ができないため、キャッシュが保存されません")
        drive_mounted = False
    else:
        drive_mounted = True
    
    # プロジェクトディレクトリのパスを設定
    if drive_mounted:
        project_path = f'/content/drive/MyDrive/{project_name}'
    else:
        project_path = f'/content/{project_name}'
    
    # プロジェクトディレクトリが存在しない場合はGitからクローン
    if not os.path.exists(project_path) or force_clone:
        if os.path.exists(project_path) and force_clone:
            print(f"🔄 強制再クローン指定のため、既存のディレクトリを削除します: {project_path}")
            try:
                os.system(f"rm -rf {project_path}")
            except Exception as e:
                print(f"⚠️ 既存ディレクトリの削除エラー: {str(e)}")
        
        print(f"📥 Gitからプロジェクトをクローンしています...")
        
        # リポジトリのURLを指定（あなたのGitHubリポジトリに書き換えてください）
        repo_url = f"https://github.com/your-username/{project_name}.git"
        
        try:
            if drive_mounted:
                # ドライブ内にクローン
                clone_command = f"cd /content/drive/MyDrive && git clone {repo_url}"
            else:
                # Colab環境内にクローン
                clone_command = f"cd /content && git clone {repo_url}"
            
            subprocess.run(clone_command, shell=True, check=True)
            print(f"✅ リポジトリのクローン成功: {project_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ リポジトリのクローンエラー: {str(e)}")
            # 基本的なディレクトリだけ作成する
            os.makedirs(project_path, exist_ok=True)
            print(f"⚠️ 空のプロジェクトディレクトリを作成しました: {project_path}")
    else:
        print(f"✅ 既存のプロジェクトディレクトリを使用します: {project_path}")
    
    # 作業ディレクトリをプロジェクトパスに変更
    os.chdir(project_path)
    print(f"📂 作業ディレクトリを変更しました: {os.getcwd()}")
    
    # プロジェクトパスをPYTHONPATHに追加
    if project_path not in sys.path:
        sys.path.append(project_path)
        print("✅ PYTHONPATHにプロジェクトディレクトリを追加しました")
    
    # データディレクトリ構造を作成
    data_dirs = [
        "data",
        "data/cache", 
        "data/raw", 
        "data/processed", 
        "data/results",
        "data/results/signals",
        "data/results/parameters",
        "data/results/validation",
        "data/logs"  # ログ用ディレクトリを追加
    ]
    
    for data_dir in data_dirs:
        os.makedirs(os.path.join(project_path, data_dir), exist_ok=True)
    
    print("✅ データディレクトリ構造を作成しました")
    
    # 必要なライブラリをインストール - 特定バージョンを指定
    print("📦 必要なライブラリをインストールしています...")
    
    try:
        requirements_file = os.path.join(project_path, "requirements.txt")
        if os.path.exists(requirements_file):
            subprocess.run(f"pip install -q -r {requirements_file}", shell=True, check=True)
            print("✅ 依存ライブラリのインストール成功")
        else:
            print("⚠️ requirements.txtが見つかりません。代わりに主要ライブラリを個別にインストールします")
            # umap-learnとhdbscanを含まない必須ライブラリのみをインストール
            subprocess.run("pip install -q pandas>=1.3.0,<2.0.0 numpy>=1.20.0,<2.0.0 scikit-learn>=1.0.0,<2.0.0 matplotlib>=3.4.0,<3.8.0 seaborn>=0.11.2,<0.13.0 yfinance>=0.1.70,<0.2.0 ta>=0.10.0,<0.11.0", 
                         shell=True, check=True)
            print("✅ 主要ライブラリのインストール成功")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 一部のライブラリインストールに失敗しました: {str(e)}")
    
    # サンプルETFモードの設定
    if use_sample_etfs:
        print("🔍 サンプルETFモードが有効です（処理時間を短縮するため少数のETFを使用）")
        # src/universe ディレクトリの存在確認
        universe_dir = os.path.join(project_path, "src", "universe")
        if not os.path.exists(universe_dir):
            os.makedirs(universe_dir, exist_ok=True)
            print(f"✅ ディレクトリを作成しました: {universe_dir}")
        
        # sample_etfs.py ファイルを作成または更新
        sample_etfs_path = os.path.join(universe_dir, "sample_etfs.py")
        with open(sample_etfs_path, 'w') as f:
            f.write('''"""MVPテスト用のサンプルETFモジュール"""

def get_sample_etfs():
    """MVPテスト用の厳選ETFリスト"""
    return [
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "category": "US Large Cap"},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust", "category": "US Tech"},
        {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "category": "US Small Cap"},
        {"symbol": "EFA", "name": "iShares MSCI EAFE ETF", "category": "Developed Markets"},
        {"symbol": "EEM", "name": "iShares MSCI Emerging Markets ETF", "category": "Emerging Markets"},
        {"symbol": "GLD", "name": "SPDR Gold Shares", "category": "Commodities"},
        {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "category": "US Treasury Bonds"},
        {"symbol": "VNQ", "name": "Vanguard Real Estate Index Fund", "category": "Real Estate"}
    ]
''')
        print(f"✅ サンプルETFリストを作成しました: {sample_etfs_path}")
    
    # メモリ使用量モニタリングヘルパー関数のインポート
    print("🔄 メモリ使用量モニタリング機能を設定しています...")
    
    try:
        # モニタリングモジュールが存在するか確認
        monitoring_path = os.path.join(project_path, "src", "utils", "monitoring.py")
        if not os.path.exists(monitoring_path):
            # utilsディレクトリの確認
            utils_dir = os.path.join(project_path, "src", "utils")
            if not os.path.exists(utils_dir):
                os.makedirs(utils_dir, exist_ok=True)
            
            # monitoringモジュールを作成
            with open(monitoring_path, 'w') as f:
                f.write('''"""メモリ使用量モニタリング機能"""
import os
import psutil
from IPython.display import display, HTML

def display_memory_usage():
    """現在のメモリ使用量を表示する"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024 ** 3)  # GB単位に変換
        
        # メモリ使用状況を表示
        memory_status = "正常" if memory_usage_gb < 10 else "警告" if memory_usage_gb < 12 else "危険"
        color = "green" if memory_usage_gb < 10 else "orange" if memory_usage_gb < 12 else "red"
        
        html = f"""
        <div style="margin:10px; padding:10px; border-radius:10px; border:1px solid #ddd;">
            <h3 style="margin-top:0; color:{color};">メモリ使用状況: {memory_status}</h3>
            <p>現在のメモリ使用量: <b>{memory_usage_gb:.2f} GB</b></p>
            <p>Colabの制限: 約12-13 GB</p>
        </div>
        """
        
        display(HTML(html))
        
        return memory_usage_gb
    except Exception as e:
        print(f"⚠️ メモリ使用量の確認ができません: {str(e)}")
        return None

def check_memory(checkpoint_name=""):
    """メモリ使用量をチェックして値を返す"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024 ** 3)  # GB単位に変換
        
        if checkpoint_name:
            print(f"{checkpoint_name}: {memory_usage_gb:.2f} GB")
        
        return memory_usage_gb
    except Exception as e:
        print(f"⚠️ メモリ使用量の確認に失敗: {str(e)}")
        return None
''')
            print("✅ メモリモニタリングモジュールを作成しました")
        
        # グローバルにメモリ使用量モニタリング関数を提供
        from src.utils.monitoring import display_memory_usage, check_memory
        globals()['display_memory_usage'] = display_memory_usage
        globals()['check_memory'] = check_memory
        
        print("✅ メモリ使用量モニタリング機能をセットアップしました")
        
        # 現在のメモリ使用量を表示
        display_memory_usage()
        
    except Exception as e:
        print(f"⚠️ メモリモニタリング機能の設定に失敗しました: {str(e)}")
    
    print(f"🎉 Colab環境のセットアップが完了しました - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 作業ディレクトリ: {os.getcwd()}")
    print(f"💡 ヒント: キャッシュは {os.path.join(project_path, 'data/cache')} に保存されます")
    
    return project_path

# Colabで実行されている場合は自動的にセットアップを実行
if 'google.colab' in sys.modules:
    project_path = setup_colab()
