"""Google Colabとの連携設定"""
import os
import sys

def setup_colab():
    """Colabでの実行環境をセットアップ"""
    # Googleドライブをマウント
    from google.colab import drive
    drive.mount('/content/drive')
    
    # リポジトリのディレクトリ
    project_dir = "etf-mean-reversion-strategy"
    
    # プロジェクトディレクトリパスを設定
    project_path = f'/content/drive/MyDrive/{project_dir}'
    
    # プロジェクトディレクトリが存在しない場合はGitからクローン
    if not os.path.exists(project_path):
        print(f"プロジェクトディレクトリが見つかりません: {project_path}")
        
        # リポジトリのURLを指定（あなたのGitHubリポジトリに書き換えてください）
        repo_url = "https://github.com/your-username/etf-mean-reversion-strategy.git"
        
        # ドライブ内にクローン
        !cd /content/drive/MyDrive && git clone {repo_url}
        
        print(f"リポジトリを {project_path} にクローンしました")
    
    # 作業ディレクトリをプロジェクトパスに変更
    os.chdir(project_path)
    
    # プロジェクトパスをPYTHONPATHに追加
    if project_path not in sys.path:
        sys.path.append(project_path)
    
    # 必要なライブラリをインストール
    !pip install -r requirements.txt
    
    print(f"Colab環境のセットアップが完了しました！")
    print(f"作業ディレクトリ: {os.getcwd()}")
    
    return project_path

# Colabで実行されている場合は自動的にセットアップを実行
if 'google.colab' in sys.modules:
    project_path = setup_colab()
