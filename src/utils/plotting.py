# src/utils/plotting.py
"""可視化ユーティリティ"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# デフォルトのスタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def setup_plot_style(theme: str = 'default'):
    """プロットのスタイルを設定する
    
    Args:
        theme: テーマ名 ('default', 'dark', 'paper', 'presentation')
    """
    if theme == 'default':
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
    elif theme == 'dark':
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.facecolor'] = '#1f1f1f'
        plt.rcParams['figure.facecolor'] = '#121212'
    elif theme == 'paper':
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.rcParams['font.family'] = 'serif'
    elif theme == 'presentation':
        plt.style.use('seaborn-v0_8-talk')
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
    else:
        raise ValueError(f"Unknown theme: {theme}")

def plot_price_with_signals(
    data: pd.DataFrame,
    buy_signal_column: str = 'Buy_Signal',
    sell_signal_column: str = 'Sell_Signal',
    price_column: str = 'Adj Close',
    sma_periods: List[int] = [50, 200],
    bollinger_window: Optional[int] = 20,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> Figure:
    """価格チャートとシグナルを可視化する
    
    Args:
        data: 価格とシグナルを含むデータフレーム
        buy_signal_column: 買いシグナル列名
        sell_signal_column: 売りシグナル列名
        price_column: 価格列名
        sma_periods: 単純移動平均の期間リスト
        bollinger_window: ボリンジャーバンドのウィンドウサイズ（Noneの場合は非表示）
        figsize: 図のサイズ
        save_path: 保存先パス（Noneの場合は保存しない）
        
    Returns:
        Figure: Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 価格の描画
    ax.plot(data.index, data[price_column], label=price_column, linewidth=1.5)
    
    # 移動平均線の描画
    for period in sma_periods:
        sma = data[price_column].rolling(window=period).mean()
        ax.plot(data.index, sma, label=f'SMA {period}', linestyle='--', alpha=0.7)
    
    # ボリンジャーバンドの描画
    if bollinger_window is not None and bollinger_window > 0:
        sma = data[price_column].rolling(window=bollinger_window).mean()
        std = data[price_column].rolling(window=bollinger_window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        ax.plot(data.index, upper_band, 'k--', alpha=0.3)
        ax.plot(data.index, lower_band, 'k--', alpha=0.3)
        ax.fill_between(data.index, lower_band, upper_band, alpha=0.1, color='gray')
    
    # 買いシグナルの描画
    if buy_signal_column in data.columns:
        buy_signals = data[data[buy_signal_column]]
        ax.scatter(
            buy_signals.index, 
            buy_signals[price_column], 
            marker='^', 
            color='green', 
            s=100, 
            label='Buy Signal'
        )
    
    # 売りシグナルの描画
    if sell_signal_column in data.columns:
        sell_signals = data[data[sell_signal_column]]
        ax.scatter(
            sell_signals.index, 
            sell_signals[price_column], 
            marker='v', 
            color='red', 
            s=100, 
            label='Sell Signal'
        )
    
    # グラフの設定
    title = f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    symbol = data.get('Symbol', None)
    if symbol is not None and isinstance(symbol, str):
        title = f"{symbol}: {title}"
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # X軸の日付フォーマット
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_equity_curve(
    equity_data: pd.DataFrame,
    benchmark_data: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (14, 8),
    log_scale: bool = False,
    include_drawdown: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """エクイティカーブを可視化する
    
    Args:
        equity_data: エクイティデータを含むデータフレーム
        benchmark_data: ベンチマークデータを含むデータフレーム（Noneの場合は表示しない）
        figsize: 図のサイズ
        log_scale: 対数スケールを使用するかどうか
        include_drawdown: ドローダウンを含めるかどうか
        save_path: 保存先パス（Noneの場合は保存しない）
        
    Returns:
        Figure: Matplotlibの図オブジェクト
    """
    if include_drawdown:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # エクイティカーブの描画
    ax1.plot(equity_data.index, equity_data, label='Strategy', linewidth=2)
    
    # ベンチマークの描画
    if benchmark_data is not None:
        ax1.plot(benchmark_data.index, benchmark_data, label='Benchmark', linewidth=1.5, alpha=0.7)
    
    # グラフの設定
    title = f"Equity Curve: {equity_data.index[0].strftime('%Y-%m-%d')} to {equity_data.index[-1].strftime('%Y-%m-%d')}"
    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity')
    
    if log_scale:
        ax1.set_yscale('log')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # X軸の日付フォーマット
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # ドローダウンの描画
    if include_drawdown:
        # ドローダウンの計算
        running_max = np.maximum.accumulate(equity_data)
        drawdown = (equity_data / running_max) - 1
        
        # ドローダウンの描画
        ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown')
        ax2.set_ylim(drawdown.min() * 1.1, 0.01)
        ax2.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_heatmap(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    title: str = '',
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    annotate: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """ヒートマップを可視化する
    
    Args:
        data: データを含むデータフレーム
        x_column: X軸の列名
        y_column: Y軸の列名
        value_column: 値の列名
        title: グラフのタイトル
        figsize: 図のサイズ
        cmap: カラーマップ
        annotate: 値をアノテーションするかどうか
        save_path: 保存先パス（Noneの場合は保存しない）
        
    Returns:
        Figure: Matplotlibの図オブジェクト
    """
    # ピボットテーブルを作成
    pivot_table = data.pivot_table(
        index=y_column,
        columns=x_column,
        values=value_column,
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # ヒートマップの描画
    sns.heatmap(
        pivot_table,
        annot=annotate,
        fmt='.3f',
        cmap=cmap,
        linewidths=0.5,
        ax=ax
    )
    
    # グラフの設定
    ax.set_title(title)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_performance_metrics(
    metrics: Dict[str, float],
    comparison_metrics: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Performance Metrics',
    save_path: Optional[str] = None
) -> Figure:
    """パフォーマンスメトリクスを棒グラフで可視化する
    
    Args:
        metrics: メトリクスの辞書
        comparison_metrics: 比較用メトリクスの辞書
        figsize: 図のサイズ
        title: グラフのタイトル
        save_path: 保存先パス（Noneの場合は保存しない）
        
    Returns:
        Figure: Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics_df = pd.DataFrame({
        'Strategy': metrics,
        'Benchmark': comparison_metrics if comparison_metrics else {}
    })
    
    # メトリクスが空の場合
    if metrics_df.empty:
        ax.text(
            0.5, 0.5,
            'No metrics available',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )
        plt.tight_layout()
        return fig
    
    # 棒グラフの描画
    metrics_df.plot(kind='bar', ax=ax)
    
    # グラフの設定
    ax.set_title(title)
    ax.set_ylabel('Value')
    ax.set_xlabel('Metric')
    ax.grid(True, alpha=0.3)
    
    # ラベルの回転
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_regime_performance(
    regime_data: Dict[str, Dict[str, float]],
    metrics: List[str] = ['win_rate', 'profit_factor', 'sharpe_ratio'],
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Performance by Market Regime',
    save_path: Optional[str] = None
) -> Figure:
    """市場レジーム別のパフォーマンスを可視化する
    
    Args:
        regime_data: レジーム別パフォーマンスデータ
        metrics: 表示するメトリクスのリスト
        figsize: 図のサイズ
        title: グラフのタイトル
        save_path: 保存先パス（Noneの場合は保存しない）
        
    Returns:
        Figure: Matplotlibの図オブジェクト
    """
    # データ準備
    regimes = list(regime_data.keys())
    
    if not regimes:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            'No regime data available',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )
        plt.tight_layout()
        return fig
    
    # メトリクスごとにプロット
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    
    # 単一メトリクスの場合
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # メトリクスの値を取得
        values = [regime_data[regime].get(metric, 0) for regime in regimes]
        
        # 棒グラフの描画
        bars = ax.bar(regimes, values)
        
        # 基準線（メトリクスによって異なる）
        if metric in ['win_rate']:
            ax.axhline(0.5, color='red', linestyle='--')
        elif metric in ['profit_factor']:
            ax.axhline(1.0, color='red', linestyle='--')
        elif metric in ['sharpe_ratio']:
            ax.axhline(0, color='red', linestyle='--')
        
        # グラフの設定
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom'
            )
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_cluster_map(
    embedding_data: pd.DataFrame,
    x_column: str = 'x',
    y_column: str = 'y',
    cluster_column: str = 'cluster',
    symbol_column: str = 'symbol',
    figsize: Tuple[int, int] = (12, 10),
    title: str = 'ETF Cluster Map',
    save_path: Optional[str] = None
) -> Figure:
    """ETFクラスタマップを可視化する
    
    Args:
        embedding_data: 埋め込みデータを含むデータフレーム
        x_column: X座標の列名
        y_column: Y座標の列名
        cluster_column: クラスタの列名
        symbol_column: シンボルの列名
        figsize: 図のサイズ
        title: グラフのタイトル
        save_path: 保存先パス（Noneの場合は保存しない）
        
    Returns:
        Figure: Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # クラスタごとに色分け
    clusters = embedding_data[cluster_column].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    
    for cluster, color in zip(clusters, colors):
        # クラスタフィルタ
        mask = embedding_data[cluster_column] == cluster
        
        # クラスタ名
        cluster_name = f'Cluster {cluster}' if cluster >= 0 else 'Noise'
        
        # 散布図の描画
        ax.scatter(
            embedding_data.loc[mask, x_column],
            embedding_data.loc[mask, y_column],
            color=color,
            label=cluster_name,
            alpha=0.7,
            s=80
        )
    
    # シンボルのアノテーション
    for _, row in embedding_data.iterrows():
        ax.annotate(
            row[symbol_column],
            (row[x_column], row[y_column]),
            fontsize=9,
            alpha=0.8
        )
    
    # グラフの設定
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_summary_dashboard(
    data: pd.DataFrame,
    validation_results: Dict[str, Any],
    etf_info: Dict[str, Any],
    output_dir: str = "data/results/dashboard",
    symbol: Optional[str] = None
) -> str:
    """戦略の結果をまとめたダッシュボードを作成する
    
    Args:
        data: 価格とシグナルを含むデータフレーム
        validation_results: 検証結果の辞書
        etf_info: ETF情報の辞書
        output_dir: 出力ディレクトリ
        symbol: ETFシンボル（Noneの場合は自動検出）
        
    Returns:
        str: ダッシュボードのHTMLパス
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    if symbol is None:
        symbol = data.get('Symbol', etf_info.get('symbol', 'Unknown'))
    
    # ファイルパス
    html_path = f"{output_dir}/{symbol}_dashboard.html"
    price_chart_path = f"{output_dir}/{symbol}_price_chart.png"
    equity_curve_path = f"{output_dir}/{symbol}_equity_curve.png"
    metrics_path = f"{output_dir}/{symbol}_metrics.png"
    regime_path = f"{output_dir}/{symbol}_regime.png"
    
    # チャートの生成
    plot_price_with_signals(data, save_path=price_chart_path)
    
    # エクイティカーブの生成（あれば）
    if 'walk_forward' in validation_results and 'equity_curve' in validation_results['walk_forward']:
        equity_data = pd.DataFrame(validation_results['walk_forward']['equity_curve'])
        if 'date' in equity_data.columns and 'equity' in equity_data.columns:
            equity_data.set_index('date', inplace=True)
            plot_equity_curve(equity_data['equity'], save_path=equity_curve_path)
    
    # パフォーマンスメトリクスの生成（あれば）
    if 'trade_metrics' in validation_results:
        plot_performance_metrics(
            validation_results['trade_metrics'],
            title=f"{symbol} Performance Metrics",
            save_path=metrics_path
        )
    
    # レジームパフォーマンスの生成（あれば）
    if 'regime_performance' in validation_results:
        plot_regime_performance(
            validation_results['regime_performance'],
            title=f"{symbol} Regime Performance",
            save_path=regime_path
        )
    
    # HTMLの生成
    with open(html_path, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>{symbol} Strategy Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .chart-container {{ margin-bottom: 30px; }}
        .chart {{ width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
        .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
        .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-right: 10px; margin-bottom: 10px; min-width: 200px; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{symbol} Strategy Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="chart-container">
            <h2>Price Chart with Signals</h2>
            <img class="chart" src="{os.path.basename(price_chart_path)}" alt="Price Chart">
        </div>
        
        <div class="chart-container">
            <h2>Equity Curve</h2>
            <img class="chart" src="{os.path.basename(equity_curve_path)}" alt="Equity Curve">
        </div>
        
        <div class="chart-container">
            <h2>Performance Metrics</h2>
            <img class="chart" src="{os.path.basename(metrics_path)}" alt="Performance Metrics">
        </div>
        
        <div class="chart-container">
            <h2>Regime Performance</h2>
            <img class="chart" src="{os.path.basename(regime_path)}" alt="Regime Performance">
        </div>
        
        <div class="metrics">
            <h2>Key Performance Indicators</h2>
        </div>
        
        <div class="metrics">
""")
        
        # KPIの追加
        if 'overall' in validation_results.get('walk_forward', {}):
            overall = validation_results['walk_forward']['overall']
            
            f.write(f"""
            <div class="metric-card">
                <h3>Win Rate</h3>
                <p>{overall.get('win_rate', 0) * 100:.1f}%</p>
            </div>
            
            <div class="metric-card">
                <h3>Profit Factor</h3>
                <p>{overall.get('profit_factor', 0):.2f}</p>
            </div>
            
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <p>{overall.get('sharpe_ratio', 0):.2f}</p>
            </div>
            
            <div class="metric-card">
                <h3>Return</h3>
                <p>{overall.get('return', 0) * 100:.1f}%</p>
            </div>
""")
        
        # ETF情報の追加
        f.write(f"""
        </div>
        
        <div class="metrics">
            <h2>ETF Information</h2>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>Symbol</h3>
                <p>{etf_info.get('symbol', 'Unknown')}</p>
            </div>
            
            <div class="metric-card">
                <h3>Name</h3>
                <p>{etf_info.get('name', 'Unknown')}</p>
            </div>
            
            <div class="metric-card">
                <h3>AUM</h3>
                <p>${etf_info.get('aum', 0) / 1e9:.1f}B</p>
            </div>
            
            <div class="metric-card">
                <h3>Avg Volume</h3>
                <p>{etf_info.get('avg_volume', 0):,.0f}</p>
            </div>
        </div>
    </div>
</body>
</html>
""")
    
    return html_path
