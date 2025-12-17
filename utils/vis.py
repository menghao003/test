"""
结果可视化模块
生成ΔG_H分布图、稳定性曲线、生成结构图等
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')


def plot_loss_curve(train_losses: List[float],
                    val_losses: Optional[List[float]] = None,
                    save_path: str = "results/loss_curve.png",
                    title: str = "Training Loss Curve"):
    """
    绘制损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表（可选）
        save_path: 保存路径
        title: 图表标题
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 标注最低点
    min_idx = np.argmin(train_losses)
    ax.annotate(f'Min: {train_losses[min_idx]:.4f}',
                xy=(min_idx + 1, train_losses[min_idx]),
                xytext=(min_idx + 1 + len(epochs) * 0.1, train_losses[min_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"损失曲线已保存至: {save_path}")


def plot_her_performance(delta_g_values: List[float],
                         labels: Optional[List[str]] = None,
                         save_path: str = "results/her_performance.png",
                         title: str = "HER Catalytic Activity (ΔG_H Distribution)"):
    """
    绘制HER催化活性分布图
    
    Args:
        delta_g_values: ΔG_H值列表
        labels: 材料标签（可选）
        save_path: 保存路径
        title: 图表标题
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图
    ax1 = axes[0]
    n, bins, patches = ax1.hist(delta_g_values, bins=20, edgecolor='black', alpha=0.7)
    
    # 根据ΔG_H值着色
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if abs(bin_center) < 0.1:
            patch.set_facecolor('green')
        elif abs(bin_center) < 0.2:
            patch.set_facecolor('yellow')
        else:
            patch.set_facecolor('red')
    
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Optimal (ΔG_H=0)')
    ax1.axvspan(-0.1, 0.1, alpha=0.2, color='green', label='Excellent region')
    ax1.set_xlabel('ΔG_H (eV)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('ΔG_H Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    
    # 右图：散点图（带误差评估）
    ax2 = axes[1]
    x = range(len(delta_g_values))
    colors = ['green' if abs(v) < 0.1 else 'yellow' if abs(v) < 0.2 else 'red' 
              for v in delta_g_values]
    
    ax2.scatter(x, delta_g_values, c=colors, s=50, alpha=0.7, edgecolors='black')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax2.axhspan(-0.1, 0.1, alpha=0.2, color='green')
    ax2.set_xlabel('Material Index', fontsize=12)
    ax2.set_ylabel('ΔG_H (eV)', fontsize=12)
    ax2.set_title('Individual Material ΔG_H', fontsize=13, fontweight='bold')
    
    # 统计信息
    mean_dg = np.mean(delta_g_values)
    std_dg = np.std(delta_g_values)
    optimal_count = sum(1 for v in delta_g_values if abs(v) < 0.1)
    
    stats_text = f'Mean: {mean_dg:.3f} eV\nStd: {std_dg:.3f} eV\nOptimal: {optimal_count}/{len(delta_g_values)}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"HER性能图已保存至: {save_path}")


def plot_stability_curve(formation_energies: List[float],
                         stability_scores: List[float],
                         synthesizability: Optional[List[float]] = None,
                         save_path: str = "results/stability_curve.png",
                         title: str = "Material Stability and Synthesizability"):
    """
    绘制稳定性与可合成性曲线
    
    Args:
        formation_energies: 形成能列表
        stability_scores: 稳定性分数列表
        synthesizability: 可合成性分数列表（可选）
        save_path: 保存路径
        title: 图表标题
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3 if synthesizability else 2, figsize=(15 if synthesizability else 12, 5))
    
    x = range(len(formation_energies))
    
    # 形成能分布
    ax1 = axes[0]
    colors = ['green' if e < 0 else 'red' for e in formation_energies]
    ax1.bar(x, formation_energies, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Material Index', fontsize=12)
    ax1.set_ylabel('Formation Energy (eV/atom)', fontsize=12)
    ax1.set_title('Formation Energy', fontsize=13, fontweight='bold')
    
    stable_count = sum(1 for e in formation_energies if e < 0)
    ax1.text(0.95, 0.95, f'Stable: {stable_count}/{len(formation_energies)}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 稳定性分数
    ax2 = axes[1]
    colors = ['green' if s > 0.7 else 'yellow' if s > 0.5 else 'red' for s in stability_scores]
    ax2.scatter(x, stability_scores, c=colors, s=60, alpha=0.7, edgecolors='black')
    ax2.axhline(y=0.7, color='green', linestyle='--', linewidth=1, label='Good threshold')
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Fair threshold')
    ax2.set_xlabel('Material Index', fontsize=12)
    ax2.set_ylabel('Stability Score', fontsize=12)
    ax2.set_title('Stability Score', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    
    # 可合成性（如果提供）
    if synthesizability:
        ax3 = axes[2]
        colors = ['green' if s > 0.7 else 'yellow' if s > 0.5 else 'red' for s in synthesizability]
        ax3.scatter(x, synthesizability, c=colors, s=60, alpha=0.7, edgecolors='black')
        ax3.axhline(y=0.7, color='green', linestyle='--', linewidth=1, label='Good threshold')
        ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Fair threshold')
        ax3.set_xlabel('Material Index', fontsize=12)
        ax3.set_ylabel('Synthesizability Score', fontsize=12)
        ax3.set_title('Synthesizability', fontsize=13, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"稳定性曲线已保存至: {save_path}")


def plot_generated_structures(structures_info: List[Dict],
                              save_path: str = "results/generated_structures.png",
                              max_show: int = 9):
    """
    绘制生成的结构信息摘要图
    
    Args:
        structures_info: 结构信息列表
        save_path: 保存路径
        max_show: 最多显示数量
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n = min(len(structures_info), max_show)
    rows = int(np.ceil(n / 3))
    
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        info = structures_info[i]
        
        # 绘制雷达图形式的属性展示
        properties = ['HER\nActivity', 'Stability', 'Synth-\nability']
        values = [
            info.get('her_score', 0.5),
            info.get('stability_score', 0.5),
            info.get('synthesizability', 0.5)
        ]
        
        # 创建条形图
        colors = ['green' if v > 0.7 else 'yellow' if v > 0.5 else 'red' for v in values]
        bars = ax.barh(properties, values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlim(0, 1)
        ax.set_title(f"{info.get('formula', f'Material {i+1}')}\nΔG_H: {info.get('delta_g', 0):.3f} eV",
                    fontsize=11, fontweight='bold')
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
        
        # 在条形上添加数值
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=10)
    
    # 隐藏多余的子图
    for i in range(n, rows * 3):
        row, col = i // 3, i % 3
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Material Structures Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"结构摘要图已保存至: {save_path}")


def plot_comparison_table(baseline_results: Dict,
                          our_results: Dict,
                          save_path: str = "results/comparison_table.png"):
    """
    绘制与baseline的对比表
    
    Args:
        baseline_results: Baseline结果
        our_results: 我们的结果
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # 表格数据
    columns = ['Method', 'Avg HER ΔG (eV)', 'Stability Score', 'Synthesis Success Rate']
    
    baseline_row = [
        'Baseline',
        f"{baseline_results.get('avg_delta_g', 0):.3f}",
        f"{baseline_results.get('stability', 0):.3f}",
        f"{baseline_results.get('synthesis_rate', 0):.1%}"
    ]
    
    our_row = [
        'Ours',
        f"{our_results.get('avg_delta_g', 0):.3f} ↓",
        f"{our_results.get('stability', 0):.3f} ↑",
        f"{our_results.get('synthesis_rate', 0):.1%} ↑"
    ]
    
    # 创建表格
    table = ax.table(
        cellText=[baseline_row, our_row],
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#4CAF50'] * 4,
        cellColours=[['#f0f0f0'] * 4, ['#e8f5e9'] * 4]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 设置标题
    ax.set_title('Performance Comparison with Baseline', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"对比表已保存至: {save_path}")


def plot_training_metrics(metrics_history: Dict[str, List[float]],
                          save_path: str = "results/training_metrics.png"):
    """
    绘制训练过程中的各项指标
    
    Args:
        metrics_history: 指标历史记录
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_metrics = len(metrics_history)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for ax, (name, values), color in zip(axes, metrics_history.items(), colors):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, color=color, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注最终值
        ax.annotate(f'{values[-1]:.4f}', xy=(len(values), values[-1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.suptitle('Training Metrics Over Time', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"训练指标图已保存至: {save_path}")


def generate_all_plots(results: Dict, save_dir: str = "results"):
    """
    生成所有可视化图表
    
    Args:
        results: 包含所有结果的字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 损失曲线
    if 'train_losses' in results:
        plot_loss_curve(
            results['train_losses'],
            results.get('val_losses'),
            os.path.join(save_dir, 'loss_curve.png')
        )
    
    # HER性能图
    if 'delta_g_values' in results:
        plot_her_performance(
            results['delta_g_values'],
            results.get('material_labels'),
            os.path.join(save_dir, 'her_performance.png')
        )
    
    # 稳定性曲线
    if 'formation_energies' in results and 'stability_scores' in results:
        plot_stability_curve(
            results['formation_energies'],
            results['stability_scores'],
            results.get('synthesizability'),
            os.path.join(save_dir, 'stability_curve.png')
        )
    
    # 生成结构图
    if 'structures_info' in results:
        plot_generated_structures(
            results['structures_info'],
            os.path.join(save_dir, 'generated_structures.png')
        )
    
    logger.info(f"所有可视化图表已保存至: {save_dir}")


if __name__ == "__main__":
    print("测试可视化模块...")
    
    # 生成测试数据
    np.random.seed(42)
    
    # 测试损失曲线
    train_losses = [1.0 - 0.8 * (1 - np.exp(-i/10)) + np.random.normal(0, 0.02) 
                    for i in range(50)]
    val_losses = [1.0 - 0.7 * (1 - np.exp(-i/10)) + np.random.normal(0, 0.03) 
                  for i in range(50)]
    plot_loss_curve(train_losses, val_losses)
    
    # 测试HER性能图
    delta_g_values = np.random.normal(0, 0.15, 30).tolist()
    plot_her_performance(delta_g_values)
    
    # 测试稳定性曲线
    formation_energies = np.random.uniform(-0.5, 0.3, 20).tolist()
    stability_scores = np.random.uniform(0.3, 1.0, 20).tolist()
    synthesizability = np.random.uniform(0.4, 1.0, 20).tolist()
    plot_stability_curve(formation_energies, stability_scores, synthesizability)
    
    # 测试结构摘要图
    structures_info = [
        {'formula': 'MoS2', 'delta_g': 0.05, 'her_score': 0.9, 
         'stability_score': 0.85, 'synthesizability': 0.95},
        {'formula': 'WS2', 'delta_g': 0.08, 'her_score': 0.85, 
         'stability_score': 0.8, 'synthesizability': 0.9},
        {'formula': 'MoSe2', 'delta_g': -0.03, 'her_score': 0.92, 
         'stability_score': 0.75, 'synthesizability': 0.85},
    ]
    plot_generated_structures(structures_info)
    
    print("所有测试图表已生成在 results/ 目录下")


