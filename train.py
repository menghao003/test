"""
训练脚本
用于训练扩散模型生成二维材料
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.diffusion_model import ConditionalDiffusionModel
from models.optimization import DiffusionLoss, PropertyPredictor, MultiTaskOptimizer
from dataset.material_dataset import MaterialDataset, create_train_val_test_split, NUM_ATOM_TYPES
from utils.vis import plot_loss_curve, plot_training_metrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    扩散模型训练器
    
    支持多任务联合训练，包括：
    - 去噪损失（扩散模型核心）
    - HER活性损失
    - 稳定性损失
    - 可合成性损失
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = 'checkpoints',
                 results_dir: str = 'results'):
        """
        初始化训练器
        
        Args:
            model: 扩散模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 计算设备
            checkpoint_dir: 检查点保存目录
            results_dir: 结果保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        
        # 创建目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # 损失函数
        self.loss_fn = DiffusionLoss(noise_weight=1.0, property_weight=0.1)
        
        # 属性预测器（用于条件训练）
        self.property_predictor = PropertyPredictor(input_dim=model.hidden_dim).to(device)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'her_loss': [],
            'stability_loss': [],
            'synth_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.epoch = 0
    
    def train_epoch(self) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_noise_loss = 0.0
        total_property_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            # 采样时间步
            t = torch.randint(
                0, self.model.num_timesteps,
                (batch.num_graphs,),
                device=self.device
            )
            
            # 添加噪声
            x, noisy_pos, atom_noise, coord_noise = self.model.add_noise(
                batch.x, batch.pos, t, batch.batch
            )
            
            # 创建带噪声的数据
            noisy_batch = batch.clone()
            noisy_batch.pos = noisy_pos
            
            # 准备条件
            conditions = torch.stack([
                batch.delta_g.squeeze(),
                batch.stability.squeeze(),
                batch.synthesizability.squeeze()
            ], dim=1).float()
            
            # 前向传播
            atom_pred, coord_pred = self.model(noisy_batch, t, conditions)
            
            # 计算损失
            losses = self.loss_fn(
                atom_pred, batch.x,
                coord_pred, coord_noise
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            total_noise_loss += losses['noise_loss'].item()
            total_property_loss += losses['property_loss'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_noise_loss = total_noise_loss / num_batches
        avg_property_loss = total_property_loss / num_batches
        
        return {
            'loss': avg_loss,
            'noise_loss': avg_noise_loss,
            'property_loss': avg_property_loss
        }
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """验证模型"""
        if self.val_loader is None:
            return {'loss': 0.0}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = batch.to(self.device)
            
            t = torch.randint(
                0, self.model.num_timesteps,
                (batch.num_graphs,),
                device=self.device
            )
            
            x, noisy_pos, atom_noise, coord_noise = self.model.add_noise(
                batch.x, batch.pos, t, batch.batch
            )
            
            noisy_batch = batch.clone()
            noisy_batch.pos = noisy_pos
            
            conditions = torch.stack([
                batch.delta_g.squeeze(),
                batch.stability.squeeze(),
                batch.synthesizability.squeeze()
            ], dim=1).float()
            
            atom_pred, coord_pred = self.model(noisy_batch, t, conditions)
            
            losses = self.loss_fn(
                atom_pred, batch.x,
                coord_pred, coord_noise
            )
            
            total_loss += losses['total_loss'].item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def train(self, num_epochs: int, save_every: int = 10):
        """
        完整训练流程
        
        Args:
            num_epochs: 训练轮数
            save_every: 每多少轮保存一次检查点
        """
        logger.info(f"开始训练，共 {num_epochs} 轮")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练样本数: {len(self.train_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)
            
            # 日志
            logger.info(
                f"Epoch {self.epoch}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                logger.info(f"保存最佳模型 (Val Loss: {self.best_val_loss:.4f})")
            
            # 定期保存
            if self.epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pt')
        
        # 保存最终模型和可视化
        self.save_checkpoint('final_model.pt')
        self.save_training_plots()
        
        logger.info("训练完成!")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        logger.info(f"加载检查点: {filename}, Epoch: {self.epoch}")
    
    def save_training_plots(self):
        """保存训练可视化"""
        # 损失曲线
        plot_loss_curve(
            self.history['train_loss'],
            self.history['val_loss'],
            str(self.results_dir / 'loss_curve.png')
        )
        
        # 训练指标
        plot_training_metrics(
            {
                'Train Loss': self.history['train_loss'],
                'Val Loss': self.history['val_loss'],
                'Learning Rate': self.history['learning_rate']
            },
            str(self.results_dir / 'training_metrics.png')
        )


def create_data_loaders(data_dir: str,
                        batch_size: int = 32,
                        num_workers: int = 0,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1):
    """创建数据加载器"""
    dataset = MaterialDataset(
        root_dir=data_dir,
        cutoff=5.0,
        max_neighbors=12
    )
    
    if len(dataset) == 0:
        raise ValueError(f"数据集为空: {data_dir}")
    
    # 划分数据集
    train_idx, val_idx, test_idx = create_train_val_test_split(
        dataset, train_ratio, val_ratio
    )
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"数据集划分: 训练 {len(train_idx)}, 验证 {len(val_idx)}, 测试 {len(test_idx)}")
    
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='训练二维材料生成扩散模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/2d_materials',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='扩散块数量')
    parser.add_argument('--num_timesteps', type=int, default=500,
                        help='扩散时间步数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点目录')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='结果目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            args.data_dir,
            args.batch_size
        )
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        logger.info("使用模拟数据进行训练...")
        train_loader, val_loader, test_loader = create_mock_data_loaders(args.batch_size)
    
    # 创建模型
    model = ConditionalDiffusionModel(
        num_atom_types=NUM_ATOM_TYPES + 1,  # +1 for unknown elements
        hidden_dim=args.hidden_dim,
        time_dim=64,
        num_blocks=args.num_blocks,
        num_timesteps=args.num_timesteps,
        condition_dim=3
    )
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(num_epochs=args.epochs)
    
    # 保存训练配置
    config = vars(args)
    config['device'] = device
    config['num_parameters'] = sum(p.numel() for p in model.parameters())
    
    with open(Path(args.results_dir) / 'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)


def create_mock_data_loaders(batch_size: int):
    """创建模拟数据加载器（用于测试）"""
    from torch_geometric.data import Data, Batch
    
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            num_atoms = np.random.randint(4, 12)
            return Data(
                x=torch.randint(0, NUM_ATOM_TYPES, (num_atoms, 1)),
                pos=torch.randn(num_atoms, 3),
                edge_index=torch.randint(0, num_atoms, (2, num_atoms * 4)),
                delta_g=torch.randn(1) * 0.2,
                stability=torch.rand(1),
                synthesizability=torch.rand(1)
            )
    
    train_dataset = MockDataset(80)
    val_dataset = MockDataset(20)
    test_dataset = MockDataset(20)
    
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()

