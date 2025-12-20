"""
优化模块
实现HER催化活性、稳定性和可合成性的多目标优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HERActivityLoss(nn.Module):
    """
    HER催化活性损失函数
    
    最小化生成材料的ΔG_H值，使其尽可能接近0 eV
    根据Sabatier原理，最优HER催化剂的ΔG_H应接近0
    """
    
    def __init__(self, target_delta_g: float = 0.0, weight: float = 1.0):
        super().__init__()
        self.target_delta_g = target_delta_g
        self.weight = weight
    
    def forward(self, predicted_delta_g: torch.Tensor,
                target_delta_g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算HER活性损失
        
        Args:
            predicted_delta_g: 预测的ΔG_H值 [B, 1]
            target_delta_g: 目标ΔG_H值（可选）
        
        Returns:
            损失值
        """
        if target_delta_g is None:
            target = torch.full_like(predicted_delta_g, self.target_delta_g)
        else:
            target = target_delta_g
        
        # L1损失使ΔG_H接近0
        loss = F.l1_loss(predicted_delta_g, target)
        
        # 添加二次惩罚项，对偏离0较远的值给予更大惩罚
        quadratic_penalty = (predicted_delta_g ** 2).mean()
        
        total_loss = self.weight * (loss + 0.1 * quadratic_penalty)
        
        return total_loss


class StabilityLoss(nn.Module):
    """
    稳定性损失函数
    
    结合热力学稳定性和动力学稳定性
    - 热力学稳定性：形成能 < 0
    - 动力学稳定性：结构不会自发分解
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, 
                formation_energy: torch.Tensor,
                predicted_stability: torch.Tensor,
                target_stability: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算稳定性损失
        
        Args:
            formation_energy: 形成能 [B, 1]
            predicted_stability: 预测的稳定性分数 [B, 1]
            target_stability: 目标稳定性（可选）
        
        Returns:
            损失值
        """
        # 热力学稳定性损失：形成能应该为负
        # 使用ReLU惩罚正的形成能
        thermodynamic_loss = F.relu(formation_energy).mean()
        
        # 动力学稳定性损失：最大化稳定性分数
        if target_stability is None:
            target_stability = torch.ones_like(predicted_stability)
        
        kinetic_loss = F.mse_loss(predicted_stability, target_stability)
        
        total_loss = self.weight * (thermodynamic_loss + kinetic_loss)
        
        return total_loss


class SynthesizabilityLoss(nn.Module):
    """
    可合成性损失函数
    
    评估生成材料的实验可合成性：
    - 元素组合合理性
    - 结构复杂度
    - 与已知可合成材料的相似度
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
        # 可合成性评估网络
        self.synthesizability_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                embeddings: torch.Tensor,
                predicted_synth: torch.Tensor,
                target_synth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算可合成性损失
        
        Args:
            embeddings: 结构嵌入 [B, hidden_dim]
            predicted_synth: 预测的可合成性 [B, 1]
            target_synth: 目标可合成性（可选）
        
        Returns:
            损失值
        """
        if target_synth is None:
            target_synth = torch.ones_like(predicted_synth) * 0.8
        
        # 可合成性预测损失
        loss = F.binary_cross_entropy(predicted_synth, target_synth)
        
        return self.weight * loss


class MultiTaskOptimizer(nn.Module):
    """
    多任务联合优化器
    
    同时优化HER催化活性、稳定性和可合成性
    使用动态权重调整策略平衡多个目标
    """
    
    def __init__(self,
                 her_weight: float = 1.0,
                 stability_weight: float = 1.0,
                 synth_weight: float = 0.5,
                 use_dynamic_weights: bool = True):
        super().__init__()
        
        self.her_loss = HERActivityLoss(weight=her_weight)
        self.stability_loss = StabilityLoss(weight=stability_weight)
        self.synth_loss = SynthesizabilityLoss(weight=synth_weight)
        
        self.use_dynamic_weights = use_dynamic_weights
        
        # 可学习的任务权重
        if use_dynamic_weights:
            self.log_weights = nn.Parameter(torch.zeros(3))
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        计算联合损失
        
        Args:
            predictions: 预测值字典
                - delta_g: ΔG_H预测值
                - formation_energy: 形成能
                - stability: 稳定性分数
                - synthesizability: 可合成性分数
                - embeddings: 结构嵌入
            targets: 目标值字典（可选）
        
        Returns:
            损失字典
        """
        if targets is None:
            targets = {}
        
        # HER活性损失
        her_loss = self.her_loss(
            predictions['delta_g'],
            targets.get('delta_g', None)
        )
        
        # 稳定性损失
        stability_loss = self.stability_loss(
            predictions['formation_energy'],
            predictions['stability'],
            targets.get('stability', None)
        )
        
        # 可合成性损失
        synth_loss = self.synth_loss(
            predictions['embeddings'],
            predictions['synthesizability'],
            targets.get('synthesizability', None)
        )
        
        # 动态权重
        if self.use_dynamic_weights:
            weights = F.softmax(self.log_weights, dim=0)
            total_loss = (weights[0] * her_loss + 
                         weights[1] * stability_loss + 
                         weights[2] * synth_loss)
        else:
            total_loss = her_loss + stability_loss + synth_loss
        
        return {
            'total_loss': total_loss,
            'her_loss': her_loss,
            'stability_loss': stability_loss,
            'synth_loss': synth_loss
        }


class PropertyPredictor(nn.Module):
    """
    材料属性预测器
    
    预测材料的HER活性、稳定性和可合成性
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # ΔG_H预测头
        self.delta_g_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 形成能预测头
        self.formation_energy_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 稳定性预测头
        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 可合成性预测头
        self.synth_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测材料属性
        
        Args:
            x: 输入特征 [B, input_dim]
        
        Returns:
            属性预测字典
        """
        h = self.encoder(x)
        
        return {
            'delta_g': self.delta_g_head(h),
            'formation_energy': self.formation_energy_head(h),
            'stability': self.stability_head(h),
            'synthesizability': self.synth_head(h),
            'embeddings': h
        }


class GuidedOptimizer:
    """
    引导优化器
    
    使用梯度引导来优化生成的结构，使其满足目标属性
    """
    
    def __init__(self,
                 property_predictor: PropertyPredictor,
                 learning_rate: float = 0.01,
                 num_steps: int = 100):
        self.predictor = property_predictor
        self.lr = learning_rate
        self.num_steps = num_steps
        self.optimizer_fn = MultiTaskOptimizer()
    
    def optimize(self,
                 initial_embedding: torch.Tensor,
                 target_delta_g: float = 0.0,
                 target_stability: float = 0.8,
                 target_synth: float = 0.8) -> Tuple[torch.Tensor, Dict]:
        """
        优化结构嵌入
        
        Args:
            initial_embedding: 初始嵌入
            target_delta_g: 目标ΔG_H
            target_stability: 目标稳定性
            target_synth: 目标可合成性
        
        Returns:
            优化后的嵌入和历史记录
        """
        embedding = initial_embedding.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([embedding], lr=self.lr)
        
        targets = {
            'delta_g': torch.tensor([[target_delta_g]]),
            'stability': torch.tensor([[target_stability]]),
            'synthesizability': torch.tensor([[target_synth]])
        }
        
        history = {
            'loss': [],
            'delta_g': [],
            'stability': [],
            'synthesizability': []
        }
        
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # 预测属性
            predictions = self.predictor(embedding)
            
            # 计算损失
            losses = self.optimizer_fn(predictions, targets)
            
            # 反向传播
            losses['total_loss'].backward()
            optimizer.step()
            
            # 记录历史
            history['loss'].append(losses['total_loss'].item())
            history['delta_g'].append(predictions['delta_g'].item())
            history['stability'].append(predictions['stability'].item())
            history['synthesizability'].append(predictions['synthesizability'].item())
        
        return embedding.detach(), history


class DiffusionLoss(nn.Module):
    """
    扩散模型训练损失
    
    结合去噪损失和属性预测损失
    """
    
    def __init__(self,
                 noise_weight: float = 1.0,
                 property_weight: float = 0.1):
        super().__init__()
        self.noise_weight = noise_weight
        self.property_weight = property_weight
        self.multi_task_optimizer = MultiTaskOptimizer()
    
    def forward(self,
                noise_pred: torch.Tensor,
                noise_true: torch.Tensor,
                coord_noise_pred: torch.Tensor,
                coord_noise_true: torch.Tensor,
                property_predictions: Optional[Dict] = None,
                property_targets: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        计算扩散模型损失
        
        Args:
            noise_pred: 预测的原子类型噪声
            noise_true: 真实的原子类型噪声
            coord_noise_pred: 预测的坐标噪声
            coord_noise_true: 真实的坐标噪声
            property_predictions: 属性预测（可选）
            property_targets: 属性目标（可选）
        
        Returns:
            损失字典
        """
        # 原子类型去噪损失
        atom_loss = F.cross_entropy(noise_pred, noise_true.long().squeeze(-1))
        
        # 坐标去噪损失
        coord_loss = F.mse_loss(coord_noise_pred, coord_noise_true)
        
        # 总噪声损失
        noise_loss = self.noise_weight * (atom_loss + coord_loss)
        
        # 属性损失
        if property_predictions is not None and property_targets is not None:
            property_losses = self.multi_task_optimizer(
                property_predictions, property_targets
            )
            property_loss = self.property_weight * property_losses['total_loss']
        else:
            property_loss = torch.tensor(0.0)
        
        total_loss = noise_loss + property_loss
        
        return {
            'total_loss': total_loss,
            'noise_loss': noise_loss,
            'atom_loss': atom_loss,
            'coord_loss': coord_loss,
            'property_loss': property_loss
        }


def compute_pareto_front(results: List[Dict]) -> List[int]:
    """
    计算Pareto前沿
    
    Args:
        results: 结果列表，每个元素包含 'delta_g', 'stability', 'synthesizability'
    
    Returns:
        Pareto前沿上的结果索引
    """
    n = len(results)
    pareto_front = []
    
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i == j:
                continue
            
            # 检查j是否支配i
            # 对于delta_g：绝对值越小越好
            # 对于stability和synthesizability：越大越好
            delta_g_i = abs(results[i]['delta_g'])
            delta_g_j = abs(results[j]['delta_g'])
            stab_i = results[i]['stability']
            stab_j = results[j]['stability']
            synth_i = results[i]['synthesizability']
            synth_j = results[j]['synthesizability']
            
            if (delta_g_j <= delta_g_i and stab_j >= stab_i and synth_j >= synth_i and
                (delta_g_j < delta_g_i or stab_j > stab_i or synth_j > synth_i)):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(i)
    
    return pareto_front


if __name__ == "__main__":
    print("测试优化模块...")
    
    # 测试属性预测器
    predictor = PropertyPredictor(input_dim=128)
    x = torch.randn(4, 128)
    predictions = predictor(x)
    
    print("预测结果:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}")
    
    # 测试多任务优化器
    optimizer = MultiTaskOptimizer()
    losses = optimizer(predictions)
    
    print("\n损失值:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # 测试引导优化
    guided_opt = GuidedOptimizer(predictor, num_steps=10)
    optimized_emb, history = guided_opt.optimize(x[0:1])
    
    print(f"\n优化后 ΔG_H: {history['delta_g'][-1]:.4f}")
    print(f"优化后稳定性: {history['stability'][-1]:.4f}")
    print(f"优化后可合成性: {history['synthesizability'][-1]:.4f}")




