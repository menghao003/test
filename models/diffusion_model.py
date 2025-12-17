"""
扩散模型核心模块
基于图神经网络的晶体结构扩散模型，用于生成二维材料
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import math
from typing import Optional, Tuple, List


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EquivariantGraphConv(MessagePassing):
    """SE(3)等变图卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 1):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 节点特征变换
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # 边特征变换
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # 坐标更新网络
        self.coord_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, 1)
        )
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, 
                edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, in_channels]
            pos: 原子坐标 [N, 3]
            edge_index: 边索引 [2, E]
            edge_attr: 边属性 [E, edge_dim]
        
        Returns:
            更新后的节点特征和坐标
        """
        # 计算相对位置
        row, col = edge_index
        rel_pos = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        if edge_attr is None:
            edge_attr = dist
        else:
            edge_attr = torch.cat([edge_attr, dist], dim=-1)
        
        # 消息传递
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, rel_pos=rel_pos)
        x_new = self.node_mlp(x) + x_new
        
        return x_new, pos
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_attr: torch.Tensor, rel_pos: torch.Tensor) -> torch.Tensor:
        """消息函数"""
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.edge_mlp(msg_input)
        return msg


class CrystalDiffusionBlock(nn.Module):
    """晶体扩散模型基本块"""
    
    def __init__(self, hidden_dim: int, time_dim: int, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 时间嵌入投影
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 图卷积层
        self.conv_layers = nn.ModuleList([
            EquivariantGraphConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, 
                edge_index: torch.Tensor, time_emb: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征
            pos: 原子坐标
            edge_index: 边索引
            time_emb: 时间嵌入
            batch: 批次索引
        """
        # 添加时间信息
        if batch is not None:
            time_proj = self.time_proj(time_emb)[batch]
        else:
            time_proj = self.time_proj(time_emb).expand(x.size(0), -1)
        
        h = x + time_proj
        
        # 图卷积
        for conv, norm in zip(self.conv_layers, self.norms):
            h_new, pos = conv(h, pos, edge_index)
            h = norm(h + h_new)
        
        return h, pos


class CrystalDiffusionModel(nn.Module):
    """
    晶体结构扩散模型
    
    基于去噪扩散概率模型(DDPM)的晶体结构生成模型，
    使用图神经网络处理晶体的原子和键结构。
    """
    
    def __init__(self, 
                 num_atom_types: int = 100,
                 hidden_dim: int = 256,
                 time_dim: int = 128,
                 num_blocks: int = 4,
                 num_layers_per_block: int = 3,
                 num_timesteps: int = 1000,
                 condition_dim: int = 3):  # ΔG_H, 稳定性, 可合成性
        super().__init__()
        
        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.condition_dim = condition_dim
        
        # 原子类型嵌入
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        # 时间嵌入
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # 条件嵌入（HER活性、稳定性、可合成性）
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_dim)
        )
        
        # 扩散块
        self.blocks = nn.ModuleList([
            CrystalDiffusionBlock(hidden_dim, time_dim, num_layers_per_block)
            for _ in range(num_blocks)
        ])
        
        # 输出层
        self.atom_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types)
        )
        
        self.coord_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # 噪声调度参数
        self._setup_noise_schedule()
    
    def _setup_noise_schedule(self):
        """设置噪声调度"""
        # 线性调度
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
    def forward(self, batch: Data, timesteps: torch.Tensor, 
                conditions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，预测噪声
        
        Args:
            batch: 批次数据
            timesteps: 时间步 [B]
            conditions: 条件向量 [B, condition_dim]
        
        Returns:
            预测的原子类型噪声和坐标噪声
        """
        # 获取输入
        x = batch.x  # 原子类型
        pos = batch.pos  # 原子坐标
        edge_index = batch.edge_index
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        
        # 原子嵌入
        h = self.atom_embedding(x.long().squeeze(-1))
        
        # 时间嵌入
        time_emb = self.time_embedding(timesteps.float())
        
        # 条件嵌入
        if conditions is not None:
            cond_emb = self.condition_embedding(conditions)
            time_emb = time_emb + cond_emb
        
        # 扩散块处理
        for block in self.blocks:
            h, pos = block(h, pos, edge_index, time_emb, batch_idx)
        
        # 输出预测
        atom_noise_pred = self.atom_output(h)
        coord_noise_pred = self.coord_output(h)
        
        return atom_noise_pred, coord_noise_pred
    
    def add_noise(self, x: torch.Tensor, pos: torch.Tensor, 
                  t: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        添加噪声到数据
        
        Args:
            x: 原子类型（one-hot或索引）
            pos: 原子坐标
            t: 时间步
            batch: 批次索引
        """
        # 获取噪声参数
        if batch is not None:
            sqrt_alpha = self.sqrt_alphas_cumprod[t][batch].unsqueeze(-1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][batch].unsqueeze(-1)
        else:
            sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        # 坐标噪声
        coord_noise = torch.randn_like(pos)
        noisy_pos = sqrt_alpha * pos + sqrt_one_minus_alpha * coord_noise
        
        # 原子类型噪声（使用mask策略）
        atom_noise = torch.randn(x.size(0), self.num_atom_types, device=x.device)
        
        return x, noisy_pos, atom_noise, coord_noise
    
    @torch.no_grad()
    def sample(self, num_atoms: int, num_samples: int = 1,
               conditions: Optional[torch.Tensor] = None,
               device: str = 'cpu') -> List[Data]:
        """
        从模型采样生成新结构
        
        Args:
            num_atoms: 每个结构的原子数
            num_samples: 生成的样本数
            conditions: 条件向量 [num_samples, condition_dim]
            device: 计算设备
        
        Returns:
            生成的结构列表
        """
        self.eval()
        
        # 初始化随机结构
        x = torch.randint(0, self.num_atom_types, (num_samples * num_atoms, 1), device=device)
        pos = torch.randn(num_samples * num_atoms, 3, device=device)
        
        # 构建批次
        batch_idx = torch.arange(num_samples, device=device).repeat_interleave(num_atoms)
        
        # 简单的全连接图（可以改进）
        edge_list = []
        for i in range(num_samples):
            start = i * num_atoms
            for j in range(num_atoms):
                for k in range(j + 1, num_atoms):
                    edge_list.append([start + j, start + k])
                    edge_list.append([start + k, start + j])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, device=device).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 反向扩散过程
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # 创建批次数据
            batch_data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch_idx)
            
            # 预测噪声
            atom_pred, coord_pred = self(batch_data, t_batch, conditions)
            
            # 去噪步骤
            if t > 0:
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                alpha_cumprod_prev = self.alphas_cumprod_prev[t]
                beta = self.betas[t]
                
                # 更新坐标
                coef1 = 1 / torch.sqrt(alpha)
                coef2 = beta / torch.sqrt(1 - alpha_cumprod)
                
                pos = coef1 * (pos - coef2 * coord_pred)
                
                # 添加噪声
                if t > 1:
                    noise = torch.randn_like(pos)
                    sigma = torch.sqrt(beta)
                    pos = pos + sigma * noise
        
        # 转换为结构列表
        structures = []
        for i in range(num_samples):
            mask = batch_idx == i
            structure = Data(
                x=x[mask],
                pos=pos[mask],
                num_atoms=num_atoms
            )
            structures.append(structure)
        
        return structures


class ConditionalDiffusionModel(CrystalDiffusionModel):
    """
    条件扩散模型
    
    支持基于目标属性（HER活性、稳定性、可合成性）的条件生成
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 属性预测头（用于引导生成）
        self.property_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.condition_dim)
        )
    
    def predict_properties(self, batch: Data) -> torch.Tensor:
        """
        预测材料属性
        
        Args:
            batch: 批次数据
        
        Returns:
            预测的属性 [B, condition_dim]
        """
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        
        # 嵌入
        h = self.atom_embedding(x.long().squeeze(-1))
        
        # 无时间步的前向传播
        time_emb = torch.zeros(1, 128, device=x.device)
        pos = batch.pos if hasattr(batch, 'pos') else torch.zeros(x.size(0), 3, device=x.device)
        
        for block in self.blocks:
            h, pos = block(h, pos, edge_index, time_emb, batch_idx)
        
        # 图级别池化
        if batch_idx is not None:
            h_graph = global_mean_pool(h, batch_idx)
        else:
            h_graph = h.mean(dim=0, keepdim=True)
        
        # 预测属性
        properties = self.property_predictor(h_graph)
        
        return properties
    
    @torch.no_grad()
    def guided_sample(self, num_atoms: int, num_samples: int = 1,
                      target_properties: Optional[torch.Tensor] = None,
                      guidance_scale: float = 1.0,
                      device: str = 'cpu') -> List[Data]:
        """
        引导采样生成符合目标属性的结构
        
        Args:
            num_atoms: 每个结构的原子数
            num_samples: 样本数
            target_properties: 目标属性 [num_samples, condition_dim]
            guidance_scale: 引导强度
            device: 计算设备
        """
        return self.sample(
            num_atoms=num_atoms,
            num_samples=num_samples,
            conditions=target_properties,
            device=device
        )


if __name__ == "__main__":
    # 测试模型
    print("测试扩散模型...")
    
    # 创建模型
    model = ConditionalDiffusionModel(
        num_atom_types=100,
        hidden_dim=128,
        time_dim=64,
        num_blocks=2,
        num_layers_per_block=2,
        num_timesteps=100
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试采样
    target_props = torch.tensor([[0.0, 0.8, 0.9]])  # ΔG_H≈0, 高稳定性, 高可合成性
    samples = model.guided_sample(
        num_atoms=6,
        num_samples=1,
        target_properties=target_props,
        device='cpu'
    )
    
    print(f"生成了 {len(samples)} 个结构")
    print(f"第一个结构: {samples[0].num_atoms} 个原子")


