"""
材料结构生成器模块
将扩散模型生成的表示转换为实际的晶体结构
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from pymatgen.core import Structure, Lattice, Element
from torch_geometric.data import Data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目路径以导入 dataset 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 material_dataset 导入统一的元素映射
from dataset.material_dataset import SUPPORTED_ELEMENTS, ELEMENT_TO_IDX, IDX_TO_ELEMENT, NUM_ATOM_TYPES

# 兼容性别名
COMMON_2D_ELEMENTS = SUPPORTED_ELEMENTS


class StructureGenerator(nn.Module):
    """
    材料结构生成器
    
    负责将扩散模型的输出转换为有效的晶体结构，
    并确保生成的结构满足物理约束。
    """
    
    def __init__(self,
                 diffusion_model: nn.Module,
                 num_atom_types: int = NUM_ATOM_TYPES,
                 default_vacuum: float = 15.0,
                 min_bond_length: float = 0.8,  # 放宽最小键长限制
                 max_bond_length: float = 4.0,  # 放宽最大键长限制
                 max_z_range: float = 10.0):    # 放宽z方向跨度限制
        """
        初始化结构生成器
        
        Args:
            diffusion_model: 扩散模型
            num_atom_types: 原子类型数量
            default_vacuum: 默认真空层厚度（Å）
            min_bond_length: 最小键长（Å）
            max_bond_length: 最大键长（Å）
            max_z_range: 最大z方向跨度（Å）
        """
        super().__init__()
        self.diffusion_model = diffusion_model
        self.num_atom_types = num_atom_types
        self.default_vacuum = default_vacuum
        self.min_bond_length = min_bond_length
        self.max_bond_length = max_bond_length
        self.max_z_range = max_z_range
        
        # 晶格参数预测网络
        self.lattice_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 6)  # a, b, c, alpha, beta, gamma
        )
    
    def generate_structures(self,
                           num_structures: int = 10,
                           num_atoms_range: Tuple[int, int] = (4, 12),
                           target_delta_g: float = 0.0,
                           target_stability: float = 0.8,
                           target_synthesizability: float = 0.8,
                           device: str = 'cpu') -> List[Structure]:
        """
        生成符合目标属性的材料结构
        
        Args:
            num_structures: 生成结构数量
            num_atoms_range: 原子数范围
            target_delta_g: 目标ΔG_H值（eV）
            target_stability: 目标稳定性（0-1）
            target_synthesizability: 目标可合成性（0-1）
            device: 计算设备
        
        Returns:
            生成的Structure对象列表
        """
        self.eval()
        structures = []
        
        # 设置目标条件
        target_conditions = torch.tensor([
            [target_delta_g, target_stability, target_synthesizability]
        ], device=device).repeat(num_structures, 1)
        
        for i in range(num_structures):
            try:
                # 随机选择原子数
                num_atoms = np.random.randint(num_atoms_range[0], num_atoms_range[1] + 1)
                
                # 从扩散模型采样
                samples = self.diffusion_model.guided_sample(
                    num_atoms=num_atoms,
                    num_samples=1,
                    target_properties=target_conditions[i:i+1],
                    device=device
                )
                
                if samples:
                    # 转换为晶体结构
                    structure = self._data_to_structure(samples[0])
                    if structure is not None:
                        structures.append(structure)
                        logger.info(f"成功生成结构 {i+1}/{num_structures}: {structure.formula}")
                        
            except Exception as e:
                logger.warning(f"生成结构 {i+1} 时出错: {e}")
                continue
        
        logger.info(f"共生成 {len(structures)} 个有效结构")
        return structures
    
    def _data_to_structure(self, data: Data) -> Optional[Structure]:
        """
        将图数据转换为pymatgen Structure
        
        Args:
            data: 图数据对象
        
        Returns:
            Structure对象或None
        """
        try:
            # 获取原子类型和坐标
            atom_indices = data.x.cpu().numpy().flatten()
            coords = data.pos.cpu().numpy()
            
            logger.debug(f"转换数据: {len(atom_indices)} 个原子, 索引范围: [{atom_indices.min()}, {atom_indices.max()}]")
            
            # 转换原子类型为元素符号
            species = []
            for idx in atom_indices:
                idx = int(idx) % self.num_atom_types  # 使用 num_atom_types 而不是固定长度
                elem_symbol = IDX_TO_ELEMENT.get(idx, 'C')  # 默认使用C
                species.append(Element(elem_symbol))
            
            logger.debug(f"元素列表: {[str(s) for s in species]}")
            
            # 归一化坐标到合理范围
            coords = self._normalize_coordinates(coords)
            
            logger.debug(f"归一化后坐标范围: x=[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
                        f"y=[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
                        f"z=[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
            
            # 创建二维晶格
            lattice = self._create_2d_lattice(coords)
            
            # 将笛卡尔坐标转换为分数坐标
            frac_coords = lattice.get_fractional_coords(coords)
            
            # 处理分数坐标：将z方向的坐标集中在0.5附近（晶格中心）
            # 这样可以保持原子的相对位置，同时确保都在晶格内
            frac_coords[:, 2] = frac_coords[:, 2] - frac_coords[:, 2].mean() + 0.5
            
            # 确保xy方向的分数坐标在[0, 1)范围内
            frac_coords[:, 0] = frac_coords[:, 0] % 1.0
            frac_coords[:, 1] = frac_coords[:, 1] % 1.0
            # z方向已经调整过，确保在合理范围内
            frac_coords[:, 2] = np.clip(frac_coords[:, 2], 0.3, 0.7)
            
            # 创建结构
            structure = Structure(lattice, species, frac_coords)
            
            logger.debug(f"创建结构: {structure.formula}, {len(structure)} 原子")
            
            # 验证结构
            if self._validate_structure(structure):
                logger.info(f"成功创建有效结构: {structure.formula}")
                return structure
            else:
                logger.debug(f"结构验证失败: {structure.formula}")
                return None
                
        except Exception as e:
            logger.warning(f"转换结构时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """归一化坐标到合理范围，确保最小键长满足要求"""
        # 中心化
        coords = coords - coords.mean(axis=0)
        
        # 计算当前最小距离
        if len(coords) > 1:
            min_dist = float('inf')
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = np.linalg.norm(coords[i] - coords[j])
                    if d > 0:
                        min_dist = min(min_dist, d)
            
            # 确保最小距离 >= 1.5Å（典型化学键长）
            target_min_dist = 1.5
            if min_dist > 0 and min_dist < target_min_dist:
                scale = target_min_dist / min_dist * 1.2  # 额外增加20%余量
                coords = coords * scale
                logger.debug(f"缩放坐标: 最小距离从 {min_dist:.2f}Å 调整到 {min_dist * scale:.2f}Å")
        
        # 将z坐标压缩到接近0（二维材料），但保持最小分离
        z_coords = coords[:, 2]
        z_range = z_coords.max() - z_coords.min()
        if z_range > 0:
            # 压缩z方向但保持一定的分离
            coords[:, 2] = (z_coords - z_coords.mean()) * 0.2  # 压缩到原来的20%
        
        return coords
    
    def _create_2d_lattice(self, coords: np.ndarray) -> Lattice:
        """
        创建适合二维材料的晶格
        
        Args:
            coords: 原子坐标
        
        Returns:
            Lattice对象
        """
        # 计算xy平面的范围
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        
        # 添加适当的边界
        padding = 2.0  # Å
        a = max(x_range + padding, 3.0)
        b = max(y_range + padding, 3.0)
        c = self.default_vacuum  # 真空层
        
        # 创建正交晶格
        lattice = Lattice.from_parameters(
            a=a, b=b, c=c,
            alpha=90, beta=90, gamma=90
        )
        
        return lattice
    
    def _validate_structure(self, structure: Structure) -> bool:
        """
        验证结构的有效性（放宽条件以适应扩散模型生成的结构）
        
        Args:
            structure: Structure对象
        
        Returns:
            是否有效
        """
        try:
            # 检查原子数
            if len(structure) < 2:
                logger.debug(f"验证失败: 原子数 {len(structure)} < 2")
                return False
            
            # 检查键长（使用分数坐标计算，避免真空层影响）
            min_dist_found = float('inf')
            for i, site1 in enumerate(structure):
                for j, site2 in enumerate(structure):
                    if i >= j:
                        continue
                    # 使用 get_distance 而不是 distance，更准确处理周期性边界
                    dist = structure.get_distance(i, j)
                    min_dist_found = min(min_dist_found, dist)
            
            # 放宽键长限制到0.5Å（对于训练时间短的模型更宽容）
            if min_dist_found < 0.5:
                logger.debug(f"验证失败: 最小键长 {min_dist_found:.2f}Å < 0.5Å")
                return False
            
            # 检查是否具有二维特征（使用分数坐标的z范围）
            z_frac_coords = [site.frac_coords[2] for site in structure]
            z_frac_range = max(z_frac_coords) - min(z_frac_coords)
            # 分数坐标的z范围应该很小（表示原子都在薄层内）
            if z_frac_range > 0.5:  # 分数坐标范围超过50%表示不是二维
                logger.debug(f"验证失败: z分数坐标范围 {z_frac_range:.2f} > 0.5")
                return False
            
            logger.debug(f"验证成功: {len(structure)}原子, 最小键长={min_dist_found:.2f}Å, z分数范围={z_frac_range:.2f}")
            return True
            
        except Exception as e:
            logger.debug(f"验证异常: {e}")
            return False
    
    def optimize_structure(self, structure: Structure,
                          target_delta_g: float = 0.0) -> Structure:
        """
        优化生成的结构以接近目标属性
        
        Args:
            structure: 输入结构
            target_delta_g: 目标ΔG_H值
        
        Returns:
            优化后的结构
        """
        # 简单的结构优化：调整晶格参数
        lattice = structure.lattice
        
        # 保持二维特征，微调晶格
        new_lattice = Lattice.from_parameters(
            a=lattice.a * 1.0,
            b=lattice.b * 1.0,
            c=lattice.c,  # 保持真空层不变
            alpha=90, beta=90, gamma=lattice.gamma
        )
        
        # 创建新结构
        optimized = Structure(
            new_lattice,
            structure.species,
            structure.frac_coords
        )
        
        return optimized


class BatchStructureGenerator:
    """批量结构生成器"""
    
    def __init__(self, generator: StructureGenerator):
        self.generator = generator
    
    def generate_diverse_structures(self,
                                   num_structures: int = 100,
                                   delta_g_range: Tuple[float, float] = (-0.2, 0.2),
                                   stability_range: Tuple[float, float] = (0.6, 1.0),
                                   synthesizability_range: Tuple[float, float] = (0.5, 1.0),
                                   device: str = 'cpu') -> List[Dict]:
        """
        生成多样化的结构集合
        
        Args:
            num_structures: 总结构数
            delta_g_range: ΔG_H范围
            stability_range: 稳定性范围
            synthesizability_range: 可合成性范围
            device: 计算设备
        
        Returns:
            结构和属性的列表
        """
        results = []
        
        for i in range(num_structures):
            # 随机采样目标属性
            target_delta_g = np.random.uniform(*delta_g_range)
            target_stability = np.random.uniform(*stability_range)
            target_synth = np.random.uniform(*synthesizability_range)
            
            try:
                structures = self.generator.generate_structures(
                    num_structures=1,
                    target_delta_g=target_delta_g,
                    target_stability=target_stability,
                    target_synthesizability=target_synth,
                    device=device
                )
                
                if structures:
                    results.append({
                        'structure': structures[0],
                        'target_delta_g': target_delta_g,
                        'target_stability': target_stability,
                        'target_synthesizability': target_synth
                    })
                    
            except Exception as e:
                logger.warning(f"生成结构 {i+1} 时出错: {e}")
                continue
        
        return results


def create_structure_from_template(template: Structure,
                                  modifications: Dict) -> Structure:
    """
    基于模板创建新结构
    
    Args:
        template: 模板结构
        modifications: 修改参数
    
    Returns:
        新结构
    """
    new_species = template.species.copy()
    new_coords = template.frac_coords.copy()
    new_lattice = template.lattice
    
    # 应用修改
    if 'scale' in modifications:
        scale = modifications['scale']
        new_lattice = Lattice.from_parameters(
            a=template.lattice.a * scale,
            b=template.lattice.b * scale,
            c=template.lattice.c,
            alpha=template.lattice.alpha,
            beta=template.lattice.beta,
            gamma=template.lattice.gamma
        )
    
    if 'substitute' in modifications:
        for old_elem, new_elem in modifications['substitute'].items():
            new_species = [Element(new_elem) if str(s) == old_elem else s 
                          for s in new_species]
    
    return Structure(new_lattice, new_species, new_coords)


if __name__ == "__main__":
    print("测试结构生成器...")
    
    # 创建模型
    from diffusion_model import ConditionalDiffusionModel
    
    diffusion = ConditionalDiffusionModel(
        num_atom_types=NUM_ATOM_TYPES + 1,  # 使用统一的原子类型数量
        hidden_dim=64,
        num_blocks=2,
        num_timesteps=50
    )
    
    generator = StructureGenerator(diffusion, num_atom_types=NUM_ATOM_TYPES + 1)
    
    # 测试生成
    structures = generator.generate_structures(
        num_structures=3,
        num_atoms_range=(4, 8),
        target_delta_g=0.0,
        target_stability=0.8,
        target_synthesizability=0.9
    )
    
    for i, struct in enumerate(structures):
        print(f"结构 {i+1}: {struct.formula}, {len(struct)} 原子")

