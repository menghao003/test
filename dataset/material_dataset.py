"""
材料数据集模块
用于加载和处理晶体结构数据
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from pymatgen.core import Structure, Element
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 常见元素列表（支持的原子类型）
SUPPORTED_ELEMENTS = [
    'H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Se', 'Br', 'Te', 'I',  # 非金属
    'B', 'Si', 'P', 'As', 'Sb', 'Bi',                            # 类金属
    'Li', 'Na', 'K', 'Mg', 'Ca', 'Sr', 'Ba',                     # 碱金属/碱土金属
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',        # 3d过渡金属
    'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',             # 4d过渡金属
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',              # 5d过渡金属
    'Al', 'Ga', 'Ge', 'In', 'Sn', 'Pb',                         # 主族金属
]

ELEMENT_TO_IDX = {elem: idx for idx, elem in enumerate(SUPPORTED_ELEMENTS)}
IDX_TO_ELEMENT = {idx: elem for elem, idx in ELEMENT_TO_IDX.items()}
NUM_ATOM_TYPES = len(SUPPORTED_ELEMENTS)


class CrystalGraphData(Data):
    """晶体图数据类"""
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class MaterialDataset(Dataset):
    """
    材料数据集
    
    从CIF文件加载晶体结构，转换为图数据格式
    """
    
    def __init__(self,
                 root_dir: str,
                 properties_file: Optional[str] = None,
                 cutoff: float = 5.0,
                 max_neighbors: int = 12,
                 transform=None):
        """
        初始化数据集
        
        Args:
            root_dir: CIF文件目录
            properties_file: 属性CSV文件路径
            cutoff: 邻居截断距离（Å）
            max_neighbors: 最大邻居数
            transform: 数据变换
        """
        self.root_dir = Path(root_dir)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.transform = transform
        
        # 加载CIF文件列表
        self.cif_files = []
        if self.root_dir.exists():
            self.cif_files = sorted(list(self.root_dir.glob("*.cif")))
        
        logger.info(f"找到 {len(self.cif_files)} 个CIF文件")
        
        # 加载属性数据
        self.properties = {}
        if properties_file and os.path.exists(properties_file):
            self._load_properties(properties_file)
        
        # 缓存已处理的数据
        self.cache = {}
    
    def _load_properties(self, filepath: str):
        """加载属性数据"""
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            material_id = row.get('material_id', row.get('id', ''))
            self.properties[material_id] = {
                'delta_g': row.get('delta_g', row.get('dgH', 0.0)),
                'formation_energy': row.get('formation_energy', row.get('hform', 0.0)),
                'stability': row.get('stability', 0.5),
                'synthesizability': row.get('synthesizability', 0.5)
            }
        
        logger.info(f"加载了 {len(self.properties)} 条属性数据")
    
    def __len__(self) -> int:
        return len(self.cif_files)
    
    def __getitem__(self, idx: int) -> CrystalGraphData:
        """获取单个数据样本"""
        if idx in self.cache:
            return self.cache[idx]
        
        cif_path = self.cif_files[idx]
        
        try:
            # 加载结构
            structure = Structure.from_file(str(cif_path))
            
            # 转换为图数据
            data = self._structure_to_graph(structure)
            
            # 添加属性标签
            material_id = cif_path.stem
            if material_id in self.properties:
                props = self.properties[material_id]
                data.delta_g = torch.tensor([props['delta_g']], dtype=torch.float)
                data.formation_energy = torch.tensor([props['formation_energy']], dtype=torch.float)
                data.stability = torch.tensor([props['stability']], dtype=torch.float)
                data.synthesizability = torch.tensor([props['synthesizability']], dtype=torch.float)
            else:
                # 默认值
                data.delta_g = torch.tensor([0.0], dtype=torch.float)
                data.formation_energy = torch.tensor([0.0], dtype=torch.float)
                data.stability = torch.tensor([0.5], dtype=torch.float)
                data.synthesizability = torch.tensor([0.5], dtype=torch.float)
            
            data.material_id = material_id
            
            if self.transform:
                data = self.transform(data)
            
            # 缓存数据
            self.cache[idx] = data
            
            return data
            
        except Exception as e:
            logger.warning(f"处理 {cif_path.name} 时出错: {e}")
            # 返回空数据
            return self._create_dummy_data()
    
    def _structure_to_graph(self, structure: Structure) -> CrystalGraphData:
        """
        将晶体结构转换为图数据
        
        Args:
            structure: pymatgen Structure对象
        
        Returns:
            CrystalGraphData对象
        """
        # 节点特征：原子类型
        atom_types = []
        positions = []
        
        for site in structure:
            elem = str(site.specie)
            if elem in ELEMENT_TO_IDX:
                atom_types.append(ELEMENT_TO_IDX[elem])
            else:
                atom_types.append(0)  # 未知元素映射到0
            positions.append(site.coords)
        
        x = torch.tensor(atom_types, dtype=torch.long).unsqueeze(-1)
        pos = torch.tensor(positions, dtype=torch.float)
        
        # 构建边（基于距离截断）
        edge_index = []
        edge_attr = []
        
        for i, site1 in enumerate(structure):
            neighbors = structure.get_neighbors(site1, r=self.cutoff)
            neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[:self.max_neighbors]
            
            for neighbor in neighbors:
                j = neighbor.index
                dist = neighbor.nn_distance
                
                edge_index.append([i, j])
                edge_attr.append([dist])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        
        # 晶格参数
        lattice = structure.lattice
        lattice_params = torch.tensor([
            lattice.a, lattice.b, lattice.c,
            lattice.alpha, lattice.beta, lattice.gamma
        ], dtype=torch.float)
        
        # 计算2D特征分数
        c_over_a = lattice.c / lattice.a if lattice.a > 0 else 1.0
        is_2d = 1.0 if c_over_a > 2.0 else 0.0
        
        return CrystalGraphData(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            lattice=lattice_params,
            is_2d=torch.tensor([is_2d], dtype=torch.float),
            num_atoms=len(structure),
            formula=structure.composition.reduced_formula
        )
    
    def _create_dummy_data(self) -> CrystalGraphData:
        """创建空数据"""
        return CrystalGraphData(
            x=torch.zeros((1, 1), dtype=torch.long),
            pos=torch.zeros((1, 3), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1), dtype=torch.float),
            lattice=torch.zeros(6, dtype=torch.float),
            is_2d=torch.tensor([0.0], dtype=torch.float),
            delta_g=torch.tensor([0.0], dtype=torch.float),
            formation_energy=torch.tensor([0.0], dtype=torch.float),
            stability=torch.tensor([0.5], dtype=torch.float),
            synthesizability=torch.tensor([0.5], dtype=torch.float),
            num_atoms=1,
            formula="Unknown"
        )
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'num_samples': len(self),
            'num_with_properties': len(self.properties),
            'atom_types': set(),
            'avg_num_atoms': 0,
            'avg_num_edges': 0
        }
        
        total_atoms = 0
        total_edges = 0
        
        for i in range(min(100, len(self))):  # 采样前100个
            data = self[i]
            total_atoms += data.x.size(0)
            total_edges += data.edge_index.size(1) if data.edge_index.size(1) > 0 else 0
            
            for atom_idx in data.x.numpy().flatten():
                if atom_idx in IDX_TO_ELEMENT:
                    stats['atom_types'].add(IDX_TO_ELEMENT[atom_idx])
        
        n = min(100, len(self))
        if n > 0:
            stats['avg_num_atoms'] = total_atoms / n
            stats['avg_num_edges'] = total_edges / n
        
        stats['atom_types'] = list(stats['atom_types'])
        
        return stats


class InMemoryMaterialDataset(InMemoryDataset):
    """
    内存数据集（用于小数据集）
    
    将所有数据加载到内存中以加速训练
    """
    
    def __init__(self,
                 root: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['materials.csv']
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        # 加载CIF文件
        cif_dir = Path(self.root) / 'raw'
        dataset = MaterialDataset(str(cif_dir))
        
        data_list = []
        for i in range(len(dataset)):
            data = dataset[i]
            if data.x.size(0) > 1:  # 过滤无效数据
                data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_train_val_test_split(dataset: MaterialDataset,
                                train_ratio: float = 0.8,
                                val_ratio: float = 0.1,
                                seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    创建训练/验证/测试集划分
    
    Args:
        dataset: 数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        训练/验证/测试集索引
    """
    np.random.seed(seed)
    
    n = len(dataset)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size + val_size].tolist()
    test_indices = indices[train_size + val_size:].tolist()
    
    return train_indices, val_indices, test_indices


def collate_fn(batch: List[CrystalGraphData]) -> CrystalGraphData:
    """自定义批处理函数"""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


class DataAugmentation:
    """数据增强"""
    
    @staticmethod
    def random_rotation(data: CrystalGraphData) -> CrystalGraphData:
        """随机旋转"""
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float)
        
        data.pos = data.pos @ rotation_matrix.T
        return data
    
    @staticmethod
    def random_noise(data: CrystalGraphData, noise_level: float = 0.01) -> CrystalGraphData:
        """添加随机噪声"""
        noise = torch.randn_like(data.pos) * noise_level
        data.pos = data.pos + noise
        return data
    
    @staticmethod
    def random_scale(data: CrystalGraphData, scale_range: Tuple[float, float] = (0.95, 1.05)) -> CrystalGraphData:
        """随机缩放"""
        scale = np.random.uniform(*scale_range)
        data.pos = data.pos * scale
        return data


if __name__ == "__main__":
    print("测试材料数据集...")
    
    # 创建数据集
    dataset = MaterialDataset(
        root_dir="data/2d_materials",
        cutoff=5.0,
        max_neighbors=12
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        # 测试获取数据
        data = dataset[0]
        print(f"\n第一个样本:")
        print(f"  原子数: {data.x.size(0)}")
        print(f"  边数: {data.edge_index.size(1)}")
        print(f"  分子式: {data.formula}")
        print(f"  是否2D: {data.is_2d.item()}")
        
        # 统计信息
        stats = dataset.get_statistics()
        print(f"\n数据集统计:")
        print(f"  样本数: {stats['num_samples']}")
        print(f"  平均原子数: {stats['avg_num_atoms']:.1f}")
        print(f"  元素种类: {len(stats['atom_types'])}")




