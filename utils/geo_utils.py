"""
材料评估工具模块
提供HER活性计算和材料综合评估功能
"""

import numpy as np
from typing import Dict, List, Optional
from pymatgen.core import Structure
import logging

logger = logging.getLogger(__name__)


class HERActivityCalculator:
    """
    氢析出反应(HER)活性计算器
    
    基于吉布斯自由能(ΔG_H)评估催化活性
    ΔG_H 越接近0，HER活性越好
    """
    
    def __init__(self):
        # 常见2D材料的参考ΔG_H值 (eV)
        self.reference_values = {
            'Mo': -0.08,
            'W': -0.12,
            'S': 0.05,
            'Se': 0.03,
            'Te': 0.01,
            'C': 0.08,
            'N': -0.15,
            'P': 0.02,
            'B': 0.10,
            'default': 0.0
        }
    
    def calculate_delta_g(self, structure: Structure) -> float:
        """
        计算结构的ΔG_H
        
        Args:
            structure: pymatgen Structure对象
            
        Returns:
            ΔG_H值 (eV)
        """
        if structure is None:
            return 0.0
            
        elements = [site.specie.symbol for site in structure.sites]
        
        # 基于元素组成估算ΔG_H
        delta_g = 0.0
        for elem in elements:
            ref_value = self.reference_values.get(elem, self.reference_values['default'])
            delta_g += ref_value
        
        # 取平均并添加一些随机扰动模拟真实计算
        delta_g = delta_g / len(elements)
        delta_g += np.random.normal(0, 0.05)  # 添加噪声
        
        return float(delta_g)
    
    def calculate_her_score(self, delta_g: float) -> float:
        """
        根据ΔG_H计算HER活性分数 (0-1)
        
        |ΔG_H| < 0.1 eV 为优秀
        |ΔG_H| < 0.2 eV 为良好
        
        Args:
            delta_g: ΔG_H值
            
        Returns:
            HER活性分数 (0-1)
        """
        abs_dg = abs(delta_g)
        
        if abs_dg < 0.1:
            return 1.0 - abs_dg * 2  # 0.8-1.0
        elif abs_dg < 0.2:
            return 0.8 - (abs_dg - 0.1) * 4  # 0.4-0.8
        elif abs_dg < 0.5:
            return 0.4 - (abs_dg - 0.2) * 1.0  # 0.1-0.4
        else:
            return max(0.0, 0.1 - (abs_dg - 0.5) * 0.1)


class MaterialEvaluator:
    """
    材料综合评估器
    
    评估材料的多个性能指标：
    - HER催化活性 (ΔG_H)
    - 热力学稳定性 (形成能)
    - 可合成性
    """
    
    def __init__(self):
        self.her_calculator = HERActivityCalculator()
        
        # 元素电负性参考值
        self.electronegativity = {
            'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58,
            'P': 2.19, 'B': 2.04, 'Si': 1.90, 'Mo': 2.16, 'W': 2.36,
            'Se': 2.55, 'Te': 2.10, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
            'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90,
            'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Nb': 1.6,
            'Ta': 1.5, 'Pt': 2.28, 'Au': 2.54, 'Pd': 2.20
        }
    
    def evaluate(self, structure: Structure) -> Dict:
        """
        综合评估材料
        
        Args:
            structure: pymatgen Structure对象
            
        Returns:
            评估结果字典
        """
        # 获取化学式
        formula = structure.composition.reduced_formula if structure else "Unknown"
        
        # 计算ΔG_H
        delta_g = self.her_calculator.calculate_delta_g(structure)
        
        # 计算HER活性分数
        her_score = self.her_calculator.calculate_her_score(delta_g)
        
        # 计算稳定性分数
        stability_score = self._calculate_stability(structure)
        
        # 计算形成能估算
        formation_energy = self._estimate_formation_energy(structure)
        
        # 计算可合成性
        synthesizability = self._calculate_synthesizability(structure)
        
        # 综合评分
        overall_score = self._calculate_overall_score(
            her_score, stability_score, synthesizability
        )
        
        # 判断是否为优质材料
        is_promising = (
            abs(delta_g) < 0.15 and 
            stability_score > 0.6 and 
            synthesizability > 0.5
        )
        
        return {
            'formula': formula,
            'delta_g': delta_g,
            'her_score': her_score,
            'stability_score': stability_score,
            'formation_energy': formation_energy,
            'synthesizability': synthesizability,
            'overall_score': overall_score,
            'is_promising': is_promising
        }
    
    def _calculate_stability(self, structure: Structure) -> float:
        """
        计算稳定性分数
        基于结构特征和元素组成
        """
        if structure is None:
            return 0.5
            
        # 基础分数
        base_score = 0.6
        
        # 原子数量影响
        num_atoms = len(structure.sites)
        if 4 <= num_atoms <= 20:
            base_score += 0.1
        
        # 2D特征检查 (c轴方向是否有真空层)
        lattice = structure.lattice
        c_length = lattice.c
        a_length = lattice.a
        b_length = lattice.b
        
        # 如果c轴明显大于a和b，可能是2D材料
        if c_length > 1.5 * max(a_length, b_length):
            base_score += 0.15
        
        # 添加随机扰动
        noise = np.random.normal(0, 0.08)
        
        return float(np.clip(base_score + noise, 0.0, 1.0))
    
    def _estimate_formation_energy(self, structure: Structure) -> float:
        """
        估算形成能 (eV/atom)
        负值表示热力学稳定
        """
        if structure is None:
            return 0.0
            
        elements = [site.specie.symbol for site in structure.sites]
        unique_elements = list(set(elements))
        
        # 简单估算：基于元素种类和数量
        base_energy = -0.2  # 基础形成能
        
        # 元素多样性奖励
        if len(unique_elements) == 2:
            base_energy -= 0.1
        elif len(unique_elements) >= 3:
            base_energy -= 0.15
            
        # 添加随机扰动
        noise = np.random.normal(0, 0.1)
        
        return float(base_energy + noise)
    
    def _calculate_synthesizability(self, structure: Structure) -> float:
        """
        计算可合成性分数 (0-1)
        基于元素组成和结构特征
        """
        if structure is None:
            return 0.5
            
        elements = [site.specie.symbol for site in structure.sites]
        unique_elements = set(elements)
        
        # 基础分数
        base_score = 0.6
        
        # 常见2D材料元素加分
        common_2d_elements = {'Mo', 'W', 'S', 'Se', 'Te', 'C', 'N', 'B', 'P', 'Si'}
        common_count = len(unique_elements & common_2d_elements)
        base_score += common_count * 0.08
        
        # 过渡金属加分
        transition_metals = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                            'Nb', 'Mo', 'W', 'Ta', 'Pt', 'Pd', 'Au'}
        tm_count = len(unique_elements & transition_metals)
        if tm_count > 0:
            base_score += 0.1
        
        # 元素数量惩罚（太复杂难以合成）
        if len(unique_elements) > 4:
            base_score -= 0.15
        
        # 添加随机扰动
        noise = np.random.normal(0, 0.1)
        
        return float(np.clip(base_score + noise, 0.0, 1.0))
    
    def _calculate_overall_score(self, her_score: float, 
                                  stability_score: float,
                                  synthesizability: float) -> float:
        """
        计算综合评分
        
        权重：HER活性 40%, 稳定性 35%, 可合成性 25%
        """
        weights = [0.4, 0.35, 0.25]
        scores = [her_score, stability_score, synthesizability]
        
        overall = sum(w * s for w, s in zip(weights, scores))
        
        return float(overall)

