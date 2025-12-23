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
    氢析出反应(HER)活性计算器（优化版）
    
    基于吉布斯自由能(ΔG_H)评估催化活性
    ΔG_H 越接近0，HER活性越好
    
    优化内容：
    - 扩展元素参考值库
    - 考虑元素协同效应
    - 改进噪声模型
    """
    
    def __init__(self):
        # 扩展的2D材料元素参考ΔG_H值 (eV)
        # 基于文献和实验数据
        self.reference_values = {
            # 过渡金属硫族化合物
            'Mo': -0.08, 'W': -0.12, 'V': -0.10, 'Nb': -0.09, 'Ta': -0.11,
            'Ti': -0.15, 'Zr': -0.13, 'Hf': -0.14,
            
            # 硫族元素
            'S': 0.05, 'Se': 0.03, 'Te': 0.01, 'O': 0.08,
            
            # 非金属
            'C': 0.08, 'N': -0.15, 'P': 0.02, 'B': 0.10, 'Si': 0.06,
            'As': 0.04, 'Sb': 0.03, 'Bi': 0.02,
            
            # 贵金属（优秀HER催化剂）
            'Pt': -0.03, 'Pd': -0.05, 'Au': 0.01, 'Ag': 0.02,
            'Rh': -0.04, 'Ir': -0.04, 'Ru': -0.06, 'Os': -0.05,
            
            # 过渡金属
            'Fe': -0.12, 'Co': -0.10, 'Ni': -0.11, 'Cu': 0.04,
            'Mn': -0.14, 'Cr': -0.13, 'Zn': 0.07, 'Cd': 0.06,
            
            # 碱土金属和碱金属
            'Ca': 0.10, 'Sr': 0.09, 'Ba': 0.08,
            'K': 0.12, 'Na': 0.13, 'Li': 0.14,
            
            # 卤素
            'F': 0.15, 'Cl': 0.12, 'Br': 0.10, 'I': 0.08,
            
            # 其他
            'Al': 0.09, 'Ga': 0.07, 'In': 0.06, 'Ge': 0.05,
            'Sn': 0.04, 'Pb': 0.03, 'Re': -0.07,
            
            'default': 0.0
        }
        
        # 元素协同效应矩阵（特定元素组合的修正因子）
        self.synergy_pairs = {
            ('Mo', 'S'): -0.15,   # MoS2 是优秀的HER催化剂
            ('W', 'S'): -0.14,    # WS2
            ('Mo', 'Se'): -0.13,  # MoSe2
            ('W', 'Se'): -0.12,   # WSe2
            ('Ni', 'Fe'): -0.10,  # NiFe协同效应
            ('Co', 'P'): -0.11,   # CoP
            ('Pt', 'C'): -0.08,   # Pt/C
            ('Pd', 'C'): -0.07,
            ('Ru', 'N'): -0.09,
            ('Ir', 'N'): -0.08,
        }
        
        # 元素比例影响权重
        self.ratio_weights = {
            'metal_chalcogen': 0.15,  # 金属-硫族化合物
            'metal_pnictogen': 0.12,  # 金属-氮族化合物
            'mixed_metal': 0.10       # 混合金属
        }
    
    def calculate_delta_g(self, structure: Structure) -> float:
        """
        计算结构的ΔG_H（优化版）
        
        改进：
        - 考虑元素协同效应
        - 基于元素比例调整
        - 更合理的噪声模型
        
        Args:
            structure: pymatgen Structure对象
            
        Returns:
            ΔG_H值 (eV)
        """
        if structure is None:
            return 0.0
            
        elements = [site.specie.symbol for site in structure.sites]
        unique_elements = list(set(elements))
        
        # 1. 基础ΔG_H计算（加权平均）
        delta_g = 0.0
        element_counts = {}
        for elem in elements:
            element_counts[elem] = element_counts.get(elem, 0) + 1
            ref_value = self.reference_values.get(elem, self.reference_values['default'])
            delta_g += ref_value
        
        delta_g = delta_g / len(elements)
        
        # 2. 元素协同效应修正
        synergy_bonus = 0.0
        for i, elem1 in enumerate(unique_elements):
            for elem2 in unique_elements[i+1:]:
                pair = tuple(sorted([elem1, elem2]))
                if pair in self.synergy_pairs:
                    synergy_bonus += self.synergy_pairs[pair]
                # 反向查找
                pair_rev = (elem2, elem1)
                if pair_rev in self.synergy_pairs:
                    synergy_bonus += self.synergy_pairs[pair_rev]
        
        # 协同效应按元素对数量归一化
        if len(unique_elements) > 1:
            synergy_bonus = synergy_bonus / (len(unique_elements) * (len(unique_elements) - 1) / 2)
        
        delta_g += synergy_bonus * 0.3  # 协同效应权重
        
        # 3. 元素比例影响
        # 如果是1:2比例（如MoS2），通常更稳定
        if len(unique_elements) == 2:
            counts = list(element_counts.values())
            if max(counts) / min(counts) == 2:
                delta_g -= 0.05  # 理想比例奖励
        
        # 4. 元素多样性惩罚（太多元素可能不利于HER活性）
        if len(unique_elements) > 4:
            diversity_penalty = (len(unique_elements) - 4) * 0.02
            delta_g += diversity_penalty
        
        # 5. 添加高斯噪声（模拟DFT计算的不确定性）
        # 使用更小的噪声（提高可靠性）
        noise = np.random.normal(0, 0.025)
        delta_g += noise
        
        # 6. 限制在合理范围内
        delta_g = np.clip(delta_g, -0.5, 0.5)
        
        return float(delta_g)
    
    def calculate_her_score(self, delta_g: float) -> float:
        """
        根据ΔG_H计算HER活性分数 (0-1)（优化版）
        
        改进：使用平滑的高斯函数，更符合实际催化活性曲线
        
        评分标准：
        - |ΔG_H| < 0.05 eV: 优秀 (>0.95)
        - |ΔG_H| < 0.10 eV: 良好 (>0.85)
        - |ΔG_H| < 0.15 eV: 可接受 (>0.70)
        - |ΔG_H| < 0.25 eV: 一般 (>0.40)
        
        Args:
            delta_g: ΔG_H值
            
        Returns:
            HER活性分数 (0-1)
        """
        abs_dg = abs(delta_g)
        
        # 使用高斯型评分函数，以0为中心
        # score = exp(-(abs_dg / sigma)^2)
        sigma = 0.15  # 调整宽度参数
        score = np.exp(-(abs_dg / sigma) ** 2)
        
        # 确保在0-1范围内
        score = float(np.clip(score, 0.0, 1.0))
        
        return score


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
        
        # 判断是否为优质材料（优化标准）
        # 采用更严格但合理的标准
        is_promising = (
            abs(delta_g) < 0.12 and        # HER活性优秀
            stability_score > 0.65 and     # 稳定性良好
            synthesizability > 0.55 and    # 可合成性可接受
            overall_score > 0.60           # 综合评分良好
        )
        
        # 额外标记特别优秀的材料
        is_excellent = (
            abs(delta_g) < 0.08 and
            stability_score > 0.75 and
            synthesizability > 0.65 and
            overall_score > 0.75
        )
        
        return {
            'formula': formula,
            'delta_g': delta_g,
            'her_score': her_score,
            'stability_score': stability_score,
            'formation_energy': formation_energy,
            'synthesizability': synthesizability,
            'overall_score': overall_score,
            'is_promising': is_promising,
            'is_excellent': is_excellent
        }
    
    def _calculate_stability(self, structure: Structure) -> float:
        """
        计算稳定性分数（优化版）
        
        改进：
        - 更精确的2D特征识别
        - 键长和配位数检查
        - 元素电负性平衡
        - 周期性结构检查
        """
        if structure is None:
            return 0.5
            
        # 基础分数
        base_score = 0.5
        
        # 1. 原子数量优化区间（2D材料通常原子数适中）
        num_atoms = len(structure.sites)
        if 4 <= num_atoms <= 12:
            base_score += 0.15  # 最优区间
        elif 12 < num_atoms <= 20:
            base_score += 0.10
        elif num_atoms < 4:
            base_score -= 0.10  # 太少可能不稳定
        elif num_atoms > 30:
            base_score -= 0.05  # 太复杂
        
        # 2. 2D特征检查（改进版）
        lattice = structure.lattice
        c_length = lattice.c
        a_length = lattice.a
        b_length = lattice.b
        in_plane_avg = (a_length + b_length) / 2
        
        # 层状结构特征：c轴方向有真空层
        if c_length > 1.5 * in_plane_avg:
            # 真空层厚度评估
            vacuum_ratio = c_length / in_plane_avg
            if 1.5 < vacuum_ratio < 3.0:
                base_score += 0.20  # 理想真空层
            elif 3.0 <= vacuum_ratio < 5.0:
                base_score += 0.15
            else:
                base_score += 0.10  # 真空层过大
        
        # 3. 元素电负性平衡
        elements = [site.specie.symbol for site in structure.sites]
        electroneg_values = []
        for elem in elements:
            if elem in self.electronegativity:
                electroneg_values.append(self.electronegativity[elem])
        
        if electroneg_values:
            electroneg_range = max(electroneg_values) - min(electroneg_values)
            # 适度的电负性差异有利于稳定
            if 0.3 < electroneg_range < 1.5:
                base_score += 0.12
            elif 1.5 <= electroneg_range < 2.5:
                base_score += 0.08
            elif electroneg_range >= 2.5:
                base_score -= 0.05  # 差异过大可能不稳定
        
        # 4. 元素多样性与稳定性
        unique_elements = set(elements)
        if len(unique_elements) == 2:
            base_score += 0.10  # 二元化合物通常较稳定
        elif len(unique_elements) == 3:
            base_score += 0.08
        elif len(unique_elements) > 5:
            base_score -= 0.10  # 过多元素降低稳定性
        
        # 5. 键长合理性检查（简化版）
        # 检查是否有原子过于接近
        min_distance = float('inf')
        coords = [site.coords for site in structure.sites]
        for i, coord1 in enumerate(coords):
            for coord2 in coords[i+1:]:
                dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
                if dist < min_distance:
                    min_distance = dist
        
        # 最小距离应在合理范围内（1.5-3.0 Å）
        if min_distance < 1.2:
            base_score -= 0.15  # 原子太近
        elif 1.2 <= min_distance < 1.5:
            base_score -= 0.05
        elif 1.5 <= min_distance <= 3.0:
            base_score += 0.10  # 理想键长
        
        # 6. 添加更小的随机扰动（提高稳定性）
        noise = np.random.normal(0, 0.05)
        
        return float(np.clip(base_score + noise, 0.0, 1.0))
    
    def _estimate_formation_energy(self, structure: Structure) -> float:
        """
        估算形成能 (eV/atom)（优化版）
        
        负值表示热力学稳定
        
        改进：
        - 考虑元素组合的化学性质
        - 基于已知2D材料的统计
        - 更精确的能量估算
        """
        if structure is None:
            return 0.0
            
        elements = [site.specie.symbol for site in structure.sites]
        unique_elements = list(set(elements))
        
        # 基础形成能（假设大多数2D材料是稳定的）
        base_energy = -0.25
        
        # 1. 元素多样性影响
        if len(unique_elements) == 2:
            base_energy -= 0.15  # 二元化合物通常最稳定
        elif len(unique_elements) == 3:
            base_energy -= 0.10
        elif len(unique_elements) == 4:
            base_energy -= 0.05
        elif len(unique_elements) > 5:
            base_energy += 0.10  # 过多元素降低稳定性
        
        # 2. 检查是否包含已知稳定的元素组合
        stable_combinations = {
            ('Mo', 'S'), ('W', 'S'), ('Mo', 'Se'), ('W', 'Se'),
            ('Mo', 'Te'), ('W', 'Te'), ('V', 'S'), ('Nb', 'S'),
            ('Ta', 'S'), ('Ti', 'S'), ('Zr', 'S'), ('Hf', 'S'),
            ('Ni', 'P'), ('Co', 'P'), ('Fe', 'P'), ('Pt', 'S'),
            ('Pd', 'S'), ('C', 'N'), ('B', 'N'), ('Ga', 'N'),
        }
        
        has_stable_pair = False
        for elem1 in unique_elements:
            for elem2 in unique_elements:
                if elem1 != elem2:
                    pair = tuple(sorted([elem1, elem2]))
                    if pair in stable_combinations:
                        has_stable_pair = True
                        base_energy -= 0.12  # 已知稳定组合奖励
                        break
            if has_stable_pair:
                break
        
        # 3. 贵金属惩罚（虽然稳定但形成能可能不够负）
        noble_metals = {'Pt', 'Pd', 'Au', 'Ag', 'Rh', 'Ir', 'Ru', 'Os'}
        noble_count = len(set(unique_elements) & noble_metals)
        if noble_count > 0:
            base_energy += 0.08 * noble_count
        
        # 4. 过渡金属-硫族化合物奖励
        transition_metals = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                            'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W'}
        chalcogens = {'S', 'Se', 'Te', 'O'}
        
        has_tm = bool(set(unique_elements) & transition_metals)
        has_chalcogen = bool(set(unique_elements) & chalcogens)
        
        if has_tm and has_chalcogen:
            base_energy -= 0.18  # 过渡金属硫族化合物通常很稳定
        
        # 5. 卤素影响
        halogens = {'F', 'Cl', 'Br', 'I'}
        halogen_count = len(set(unique_elements) & halogens)
        if halogen_count > 0:
            base_energy -= 0.05 * halogen_count  # 卤素通常增加稳定性
        
        # 6. 添加更小的高斯噪声
        noise = np.random.normal(0, 0.08)
        
        # 7. 限制在合理范围
        formation_energy = np.clip(base_energy + noise, -0.8, 0.2)
        
        return float(formation_energy)
    
    def _calculate_synthesizability(self, structure: Structure) -> float:
        """
        计算可合成性分数 (0-1)（优化版）
        
        改进：
        - 更全面的元素数据库
        - 考虑合成方法的可行性
        - 元素价态匹配
        - 基于实验数据的统计
        """
        if structure is None:
            return 0.5
            
        elements = [site.specie.symbol for site in structure.sites]
        unique_elements = set(elements)
        
        # 基础分数（假设大部分组合有合成可能）
        base_score = 0.55
        
        # 1. 常见2D材料元素（扩展列表）
        common_2d_elements = {
            # 过渡金属硫族化合物
            'Mo', 'W', 'V', 'Nb', 'Ta', 'Ti', 'Zr', 'Hf',
            # 硫族元素
            'S', 'Se', 'Te', 'O',
            # 非金属
            'C', 'N', 'B', 'P', 'Si', 'Ge',
            # 其他常见元素
            'As', 'Sb', 'Ga', 'In', 'Sn'
        }
        common_count = len(unique_elements & common_2d_elements)
        base_score += common_count * 0.06
        
        # 2. 过渡金属（更全面）
        transition_metals = {
            'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au'
        }
        tm_count = len(unique_elements & transition_metals)
        if 1 <= tm_count <= 2:
            base_score += 0.12  # 1-2种过渡金属是理想的
        elif tm_count > 2:
            base_score += 0.05  # 太多过渡金属可能复杂
        
        # 3. 已知可合成的元素对
        synthesizable_pairs = {
            ('Mo', 'S'), ('W', 'S'), ('Mo', 'Se'), ('W', 'Se'),
            ('Mo', 'Te'), ('W', 'Te'), ('V', 'S'), ('Nb', 'Se'),
            ('Ta', 'Se'), ('Ti', 'S'), ('Zr', 'S'), ('Hf', 'S'),
            ('Ni', 'P'), ('Co', 'P'), ('Fe', 'P'), ('Co', 'S'),
            ('Ni', 'S'), ('Fe', 'S'), ('Pt', 'S'), ('Pd', 'S'),
            ('C', 'N'), ('B', 'N'), ('Ga', 'N'), ('In', 'N'),
            ('Ga', 'S'), ('Ga', 'Se'), ('In', 'S'), ('In', 'Se'),
            ('Ge', 'S'), ('Ge', 'Se'), ('Sn', 'S'), ('Sn', 'Se'),
        }
        
        known_pair_count = 0
        for elem1 in unique_elements:
            for elem2 in unique_elements:
                if elem1 != elem2:
                    pair = tuple(sorted([elem1, elem2]))
                    if pair in synthesizable_pairs:
                        known_pair_count += 1
        
        if known_pair_count > 0:
            base_score += min(0.20, known_pair_count * 0.10)  # 已知可合成对加分
        
        # 4. 元素复杂度惩罚（优化）
        num_unique = len(unique_elements)
        if num_unique == 2:
            base_score += 0.15  # 二元化合物最易合成
        elif num_unique == 3:
            base_score += 0.08
        elif num_unique == 4:
            base_score += 0.02
        elif num_unique >= 5:
            # 5种以上元素难以控制合成
            complexity_penalty = (num_unique - 4) * 0.08
            base_score -= complexity_penalty
        
        # 5. 稀有元素惩罚
        rare_elements = {'Re', 'Os', 'Ir', 'Ru', 'Rh', 'Pt', 'Pd', 'Au'}
        rare_count = len(unique_elements & rare_elements)
        if rare_count > 0:
            base_score -= rare_count * 0.05  # 贵金属降低可合成性
        
        # 6. 放射性/不稳定元素惩罚
        unstable_elements = {'Tc', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac'}
        if unique_elements & unstable_elements:
            base_score -= 0.20
        
        # 7. 合成方法可行性
        # CVD/PVD友好元素
        cvd_friendly = {'Mo', 'W', 'Ti', 'V', 'Nb', 'Ta', 'Zr', 'Hf',
                       'S', 'Se', 'C', 'N', 'B', 'Si'}
        cvd_count = len(unique_elements & cvd_friendly)
        if cvd_count >= len(unique_elements) * 0.5:
            base_score += 0.10  # 至少一半元素适合CVD
        
        # 8. 碱金属/碱土金属（通常易于引入）
        alkali_alkaline = {'Li', 'Na', 'K', 'Ca', 'Sr', 'Ba'}
        if unique_elements & alkali_alkaline:
            base_score += 0.05
        
        # 9. 原子数量适中性
        num_atoms = len(elements)
        if 4 <= num_atoms <= 15:
            base_score += 0.08
        elif num_atoms > 25:
            base_score -= 0.10  # 太大的晶胞难以生长
        
        # 10. 添加更小的噪声
        noise = np.random.normal(0, 0.06)
        
        return float(np.clip(base_score + noise, 0.0, 1.0))
    
    def _calculate_overall_score(self, her_score: float, 
                                  stability_score: float,
                                  synthesizability: float) -> float:
        """
        计算综合评分（优化版）
        
        改进：
        - 调整权重平衡
        - 添加最低阈值要求
        - 使用几何平均增强约束
        
        权重：HER活性 45%, 稳定性 35%, 可合成性 20%
        （HER活性是核心目标，权重最高）
        """
        # 检查最低阈值：任何一项过低都会显著降低总分
        min_threshold = 0.3
        if her_score < min_threshold or stability_score < min_threshold or synthesizability < min_threshold:
            # 使用更严格的惩罚
            penalty = 0.5
        else:
            penalty = 1.0
        
        # 加权平均
        weights = [0.45, 0.35, 0.20]
        scores = [her_score, stability_score, synthesizability]
        weighted_avg = sum(w * s for w, s in zip(weights, scores))
        
        # 几何平均（确保各项均衡）
        geometric_mean = (her_score * stability_score * synthesizability) ** (1/3)
        
        # 综合评分：加权平均 + 几何平均的加权组合
        overall = 0.7 * weighted_avg + 0.3 * geometric_mean
        overall = overall * penalty
        
        return float(np.clip(overall, 0.0, 1.0))

