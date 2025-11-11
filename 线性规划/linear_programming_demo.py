#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 说明：本文件演示线性规划（LP）在生产计划中的应用，包含建模、求解、可视化、敏感性分析与报告。
# 语法与规则：严格使用PuLP进行线性规划建模；中文可视化需加载字体；遵循项目的可视化与编码规范。
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线性规划优化演示
Linear Programming Optimization Demo

演示内容：生产计划问题
- 目标：最大化利润
- 约束：劳动力和原材料限制
- 方法：使用PuLP求解器

作者: AI Assistant
日期: 2024
"""

# 数值计算与数据处理库（常用缩写：numpy→np，pandas→pd）；
# 绘图库matplotlib用于静态图；pulp用于LP建模与求解；warnings用于抑制非关键警告。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import warnings
warnings.filterwarnings('ignore')

# 路径与中文字体：确保无论从根目录或子目录运行，都能导入根目录的字体配置
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from font_config import setup_chinese_font
setup_chinese_font()

class LinearProgrammingDemo:
    """线性规划演示类
    作用：封装生产计划LP的各步骤（求解、可视化、敏感性、报告）。
    设计：面向对象封装，便于复用与扩展；共享状态通过self.results传递。
    """
    
    def __init__(self):
        # 初始化结果容器；打印统一的演示标题，提升交互体验
        self.results = {}
        print("=" * 50)
        print("线性规划优化演示")
        print("=" * 50)

    def solve_production_planning(self):
        """
        线性规划演示 - 生产计划问题
        
        作用：构建并求解LP模型，得到最优产量与利润；保存用于后续分析与可视化。
        语法要点：
        - LpProblem(name, LpMaximize/LpMinimize) 定义优化方向
        - LpVariable(name, lowBound=0) 定义非负连续变量
        - lpSum([...]) 构造线性目标与约束表达式
        - prob += expr 依次添加目标（第一条）与约束
        - prob.solve(PULP_CBC_CMD(msg=0)) 使用CBC求解器静默求解
        原理：线性规划可行域为凸多边形，最优解位于可行域的极点（单纯形法思想）。
        """
        print("\n生产计划优化问题")
        print("-" * 40)
        
        # 问题数据（与题目集说明一致，确保教学与代码对齐）
        # products：产品名称列表；profit：单位利润系数（目标函数系数）
        products = ['产品A', '产品B', '产品C']
        profit = [40, 30, 50]  # 每单位产品利润
        
        # 资源需求矩阵（约束的系数）：每单位产品消耗的劳动力/原材料
        labor_req = [2, 1, 3]      # 劳动力需求（小时/单位）
        material_req = [1, 2, 1]   # 原材料需求（kg/单位）
        
        # 资源约束（约束右端项）：容量限制
        labor_available = 100      # 可用劳动力（小时）
        material_available = 80    # 可用原材料（kg）
        
        # 说明性打印，帮助理解数据结构与参数含义
        print(f"产品利润：{dict(zip(products, profit))}")
        print(f"劳动力需求：{dict(zip(products, labor_req))}")
        print(f"原材料需求：{dict(zip(products, material_req))}")
        print(f"可用劳动力：{labor_available} 小时")
        print(f"可用原材料：{material_available} kg")
        
        # 使用PuLP定义优化问题：maximization模型
        prob = pulp.LpProblem("生产计划", pulp.LpMaximize)
        
        # 决策变量：x0,x1,x2分别表示A/B/C的产量；lowBound=0保证非负
        x = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(3)]
        
        # 目标函数：最大化利润 Σ profit[i] * x[i]
        prob += pulp.lpSum([profit[i] * x[i] for i in range(3)])
        
        # 约束条件：
        # 劳动力 Σ labor_req[i]*x[i] ≤ labor_available
        prob += pulp.lpSum([labor_req[i] * x[i] for i in range(3)]) <= labor_available
        # 原材料 Σ material_req[i]*x[i] ≤ material_available
        prob += pulp.lpSum([material_req[i] * x[i] for i in range(3)]) <= material_available
        
        # 求解：CBC开源求解器；msg=0静默输出更适合教学
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果：读取变量值与目标值；varValue返回最优解数值
        solution = [x[i].varValue for i in range(3)]
        max_profit = pulp.value(prob.objective)
        
        print(f"\n最优解：")
        for i, product in enumerate(products):
            print(f"  {product}: {solution[i]:.2f} 单位")
        print(f"  最大利润: {max_profit:.2f} 元")
        
        # 资源利用率：用于诊断紧约束与松弛
        labor_used = sum(labor_req[i] * solution[i] for i in range(3))
        material_used = sum(material_req[i] * solution[i] for i in range(3))
        
        print(f"\n资源利用率：")
        print(f"  劳动力：{labor_used:.2f}/{labor_available} ({labor_used/labor_available*100:.1f}%)")
        print(f"  原材料：{material_used:.2f}/{material_available} ({material_used/material_available*100:.1f}%)")
        
        # 保存结果用于可视化与后续分析（避免重复求解，提升复用性）
        self.results = {
            'products': products,
            'solution': solution,
            'profit': profit,
            'max_profit': max_profit,
            'labor_used': labor_used,
            'material_used': material_used,
            'labor_available': labor_available,
            'material_available': material_available,
            'labor_req': labor_req,
            'material_req': material_req
        }
        
        return solution, max_profit
    
    def visualize_results(self):
        """可视化结果
        作用：生成多维度分析图表，包括最优产量、资源利用率、利润贡献分析和资源需求对比。
        规则：统一图表风格、中文标题、网格、PNG输出（dpi=300）。
        """
        if not self.results:
            print("请先运行求解方法")
            return
        
        print("\n生成可视化图表...")
        
        # 设置统一图表样式
        plt.style.use('seaborn-v0_8')
        
        # 创建2x2子图布局，展示更丰富的分析内容
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 最优生产计划
        bars1 = ax1.bar(self.results['products'], self.results['solution'], 
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('最优生产计划', fontsize=14, fontweight='bold')
        ax1.set_ylabel('产量 (单位)')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, self.results['solution']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. 资源利用率分析
        resources = ['劳动力', '原材料']
        utilization = [
            self.results['labor_used'] / self.results['labor_available'] * 100,
            self.results['material_used'] / self.results['material_available'] * 100
        ]
        colors2 = ['#FF9999' if u > 95 else '#99FF99' for u in utilization]
        
        bars2 = ax2.bar(resources, utilization, color=colors2)
        ax2.set_title('资源利用率分析', fontsize=14, fontweight='bold')
        ax2.set_ylabel('利用率 (%)')
        ax2.set_ylim(0, 110)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='满负荷线')
        
        # 添加利用率标签
        for bar, value in zip(bars2, utilization):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{value:.1f}%', ha='center', va='bottom')
        ax2.legend()
        
        # 3. 利润贡献分析
        profit_contribution = [self.results['profit'][i] * self.results['solution'][i] 
                              for i in range(len(self.results['products']))]
        
        bars3 = ax3.bar(self.results['products'], profit_contribution, 
                       color=['#FFD93D', '#6BCF7F', '#4D96FF'])
        ax3.set_title('各产品利润贡献', fontsize=14, fontweight='bold')
        ax3.set_ylabel('利润贡献 (元)')
        ax3.grid(True, alpha=0.3)
        
        # 添加利润贡献标签和百分比
        total_profit = sum(profit_contribution)
        for bar, value in zip(bars3, profit_contribution):
            percentage = value / total_profit * 100 if total_profit > 0 else 0
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{value:.0f}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # 4. 资源需求vs可用量对比
        labor_demand = sum(self.results['labor_req'][i] * self.results['solution'][i] 
                          for i in range(len(self.results['products'])))
        material_demand = sum(self.results['material_req'][i] * self.results['solution'][i] 
                             for i in range(len(self.results['products'])))
        
        x_pos = np.arange(len(resources))
        width = 0.35
        
        bars4_1 = ax4.bar(x_pos - width/2, [labor_demand, material_demand], 
                         width, label='实际需求', color='#FF6B6B', alpha=0.8)
        bars4_2 = ax4.bar(x_pos + width/2, [self.results['labor_available'], 
                                           self.results['material_available']], 
                         width, label='可用资源', color='#4ECDC4', alpha=0.8)
        
        ax4.set_title('资源需求vs可用量', fontsize=14, fontweight='bold')
        ax4.set_ylabel('数量')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(resources)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 添加数值标签
        for bars in [bars4_1, bars4_2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.1f}', ha='center', va='bottom')
        
        # 布局与保存
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, 'linear_programming_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print("可视化图表已保存为 'linear_programming_results.png'")
    
    def sensitivity_analysis(self):
        """敏感性分析
        作用：通过改变单个产品的利润系数（±10%、±20%），重新求解并比较总利润变化。
        原理：目标系数变化影响最优解与最优值；可用作影子价格与稳定区间的直觉参考。
        """
        if not self.results:
            print("请先运行求解方法")
            return
        
        print("\n敏感性分析")
        print("-" * 30)
        
        # 分析利润系数变化的影响：逐产品与多档变化百分比遍历
        print("1. 利润系数敏感性分析：")
        base_profits = self.results['profit']
        
        for i, product in enumerate(self.results['products']):
            print(f"\n  {product} 利润变化影响：")
            for change in [-20, -10, 10, 20]:  # 变化百分比
                new_profit = base_profits[i] * (1 + change/100)
                
                # 重新求解：重建模型以隔离影响，避免共享状态污染
                prob = pulp.LpProblem("敏感性分析", pulp.LpMaximize)
                x = [pulp.LpVariable(f"x{j}", lowBound=0) for j in range(3)]
                
                # 修改目标函数：仅替换一个产品的利润系数
                modified_profits = base_profits.copy()
                modified_profits[i] = new_profit
                prob += pulp.lpSum([modified_profits[j] * x[j] for j in range(3)])
                
                # 约束条件：沿用原始资源需求与容量
                prob += pulp.lpSum([self.results['labor_req'][j] * x[j] for j in range(3)]) <= self.results['labor_available']
                prob += pulp.lpSum([self.results['material_req'][j] * x[j] for j in range(3)]) <= self.results['material_available']
                
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                new_max_profit = pulp.value(prob.objective)
                
                print(f"    利润{change:+d}% → 总利润: {new_max_profit:.2f} 元 "
                      f"(变化: {new_max_profit - self.results['max_profit']:+.2f})")
    
    def generate_report(self):
        """生成详细报告
        作用：以结构化文本形式输出问题概要、最优解、资源利用、管理建议与洞察。
        规则：条理清晰、中文输出；将技术结果转化为管理语言便于决策。
        """
        if not self.results:
            print("请先运行求解方法")
            return
        
        print("\n" + "="*50)
        print("线性规划优化报告")
        print("="*50)
        
        print(f"\n问题描述：")
        print(f"  优化目标：最大化生产利润")
        print(f"  决策变量：三种产品的生产数量")
        print(f"  约束条件：劳动力和原材料限制")
        
        print(f"\n最优解：")
        for i, product in enumerate(self.results['products']):
            print(f"  {product}：{self.results['solution'][i]:.2f} 单位")
        print(f"  最大利润：{self.results['max_profit']:.2f} 元")
        
        print(f"\n资源利用情况：")
        labor_util = self.results['labor_used'] / self.results['labor_available'] * 100
        material_util = self.results['material_used'] / self.results['material_available'] * 100
        print(f"  劳动力利用率：{labor_util:.1f}%")
        print(f"  原材料利用率：{material_util:.1f}%")
        
        print(f"\n管理建议：")
        if labor_util > 95:
            print(f"  劳动力资源接近满负荷，建议考虑增加人力")
        if material_util > 95:
            print(f"  原材料资源接近满负荷，建议优化采购计划")
        
        # 找出最有价值的产品：单位利润最高者
        profit_per_unit = self.results['profit']
        max_profit_idx = profit_per_unit.index(max(profit_per_unit))
        print(f"  单位利润最高产品：{self.results['products'][max_profit_idx]} "
              f"({profit_per_unit[max_profit_idx]} 元/单位)")
        
        print("="*50)

def main():
    """主函数
    作用：提供“一键演示”入口，按顺序执行求解→可视化→敏感性→报告。
    使用规则：仅当作为脚本运行时触发；导入为模块时不自动执行。
    """
    # 创建演示实例
    demo = LinearProgrammingDemo()
    
    # 求解生产计划问题
    solution, max_profit = demo.solve_production_planning()
    
    # 生成可视化
    demo.visualize_results()
    
    # 敏感性分析
    demo.sensitivity_analysis()
    
    # 生成报告
    demo.generate_report()
    
    print(f"\n线性规划演示完成。")
    print(f"最优解：{[f'{x:.1f}' for x in solution]}")
    print(f"最大利润：{max_profit:.2f} 元")

if __name__ == "__main__":
    # 入口保护：确保脚本直接运行时才执行主流程，导入时不执行
    main()