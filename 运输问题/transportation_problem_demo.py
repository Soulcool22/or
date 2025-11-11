#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 说明：本文件演示运输问题（TP）在供应链中的应用，包含基础运输与多产品运输的建模、求解、可视化与报告。
# 语法与规则：使用PuLP连续非负变量；中文图表需加载字体；遵循项目的可视化与编码规范。
"""
运输问题优化演示
Transportation Problem Optimization Demo

演示内容：供应链优化问题
- 目标：最小化运输成本
- 约束：供应量和需求量平衡
- 方法：使用PuLP求解器和运输单纯形法

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pulp
import warnings
# 抑制非关键警告，保证教学输出清爽
warnings.filterwarnings('ignore')

# 路径与中文字体：移动到子目录后也能导入根目录的配置
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from font_config import setup_chinese_font
setup_chinese_font()

class TransportationProblemDemo:
    """运输问题演示类
    作用：封装基础运输与多产品运输的求解、可视化、敏感性分析与报告生成。
    设计：面向对象组织流程；共享结果通过 self.results 以便各方法复用。
    """
    
    def __init__(self):
        self.results = {}
        print("=" * 50)
        print("运输问题优化演示")
        print("=" * 50)
    
    def solve_basic_transportation(self):
        """
        基础运输问题演示 - 供应链优化
        
        作用：构建并求解经典运输问题（平衡或通过虚拟节点平衡），最小化运输成本。
        语法要点：
        - LpProblem(name, LpMinimize)
        - LpVariable(f"x_{i}_{j}", lowBound=0) 连续非负变量表示从工厂i到仓库j的运输量
        - 目标：Σ c_ij x_ij；约束：每个工厂的供应等式、每个仓库的需求等式
        - 非平衡时添加“虚拟工厂/虚拟仓库”，成本为0以吸收差额
        原理：线性规划的特殊结构（完全单调矩阵），可用运输单纯形法；此处用CBC求解器。
        """
        print("\n基础运输问题 - 供应链优化")
        print("-" * 40)
        
        # 工厂和仓库
        factories = ['工厂A', '工厂B', '工厂C']
        warehouses = ['仓库1', '仓库2', '仓库3', '仓库4']
        
        # 供应量（吨）
        supply = [300, 400, 500]
        
        # 需求量（吨）
        demand = [250, 350, 400, 200]
        
        # 运输成本矩阵（元/吨）
        cost_matrix = np.array([
            [8, 6, 10, 9],   # 工厂A到各仓库
            [9, 12, 13, 7],  # 工厂B到各仓库
            [14, 9, 16, 5]   # 工厂C到各仓库
        ])
        
        print("供需信息：")
        print(f"工厂供应量: {dict(zip(factories, supply))}")
        print(f"仓库需求量: {dict(zip(warehouses, demand))}")
        print(f"总供应量: {sum(supply)} 吨")
        print(f"总需求量: {sum(demand)} 吨")

        print("\n运输成本矩阵（元/吨）：")
        cost_df = pd.DataFrame(cost_matrix, index=factories, columns=warehouses)
        print(cost_df)
        
        # 检查平衡性：供应 ≠ 需求时增加虚拟节点以形成平衡问题
        original_warehouses = warehouses.copy()
        original_demand = demand.copy()
        
        if sum(supply) != sum(demand):
            print("非平衡运输问题：供应量 ≠ 需求量")
            if sum(supply) > sum(demand):
                # 添加虚拟仓库
                demand.append(sum(supply) - sum(demand))
                warehouses.append('虚拟仓库')
                cost_matrix = np.column_stack([cost_matrix, np.zeros(3)])
                print(f"添加虚拟仓库，需求量: {demand[-1]} 吨")
            else:
                # 添加虚拟工厂
                supply.append(sum(demand) - sum(supply))
                factories.append('虚拟工厂')
                cost_matrix = np.vstack([cost_matrix, np.zeros(len(warehouses))])
                print(f"添加虚拟工厂，供应量: {supply[-1]} 吨")
        
        # 使用PuLP定义优化问题：最小化总运输成本
        prob = pulp.LpProblem("运输问题", pulp.LpMinimize)
        
        # 决策变量：从工厂i到仓库j的运输量（非负连续）
        x = {}
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                x[i,j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        
        # 目标函数：最小化运输成本 Σ c_ij x_ij
        prob += pulp.lpSum([cost_matrix[i][j] * x[i,j] 
                           for i in range(len(factories)) 
                           for j in range(len(warehouses))])
        
        # 约束条件：
        # 1) 供应约束（每个工厂的发货量等于其供应）
        for i in range(len(factories)):
            prob += pulp.lpSum([x[i,j] for j in range(len(warehouses))]) == supply[i]
        
        # 2) 需求约束（每个仓库的收货量等于其需求）
        for j in range(len(warehouses)):
            prob += pulp.lpSum([x[i,j] for i in range(len(factories))]) == demand[j]
        
        # 求解：CBC开源求解器，msg=0静默输出
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果：读取运输矩阵与目标值
        solution_matrix = np.zeros((len(factories), len(warehouses)))
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                solution_matrix[i][j] = x[i,j].varValue
        
        min_transport_cost = pulp.value(prob.objective)
        
        print("\n最优运输方案：")
        solution_df = pd.DataFrame(solution_matrix, 
                                 index=factories, 
                                 columns=warehouses)
        print(solution_df.round(1))
        
        print("\n运输成本分析：")
        print(f"  最小运输成本：{min_transport_cost:.2f} 元")
        
        # 计算各路线成本：便于识别高成本路线与优化机会
        print("\n主要运输路线：")
        route_details = []
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                if solution_matrix[i][j] > 0:
                    route_cost = solution_matrix[i][j] * cost_matrix[i][j]
                    route_details.append({
                        'from': factories[i],
                        'to': warehouses[j],
                        'quantity': solution_matrix[i][j],
                        'unit_cost': cost_matrix[i][j],
                        'total_cost': route_cost
                    })
                    print(f"  {factories[i]} → {warehouses[j]}: "
                          f"{solution_matrix[i][j]:.1f}吨, 成本: {route_cost:.2f}元")
        
        # 保存结果以供后续可视化与报告
        self.results['basic'] = {
            'factories': factories,
            'warehouses': warehouses,
            'original_warehouses': original_warehouses,
            'supply': supply,
            'demand': demand,
            'original_demand': original_demand,
            'cost_matrix': cost_matrix,
            'solution_matrix': solution_matrix,
            'min_cost': min_transport_cost,
            'route_details': route_details
        }
        
        return solution_matrix, min_transport_cost
    
    def solve_multi_product_transportation(self):
        """
        多产品运输问题演示
        
        作用：构建多索引运输模型（工厂×产品×市场），最小化总成本。
        语法要点：
        - 决策变量 x[i,p,j] 表示工厂i的产品p送至市场j的数量
        - 供应约束：每个工厂每种产品的总发货量 ≤ 供应
        - 需求约束：每个市场每种产品的总收货量 ≥ 需求
        原理：仍为线性规划，但维度更高，适合展示结构化建模方法。
        """
        print("\n多产品运输问题")
        print("-" * 30)
        
        # 工厂、产品、市场
        factories = ['工厂X', '工厂Y']
        products = ['产品P1', '产品P2']
        markets = ['市场M1', '市场M2', '市场M3']
        
        # 各工厂各产品的供应量
        supply_matrix = np.array([
            [200, 150],  # 工厂X的P1, P2供应量
            [180, 220]   # 工厂Y的P1, P2供应量
        ])
        
        # 各市场各产品的需求量
        demand_matrix = np.array([
            [120, 100],  # 市场M1的P1, P2需求量
            [140, 130],  # 市场M2的P1, P2需求量
            [120, 140]   # 市场M3的P1, P2需求量
        ])
        
        # 运输成本矩阵 [工厂][产品][市场]
        cost_tensor = np.array([
            [[5, 7, 6],   # 工厂X的P1到各市场
             [6, 8, 7]],  # 工厂X的P2到各市场
            [[8, 6, 9],   # 工厂Y的P1到各市场
             [7, 5, 8]]   # 工厂Y的P2到各市场
        ])
        
        print("供应信息：")
        supply_df = pd.DataFrame(supply_matrix, index=factories, columns=products)
        print(supply_df)
        
        print("\n需求信息：")
        demand_df = pd.DataFrame(demand_matrix, index=markets, columns=products)
        print(demand_df)
        
        print(f"\n各产品总供应量：P1={supply_matrix[:, 0].sum()}, P2={supply_matrix[:, 1].sum()}")
        print(f"各产品总需求量：P1={demand_matrix[:, 0].sum()}, P2={demand_matrix[:, 1].sum()}")
        
        # 使用PuLP定义优化问题：最小化总运输成本
        prob = pulp.LpProblem("多产品运输问题", pulp.LpMinimize)
        
        # 决策变量：从工厂i的产品p到市场j的运输量（非负连续）
        x = {}
        for i in range(len(factories)):
            for p in range(len(products)):
                for j in range(len(markets)):
                    x[i,p,j] = pulp.LpVariable(f"x_{i}_{p}_{j}", lowBound=0)
        
        # 目标函数：最小化总运输成本 Σ c_{i,p,j} x_{i,p,j}
        prob += pulp.lpSum([cost_tensor[i][p][j] * x[i,p,j] 
                           for i in range(len(factories))
                           for p in range(len(products))
                           for j in range(len(markets))])
        
        # 约束条件：
        # 1) 供应约束：每个工厂每种产品的供应量限制
        for i in range(len(factories)):
            for p in range(len(products)):
                prob += pulp.lpSum([x[i,p,j] for j in range(len(markets))]) <= supply_matrix[i][p]
        
        # 2) 需求约束：每个市场每种产品的需求量满足
        for j in range(len(markets)):
            for p in range(len(products)):
                prob += pulp.lpSum([x[i,p,j] for i in range(len(factories))]) >= demand_matrix[j][p]
        
        # 求解：CBC开源求解器，msg=0静默输出
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        min_cost = pulp.value(prob.objective)
        
        print("\n最优运输方案：")
        print(f"  最小运输成本：{min_cost:.2f} 元")
        
        print("\n运输路线详情：")
        multi_route_details = []
        for i in range(len(factories)):
            for p in range(len(products)):
                for j in range(len(markets)):
                    quantity = x[i,p,j].varValue
                    if quantity > 0:
                        cost = quantity * cost_tensor[i][p][j]
                        multi_route_details.append({
                            'factory': factories[i],
                            'product': products[p],
                            'market': markets[j],
                            'quantity': quantity,
                            'unit_cost': cost_tensor[i][p][j],
                            'total_cost': cost
                        })
                        print(f"  {factories[i]} {products[p]} → {markets[j]}: "
                              f"{quantity:.1f}单位, 成本: {cost:.2f}元")
        
        # 保存多产品运输结果以供可视化与报告
        self.results['multi_product'] = {
            'factories': factories,
            'products': products,
            'markets': markets,
            'supply_matrix': supply_matrix,
            'demand_matrix': demand_matrix,
            'cost_tensor': cost_tensor,
            'min_cost': min_cost,
            'route_details': multi_route_details
        }
        
        return min_cost
    
    def visualize_results(self):
        """可视化结果
        作用：多维度展示运输网络图、成本热力图、供需分析和路线优化，统一中文标签和样式。
        规则：figsize统一；网格 alpha=0.3；PNG输出（dpi=300）。
        """
        if not self.results:
            print("请先运行求解方法")
            return
        
        print("\n生成可视化图表…")
        
        # 创建2x3子图布局，展示更全面的运输分析
        if 'multi_product' in self.results:
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        if 'basic' in self.results:
            basic = self.results['basic']
            
            # 1. 运输网络图
            import networkx as nx
            G = nx.Graph()
            
            # 添加节点
            factories = basic['factories'][:len(basic['cost_matrix'])]
            warehouses = basic['original_warehouses']
            
            # 工厂节点（红色）
            for factory in factories:
                G.add_node(factory, node_type='factory')
            
            # 仓库节点（蓝色）
            for warehouse in warehouses:
                G.add_node(warehouse, node_type='warehouse')
            
            # 添加边（运输路线）
            for detail in basic['route_details']:
                if detail['quantity'] > 0:
                    G.add_edge(detail['from'], detail['to'], 
                              weight=detail['quantity'], 
                              cost=detail['unit_cost'])
            
            # 绘制网络图
            pos = {}
            # 工厂位置（左侧）
            for i, factory in enumerate(factories):
                pos[factory] = (0, i * 2)
            
            # 仓库位置（右侧）
            for i, warehouse in enumerate(warehouses):
                pos[warehouse] = (3, i * 1.5)
            
            # 绘制节点
            factory_nodes = [n for n in G.nodes() if n in factories]
            warehouse_nodes = [n for n in G.nodes() if n in warehouses]
            
            nx.draw_networkx_nodes(G, pos, nodelist=factory_nodes, 
                                 node_color='#FF6B6B', node_size=800, ax=ax1)
            nx.draw_networkx_nodes(G, pos, nodelist=warehouse_nodes, 
                                 node_color='#4ECDC4', node_size=800, ax=ax1)
            
            # 绘制边（运输路线）
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            
            for (u, v) in edges:
                weight = G[u][v]['weight']
                width = (weight / max_weight) * 5 + 1
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                     width=width, alpha=0.7, ax=ax1)
            
            # 添加标签
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax1)
            
            # 添加边标签（运输量）
            edge_labels = {(u, v): f'{G[u][v]["weight"]:.0f}' for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax1)
            
            ax1.set_title('运输网络图', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # 2. 运输成本热力图
            original_cost_matrix = basic['cost_matrix'][:, :len(basic['original_warehouses'])]
            im2 = ax2.imshow(original_cost_matrix, cmap='YlOrRd', aspect='auto')
            
            # 添加数值标注
            for i in range(len(factories)):
                for j in range(len(warehouses)):
                    if i < len(original_cost_matrix) and j < len(original_cost_matrix[0]):
                        text = ax2.text(j, i, f'{original_cost_matrix[i, j]:.0f}',
                                       ha="center", va="center", color="black", fontweight='bold')
            
            ax2.set_xticks(range(len(warehouses)))
            ax2.set_xticklabels(warehouses, rotation=45)
            ax2.set_yticks(range(len(factories)))
            ax2.set_yticklabels(factories)
            ax2.set_title('运输成本热力图 (元/吨)', fontsize=14, fontweight='bold')
            
            # 添加颜色条
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            # 3. 最优运输方案
            original_solution = basic['solution_matrix'][:len(original_cost_matrix), :len(basic['original_warehouses'])]
            im3 = ax3.imshow(original_solution, cmap='Blues', aspect='auto')
            
            # 添加数值标注
            for i in range(len(factories)):
                for j in range(len(warehouses)):
                    if i < len(original_solution) and j < len(original_solution[0]):
                        if original_solution[i, j] > 0:
                            text = ax3.text(j, i, f'{original_solution[i, j]:.0f}',
                                           ha="center", va="center", color="white", fontweight='bold')
            
            ax3.set_xticks(range(len(warehouses)))
            ax3.set_xticklabels(warehouses, rotation=45)
            ax3.set_yticks(range(len(factories)))
            ax3.set_yticklabels(factories)
            ax3.set_title('最优运输方案 (吨)', fontsize=14, fontweight='bold')
            
            # 添加颜色条
            plt.colorbar(im3, ax=ax3, shrink=0.8)
            
            # 4. 供需平衡分析
            supply = basic['supply'][:len(factories)]
            demand = basic['demand'][:len(warehouses)]
            
            x_pos = np.arange(max(len(supply), len(demand)))
            width = 0.35
            
            # 供应量
            supply_padded = list(supply) + [0] * (len(demand) - len(supply))
            demand_padded = list(demand) + [0] * (len(supply) - len(demand))
            
            bars1 = ax4.bar(x_pos - width/2, supply_padded[:len(x_pos)], width, 
                           label='供应量', color='#FF9999', alpha=0.8)
            bars2 = ax4.bar(x_pos + width/2, demand_padded[:len(x_pos)], width, 
                           label='需求量', color='#99CCFF', alpha=0.8)
            
            ax4.set_title('供需平衡分析', fontsize=14, fontweight='bold')
            ax4.set_ylabel('数量 (吨)')
            ax4.set_xlabel('节点')
            ax4.set_xticks(x_pos)
            labels = factories + warehouses
            ax4.set_xticklabels(labels[:len(x_pos)], rotation=45)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2, height + 5,
                                f'{height:.0f}', ha='center', va='bottom')
        
        if 'multi_product' in self.results:
            multi = self.results['multi_product']
            
            # 5. 多产品运输成本对比
            product_costs = {}
            product_quantities = {}
            
            for detail in multi['route_details']:
                product = detail['product']
                if product not in product_costs:
                    product_costs[product] = 0
                    product_quantities[product] = 0
                product_costs[product] += detail['total_cost']
                product_quantities[product] += detail['quantity']
            
            products = list(product_costs.keys())
            costs = list(product_costs.values())
            quantities = list(product_quantities.values())
            
            # 成本对比
            bars5 = ax5.bar(products, costs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax5.set_title('各产品运输成本对比', fontsize=14, fontweight='bold')
            ax5.set_ylabel('总成本 (元)')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
            
            # 添加成本标签和百分比
            total_cost = sum(costs)
            for bar, cost in zip(bars5, costs):
                percentage = cost / total_cost * 100
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{cost:.0f}\n({percentage:.1f}%)', 
                        ha='center', va='bottom')
            
            # 6. 产品运输效率分析
            efficiency = [cost/qty if qty > 0 else 0 for cost, qty in zip(costs, quantities)]
            
            bars6 = ax6.bar(products, efficiency, color=['#32CD32', '#FFD700', '#FF6347'])
            ax6.set_title('产品运输效率 (元/单位)', fontsize=14, fontweight='bold')
            ax6.set_ylabel('单位运输成本 (元)')
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)
            
            # 添加效率标签
            for bar, eff in zip(bars6, efficiency):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{eff:.2f}', ha='center', va='bottom')
        else:
            # 如果没有多产品问题，显示路线成本分析
            if 'basic' in self.results and len(basic['route_details']) > 0:
                routes = [f"{r['from']}\n→{r['to']}" for r in basic['route_details']]
                unit_costs = [r['unit_cost'] for r in basic['route_details']]
                total_costs = [r['total_cost'] for r in basic['route_details']]
                
                # 路线单位成本
                bars4_alt = ax4.bar(routes, unit_costs, color='#FF9999', alpha=0.8)
                ax4.set_title('各路线单位成本', fontsize=14, fontweight='bold')
                ax4.set_ylabel('单位成本 (元/吨)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
                
                for bar, cost in zip(bars4_alt, unit_costs):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                            f'{cost:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, 'transportation_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("可视化图表已保存为 'transportation_results.png'")
    
    def cost_sensitivity_analysis(self):
        """运输成本敏感性分析
        作用：在不重新优化的简化前提下评估关键路线成本变化对总成本的影响，作为直觉参考。
        说明：严谨分析需在成本变动下重新求解模型，这里为教学简化演示。
        """
        if 'basic' not in self.results:
            print("请先运行基础运输问题求解")
            return
        
        print("\n运输成本敏感性分析")
        print("-" * 30)
        
        basic = self.results['basic']
        base_cost = basic['min_cost']
        
        # 分析关键路线成本变化的影响
        print("关键路线成本变化影响：")
        
        for route in basic['route_details'][:3]:  # 分析前3条主要路线
            print(f"\n  {route['from']} → {route['to']} 路线:")
            
            for cost_change in [-20, -10, 10, 20]:  # 成本变化百分比
                # 这里简化处理，实际应该重新求解整个问题
                estimated_cost_change = route['total_cost'] * cost_change / 100
                new_total_cost = base_cost + estimated_cost_change
                
                print(f"    成本{cost_change:+d}% → 预估总成本：{new_total_cost:.2f} 元 "
                      f"(变化：{estimated_cost_change:+.2f})")
    
    def generate_report(self):
        """生成详细报告
        作用：以结构化中文输出总结运输方案、成本统计与优化建议，便于业务决策。
        规则：条理清晰、教学友好；将技术结果转化为业务可读信息。
        """
        if not self.results:
            print("请先运行求解方法")
            return
        
        print("\n" + "="*50)
        print("运输问题优化报告")
        print("="*50)
        
        if 'basic' in self.results:
            basic = self.results['basic']
            print("\n基础运输问题：")
            print(f"  • 优化目标：最小化运输成本")
            print(f"  • 工厂数量：{len(basic['factories'])}")
            print(f"  • 仓库数量：{len(basic['original_warehouses'])}")
            print(f"  • 最小运输成本：{basic['min_cost']:.2f} 元")
            
            print("\n运输方案统计：")
            total_quantity = sum(detail['quantity'] for detail in basic['route_details'])
            print(f"  • 总运输量：{total_quantity:.1f} 吨")
            print(f"  • 平均运输成本：{basic['min_cost']/total_quantity:.2f} 元/吨")
            print(f"  • 活跃路线数：{len(basic['route_details'])}")
            
            # 找出成本最高和最低的路线
            if basic['route_details']:
                max_cost_route = max(basic['route_details'], key=lambda x: x['unit_cost'])
                min_cost_route = min(basic['route_details'], key=lambda x: x['unit_cost'])
                
                print("\n路线成本分析：")
                print(f"  • 最高成本路线: {max_cost_route['from']} → {max_cost_route['to']} "
                      f"({max_cost_route['unit_cost']} 元/吨)")
                print(f"  • 最低成本路线: {min_cost_route['from']} → {min_cost_route['to']} "
                      f"({min_cost_route['unit_cost']} 元/吨)")
        
        if 'multi_product' in self.results:
            multi = self.results['multi_product']
            print("\n多产品运输问题：")
            print(f"  • 工厂数量：{len(multi['factories'])}")
            print(f"  • 产品种类：{len(multi['products'])}")
            print(f"  • 市场数量：{len(multi['markets'])}")
            print(f"  • 最小运输成本：{multi['min_cost']:.2f} 元")
            
            # 各产品的运输成本分析
            product_costs = {}
            for detail in multi['route_details']:
                product = detail['product']
                if product not in product_costs:
                    product_costs[product] = 0
                product_costs[product] += detail['total_cost']
            
            print("\n各产品运输成本：")
            for product, cost in product_costs.items():
                percentage = cost / multi['min_cost'] * 100
                print(f"  • {product}: {cost:.2f} 元 ({percentage:.1f}%)")
        
        print("\n优化建议：")
        if 'basic' in self.results:
            basic = self.results['basic']
            if basic['route_details']:
                # 建议优化高成本路线
                high_cost_routes = [r for r in basic['route_details'] if r['unit_cost'] > 10]
                if high_cost_routes:
                    print(f"  • 考虑优化高成本路线，寻找替代运输方案")
                
                # 建议增加低成本路线的利用
                low_cost_routes = [r for r in basic['route_details'] if r['unit_cost'] < 8]
                if low_cost_routes:
                    print(f"  • 充分利用低成本路线，提高运输效率")
        
        print("="*50)

def main():
    """主函数
    作用：按顺序执行基础运输→多产品运输→可视化→敏感性→报告，一键演示完整流程。
    使用规则：脚本运行时触发；导入为模块时不自动执行。
    """
    # 创建演示实例
    demo = TransportationProblemDemo()
    
    # 求解基础运输问题
    solution_matrix, min_cost = demo.solve_basic_transportation()
    
    # 求解多产品运输问题
    multi_min_cost = demo.solve_multi_product_transportation()
    
    # 生成可视化
    demo.visualize_results()
    
    # 敏感性分析
    demo.cost_sensitivity_analysis()
    
    # 生成报告
    demo.generate_report()
    
    print("\n运输问题演示完成。")
    print(f"基础运输最小成本：{min_cost:.2f} 元")
    print(f"多产品运输最小成本：{multi_min_cost:.2f} 元")

if __name__ == "__main__":
    main()

