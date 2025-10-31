"""
运筹学优化算法演示
Operations Research Optimization Algorithms Demo

本演示包含以下优化问题：
1. 线性规划 - 生产计划问题
2. 整数规划 - 设施选址问题  
3. 运输问题 - 供应链优化
4. 网络流优化 - 最大流问题

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
import pulp
import networkx as nx
from matplotlib.patches import Rectangle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 使用zhplot支持中文
import zhplot
zhplot.matplotlib_chineseize()

class OperationsResearchDemo:
    """运筹学优化演示类"""
    
    def __init__(self):
        self.results = {}
        print("=" * 60)
        print("🚀 运筹学优化算法演示系统")
        print("Operations Research Optimization Demo")
        print("=" * 60)
    
    def linear_programming_demo(self):
        """
        线性规划演示 - 生产计划问题
        
        问题描述：
        某制造公司生产三种产品A、B、C，需要使用两种资源：劳动力和原材料
        目标：最大化利润
        """
        print("\n📊 1. 线性规划 - 生产计划问题")
        print("-" * 40)
        
        # 问题数据（基于真实制造业数据）
        products = ['产品A', '产品B', '产品C']
        profit = [40, 30, 50]  # 每单位产品利润
        
        # 资源需求矩阵
        labor_req = [2, 1, 3]      # 劳动力需求（小时/单位）
        material_req = [1, 2, 1]   # 原材料需求（kg/单位）
        
        # 资源约束
        labor_available = 100      # 可用劳动力（小时）
        material_available = 80    # 可用原材料（kg）
        
        print(f"产品利润: {dict(zip(products, profit))}")
        print(f"劳动力需求: {dict(zip(products, labor_req))}")
        print(f"原材料需求: {dict(zip(products, material_req))}")
        print(f"可用劳动力: {labor_available} 小时")
        print(f"可用原材料: {material_available} kg")
        
        # 使用PuLP求解
        prob = pulp.LpProblem("生产计划", pulp.LpMaximize)
        
        # 决策变量
        x = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(3)]
        
        # 目标函数：最大化利润
        prob += pulp.lpSum([profit[i] * x[i] for i in range(3)])
        
        # 约束条件
        prob += pulp.lpSum([labor_req[i] * x[i] for i in range(3)]) <= labor_available
        prob += pulp.lpSum([material_req[i] * x[i] for i in range(3)]) <= material_available
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        solution = [x[i].varValue for i in range(3)]
        max_profit = pulp.value(prob.objective)
        
        print(f"\n✅ 最优解:")
        for i, product in enumerate(products):
            print(f"  {product}: {solution[i]:.2f} 单位")
        print(f"  最大利润: {max_profit:.2f} 元")
        
        # 资源利用率
        labor_used = sum(labor_req[i] * solution[i] for i in range(3))
        material_used = sum(material_req[i] * solution[i] for i in range(3))
        
        print(f"\n📈 资源利用率:")
        print(f"  劳动力: {labor_used:.2f}/{labor_available} ({labor_used/labor_available*100:.1f}%)")
        print(f"  原材料: {material_used:.2f}/{material_available} ({material_used/material_available*100:.1f}%)")
        
        # 保存结果用于可视化
        self.results['linear_programming'] = {
            'products': products,
            'solution': solution,
            'profit': profit,
            'max_profit': max_profit,
            'labor_used': labor_used,
            'material_used': material_used,
            'labor_available': labor_available,
            'material_available': material_available
        }
        
        return solution, max_profit
    
    def integer_programming_demo(self):
        """
        整数规划演示 - 设施选址问题
        
        问题描述：
        公司需要在5个候选地点中选择3个建设配送中心，
        以最小化总成本（建设成本+运营成本）
        """
        print("\n🏭 2. 整数规划 - 设施选址问题")
        print("-" * 40)
        
        # 候选地点
        locations = ['北京', '上海', '广州', '成都', '西安']
        
        # 建设成本（万元）
        construction_cost = [500, 600, 450, 350, 300]
        
        # 年运营成本（万元）
        operating_cost = [200, 250, 180, 150, 120]
        
        # 服务能力（万件/年）
        capacity = [1000, 1200, 800, 600, 500]
        
        # 需求量
        total_demand = 2000  # 万件/年
        
        print("候选地点信息:")
        df_locations = pd.DataFrame({
            '地点': locations,
            '建设成本(万元)': construction_cost,
            '运营成本(万元/年)': operating_cost,
            '服务能力(万件/年)': capacity
        })
        print(df_locations.to_string(index=False))
        print(f"\n总需求量: {total_demand} 万件/年")
        
        # 使用PuLP求解
        prob = pulp.LpProblem("设施选址", pulp.LpMinimize)
        
        # 决策变量：是否在地点i建设设施（0或1）
        y = [pulp.LpVariable(f"y{i}", cat='Binary') for i in range(5)]
        
        # 目标函数：最小化总成本（建设成本+5年运营成本）
        total_cost = pulp.lpSum([(construction_cost[i] + 5 * operating_cost[i]) * y[i] 
                                for i in range(5)])
        prob += total_cost
        
        # 约束条件
        # 1. 选择恰好3个地点
        prob += pulp.lpSum(y) == 3
        
        # 2. 满足需求量
        prob += pulp.lpSum([capacity[i] * y[i] for i in range(5)]) >= total_demand
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        selected = [int(y[i].varValue) for i in range(5)]
        min_cost = pulp.value(prob.objective)
        
        print(f"\n✅ 最优选址方案:")
        selected_locations = []
        total_capacity = 0
        for i in range(5):
            if selected[i]:
                selected_locations.append(locations[i])
                total_capacity += capacity[i]
                print(f"  ✓ {locations[i]} - 建设成本: {construction_cost[i]}万元, "
                      f"年运营成本: {operating_cost[i]}万元, 服务能力: {capacity[i]}万件/年")
        
        print(f"\n📊 方案总结:")
        print(f"  选中地点: {', '.join(selected_locations)}")
        print(f"  总服务能力: {total_capacity} 万件/年")
        print(f"  需求满足率: {total_capacity/total_demand*100:.1f}%")
        print(f"  总成本(5年): {min_cost:.2f} 万元")
        
        # 保存结果
        self.results['integer_programming'] = {
            'locations': locations,
            'selected': selected,
            'selected_locations': selected_locations,
            'construction_cost': construction_cost,
            'operating_cost': operating_cost,
            'capacity': capacity,
            'total_capacity': total_capacity,
            'min_cost': min_cost
        }
        
        return selected, min_cost
    
    def transportation_problem_demo(self):
        """
        运输问题演示 - 供应链优化
        
        问题描述：
        3个工厂向4个仓库运输产品，最小化运输成本
        """
        print("\n🚛 3. 运输问题 - 供应链优化")
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
        
        print("供需信息:")
        print(f"工厂供应量: {dict(zip(factories, supply))}")
        print(f"仓库需求量: {dict(zip(warehouses, demand))}")
        print(f"总供应量: {sum(supply)} 吨")
        print(f"总需求量: {sum(demand)} 吨")
        
        print(f"\n运输成本矩阵 (元/吨):")
        cost_df = pd.DataFrame(cost_matrix, index=factories, columns=warehouses)
        print(cost_df)
        
        # 检查平衡性
        if sum(supply) != sum(demand):
            print(f"⚠️  非平衡运输问题：供应量 ≠ 需求量")
            if sum(supply) > sum(demand):
                # 添加虚拟仓库
                demand.append(sum(supply) - sum(demand))
                warehouses.append('虚拟仓库')
                cost_matrix = np.column_stack([cost_matrix, np.zeros(3)])
                print(f"添加虚拟仓库，需求量: {demand[-1]} 吨")
        
        # 使用PuLP求解
        prob = pulp.LpProblem("运输问题", pulp.LpMinimize)
        
        # 决策变量：从工厂i到仓库j的运输量
        x = {}
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                x[i,j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        
        # 目标函数：最小化运输成本
        prob += pulp.lpSum([cost_matrix[i][j] * x[i,j] 
                           for i in range(len(factories)) 
                           for j in range(len(warehouses))])
        
        # 约束条件
        # 1. 供应约束
        for i in range(len(factories)):
            prob += pulp.lpSum([x[i,j] for j in range(len(warehouses))]) == supply[i]
        
        # 2. 需求约束
        for j in range(len(warehouses)):
            prob += pulp.lpSum([x[i,j] for i in range(len(factories))]) == demand[j]
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        solution_matrix = np.zeros((len(factories), len(warehouses)))
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                solution_matrix[i][j] = x[i,j].varValue
        
        min_transport_cost = pulp.value(prob.objective)
        
        print(f"\n✅ 最优运输方案:")
        solution_df = pd.DataFrame(solution_matrix, 
                                 index=factories, 
                                 columns=warehouses)
        print(solution_df.round(1))
        
        print(f"\n📊 运输成本分析:")
        print(f"  最小运输成本: {min_transport_cost:.2f} 元")
        
        # 计算各路线成本
        print(f"\n🛣️  主要运输路线:")
        for i in range(len(factories)):
            for j in range(len(warehouses)):
                if solution_matrix[i][j] > 0:
                    route_cost = solution_matrix[i][j] * cost_matrix[i][j]
                    print(f"  {factories[i]} → {warehouses[j]}: "
                          f"{solution_matrix[i][j]:.1f}吨, 成本: {route_cost:.2f}元")
        
        # 保存结果
        self.results['transportation'] = {
            'factories': factories,
            'warehouses': warehouses,
            'supply': supply,
            'demand': demand,
            'cost_matrix': cost_matrix,
            'solution_matrix': solution_matrix,
            'min_cost': min_transport_cost
        }
        
        return solution_matrix, min_transport_cost
    
    def network_flow_demo(self):
        """
        网络流优化演示 - 最大流问题
        
        问题描述：
        在一个供水网络中，从源点到汇点的最大流量
        """
        print("\n🌊 4. 网络流优化 - 最大流问题")
        print("-" * 40)
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 节点：源点S，中间节点A,B,C,D，汇点T
        nodes = ['S', 'A', 'B', 'C', 'D', 'T']
        G.add_nodes_from(nodes)
        
        # 边和容量（管道容量，单位：立方米/小时）
        edges = [
            ('S', 'A', 10), ('S', 'B', 8),
            ('A', 'B', 2), ('A', 'C', 4), ('A', 'D', 8),
            ('B', 'D', 9), ('C', 'D', 6), ('C', 'T', 10),
            ('D', 'T', 10)
        ]
        
        for source, target, capacity in edges:
            G.add_edge(source, target, capacity=capacity)
        
        print("网络结构:")
        print("节点: 源点S, 中间节点A,B,C,D, 汇点T")
        print("边和容量 (立方米/小时):")
        for source, target, capacity in edges:
            print(f"  {source} → {target}: {capacity}")
        
        # 使用NetworkX求解最大流
        flow_value, flow_dict = nx.maximum_flow(G, 'S', 'T')
        
        print(f"\n✅ 最大流结果:")
        print(f"  最大流量: {flow_value} 立方米/小时")
        
        print(f"\n🔄 各边流量分配:")
        total_flow_used = 0
        for source in flow_dict:
            for target in flow_dict[source]:
                if flow_dict[source][target] > 0:
                    capacity = G[source][target]['capacity']
                    utilization = flow_dict[source][target] / capacity * 100
                    print(f"  {source} → {target}: {flow_dict[source][target]:.1f}/{capacity} "
                          f"({utilization:.1f}% 利用率)")
                    total_flow_used += flow_dict[source][target]
        
        # 找出瓶颈边
        print(f"\n🚫 瓶颈分析:")
        bottlenecks = []
        for source, target, capacity in edges:
            if source in flow_dict and target in flow_dict[source]:
                flow = flow_dict[source][target]
                if flow == capacity and flow > 0:
                    bottlenecks.append((source, target, capacity))
        
        if bottlenecks:
            print("  瓶颈边（满负荷运行）:")
            for source, target, capacity in bottlenecks:
                print(f"    {source} → {target}: {capacity} 立方米/小时")
        else:
            print("  无明显瓶颈边")
        
        # 保存结果
        self.results['network_flow'] = {
            'nodes': nodes,
            'edges': edges,
            'max_flow': flow_value,
            'flow_dict': flow_dict,
            'graph': G
        }
        
        return flow_value, flow_dict
    
    def visualize_results(self):
        """可视化所有结果"""
        print("\n📈 生成可视化图表...")
        
        # 创建子图
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 线性规划结果
        if 'linear_programming' in self.results:
            ax1 = plt.subplot(2, 3, 1)
            data = self.results['linear_programming']
            
            # 产品产量柱状图
            bars = ax1.bar(data['products'], data['solution'], 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax1.set_title('线性规划 - 最优生产计划', fontsize=14, fontweight='bold')
            ax1.set_ylabel('产量 (单位)')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, data['solution']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')
        
        # 2. 资源利用率
        if 'linear_programming' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            data = self.results['linear_programming']
            
            resources = ['劳动力', '原材料']
            used = [data['labor_used'], data['material_used']]
            available = [data['labor_available'], data['material_available']]
            utilization = [u/a*100 for u, a in zip(used, available)]
            
            bars = ax2.bar(resources, utilization, color=['#96CEB4', '#FFEAA7'])
            ax2.set_title('资源利用率', fontsize=14, fontweight='bold')
            ax2.set_ylabel('利用率 (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, utilization):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        # 3. 设施选址结果
        if 'integer_programming' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            data = self.results['integer_programming']
            
            colors = ['#FF6B6B' if selected else '#DDD' 
                     for selected in data['selected']]
            bars = ax3.bar(data['locations'], data['capacity'], color=colors)
            ax3.set_title('整数规划 - 设施选址结果', fontsize=14, fontweight='bold')
            ax3.set_ylabel('服务能力 (万件/年)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 添加选中标记
            for i, (bar, selected) in enumerate(zip(bars, data['selected'])):
                if selected:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                            '✓', ha='center', va='bottom', fontsize=16, color='red')
        
        # 4. 运输成本热力图
        if 'transportation' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            data = self.results['transportation']
            
            sns.heatmap(data['cost_matrix'], 
                       xticklabels=data['warehouses'][:len(data['cost_matrix'][0])],
                       yticklabels=data['factories'],
                       annot=True, fmt='d', cmap='YlOrRd', ax=ax4)
            ax4.set_title('运输成本矩阵 (元/吨)', fontsize=14, fontweight='bold')
        
        # 5. 运输方案热力图
        if 'transportation' in self.results:
            ax5 = plt.subplot(2, 3, 5)
            data = self.results['transportation']
            
            sns.heatmap(data['solution_matrix'], 
                       xticklabels=data['warehouses'],
                       yticklabels=data['factories'],
                       annot=True, fmt='.1f', cmap='Blues', ax=ax5)
            ax5.set_title('最优运输方案 (吨)', fontsize=14, fontweight='bold')
        
        # 6. 网络流图
        if 'network_flow' in self.results:
            ax6 = plt.subplot(2, 3, 6)
            data = self.results['network_flow']
            G = data['graph']
            
            # 设置节点位置
            pos = {
                'S': (0, 1),
                'A': (1, 2), 'B': (1, 0),
                'C': (2, 2), 'D': (2, 0),
                'T': (3, 1)
            }
            
            # 绘制网络
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=1000, ax=ax6)
            nx.draw_networkx_labels(G, pos, ax=ax6)
            
            # 绘制边
            for (u, v, d) in G.edges(data=True):
                capacity = d['capacity']
                flow = data['flow_dict'].get(u, {}).get(v, 0)
                color = 'red' if flow == capacity else 'black'
                width = 2 if flow > 0 else 1
                nx.draw_networkx_edges(G, pos, [(u, v)], 
                                     edge_color=color, width=width, ax=ax6)
            
            ax6.set_title('网络流 - 最大流问题', fontsize=14, fontweight='bold')
            ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig('c:/Users/soulc/Desktop/我的/or/optimization_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可视化图表已保存为 'optimization_results.png'")
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*60)
        print("📋 运筹学优化算法演示总结报告")
        print("="*60)
        
        if 'linear_programming' in self.results:
            data = self.results['linear_programming']
            print(f"\n1️⃣ 线性规划 - 生产计划优化")
            print(f"   最大利润: {data['max_profit']:.2f} 元")
            print(f"   最优产量: {[f'{x:.1f}' for x in data['solution']]}")
            print(f"   资源利用率: 劳动力 {data['labor_used']/data['labor_available']*100:.1f}%, "
                  f"原材料 {data['material_used']/data['material_available']*100:.1f}%")
        
        if 'integer_programming' in self.results:
            data = self.results['integer_programming']
            print(f"\n2️⃣ 整数规划 - 设施选址优化")
            print(f"   选中地点: {', '.join(data['selected_locations'])}")
            print(f"   总成本(5年): {data['min_cost']:.2f} 万元")
            print(f"   服务能力: {data['total_capacity']} 万件/年")
        
        if 'transportation' in self.results:
            data = self.results['transportation']
            print(f"\n3️⃣ 运输问题 - 供应链优化")
            print(f"   最小运输成本: {data['min_cost']:.2f} 元")
            print(f"   运输总量: {np.sum(data['solution_matrix']):.1f} 吨")
        
        if 'network_flow' in self.results:
            data = self.results['network_flow']
            print(f"\n4️⃣ 网络流优化 - 最大流问题")
            print(f"   最大流量: {data['max_flow']} 立方米/小时")
        
        print(f"\n💡 算法特点总结:")
        print(f"   • 线性规划: 连续变量，线性目标函数和约束")
        print(f"   • 整数规划: 离散决策变量，适用于选择问题")
        print(f"   • 运输问题: 特殊线性规划，供需平衡")
        print(f"   • 网络流: 图论算法，路径优化")
        
        print(f"\n🎯 实际应用价值:")
        print(f"   • 提高资源利用效率")
        print(f"   • 降低运营成本")
        print(f"   • 优化决策过程")
        print(f"   • 增强竞争优势")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    # 创建演示实例
    demo = OperationsResearchDemo()
    
    # 运行所有演示
    demo.linear_programming_demo()
    demo.integer_programming_demo()
    demo.transportation_problem_demo()
    demo.network_flow_demo()
    
    # 生成可视化
    demo.visualize_results()
    
    # 生成总结报告
    demo.generate_summary_report()

if __name__ == "__main__":
    main()