#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络流优化演示
Network Flow Optimization Demo

演示内容：网络流问题
- 最大流问题：网络容量限制下的最大流量
- 最小费用流问题：在满足需求的前提下最小化成本
- 最短路径问题：寻找两点间的最短路径

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pulp
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 路径与中文字体：移动到子目录后也能导入根目录的配置
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from font_config import setup_chinese_font
setup_chinese_font()

class NetworkFlowDemo:
    """网络流优化演示类
    作用：封装最大流、最小费用流与最短路径的建模求解、可视化与报告。
    设计：面向对象组织流程；结果保存在 self.results/self.graphs 以便复用。
    规则：中文输出、统一可视化样式、PNG高分辨率保存。
    """
    
    def __init__(self):
        self.results = {}
        self.graphs = {}
        print("=" * 50)
        print("网络流优化演示")
        print("=" * 50)
    
    def solve_max_flow_problem(self):
        """最大流问题 - 供水网络优化
        作用：基于有向图与容量约束，计算从源点到汇点的最大流量。
        语法要点：
        - 使用 NetworkX 的 maximum_flow (Edmonds–Karp)
        - 边属性包含 capacity 与 flow，便于可视化展示利用率
        原理：最大流-最小割定理；瓶颈边决定整体可达流量。
        规则：中文输出与统一风格；结果存储供后续图表与报告使用。
        """
        print("\n最大流问题 - 供水网络优化")
        print("-" * 40)
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 节点：源点S，中间节点A,B,C,D，汇点T
        nodes = ['S', 'A', 'B', 'C', 'D', 'T']
        G.add_nodes_from(nodes)
        
        # 边和容量 (起点, 终点, 容量)
        edges_capacity = [
            ('S', 'A', 16), ('S', 'B', 13),
            ('A', 'B', 4), ('A', 'C', 12),
            ('B', 'D', 14), ('C', 'B', 9),
            ('C', 'T', 20), ('D', 'C', 7),
            ('D', 'T', 4)
        ]
        
        # 添加边
        for start, end, capacity in edges_capacity:
            G.add_edge(start, end, capacity=capacity, flow=0)
        
        print("网络结构:")
        print("节点: 水源S → 中间节点A,B,C,D → 用户T")
        print("边容量 (管道最大流量):")
        for start, end, capacity in edges_capacity:
            print(f"  {start} → {end}: {capacity} 单位/小时")
        
        # 使用NetworkX求解最大流（Edmonds–Karp）
        max_flow_value, max_flow_dict = nx.maximum_flow(G, 'S', 'T')
        
        print(f"\n最大流结果：")
        print(f"  最大流量: {max_flow_value} 单位/小时")
        
        print(f"\n最优流量分配：")
        total_flow_used = 0
        flow_details = []
        for start in max_flow_dict:
            for end in max_flow_dict[start]:
                flow = max_flow_dict[start][end]
                if flow > 0:
                    capacity = G[start][end]['capacity']
                    utilization = flow / capacity * 100
                    flow_details.append({
                        'from': start,
                        'to': end,
                        'flow': flow,
                        'capacity': capacity,
                        'utilization': utilization
                    })
                    print(f"  {start} → {end}: {flow}/{capacity} "
                          f"(利用率: {utilization:.1f}%)")
                    total_flow_used += flow
        
        # 找出瓶颈边
        bottleneck_edges = [detail for detail in flow_details 
                           if detail['utilization'] >= 99.9]
        
        if bottleneck_edges:
            print(f"\n网络瓶颈：")
            for edge in bottleneck_edges:
                print(f"  {edge['from']} → {edge['to']}: 满负荷运行")
        
        # 保存结果
        self.results['max_flow'] = {
            'graph': G,
            'max_flow_value': max_flow_value,
            'flow_dict': max_flow_dict,
            'flow_details': flow_details,
            'bottleneck_edges': bottleneck_edges
        }
        self.graphs['max_flow'] = G
        
        return max_flow_value, max_flow_dict
    
    def solve_min_cost_flow_problem(self):
        """最小费用流问题 - 物流配送优化
        作用：在供应与需求约束下，决定各路线流量以最小化总成本。
        语法要点：
        - PuLP 非负连续变量 x_{i,j}
        - 目标函数：Σ cost · x；约束：供应等式、需求等式、容量上限
        原理：网络流的线性规划形式；影子价格反映路线紧张程度。
        规则：中文输出、统一样式；结果保存供可视化与报告。
        """
        print("\n最小费用流问题 - 物流配送优化")
        print("-" * 40)
        
        # 网络节点
        warehouses = ['仓库1', '仓库2']
        customers = ['客户A', '客户B', '客户C']
        
        # 供应量和需求量
        supply = {'仓库1': 100, '仓库2': 150}
        demand = {'客户A': 80, '客户B': 90, '客户C': 80}
        
        print("供需信息：")
        print(f"仓库供应量：{supply}")
        print(f"客户需求量：{demand}")
        print(f"总供应量：{sum(supply.values())}")
        print(f"总需求量：{sum(demand.values())}")
        
        # 运输成本和容量
        # (起点, 终点, 单位成本, 容量)
        transport_data = [
            ('仓库1', '客户A', 4, 60),
            ('仓库1', '客户B', 6, 70),
            ('仓库1', '客户C', 8, 50),
            ('仓库2', '客户A', 5, 50),
            ('仓库2', '客户B', 3, 80),
            ('仓库2', '客户C', 7, 60)
        ]
        
        print("\n运输成本和容量限制：")
        for start, end, cost, capacity in transport_data:
            print(f"  {start} → {end}：成本{cost}元/单位，容量{capacity}单位")
        
        # 使用PuLP求解最小费用流
        prob = pulp.LpProblem("最小费用流问题", pulp.LpMinimize)
        
        # 决策变量：从仓库i到客户j的运输量
        x = {}
        for start, end, cost, capacity in transport_data:
            x[start, end] = pulp.LpVariable(f"x_{start}_{end}", 
                                          lowBound=0, upBound=capacity)
        
        # 目标函数：最小化总运输成本
        prob += pulp.lpSum([cost * x[start, end] 
                           for start, end, cost, capacity in transport_data])
        
        # 约束条件
        # 1. 供应约束
        for warehouse in warehouses:
            prob += pulp.lpSum([x[warehouse, customer] 
                               for customer in customers 
                               if (warehouse, customer) in x]) <= supply[warehouse]
        
        # 2. 需求约束
        for customer in customers:
            prob += pulp.lpSum([x[warehouse, customer] 
                               for warehouse in warehouses 
                               if (warehouse, customer) in x]) >= demand[customer]
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 结果
        min_cost = pulp.value(prob.objective)
        
        print("\n最小费用流结果：")
        print(f"  最小运输成本：{min_cost:.2f} 元")
        
        print(f"\n最优配送方案：")
        flow_solution = {}
        cost_details = []
        for start, end, cost, capacity in transport_data:
            flow = x[start, end].varValue
            if flow > 0:
                total_cost = flow * cost
                cost_details.append({
                    'from': start,
                    'to': end,
                    'flow': flow,
                    'unit_cost': cost,
                    'total_cost': total_cost,
                    'capacity': capacity,
                    'utilization': flow / capacity * 100
                })
                flow_solution[start, end] = flow
                print(f"  {start} → {end}：{flow:.1f}单位，成本：{total_cost:.2f}元")
        
        # 保存结果
        self.results['min_cost_flow'] = {
            'warehouses': warehouses,
            'customers': customers,
            'supply': supply,
            'demand': demand,
            'transport_data': transport_data,
            'min_cost': min_cost,
            'flow_solution': flow_solution,
            'cost_details': cost_details
        }
        
        return min_cost, flow_solution
    
    def solve_shortest_path_problem(self):
        """最短路径问题 - 城市交通网络
        作用：计算两点间的最短路径及距离，并统计所有源的最短路径。
        语法要点：NetworkX shortest_path 与 shortest_path_length；边权为距离 `weight`。
        原理：最短路径的图论算法；用于交通/通信/物流的路径优化。
        规则：中文输出，结果保存供可视化。
        """
        print("\n最短路径问题 - 城市交通网络")
        print("-" * 40)
        
        # 创建城市交通网络
        G = nx.Graph()
        
        # 城市节点
        cities = ['起点', '城市A', '城市B', '城市C', '城市D', '终点']
        G.add_nodes_from(cities)
        
        # 道路和距离 (城市1, 城市2, 距离km)
        roads = [
            ('起点', '城市A', 10), ('起点', '城市B', 15),
            ('城市A', '城市C', 12), ('城市A', '城市D', 15),
            ('城市B', '城市C', 8), ('城市B', '城市D', 7),
            ('城市C', '终点', 10), ('城市D', '终点', 12),
            ('城市A', '城市B', 6), ('城市C', '城市D', 5)
        ]
        
        # 添加边
        for city1, city2, distance in roads:
            G.add_edge(city1, city2, weight=distance)
        
        print("交通网络：")
        print("城市节点：", cities)
        print("道路距离：")
        for city1, city2, distance in roads:
            print(f"  {city1} ↔ {city2}: {distance} km")
        
        # 使用Dijkstra算法求最短路径
        shortest_path = nx.shortest_path(G, '起点', '终点', weight='weight')
        # 计算最短路径与距离（Dijkstra，权重字段为 'weight'）
        shortest_distance = nx.shortest_path_length(G, '起点', '终点', weight='weight')
        
        print("\n最短路径结果：")
        print(f"  最短距离：{shortest_distance} km")
        print(f"  最短路径：{' → '.join(shortest_path)}")
        
        # 计算路径详情
        path_details = []
        total_distance = 0
        for i in range(len(shortest_path) - 1):
            start = shortest_path[i]
            end = shortest_path[i + 1]
            distance = G[start][end]['weight']
            total_distance += distance
            path_details.append({
                'from': start,
                'to': end,
                'distance': distance,
                'cumulative': total_distance
            })
            print(f"  第{i+1}段：{start} → {end}，{distance} km "
                  f"(累计：{total_distance} km)")
        
        # 计算所有节点间的最短路径（用于分析网络连通性）
        all_shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        
        print(f"\n网络连通性分析：")
        print(f"  网络直径: {nx.diameter(G, weight='weight'):.1f} km")
        print(f"  平均路径长度: {nx.average_shortest_path_length(G, weight='weight'):.1f} km")
        
        # 保存结果
        self.results['shortest_path'] = {
            'graph': G,
            'cities': cities,
            'roads': roads,
            'shortest_path': shortest_path,
            'shortest_distance': shortest_distance,
            'path_details': path_details,
            'all_shortest_paths': all_shortest_paths
        }
        self.graphs['shortest_path'] = G
        
        return shortest_path, shortest_distance
    
    def visualize_results(self):
        """可视化网络流结果
        作用：多维度展示最大流网络、流量分布、最小费用流、最短路径和网络性能分析，统一中文标签和样式。
        规则：中文标签、统一样式、网格 alpha=0.3、PNG输出（dpi=300）。
        """
        if not self.results:
            print("请先运行求解方法")
            return
        
        print("\n生成网络流可视化图表…")
        
        # 设置统一图表样式
        plt.style.use('seaborn-v0_8')
        
        # 创建2x3子图布局，展示更全面的网络流分析
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 最大流网络图（改进布局）
        if 'max_flow' in self.results:
            max_flow_data = self.results['max_flow']
            G = max_flow_data['graph']
            
            # 改进的节点位置布局 - 更清晰的层次结构
            pos = {
                'S': (0, 2),      # 源点居中
                'A': (2, 3),      # 第一层上方
                'B': (2, 1),      # 第一层下方
                'C': (4, 3),      # 第二层上方
                'D': (4, 1),      # 第二层下方
                'T': (6, 2)       # 汇点居中
            }
            
            # 绘制节点 - 源汇点特殊标记
            source_sink = ['S', 'T']
            intermediate = [n for n in G.nodes() if n not in source_sink]
            
            nx.draw_networkx_nodes(G, pos, nodelist=source_sink, 
                                 node_color='#FF6B6B', node_size=1200, ax=ax1)
            nx.draw_networkx_nodes(G, pos, nodelist=intermediate, 
                                 node_color='#4ECDC4', node_size=1000, ax=ax1)
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax1)
            
            # 绘制边 - 根据流量设置颜色和粗细
            for start, end in G.edges():
                flow = max_flow_data['flow_dict'][start][end]
                capacity = G[start][end]['capacity']
                
                if flow > 0:
                    # 有流量的边 - 红色，粗细根据流量比例
                    width = max(2, (flow / capacity) * 8)
                    alpha = 0.6 + 0.4 * (flow / capacity)
                    nx.draw_networkx_edges(G, pos, [(start, end)], 
                                         width=width, edge_color='red', 
                                         alpha=alpha, ax=ax1)
                    
                    # 流量标签 - 更好的位置和样式
                    x1, y1 = pos[start]
                    x2, y2 = pos[end]
                    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                    
                    # 根据边的方向调整标签位置
                    offset_y = 0.15 if y1 == y2 else 0
                    offset_x = 0.15 if x1 == x2 else 0
                    
                    ax1.text(mid_x + offset_x, mid_y + offset_y, 
                            f'{flow}/{capacity}', 
                            fontsize=9, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.9, edgecolor='red'))
                else:
                    # 无流量的边 - 灰色虚线
                    nx.draw_networkx_edges(G, pos, [(start, end)], 
                                         width=1, edge_color='gray', 
                                         style='dashed', alpha=0.5, ax=ax1)
            
            ax1.set_title(f'最大流网络图\n最大流量: {max_flow_data["max_flow_value"]} 单位/小时', 
                         fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # 2. 边流量利用率分析
            edges = list(G.edges())
            utilization_rates = []
            edge_labels = []
            
            for start, end in edges:
                flow = max_flow_data['flow_dict'][start][end]
                capacity = G[start][end]['capacity']
                utilization = (flow / capacity) * 100 if capacity > 0 else 0
                utilization_rates.append(utilization)
                edge_labels.append(f'{start}→{end}')
            
            colors = ['#FF6B6B' if rate > 80 else '#FFD93D' if rate > 50 else '#4ECDC4' 
                     for rate in utilization_rates]
            
            bars2 = ax2.bar(range(len(edges)), utilization_rates, color=colors)
            ax2.set_title('边流量利用率分析', fontsize=14, fontweight='bold')
            ax2.set_ylabel('利用率 (%)')
            ax2.set_xlabel('边')
            ax2.set_xticks(range(len(edges)))
            ax2.set_xticklabels(edge_labels, rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='满负荷')
            ax2.legend()
            
            # 添加利用率标签
            for bar, rate in zip(bars2, utilization_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. 最小费用流网络
        if 'min_cost_flow' in self.results:
            mcf_data = self.results['min_cost_flow']
            
            # 创建图结构用于可视化
            G_mcf = nx.DiGraph()
            
            # 添加节点
            warehouses = mcf_data['warehouses']
            customers = mcf_data['customers']
            G_mcf.add_nodes_from(warehouses)
            G_mcf.add_nodes_from(customers)
            
            # 添加边和成本信息
            for start, end, cost, capacity in mcf_data['transport_data']:
                G_mcf.add_edge(start, end, cost=cost, capacity=capacity)
            
            # 使用改进的布局
            pos_mcf = nx.spring_layout(G_mcf, k=2, iterations=50, seed=42)
            
            # 绘制节点 - 仓库和客户区分颜色
            nx.draw_networkx_nodes(G_mcf, pos_mcf, nodelist=warehouses,
                                 node_color='#FF6B6B', node_size=1000, ax=ax3)
            nx.draw_networkx_nodes(G_mcf, pos_mcf, nodelist=customers,
                                 node_color='#4ECDC4', node_size=800, ax=ax3)
            nx.draw_networkx_labels(G_mcf, pos_mcf, font_size=10, ax=ax3)
            
            # 绘制边 - 根据成本设置颜色
            edges_mcf = G_mcf.edges()
            costs = [G_mcf[u][v]['cost'] for u, v in edges_mcf]
            max_cost = max(costs) if costs else 1
            
            for (u, v) in edges_mcf:
                cost = G_mcf[u][v]['cost']
                # 成本越高颜色越红
                color_intensity = cost / max_cost
                color = plt.cm.Reds(0.3 + 0.7 * color_intensity)
                
                # 检查是否有流量
                flow = mcf_data['flow_solution'].get((u, v), 0)
                width = 3 if flow > 0 else 1
                alpha = 1.0 if flow > 0 else 0.5
                
                nx.draw_networkx_edges(G_mcf, pos_mcf, [(u, v)], 
                                     edge_color=[color], width=width, alpha=alpha, ax=ax3)
            
            # 添加成本标签
            edge_labels_mcf = {(u, v): f'{G_mcf[u][v]["cost"]}' for u, v in edges_mcf}
            nx.draw_networkx_edge_labels(G_mcf, pos_mcf, edge_labels_mcf, 
                                       font_size=8, ax=ax3)
            
            ax3.set_title(f'最小费用流网络\n最小成本: {mcf_data["min_cost"]:.0f} 元', 
                         fontsize=14, fontweight='bold')
            ax3.axis('off')
        
        # 4. 最短路径网络（改进布局）
        if 'shortest_path' in self.results:
            sp_data = self.results['shortest_path']
            G_sp = sp_data['graph']
            
            # 使用更好的布局算法
            pos_sp = nx.kamada_kawai_layout(G_sp)
            
            # 绘制所有边
            nx.draw_networkx_edges(G_sp, pos_sp, edge_color='lightgray', 
                                 width=1, alpha=0.5, ax=ax4)
            
            # 高亮最短路径
            shortest_path = sp_data['shortest_path']
            path_edges = [(shortest_path[i], shortest_path[i+1]) 
                         for i in range(len(shortest_path)-1)]
            
            # 绘制最短路径 - 渐变效果
            for i, (u, v) in enumerate(path_edges):
                color_intensity = 1 - (i / len(path_edges)) * 0.5
                nx.draw_networkx_edges(G_sp, pos_sp, [(u, v)], 
                                     edge_color='red', width=4, 
                                     alpha=color_intensity, ax=ax4)
            
            # 绘制节点 - 路径上的节点特殊标记
            path_nodes = set(shortest_path)
            other_nodes = [n for n in G_sp.nodes() if n not in path_nodes]
            
            nx.draw_networkx_nodes(G_sp, pos_sp, nodelist=list(path_nodes), 
                                 node_color='#FF6B6B', node_size=900, ax=ax4)
            nx.draw_networkx_nodes(G_sp, pos_sp, nodelist=other_nodes, 
                                 node_color='lightblue', node_size=600, ax=ax4)
            nx.draw_networkx_labels(G_sp, pos_sp, font_size=10, ax=ax4)
            
            # 添加距离标签
            edge_labels_sp = nx.get_edge_attributes(G_sp, 'weight')
            nx.draw_networkx_edge_labels(G_sp, pos_sp, edge_labels_sp, 
                                       font_size=8, ax=ax4)
            
            ax4.set_title(f'最短路径网络\n最短距离: {sp_data["shortest_distance"]} km', 
                         fontsize=14, fontweight='bold')
            ax4.axis('off')
        
        # 5. 网络性能指标对比
        if self.results:
            metrics = []
            values = []
            
            if 'max_flow' in self.results:
                metrics.append('最大流量')
                values.append(self.results['max_flow']['max_flow_value'])
            
            if 'min_cost_flow' in self.results:
                metrics.append('最小成本')
                values.append(self.results['min_cost_flow']['min_cost'])
            
            if 'shortest_path' in self.results:
                metrics.append('最短距离')
                values.append(self.results['shortest_path']['shortest_distance'])
            
            if metrics:
                # 标准化数值以便比较
                normalized_values = [(v / max(values)) * 100 for v in values]
                colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(metrics)]
                
                bars5 = ax5.bar(metrics, normalized_values, color=colors_metrics)
                ax5.set_title('网络性能指标对比\n(标准化至100%)', fontsize=14, fontweight='bold')
                ax5.set_ylabel('标准化值 (%)')
                ax5.tick_params(axis='x', rotation=45)
                ax5.grid(True, alpha=0.3)
                
                # 添加原始数值标签
                for bar, original_val, norm_val in zip(bars5, values, normalized_values):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                            f'{original_val}\n({norm_val:.1f}%)', 
                            ha='center', va='bottom')
        
        # 6. 网络拓扑分析
        if 'max_flow' in self.results:
            G_topo = self.results['max_flow']['graph']
            
            # 计算网络拓扑指标
            degree_centrality = nx.degree_centrality(G_topo)
            betweenness_centrality = nx.betweenness_centrality(G_topo)
            
            nodes = list(G_topo.nodes())
            degree_values = [degree_centrality[node] * 100 for node in nodes]
            betweenness_values = [betweenness_centrality[node] * 100 for node in nodes]
            
            x_pos = np.arange(len(nodes))
            width = 0.35
            
            bars6_1 = ax6.bar(x_pos - width/2, degree_values, width, 
                             label='度中心性', color='#FF9999', alpha=0.8)
            bars6_2 = ax6.bar(x_pos + width/2, betweenness_values, width, 
                             label='介数中心性', color='#99CCFF', alpha=0.8)
            
            ax6.set_title('节点重要性分析', fontsize=14, fontweight='bold')
            ax6.set_ylabel('中心性 (%)')
            ax6.set_xlabel('节点')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(nodes)
            ax6.grid(True, alpha=0.3)
            ax6.legend()
            
            # 添加数值标签
            for bars in [bars6_1, bars6_2]:
                for bar in bars:
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2, height + 1,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(BASE_DIR, 'network_flow_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("网络流可视化图表已保存为 'network_flow_results.png'")
    
    def network_analysis(self):
        """网络结构分析
        作用：输出节点数、边数、密度、连通性等指标，并给出业务解读与建议。
        规则：中文输出、结构化信息。
        """
        if not self.graphs:
            print("请先运行求解方法")
            return
        
        print("\n网络结构分析")
        print("-" * 30)
        
        for problem_type, G in self.graphs.items():
            print(f"\n{problem_type.upper()} 网络:")
            print(f"  • 节点数: {G.number_of_nodes()}")
            print(f"  • 边数: {G.number_of_edges()}")
            print(f"  • 网络密度: {nx.density(G):.3f}")
            
            if nx.is_connected(G.to_undirected()):
                print(f"  • 网络连通性: 连通")
                if problem_type == 'shortest_path':
                    print(f"  • 网络直径: {nx.diameter(G):.1f}")
                    print(f"  • 平均路径长度: {nx.average_shortest_path_length(G):.1f}")
            else:
                print(f"  • 网络连通性: 非连通")
    
    def generate_report(self):
        """生成详细报告
        作用：结构化总结最大流、最小费用流与最短路径的关键结果与管理建议。
        规则：条理清晰、教学友好；将技术结果转化为业务可读信息。
        """
        if not self.results:
            print("请先运行求解方法")
            return
        
        print("\n" + "="*50)
        print("网络流优化报告")
        print("="*50)
        
        if 'max_flow' in self.results:
            max_flow_data = self.results['max_flow']
            print(f"\n最大流问题：")
            print(f"  • 优化目标: 最大化网络流量")
            print(f"  • 最大流量: {max_flow_data['max_flow_value']} 单位/小时")
            print(f"  • 活跃边数: {len(max_flow_data['flow_details'])}")
            
            if max_flow_data['bottleneck_edges']:
                print(f"  • 瓶颈边数: {len(max_flow_data['bottleneck_edges'])}")
                print(f"  • 瓶颈位置: {', '.join([f"{e['from']}-{e['to']}" for e in max_flow_data['bottleneck_edges']])}")
        
        if 'min_cost_flow' in self.results:
            mcf_data = self.results['min_cost_flow']
            print(f"\n最小费用流问题：")
            print(f"  • 优化目标: 最小化运输成本")
            print(f"  • 最小成本: {mcf_data['min_cost']:.2f} 元")
            print(f"  • 总供应量: {sum(mcf_data['supply'].values())} 单位")
            print(f"  • 总需求量: {sum(mcf_data['demand'].values())} 单位")
            
            if mcf_data['cost_details']:
                avg_cost = mcf_data['min_cost'] / sum(detail['flow'] for detail in mcf_data['cost_details'])
                print(f"  • 平均运输成本: {avg_cost:.2f} 元/单位")
        
        if 'shortest_path' in self.results:
            sp_data = self.results['shortest_path']
            print(f"\n最短路径问题：")
            print(f"  • 优化目标: 最小化路径距离")
            print(f"  • 最短距离: {sp_data['shortest_distance']} km")
            print(f"  • 路径长度: {len(sp_data['shortest_path'])} 个节点")
            print(f"  • 路径: {' → '.join(sp_data['shortest_path'])}")
        
        print(f"\n优化建议：")
        
        if 'max_flow' in self.results and max_flow_data['bottleneck_edges']:
            print(f"  • 最大流: 考虑扩容瓶颈边以提高网络流量")
        
        if 'min_cost_flow' in self.results:
            print(f"  • 最小费用流: 优化高成本路线，寻找替代方案")
        
        if 'shortest_path' in self.results:
            print(f"  • 最短路径: 考虑建设新道路缩短关键路径")
        
        print("="*50)

def main():
    """主函数
    作用：按顺序执行最大流→最小费用流→最短路径→可视化→分析→报告，一键演示完整流程。
    使用规则：脚本运行时触发；导入为模块时不自动执行。
    """
    # 创建演示实例
    demo = NetworkFlowDemo()
    
    # 求解最大流问题
    max_flow_value, max_flow_dict = demo.solve_max_flow_problem()
    
    # 求解最小费用流问题
    min_cost, flow_solution = demo.solve_min_cost_flow_problem()
    
    # 求解最短路径问题
    shortest_path, shortest_distance = demo.solve_shortest_path_problem()
    
    # 生成可视化
    demo.visualize_results()
    
    # 网络分析
    demo.network_analysis()
    
    # 生成报告
    demo.generate_report()
    
    print("\n网络流优化演示完成。")
    print(f"最大流量：{max_flow_value} 单位/小时")
    print(f"最小运输成本：{min_cost:.2f} 元")
    print(f"最短路径距离：{shortest_distance} km")

if __name__ == "__main__":
    main()