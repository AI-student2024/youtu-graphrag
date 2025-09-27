"""
重新设计的Graph-First策略实现
基于空间节点的锚点图谱遍历方法
"""

import re
from typing import List, Tuple, Dict, Set, Optional
import json
import networkx as nx


class RedesignedGraphFirstStrategy:
    """重新设计的Graph-First策略"""
    
    def __init__(self, graph_data: List[Dict]):
        self.graph_data = graph_data
        self.graph = self._build_networkx_graph()
        self.nodes = dict(self.graph.nodes(data=True))
        self.relations = self._build_relation_index()
    
    def _build_networkx_graph(self) -> nx.MultiDiGraph:
        """构建NetworkX图"""
        graph = nx.MultiDiGraph()
        
        # Track nodes to avoid duplicates and assign consistent IDs
        node_mapping = {}  # (label, name) -> node_id
        node_counter = 0
        
        for rel in self.graph_data:
            start_node_data = rel["start_node"]
            end_node_data = rel["end_node"]
            relation = rel["relation"]
            
            # Create unique key for start node
            start_name = start_node_data["properties"].get("name", "")
            if isinstance(start_name, list):
                start_name = ", ".join(str(item) for item in start_name)
            elif not isinstance(start_name, str):
                start_name = str(start_name)
            
            start_key = (start_node_data["label"], start_name)
            if start_key not in node_mapping:
                node_id = f"{start_node_data['label']}_{node_counter}"
                node_mapping[start_key] = node_id
                node_counter += 1
                
                # Add node with all properties
                def _normalize_props(p: dict) -> dict:
                    if not isinstance(p, dict):
                        return {"name": str(p)}
                    props_norm = dict(p)
                    # Standardize chunk id key
                    for k in ["chunk_id", "chunkId", "source_chunk", "source_chunk_id", "chunk"]:
                        if k in props_norm and "chunk id" not in props_norm:
                            props_norm["chunk id"] = props_norm[k]
                            break
                    return props_norm
                
                node_attrs = {
                    "label": start_node_data["label"],
                    "properties": _normalize_props(start_node_data["properties"])
                }
                
                graph.add_node(node_id, **node_attrs)
            
            # Create unique key for end node
            end_name = end_node_data["properties"].get("name", "")
            if isinstance(end_name, list):
                end_name = ", ".join(str(item) for item in end_name)
            elif not isinstance(end_name, str):
                end_name = str(end_name)
            
            end_key = (end_node_data["label"], end_name)
            if end_key not in node_mapping:
                node_id = f"{end_node_data['label']}_{node_counter}"
                node_mapping[end_key] = node_id
                node_counter += 1
                
                # Add node with all properties
                def _normalize_props(p: dict) -> dict:
                    if not isinstance(p, dict):
                        return {"name": str(p)}
                    props_norm = dict(p)
                    # Standardize chunk id key
                    for k in ["chunk_id", "chunkId", "source_chunk", "source_chunk_id", "chunk"]:
                        if k in props_norm and "chunk id" not in props_norm:
                            props_norm["chunk id"] = props_norm[k]
                            break
                    return props_norm
                
                node_attrs = {
                    "label": end_node_data["label"],
                    "properties": _normalize_props(end_node_data["properties"])
                }
                
                graph.add_node(node_id, **node_attrs)
            
            # Add edge
            start_id = node_mapping[start_key]
            end_id = node_mapping[end_key]
            graph.add_edge(start_id, end_id, relation=relation)
        
        return graph
    
    def _build_relation_index(self) -> Dict[str, List[Dict]]:
        """构建关系索引"""
        relations = {}
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation')
            if relation:
                if u not in relations:
                    relations[u] = []
                relations[u].append({
                    'target_id': v,
                    'relation': relation,
                    'target_node': self.nodes[v]
                })
                
                if v not in relations:
                    relations[v] = []
                relations[v].append({
                    'target_id': u,
                    'relation': relation,
                    'target_node': self.nodes[u]
                })
        
        return relations
    
    def extract_location_keywords(self, question: str) -> Dict[str, List[str]]:
        """从问题中提取位置关键词"""
        keywords = {
            'buildings': [],
            'floors': [],
            'rooms': []
        }
        
        # 建筑关键词
        building_patterns = [
            r'([AB]栋)',
            r'(A栋|B栋)',
            r'(公共区域)'
        ]
        
        for pattern in building_patterns:
            matches = re.findall(pattern, question)
            keywords['buildings'].extend(matches)
        
        # 楼层关键词
        floor_patterns = [
            r'(\d+层)',
            r'(地下一层)',
            r'(B1层)',
            r'(一层|二层|三层|四层|五层|六层|七层|八层|九层)'
        ]
        
        for pattern in floor_patterns:
            matches = re.findall(pattern, question)
            keywords['floors'].extend(matches)
        
        # 房间关键词
        room_patterns = [
            r'(机房)',
            r'(机电房)',
            r'(办公区)',
            r'(敞开办公区)',
            r'(会议室)',
            r'(电梯厅)',
            r'(走廊)',
            r'(卫生间)'
        ]
        
        for pattern in room_patterns:
            matches = re.findall(pattern, question)
            keywords['rooms'].extend(matches)
        
        return keywords
    
    def build_anchor_nodes(self, keywords: Dict[str, List[str]]) -> List[str]:
        """构建锚点节点列表"""
        anchor_nodes = []
        
        # 构建楼层级别空间节点name模式
        for building in keywords['buildings']:
            for floor in keywords['floors']:
                # 楼层节点: {建筑}栋{楼层}
                floor_name = f"{building}{floor}"
                anchor_nodes.append(floor_name)
        
        # 构建房间级别空间节点name模式
        for building in keywords['buildings']:
            for floor in keywords['floors']:
                # 房间节点: LOC-{建筑}-{楼层}-*
                if 'B' in building and '地下' in floor:
                    room_pattern = "LOC-B-B1-*"
                    anchor_nodes.append(room_pattern)
                elif 'A' in building and '地下' in floor:
                    room_pattern = "LOC-A-B1-*"
                    anchor_nodes.append(room_pattern)
        
        return anchor_nodes
    
    def find_matching_space_nodes(self, anchor_nodes: List[str]) -> List[Dict]:
        """查找匹配的空间节点"""
        matching_nodes = []
        
        print(f"调试: 总节点数: {len(self.nodes)}")
        print(f"调试: 锚点节点: {anchor_nodes}")
        
        for node_id, node in self.nodes.items():
            # 获取节点属性
            properties = node.get('properties', {})
            node_name = properties.get('name', '')
            schema_type = properties.get('schema_type', '')
            label = node.get('label', '')
            
            # 只处理空间节点
            if schema_type != 'location':
                continue
            
            print(f"调试: 找到空间节点: {node_name} (schema_type: {schema_type}, label: {label})")
            
            # 检查是否匹配锚点
            for anchor in anchor_nodes:
                if anchor.endswith('*'):
                    # 模式匹配
                    pattern = anchor.replace('*', '.*')
                    if re.match(pattern, node_name):
                        print(f"调试: 模式匹配成功: {node_name} 匹配 {anchor}")
                        matching_nodes.append({
                            'node_id': node_id,
                            'node': node,
                            'name': node_name
                        })
                else:
                    # 精确匹配
                    if node_name == anchor:
                        print(f"调试: 精确匹配成功: {node_name} 匹配 {anchor}")
                        matching_nodes.append({
                            'node_id': node_id,
                            'node': node,
                            'name': node_name
                        })
        
        # 去重
        seen = set()
        unique_nodes = []
        for node in matching_nodes:
            if node['node_id'] not in seen:
                seen.add(node['node_id'])
                unique_nodes.append(node)
        
        print(f"调试: 最终匹配的空间节点数: {len(unique_nodes)}")
        return unique_nodes
    
    def traverse_graph_from_anchors(self, space_nodes: List[Dict]) -> List[Tuple[str, str, str, float]]:
        """从锚点开始图谱遍历"""
        all_triples = []
        visited_nodes = set()
        
        for space_node in space_nodes:
            node_id = space_node['node_id']
            node_name = space_node['name']
            
            # 从空间节点开始遍历
            triples = self._traverse_from_node(node_id, node_name, visited_nodes)
            all_triples.extend(triples)
        
        return all_triples
    
    def _traverse_from_node(self, node_id: str, node_name: str, visited_nodes: Set[str], depth: int = 0, max_depth: int = 2) -> List[Tuple[str, str, str, float]]:
        """从指定节点开始遍历"""
        if node_id in visited_nodes or depth > max_depth:
            return []
        
        visited_nodes.add(node_id)
        triples = []
        
        # 获取节点的所有关系
        relations = self.relations.get(node_id, [])
        
        for rel in relations:
            target_id = rel['target_id']
            relation = rel['relation']
            target_node = rel['target_node']
            
            # 正确获取目标节点名称和schema_type
            target_properties = target_node.get('properties', {})
            target_name = target_properties.get('name', '')
            target_schema = target_properties.get('schema_type', '')
            
            # 只保留重要的关系类型 - 重点关注空间和设备关系
            important_relations = ['located_in', 'part_of', 'installed_in']
            if relation not in important_relations:
                continue
            
            # 避免重复的双向关系
            if relation == 'located_in' and target_schema == 'location':
                continue  # 跳过从设备到空间的located_in关系
            
            # 构建三元组
            triple = (node_name, relation, target_name)
            
            # 计算相关性分数
            score = self._calculate_relevance_score(node_name, target_name, relation, target_schema)
            scored_triple = (node_name, relation, target_name, score)
            triples.append(scored_triple)
            
            # 如果是空间节点，继续遍历
            if target_schema == 'location' and depth < max_depth:
                sub_triples = self._traverse_from_node(target_id, target_name, visited_nodes, depth + 1, max_depth)
                triples.extend(sub_triples)
            
            # 如果是设备节点，只遍历一层
            elif target_schema == 'asset' and depth < 1:
                sub_triples = self._traverse_from_node(target_id, target_name, visited_nodes, depth + 1, max_depth)
                triples.extend(sub_triples)
        
        return triples
    
    def _calculate_relevance_score(self, source: str, target: str, relation: str, target_schema: str) -> float:
        """计算相关性分数"""
        score = 0.0
        
        # 基础分数
        if relation in ['part_of', 'located_in']:
            score += 0.8
        elif relation in ['has_attribute']:
            score += 0.3
        else:
            score += 0.5
        
        # 设备节点加分
        if target_schema == 'asset':
            score += 0.2
        
        # 空间节点加分
        if target_schema == 'location':
            score += 0.1
        
        return min(score, 1.0)
    
    def extract_chunk_ids_from_triples(self, triples: List[Tuple[str, str, str, float]]) -> Set[str]:
        """从三元组中提取chunk_id"""
        chunk_ids = set()
        
        for triple in triples:
            source, relation, target, score = triple
            
            # 查找source节点的chunk_id
            for node_id, node in self.nodes.items():
                properties = node.get('properties', {})
                node_name = properties.get('name', '')
                
                if node_name == source:
                    chunk_id = properties.get('chunk id')
                    if chunk_id:
                        chunk_ids.add(chunk_id)
                
                if node_name == target:
                    chunk_id = properties.get('chunk id')
                    if chunk_id:
                        chunk_ids.add(chunk_id)
        
        return chunk_ids
    
    def execute_graph_first_strategy(self, question: str) -> Dict:
        """执行Graph-First策略"""
        # 1. 提取位置关键词
        keywords = self.extract_location_keywords(question)
        print(f"提取的关键词: {keywords}")
        
        # 2. 构建锚点节点
        anchor_nodes = self.build_anchor_nodes(keywords)
        print(f"构建的锚点节点: {anchor_nodes}")
        
        # 3. 查找匹配的空间节点
        space_nodes = self.find_matching_space_nodes(anchor_nodes)
        print(f"匹配的空间节点: {[node['name'] for node in space_nodes]}")
        
        # 4. 从锚点开始图谱遍历
        triples = self.traverse_graph_from_anchors(space_nodes)
        print(f"找到的三元组数量: {len(triples)}")
        
        # 5. 提取chunk_id
        chunk_ids = self.extract_chunk_ids_from_triples(triples)
        print(f"提取的chunk_id: {list(chunk_ids)}")
        
        # 6. 按分数排序三元组
        triples.sort(key=lambda x: x[3], reverse=True)
        
        # 7. 查看边的属性
        self._analyze_edge_properties(space_nodes)
        
        return {
            'keywords': keywords,
            'anchor_nodes': anchor_nodes,
            'space_nodes': space_nodes,
            'triples': triples,
            'chunk_ids': list(chunk_ids)
        }
    
    def _analyze_edge_properties(self, space_nodes: List[Dict]):
        """分析边的属性"""
        print("\n=== 边的属性分析 ===")
        
        for space_node in space_nodes:
            node_id = space_node['node_id']
            node_name = space_node['name']
            
            print(f"\n从节点 '{node_name}' 出发的边:")
            
            # 获取该节点的所有出边
            for u, v, data in self.graph.edges(node_id, data=True):
                relation = data.get('relation', '')
                target_node = self.nodes[v]
                target_properties = target_node.get('properties', {})
                target_name = target_properties.get('name', '')
                target_schema = target_properties.get('schema_type', '')
                
                print(f"  {node_name} --[{relation}]--> {target_name} (schema: {target_schema})")
                
                # 如果是设备节点，显示更多信息
                if target_schema == 'asset':
                    chunk_id = target_properties.get('chunk id', '')
                    print(f"    设备chunk_id: {chunk_id}")
            
            # 获取该节点的所有入边
            for u, v, data in self.graph.in_edges(node_id, data=True):
                relation = data.get('relation', '')
                source_node = self.nodes[u]
                source_properties = source_node.get('properties', {})
                source_name = source_properties.get('name', '')
                source_schema = source_properties.get('schema_type', '')
                
                print(f"  {source_name} --[{relation}]--> {node_name} (schema: {source_schema})")


def test_redesigned_graph_first():
    """测试重新设计的Graph-First策略"""
    # 加载图数据
    with open('output/graphs/building_assets_new.json', 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 创建策略实例
    strategy = RedesignedGraphFirstStrategy(graph_data)
    
    # 测试问题
    question = "B栋地下一层有哪些设备？"
    
    # 执行Graph-First策略
    result = strategy.execute_graph_first_strategy(question)
    
    print("\n=== Graph-First策略执行结果 ===")
    print(f"问题: {question}")
    print(f"关键词: {result['keywords']}")
    print(f"锚点节点: {result['anchor_nodes']}")
    print(f"匹配的空间节点: {[node['name'] for node in result['space_nodes']]}")
    print(f"找到的三元组数量: {len(result['triples'])}")
    print(f"提取的chunk_id: {result['chunk_ids']}")
    
    print("\n=== 前20个三元组 ===")
    for i, triple in enumerate(result['triples'][:20]):
        print(f"{i+1}. {triple[0]} --[{triple[1]}]--> {triple[2]} (分数: {triple[3]:.3f})")
    
    print(f"\n=== 所有三元组统计 ===")
    print(f"总三元组数: {len(result['triples'])}")
    
    # 统计关系类型
    relation_counts = {}
    for triple in result['triples']:
        relation = triple[1]
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    
    print("关系类型统计:")
    for relation, count in sorted(relation_counts.items()):
        print(f"  {relation}: {count}个")
    
    return result


if __name__ == "__main__":
    test_redesigned_graph_first()
