#!/usr/bin/env python3
"""
调试图谱节点，检查是否有A栋相关的节点
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.retriever.enhanced_kt_retriever import KTRetriever
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_graph_nodes():
    """调试图谱节点"""
    
    # 初始化检索器
    config = get_config("config/base_config.yaml")
    
    retriever = KTRetriever(
        dataset="demo",
        json_path="output/graphs/demo_new.json",
        recall_paths=config.retrieval.recall_paths,
        schema_path="schemas/demo.json",
        top_k=config.retrieval.top_k_filter,
        mode="agent",
        config=config
    )
    
    print("=== 调试图谱节点 ===")
    
    # 查找A栋相关的节点
    a_building_nodes = []
    for node_id, node_data in retriever.graph.nodes(data=True):
        props = node_data.get('properties', {})
        name = props.get('name', '')
        if 'A栋' in name or 'A' in name:
            a_building_nodes.append((node_id, name, props.get('schema_type', '')))
    
    print(f"找到 {len(a_building_nodes)} 个A栋相关节点:")
    for node_id, name, schema_type in a_building_nodes:
        print(f"  {node_id}: {name} ({schema_type})")
    
    # 查找楼层相关的节点
    floor_nodes = []
    for node_id, node_data in retriever.graph.nodes(data=True):
        props = node_data.get('properties', {})
        name = props.get('name', '')
        schema_type = props.get('schema_type', '')
        if '层' in name or 'floor' in name.lower() or schema_type == 'floor':
            floor_nodes.append((node_id, name, schema_type))
    
    print(f"\n找到 {len(floor_nodes)} 个楼层相关节点:")
    for node_id, name, schema_type in floor_nodes:
        print(f"  {node_id}: {name} ({schema_type})")
    
    # 查找设备相关的节点
    asset_nodes = []
    for node_id, node_data in retriever.graph.nodes(data=True):
        props = node_data.get('properties', {})
        name = props.get('name', '')
        schema_type = props.get('schema_type', '')
        if schema_type == 'asset' or '设备' in name or '设备' in name:
            asset_nodes.append((node_id, name, schema_type))
    
    print(f"\n找到 {len(asset_nodes)} 个设备相关节点:")
    for node_id, name, schema_type in asset_nodes[:10]:  # 只显示前10个
        print(f"  {node_id}: {name} ({schema_type})")
    
    # 测试_path_strategy方法
    print(f"\n=== 测试_path_strategy方法 ===")
    question = "A栋地下一层有哪些设备"
    try:
        triples = retriever._path_strategy(question, None)
        print(f"找到三元组数量: {len(triples)}")
        for i, triple in enumerate(triples[:5]):
            print(f"  三元组 {i+1}: {triple}")
    except Exception as e:
        print(f"❌ _path_strategy失败: {e}")

if __name__ == "__main__":
    debug_graph_nodes()
