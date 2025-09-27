"""
调试LOC-B-B1-MECH的entity节点
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retriever.enhanced_kt_retriever import KTRetriever
from utils.logger import logger

def debug_loc_entity_node():
    """调试LOC-B-B1-MECH的entity节点"""
    
    # 初始化检索器
    logger.info("初始化KTRetriever...")
    retriever = KTRetriever(
        dataset="building_assets",
        json_path="output/graphs/building_assets_new.json",
        device="cpu",
        top_k=5,
        recall_paths=2
    )
    
    # 查找所有包含LOC-B-B1-MECH的节点
    logger.info("查找所有包含LOC-B-B1-MECH的节点...")
    loc_nodes = []
    for node_id, node_data in retriever.graph.nodes(data=True):
        properties = node_data.get('properties', {})
        node_name = properties.get('name', '')
        if 'LOC-B-B1-MECH' in node_name:
            loc_nodes.append((node_id, node_data))
            logger.info(f"找到节点: {node_id} -> {node_name}")
            logger.info(f"节点数据: {node_data}")
    
    # 查找冷冻水循环泵-02的located_in关系
    logger.info("查找冷冻水循环泵-02的located_in关系...")
    pump_node_id = 'entity_138'  # 从之前的调试中得知
    
    for u, v, data in retriever.graph.edges(data=True):
        if u == pump_node_id and data.get('relation') == 'located_in':
            v_text = retriever._get_node_text(v)
            logger.info(f"冷冻水循环泵-02的located_in关系: {u} --[{data.get('relation')}]--> {v}")
            logger.info(f"目标节点文本: {v_text}")
            logger.info(f"目标节点数据: {retriever.graph.nodes[v]}")
    
    # 查找LOC-B-B1-MECH entity节点的入边
    logger.info("查找LOC-B-B1-MECH entity节点的入边...")
    for node_id, node_data in loc_nodes:
        if node_data.get('label') == 'entity':
            logger.info(f"检查entity节点: {node_id}")
            in_edges = list(retriever.graph.in_edges(node_id, data=True))
            logger.info(f"入边数量: {len(in_edges)}")
            
            for u, v, data in in_edges:
                u_text = retriever._get_node_text(u)
                logger.info(f"入边: {u} --[{data.get('relation')}]--> {v}, 源节点文本: {u_text}")
                if '冷冻水循环泵-02' in u_text:
                    logger.info("✅ 找到冷冻水循环泵-02的入边!")

if __name__ == "__main__":
    debug_loc_entity_node()
