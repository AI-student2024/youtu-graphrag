"""
调试冷冻水循环泵-02的节点ID
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retriever.enhanced_kt_retriever import KTRetriever
from utils.logger import logger

def debug_pump_node_id():
    """调试冷冻水循环泵-02的节点ID"""
    
    # 初始化检索器
    logger.info("初始化KTRetriever...")
    retriever = KTRetriever(
        dataset="building_assets",
        json_path="output/graphs/building_assets_new.json",
        device="cpu",
        top_k=5,
        recall_paths=2
    )
    
    # 查找冷冻水循环泵-02的节点ID
    logger.info("查找冷冻水循环泵-02的节点ID...")
    pump_node_id = None
    for node_id, node_data in retriever.graph.nodes(data=True):
        properties = node_data.get('properties', {})
        node_name = properties.get('name', '')
        if '冷冻水循环泵-02' in node_name:
            pump_node_id = node_id
            logger.info(f"找到冷冻水循环泵-02节点ID: {node_id}")
            logger.info(f"节点数据: {node_data}")
            break
    
    if not pump_node_id:
        logger.warning("未找到冷冻水循环泵-02的节点ID")
        return
    
    # 查找LOC-B-B1-MECH的节点ID
    logger.info("查找LOC-B-B1-MECH的节点ID...")
    loc_node_id = None
    for node_id, node_data in retriever.graph.nodes(data=True):
        properties = node_data.get('properties', {})
        node_name = properties.get('name', '')
        if 'LOC-B-B1-MECH' in node_name:
            loc_node_id = node_id
            logger.info(f"找到LOC-B-B1-MECH节点ID: {node_id}")
            logger.info(f"节点数据: {node_data}")
            break
    
    if not loc_node_id:
        logger.warning("未找到LOC-B-B1-MECH的节点ID")
        return
    
    # 检查冷冻水循环泵-02到LOC-B-B1-MECH的边
    logger.info("检查冷冻水循环泵-02到LOC-B-B1-MECH的边...")
    for u, v, data in retriever.graph.edges(data=True):
        if u == pump_node_id and v == loc_node_id:
            logger.info(f"找到边: {u} --[{data.get('relation')}]--> {v}")
            logger.info(f"边数据: {data}")
            break
    else:
        logger.warning("未找到冷冻水循环泵-02到LOC-B-B1-MECH的边")
    
    # 检查LOC-B-B1-MECH的入边
    logger.info("检查LOC-B-B1-MECH的入边...")
    in_edges = list(retriever.graph.in_edges(loc_node_id, data=True))
    logger.info(f"LOC-B-B1-MECH的入边数量: {len(in_edges)}")
    
    for u, v, data in in_edges:
        u_text = retriever._get_node_text(u)
        logger.info(f"入边: {u} --[{data.get('relation')}]--> {v}, 源节点文本: {u_text}")
        if '冷冻水循环泵-02' in u_text:
            logger.info("✅ 找到冷冻水循环泵-02的入边!")
    
    # 检查设备过滤条件
    logger.info("检查设备过滤条件...")
    equipment_filters = ['设备', '冷机', '水泵', '泵', '配电', '空调', '机组', '柜', '箱', '末端']
    pump_text = retriever._get_node_text(pump_node_id)
    logger.info(f"冷冻水循环泵-02的节点文本: {pump_text}")
    
    for keyword in equipment_filters:
        if keyword in pump_text:
            logger.info(f"✅ 匹配关键词: {keyword}")
        else:
            logger.info(f"❌ 不匹配关键词: {keyword}")

if __name__ == "__main__":
    debug_pump_node_id()
