"""
调试Graph-First策略找到的三元组
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retriever.enhanced_kt_retriever import KTRetriever
from utils.logger import logger

def debug_graph_first_triples():
    """调试Graph-First策略找到的三元组"""
    
    # 初始化检索器
    logger.info("初始化KTRetriever...")
    retriever = KTRetriever(
        dataset="building_assets",
        json_path="output/graphs/building_assets_new.json",
        device="cpu",
        top_k=5,
        recall_paths=2
    )
    
    # 测试问题
    question = "B栋地下一层有哪些设备？"
    logger.info(f"测试问题: {question}")
    
    # 直接调用Graph-First策略
    logger.info("直接调用Graph-First策略...")
    try:
        # 获取问题嵌入
        question_embed = retriever._get_query_embedding(question)
        
        # 调用Graph-First策略
        triples = retriever._path_strategy(question, question_embed)
        
        logger.info(f"找到的三元组数量: {len(triples)}")
        logger.info("=== 所有三元组 ===")
        for i, triple in enumerate(triples):
            logger.info(f"{i+1}. {triple}")
        
        # 检查是否包含冷冻水循环泵-02
        logger.info("\n=== 检查冷冻水循环泵-02 ===")
        found_pump = False
        for i, triple in enumerate(triples):
            source, relation, target, score = triple
            if '冷冻水循环泵-02' in source or '冷冻水循环泵-02' in target:
                found_pump = True
                logger.info(f"✅ 找到冷冻水循环泵-02: {triple}")
                break
        
        if not found_pump:
            logger.warning("❌ 未找到冷冻水循环泵-02")
            
            # 检查图谱中是否存在冷冻水循环泵-02
            logger.info("检查图谱中是否存在冷冻水循环泵-02...")
            graph_data = retriever.graph_data
            pump_found = False
            for edge in graph_data:
                start_node = edge.get('start_node', {})
                end_node = edge.get('end_node', {})
                
                if '冷冻水循环泵-02' in start_node.get('name', ''):
                    logger.info(f"在start_node中找到: {start_node}")
                    pump_found = True
                if '冷冻水循环泵-02' in end_node.get('name', ''):
                    logger.info(f"在end_node中找到: {end_node}")
                    pump_found = True
            
            if not pump_found:
                logger.warning("图谱中也没有找到冷冻水循环泵-02")
            else:
                logger.info("图谱中存在冷冻水循环泵-02，但Graph-First策略没有找到")
        
        return triples
        
    except Exception as e:
        logger.error(f"调试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_graph_first_triples()
