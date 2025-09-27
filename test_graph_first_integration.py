"""
测试Graph-First策略在整体检索流程中的集成
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retriever.enhanced_kt_retriever import KTRetriever
from utils.logger import logger

def test_graph_first_integration():
    """测试Graph-First策略集成"""
    
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
    
    # 执行检索
    logger.info("开始执行检索...")
    try:
        retriever_result = retriever.retrieve(question)
        
        # 处理可能的元组返回 (question_embed, results)
        if isinstance(retriever_result, tuple):
            question_embed, results = retriever_result
            logger.info("检测到元组返回格式 (question_embed, results)")
        else:
            results = retriever_result
            question_embed = None
        
        logger.info("=== 检索结果 ===")
        logger.info(f"结果类型: {type(results)}")
        
        if isinstance(results, dict):
            # 检查path1_results
            if 'path1_results' in results:
                path1 = results['path1_results']
                logger.info(f"path1_results: {path1}")
                
                # 检查one_hop_triples
                if 'one_hop_triples' in path1:
                    triples = path1['one_hop_triples']
                    logger.info(f"one_hop_triples数量: {len(triples)}")
                    logger.info(f"one_hop_triples类型: {type(triples)}")
                    
                    if triples:
                        logger.info("前5个三元组:")
                        for i, triple in enumerate(triples[:5]):
                            logger.info(f"  {i+1}. {triple}")
                    else:
                        logger.warning("one_hop_triples为空!")
                
                # 检查chunk_results
                if 'chunk_results' in path1:
                    chunk_results = path1['chunk_results']
                    logger.info(f"chunk_results: {chunk_results}")
                    
                    if 'chunk_ids' in chunk_results:
                        chunk_ids = chunk_results['chunk_ids']
                        logger.info(f"chunk_ids数量: {len(chunk_ids)}")
                        logger.info(f"chunk_ids: {chunk_ids}")
                    
                    if 'chunk_contents' in chunk_results:
                        chunk_contents = chunk_results['chunk_contents']
                        logger.info(f"chunk_contents数量: {len(chunk_contents)}")
                        if chunk_contents:
                            logger.info("前3个chunk内容:")
                            for i, content in enumerate(chunk_contents[:3]):
                                logger.info(f"  {i+1}. {content[:100]}...")
            
            # 检查顶层chunk_ids
            if 'chunk_ids' in results:
                top_chunk_ids = results['chunk_ids']
                logger.info(f"顶层chunk_ids数量: {len(top_chunk_ids)}")
                logger.info(f"顶层chunk_ids: {top_chunk_ids}")
            
            # 检查顶层chunk_contents
            if 'chunk_contents' in results:
                top_chunk_contents = results['chunk_contents']
                logger.info(f"顶层chunk_contents数量: {len(top_chunk_contents)}")
                if top_chunk_contents:
                    logger.info("前3个顶层chunk内容:")
                    for i, content in enumerate(top_chunk_contents[:3]):
                        logger.info(f"  {i+1}. {content[:100]}...")
        
        # 检查是否包含冷冻水循环泵-02
        logger.info("=== 检查冷冻水循环泵-02 ===")
        found_pump = False
        
        # 检查三元组中是否包含冷冻水循环泵-02
        if 'path1_results' in results and 'one_hop_triples' in results['path1_results']:
            triples = results['path1_results']['one_hop_triples']
            logger.info(f"检查 {len(triples)} 个三元组...")
            for i, triple in enumerate(triples):
                if len(triple) >= 3:
                    h, r, t = triple[:3]
                    h_text = retriever._get_node_text(h) if hasattr(retriever, '_get_node_text') else str(h)
                    t_text = retriever._get_node_text(t) if hasattr(retriever, '_get_node_text') else str(t)
                    if '冷冻水循环泵-02' in h_text or '冷冻水循环泵-02' in t_text:
                        found_pump = True
                        logger.info(f"✅ 在三元组中找到冷冻水循环泵-02: {triple}")
                        break
        
        # 也检查chunk内容
        all_chunk_contents = []
        if 'chunk_contents' in results:
            all_chunk_contents.extend(results['chunk_contents'])
        if 'path1_results' in results and 'chunk_results' in results['path1_results']:
            chunk_results = results['path1_results']['chunk_results']
            if 'chunk_contents' in chunk_results:
                all_chunk_contents.extend(chunk_results['chunk_contents'])
        
        for content in all_chunk_contents:
            if '冷冻水循环泵-02' in content:
                found_pump = True
                logger.info("✅ 在chunk内容中找到冷冻水循环泵-02!")
                logger.info(f"相关内容: {content[:200]}...")
                break
        
        if not found_pump:
            logger.warning("❌ 未找到冷冻水循环泵-02")
        
        return results
        
    except Exception as e:
        logger.error(f"检索失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_process_retrieval_results():
    """测试process_retrieval_results方法"""
    logger.info("测试process_retrieval_results方法...")
    
    # 初始化检索器
    retriever = KTRetriever(
        dataset="building_assets",
        json_path="output/graphs/building_assets_new.json",
        device="cpu",
        top_k=5,
        recall_paths=2
    )
    
    # 测试问题
    question = "B栋地下一层有哪些设备？"
    
    # 执行检索
    results = retriever.retrieve(question)
    
    if results:
        logger.info("=== 处理检索结果 ===")
        try:
            # 获取问题嵌入
            question_embed = retriever.qa_encoder.encode([question])
            question_embed = torch.tensor(question_embed, dtype=torch.float32)
            
            # 处理检索结果
            processed_results = retriever.process_retrieval_results(results, question_embed)
            
            logger.info(f"处理后的结果类型: {type(processed_results)}")
            logger.info(f"处理后的结果: {processed_results}")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"处理检索结果失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None

if __name__ == "__main__":
    import torch
    
    logger.info("开始测试Graph-First策略集成...")
    
    # 测试1: 基本检索
    logger.info("\n=== 测试1: 基本检索 ===")
    results = test_graph_first_integration()
    
    # 测试2: 处理检索结果
    logger.info("\n=== 测试2: 处理检索结果 ===")
    processed_results = test_process_retrieval_results()
    
    logger.info("\n=== 测试完成 ===")
