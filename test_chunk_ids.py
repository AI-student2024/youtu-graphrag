#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retriever.enhanced_kt_retriever import KTRetriever
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chunk_ids():
    """测试chunk_id的具体情况"""
    logger.info("开始测试chunk_id情况...")
    
    # 初始化检索器
    retriever = KTRetriever("building_assets")
    
    # 测试问题
    question = "A栋3层有哪些设备？"
    logger.info(f"测试问题: {question}")
    
    # 执行检索
    logger.info("开始执行检索...")
    retriever_result = retriever.retrieve(question)
    
    # 处理可能的元组返回 (question_embed, results)
    if isinstance(retriever_result, tuple):
        question_embed, results = retriever_result
        logger.info("检测到元组返回格式 (question_embed, results)")
    else:
        results = retriever_result
        question_embed = None
    
    # 检查Graph-First策略的结果
    path1_triples = results['path1_results'].get('one_hop_triples', [])
    logger.info(f"path1_triples数量: {len(path1_triples)}")
    
    if path1_triples:
        logger.info("Graph-First策略被触发")
        logger.info("path1_triples内容:")
        for i, triple in enumerate(path1_triples):
            h, r, t, score = triple
            h_text = retriever._get_node_text(h)
            t_text = retriever._get_node_text(t)
            logger.info(f"  {i+1}. ({h_text}, {r}, {t_text}, {score})")
            
            # 获取chunk_id
            h_chunk_id = retriever._get_node_chunk_id(h)
            t_chunk_id = retriever._get_node_chunk_id(t)
            logger.info(f"      Head chunk_id: {h_chunk_id}")
            logger.info(f"      Tail chunk_id: {t_chunk_id}")
    
    # 检查chunk_results
    chunk_results = results['path1_results'].get('chunk_results', [])
    logger.info(f"chunk_results数量: {len(chunk_results)}")
    
    if chunk_results:
        logger.info("chunk_results内容:")
        for i, chunk_result in enumerate(chunk_results):
            logger.info(f"  {i+1}. {chunk_result}")
    
    # 测试process_retrieval_results
    logger.info("测试process_retrieval_results方法...")
    try:
        processed_results = retriever.process_retrieval_results(question)
        final_triples = processed_results[0].get('triples', [])
        final_chunk_ids = processed_results[0].get('chunk_ids', [])
        final_chunk_contents = processed_results[0].get('chunk_contents', {})
        
        logger.info(f"最终三元组数量: {len(final_triples)}")
        logger.info(f"最终chunk_ids数量: {len(final_chunk_ids)}")
        logger.info(f"最终chunk_contents数量: {len(final_chunk_contents)}")
        
        logger.info("最终chunk_ids列表:")
        for i, chunk_id in enumerate(final_chunk_ids):
            logger.info(f"  {i+1}. {chunk_id}")
        
        logger.info("最终chunk_contents:")
        for chunk_id, content in final_chunk_contents.items():
            logger.info(f"  {chunk_id}: {content[:100]}...")
        
    except Exception as e:
        logger.error(f"处理检索结果失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chunk_ids()
