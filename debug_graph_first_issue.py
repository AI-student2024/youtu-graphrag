#!/usr/bin/env python3
"""
调试Graph-First策略问题
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

def debug_graph_first_issue():
    """调试Graph-First策略问题"""
    
    # 初始化检索器
    config = get_config("config/base_config.yaml")
    
    retriever = KTRetriever(
        dataset="building_assets",
        json_path="output/graphs/building_assets_new.json",
        recall_paths=config.retrieval.recall_paths,
        schema_path="schemas/demo.json",
        top_k=config.retrieval.top_k_filter,
        mode="agent",
        config=config
    )
    
    print("=== 调试Graph-First策略问题 ===")
    
    # 测试查询
    question = "B栋地下一层有哪些设备"
    print(f"测试查询: {question}")
    print("-" * 60)
    
    try:
        # 直接调用_path_strategy方法
        print("1. 直接调用_path_strategy方法:")
        path_triples = retriever._path_strategy(question)
        print(f"   找到三元组数量: {len(path_triples)}")
        
        if path_triples:
            print("   前5个三元组:")
            for i, triple in enumerate(path_triples[:5]):
                h, r, t, score = triple
                print(f"   {i+1}. ({h}, {r}, {t}) [score: {score}]")
                
                # 检查节点是否存在于图中
                h_exists = h in retriever.graph.nodes
                t_exists = t in retriever.graph.nodes
                print(f"      h节点存在: {h_exists}, t节点存在: {t_exists}")
                
                if h_exists:
                    h_data = retriever.graph.nodes[h]
                    h_name = h_data.get('properties', {}).get('name', 'No name')
                    print(f"      h节点名称: {h_name}")
                
                if t_exists:
                    t_data = retriever.graph.nodes[t]
                    t_name = t_data.get('properties', {}).get('name', 'No name')
                    print(f"      t节点名称: {t_name}")
                print()
        
        # 测试_format_scored_triples方法
        print("2. 测试_format_scored_triples方法:")
        if path_triples:
            formatted_triples = retriever._format_scored_triples(path_triples[:3])
            print(f"   格式化三元组数量: {len(formatted_triples)}")
            for i, triple_text in enumerate(formatted_triples):
                print(f"   {i+1}. {triple_text}")
        
        # 测试完整的retrieve方法
        print("3. 测试完整的retrieve方法:")
        question_embed, results = retriever.retrieve(question)
        
        path1_results = results.get('path1_results', {})
        triples = path1_results.get('one_hop_triples', [])
        
        print(f"   完整检索找到三元组数量: {len(triples)}")
        if triples:
            print("   前3个三元组:")
            for i, triple_text in enumerate(triples[:3]):
                print(f"   {i+1}. {triple_text}")
                
        # 检查是否有Unknown Node
        unknown_count = sum(1 for triple in triples if "Unknown Node" in str(triple))
        print(f"   Unknown Node数量: {unknown_count}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    debug_graph_first_issue()
