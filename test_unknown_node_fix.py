#!/usr/bin/env python3
"""
测试"Unknown Node"问题修复效果
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

def test_unknown_node_fix():
    """测试Unknown Node问题修复"""
    
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
    
    print("=== 测试Unknown Node问题修复 ===")
    
    # 测试查询
    test_questions = [
        "A栋地下一层有哪些设备",
        "A栋B1有哪些设备", 
        "A栋3层有哪些设备",
        "What devices are in Building A Floor 3"
    ]
    
    for question in test_questions:
        print(f"\n测试查询: {question}")
        print("-" * 50)
        
        try:
            # 调用检索方法
            question_embed, results = retriever.retrieve(question)
            
            # 检查结果
            path1_results = results.get('path1_results', {})
            triples = path1_results.get('one_hop_triples', [])
            chunk_ids = results.get('chunk_ids', [])
            
            print(f"找到三元组数量: {len(triples)}")
            print(f"找到chunk数量: {len(chunk_ids)}")
            
            # 检查是否有"Unknown Node"
            unknown_count = 0
            for triple in triples[:5]:  # 只检查前5个
                if "Unknown Node" in str(triple):
                    unknown_count += 1
                    print(f"❌ 发现Unknown Node: {triple}")
                else:
                    print(f"✅ 正常三元组: {triple}")
            
            if unknown_count == 0:
                print("🎉 没有发现Unknown Node问题！")
            else:
                print(f"⚠️ 发现 {unknown_count} 个Unknown Node问题")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_unknown_node_fix()
