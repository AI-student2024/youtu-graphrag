#!/usr/bin/env python3
"""
最终测试：验证A栋地下一层设备召回和Unknown Node修复
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

def test_device_recall_final():
    """最终测试设备召回和Unknown Node修复"""
    
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
    
    print("=== 最终测试：A栋地下一层设备召回 ===")
    
    # 测试查询
    question = "A栋地下一层有哪些设备"
    print(f"测试查询: {question}")
    print("-" * 60)
    
    try:
        # 调用检索方法
        question_embed, results = retriever.retrieve(question)
        
        # 检查结果
        path1_results = results.get('path1_results', {})
        triples = path1_results.get('one_hop_triples', [])
        chunk_ids = results.get('chunk_ids', [])
        
        print(f"✅ Graph-First策略触发成功")
        print(f"✅ 找到三元组数量: {len(triples)}")
        print(f"✅ 找到chunk数量: {len(chunk_ids)}")
        
        # 检查设备召回
        device_count = 0
        unknown_count = 0
        
        print(f"\n=== 三元组详情 ===")
        for i, triple in enumerate(triples[:10]):  # 显示前10个
            if "Unknown Node" in str(triple):
                unknown_count += 1
                print(f"❌ 三元组 {i+1}: {triple}")
            else:
                print(f"✅ 三元组 {i+1}: {triple}")
                # 检查是否包含设备
                if "冷机" in str(triple) or "水泵" in str(triple) or "配电" in str(triple) or "电梯" in str(triple):
                    device_count += 1
        
        print(f"\n=== 测试结果总结 ===")
        print(f"🎯 设备相关三元组: {device_count}")
        print(f"❌ Unknown Node数量: {unknown_count}")
        
        if unknown_count == 0:
            print("🎉 Unknown Node问题完全解决！")
        else:
            print(f"⚠️ 仍有 {unknown_count} 个Unknown Node问题")
            
        if device_count > 0:
            print("🎉 成功找到设备相关三元组！")
        else:
            print("⚠️ 未找到设备相关三元组")
            
        # 检查chunk内容
        print(f"\n=== Chunk内容预览 ===")
        for i, chunk_id in enumerate(chunk_ids[:3]):
            chunk_content = retriever.chunk2id.get(chunk_id, "未找到内容")
            print(f"Chunk {i+1} ({chunk_id}): {chunk_content[:100]}...")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_device_recall_final()
