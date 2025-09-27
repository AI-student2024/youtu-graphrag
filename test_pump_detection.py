#!/usr/bin/env python3
"""
测试冷冻水循环泵-02是否能被Graph-First策略找到
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置详细日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_pump_detection():
    """测试冷冻水循环泵-02检测"""
    print("🔧 测试冷冻水循环泵-02检测...")
    
    try:
        from config import get_config
        from models.retriever.enhanced_kt_retriever import KTRetriever
        
        # 加载配置
        config = get_config()
        print("✅ 配置加载成功")
        
        # 创建检索器
        retriever = KTRetriever(
            dataset="building_assets",
            config=config
        )
        print("✅ 检索器创建成功")
        
        # 测试查询
        question = "B栋地下一层有哪些设备？"
        print(f"📝 测试查询: {question}")
        
        # 执行完整的检索流程
        print("\n🔍 执行完整检索流程...")
        retrieval_results, retrieval_time = retriever.process_retrieval_results(question, top_k=20)
        
        print(f"⏱️ 检索时间: {retrieval_time:.4f}s")
        
        # 检查三元组
        triples = retrieval_results.get('triples', [])
        print(f"\n🔗 三元组数量: {len(triples)}")
        
        # 查找冷冻水循环泵相关的三元组
        pump_triples = [triple for triple in triples if '冷冻水循环泵' in str(triple)]
        print(f"🔧 冷冻水循环泵相关三元组数量: {len(pump_triples)}")
        
        if pump_triples:
            print("📋 冷冻水循环泵相关三元组:")
            for i, triple in enumerate(pump_triples):
                print(f"  {i+1}. {triple}")
        else:
            print("❌ 没有找到冷冻水循环泵相关三元组")
        
        # 检查chunk内容
        chunk_contents = retrieval_results.get('chunk_contents', [])
        print(f"\n📄 Chunk数量: {len(chunk_contents)}")
        
        # 查找冷冻水循环泵相关的chunk
        pump_chunks = [chunk for chunk in chunk_contents if '冷冻水循环泵' in str(chunk)]
        print(f"🔧 冷冻水循环泵相关Chunk数量: {len(pump_chunks)}")
        
        if pump_chunks:
            print("📋 冷冻水循环泵相关Chunk:")
            for i, chunk in enumerate(pump_chunks):
                print(f"  {i+1}. {chunk[:200]}...")
        else:
            print("❌ 没有找到冷冻水循环泵相关Chunk")
        
        # 检查是否有任何泵相关的设备
        pump_related = [triple for triple in triples if '泵' in str(triple)]
        print(f"\n🔧 所有泵相关三元组数量: {len(pump_related)}")
        
        if pump_related:
            print("📋 所有泵相关三元组:")
            for i, triple in enumerate(pump_related):
                print(f"  {i+1}. {triple}")
        
        print("\n🎉 测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pump_detection()
