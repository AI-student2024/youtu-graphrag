#!/usr/bin/env python3
"""
修复Graph-First策略，实现真正的基于锚点的图谱遍历
"""

def create_improved_graph_first_strategy():
    """创建改进的Graph-First策略"""
    
    strategy_code = '''
    def _improved_path_strategy(self, question: str, question_embed: torch.Tensor = None) -> List[Tuple[str, str, str, float]]:
        """改进的Graph-First策略：基于锚点的图谱遍历"""
        try:
            # 1. 从问题中提取位置关键词
            location_keywords = self._extract_location_keywords_from_question(question)
            logger.info(f"[GraphFirst] 提取的位置关键词: {location_keywords}")
            
            # 2. 构建锚点节点列表
            anchor_nodes = self._build_anchor_nodes_from_keywords(location_keywords)
            logger.info(f"[GraphFirst] 锚点节点: {anchor_nodes}")
            
            if not anchor_nodes:
                logger.warning("[GraphFirst] 未找到锚点节点")
                return []
            
            # 3. 从锚点节点进行图谱遍历
            triples = self._traverse_graph_from_anchors(anchor_nodes)
            logger.info(f"[GraphFirst] 找到 {len(triples)} 个三元组")
            
            # 4. 按相关性排序
            triples = self._rank_triples_by_relevance(triples, question_embed)
            
            return triples[:self.top_k]
            
        except Exception as e:
            logger.error(f"Improved path strategy failed: {e}")
            return []
    
    def _extract_location_keywords_from_question(self, question: str) -> List[str]:
        """从问题中提取位置关键词"""
        import re
        
        keywords = []
        
        # 提取建筑关键词
        building_matches = re.findall(r'([A-Za-z\u4e00-\u9fff]+)栋', question)
        keywords.extend(building_matches)
        
        # 提取楼层关键词
        floor_matches = re.findall(r'(\d+[层F楼]|[一二三四五六七八九十]+层|地下[一二三四五六七八九十]*层?|B\d+)', question)
        keywords.extend(floor_matches)
        
        # 提取房间关键词
        room_matches = re.findall(r'([^，。！？\s]+(?:房|室|厅|井|机房))', question)
        keywords.extend(room_matches)
        
        return list(set(keywords))
    
    def _build_anchor_nodes_from_keywords(self, keywords: List[str]) -> List[str]:
        """根据关键词构建锚点节点列表"""
        anchor_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_text = self._get_node_text(node_id)
            if not node_text or node_text.startswith('[Error'):
                continue
                
            # 检查节点文本是否包含任何关键词
            for keyword in keywords:
                if keyword in node_text:
                    anchor_nodes.append(node_id)
                    break
        
        return anchor_nodes
    
    def _traverse_graph_from_anchors(self, anchor_nodes: List[str]) -> List[Tuple[str, str, str, float]]:
        """从锚点节点进行图谱遍历"""
        triples = []
        visited_edges = set()
        
        for anchor in anchor_nodes:
            # 反向检索：查找指向锚点的节点（设备 -> 位置）
            for u, v, data in self.graph.in_edges(anchor, data=True):
                edge_key = (u, v, data.get('relation', ''))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    if data.get('relation') == 'located_in':
                        triples.append((u, 'located_in', v, 0.95))
            
            # 正向检索：查找锚点指向的节点（位置 -> 子空间）
            for u, v, data in self.graph.out_edges(anchor, data=True):
                edge_key = (u, v, data.get('relation', ''))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    if data.get('relation') == 'part_of':
                        # 继续查找子空间内的设备
                        for a, b, d2 in self.graph.in_edges(v, data=True):
                            edge_key2 = (a, b, d2.get('relation', ''))
                            if edge_key2 not in visited_edges:
                                visited_edges.add(edge_key2)
                                if d2.get('relation') == 'located_in':
                                    triples.append((a, 'located_in', b, 0.90))
        
        return triples
    '''
    
    return strategy_code

if __name__ == "__main__":
    print("改进的Graph-First策略代码已生成")
    print(create_improved_graph_first_strategy())
