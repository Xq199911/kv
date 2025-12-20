"""
Group-Aware KV Eviction: Group跟踪模块

跟踪语义Group的边界，用于Group-level的KV cache驱逐
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Group:
    """表示一个语义Group"""
    group_id: int
    start_token_idx: int  # 在KV cache中的起始位置
    end_token_idx: int    # 在KV cache中的结束位置
    group_type: str       # 'source' or 'target'
    segment_idx: int      # 在source_txt_lt或target_txt_lt中的索引
    token_count: int      # token数量
    
    def __len__(self):
        return self.end_token_idx - self.start_token_idx


class GroupTracker:
    """
    跟踪每个Group的边界和元数据
    
    在GPE框架中，Group是source_txt_lt和target_txt_lt的分段
    每个Group对应一个语义单元（词、短语、句子等）
    """
    
    def __init__(self, sink_groups: int = 2):
        """
        Args:
            sink_groups: 保留的sink groups数量（类似Attention Sinks）
        """
        self.groups: List[Group] = []
        self.sink_groups = sink_groups
        self.current_token_offset = 0  # 当前KV cache中的token偏移量
        
    def add_group(
        self,
        segment_tokens: List[int],
        group_type: str,
        segment_idx: int
    ) -> Group:
        """
        添加一个新Group
        
        Args:
            segment_tokens: 该segment的token数量列表（用于计算总token数）
            group_type: 'source' or 'target'
            segment_idx: segment在source_txt_lt或target_txt_lt中的索引
        
        Returns:
            创建的Group对象
        """
        token_count = sum(segment_tokens) if isinstance(segment_tokens, list) else segment_tokens
        
        group = Group(
            group_id=len(self.groups),
            start_token_idx=self.current_token_offset,
            end_token_idx=self.current_token_offset + token_count,
            group_type=group_type,
            segment_idx=segment_idx,
            token_count=token_count
        )
        
        self.groups.append(group)
        self.current_token_offset += token_count
        
        return group
    
    def get_group_indices(self, group_id: int) -> Tuple[int, int]:
        """获取Group的token索引范围"""
        if group_id < len(self.groups):
            group = self.groups[group_id]
            return group.start_token_idx, group.end_token_idx
        return None, None
    
    def get_groups_to_evict(
        self, 
        max_groups: int,
        keep_recent: bool = True
    ) -> List[int]:
        """
        确定需要驱逐的Group IDs
        
        Args:
            max_groups: 最多保留的Group数量
            keep_recent: 是否保留最近的groups
        
        Returns:
            需要驱逐的Group ID列表
        """
        if len(self.groups) <= max_groups:
            return []
        
        # 保留sink groups（前N个）
        sink_group_ids = list(range(min(self.sink_groups, len(self.groups))))
        
        if keep_recent:
            # 保留最近的groups
            recent_group_ids = list(range(
                max(self.sink_groups, len(self.groups) - max_groups + self.sink_groups),
                len(self.groups)
            ))
            keep_group_ids = set(sink_group_ids + recent_group_ids)
        else:
            keep_group_ids = set(sink_group_ids)
        
        # 需要驱逐的groups
        evict_group_ids = [
            gid for gid in range(len(self.groups)) 
            if gid not in keep_group_ids
        ]
        
        return evict_group_ids
    
    def evict_groups(self, group_ids: List[int]) -> Tuple[int, int]:
        """
        驱逐指定的Groups
        
        Args:
            group_ids: 要驱逐的Group ID列表（必须按顺序）
        
        Returns:
            (evicted_start_idx, evicted_end_idx) - 被驱逐的token范围
        """
        if not group_ids:
            return None, None
        
        # 计算被驱逐的token范围
        evicted_start = self.groups[group_ids[0]].start_token_idx
        evicted_end = self.groups[group_ids[-1]].end_token_idx
        
        # 更新后续groups的索引
        evicted_token_count = evicted_end - evicted_start
        
        # 从后往前删除，避免索引问题
        for gid in sorted(group_ids, reverse=True):
            del self.groups[gid]
        
        # 重新编号和更新索引
        self._rebuild_indices()
        
        # 更新当前偏移量
        self.current_token_offset -= evicted_token_count
        
        return evicted_start, evicted_end
    
    def _rebuild_indices(self):
        """重建所有groups的索引"""
        current_offset = 0
        for i, group in enumerate(self.groups):
            group.group_id = i
            group.start_token_idx = current_offset
            group.end_token_idx = current_offset + group.token_count
            current_offset += group.token_count
    
    def get_total_tokens(self) -> int:
        """获取当前总token数"""
        return self.current_token_offset
    
    def get_group_count(self) -> int:
        """获取当前Group数量"""
        return len(self.groups)
    
    def reset(self):
        """重置tracker"""
        self.groups = []
        self.current_token_offset = 0

