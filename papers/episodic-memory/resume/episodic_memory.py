import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import random

@dataclass
class MemoryEntry:
    state: torch.Tensor
    surprise_score: float
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None

class EpisodicMemory:
    def __init__(self, max_size: int = 100, eviction_policy: str = "lowest_surprise",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.device = device
        self.memories: List[MemoryEntry] = []
        self.total_stored = 0
        self.total_evicted = 0

    def __len__(self) -> int:
        return len(self.memories)

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x.float().flatten()
        return x / (x.norm(p=2) + eps)

    def add(self, state: torch.Tensor, surprise_score: float,
            timestamp: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None,
            context: Optional[str] = None) -> bool:
        if timestamp is None:
            timestamp = self.total_stored
        if metadata is None:
            metadata = {}
            
        st = self._l2norm(state.detach().cpu())

        entry = MemoryEntry(
            state=st,
            surprise_score=float(surprise_score),
            timestamp=int(timestamp),
            metadata=metadata,
            context=context
        )

        if len(self.memories) >= self.max_size:
            self._evict()

        self.memories.append(entry)
        self.total_stored += 1
        return True

    def _evict(self):
        if not self.memories:
            return

        if self.eviction_policy == "lowest_surprise":
            idx = int(np.argmin([m.surprise_score for m in self.memories]))
        elif self.eviction_policy == "oldest":
            idx = int(np.argmin([m.timestamp for m in self.memories]))
        elif self.eviction_policy == "random":
            idx = random.randrange(len(self.memories))
        else:
            idx = int(np.argmin([m.surprise_score for m in self.memories]))

        self.memories.pop(idx)
        self.total_evicted += 1

    def retrieve_by_similarity(self, query_state: torch.Tensor, k: int = 5,
                               metric: str = "cosine") -> List[Tuple[MemoryEntry, float]]:
        if len(self.memories) == 0:
            return []

        q = self._l2norm(query_state.detach().cpu())

        sims: List[Tuple[MemoryEntry, float]] = []
        for mem in self.memories:
            m = mem.state 

            if metric == "cosine":
                sim = torch.nn.functional.cosine_similarity(
                    q.unsqueeze(0), m.unsqueeze(0)
                ).item()
            elif metric == "l2":
                sim = -torch.norm(q - m, p=2).item()
            else:
                sim = 0.0

            sims.append((mem, float(sim)))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[: min(k, len(sims))]

    def get_statistics(self) -> Dict[str, float]:
        size = len(self.memories)
        utilization = size / max(self.max_size, 1)

        if size == 0:
            return {
                "size": 0,
                "max_size": self.max_size,
                "utilization": utilization,
                "total_stored": self.total_stored,
                "total_evicted": self.total_evicted,
                "mean_surprise": 0.0,
                "min_surprise": 0.0,
                "max_surprise": 0.0,
                "eviction_policy": self.eviction_policy,
            }

        surprises = [m.surprise_score for m in self.memories]
        return {
            "size": size,
            "max_size": self.max_size,
            "utilization": utilization,
            "total_stored": self.total_stored,
            "total_evicted": self.total_evicted,
            "mean_surprise": float(np.mean(surprises)),
            "min_surprise": float(np.min(surprises)),
            "max_surprise": float(np.max(surprises)),
            "eviction_policy": self.eviction_policy,
        }
