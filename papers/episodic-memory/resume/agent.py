from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .episodic_memory import EpisodicMemory
from .surprise_detector import SurpriseDetector


@dataclass
class TraceEntry:
    step: int
    role: str
    text: Optional[str]
    recon_error: float
    surprise_z: float
    is_surprising: bool
    critical_event: Optional[str]
    did_write: bool
    write_id: Optional[str]
    retrieved_ids: List[str]
    retrieved_scores: List[float]
    did_use: bool


class ReSuMEAgent:
    def __init__(
        self,
        sae,
        surprise_threshold: float,
        memory_size: int = 100,
        eviction_policy: str = "lowest_surprise",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.sae = sae
        self.device = device
        self.surprise_detector = SurpriseDetector(
            sae=sae,
            surprise_threshold=surprise_threshold,
            device=device,
        )
        self.memory = EpisodicMemory(
            max_size=memory_size,
            eviction_policy=eviction_policy,
            device=device,
        )
        self.current_step = 0
        self.trace: List[TraceEntry] = []
        self._write_counter = 0

    def process(
        self,
        state: torch.Tensor,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store_in_memory: bool = True,
    ) -> Dict[str, Any]:
        meta = metadata or {}
        role = meta.get("role", "user")
        critical_event = meta.get("critical_event")

        surprise_metrics = self.surprise_detector.compute_surprise(
            state,
            update_stats=(role == "user"),
        )

        stored = False
        write_id = None
        allow_store = store_in_memory and (role == "user")
        if allow_store and surprise_metrics.is_surprising:
            stored = self.memory.add(
                state=state,
                surprise_score=surprise_metrics.surprise_score,
                timestamp=self.current_step,
                metadata=meta,
                context=context,
            )
            if stored:
                write_id = f"mem_{self._write_counter}"
                self._write_counter += 1

        similar_memories: List[Tuple[Any, float]] = []
        if len(self.memory) > 0:
            similar_memories = self.memory.retrieve_by_similarity(
                state,
                k=min(3, len(self.memory)),
            )

        retrieved_ids: List[str] = []
        retrieved_scores: List[float] = []
        for mem, sim in similar_memories:
            retrieved_ids.append(f"t{mem.timestamp}")
            retrieved_scores.append(float(sim))

        use_thresh = float(meta.get("use_thresh", 0.65))
        did_use = (
            role == "assistant"
            and len(retrieved_scores) > 0
            and max(retrieved_scores) >= use_thresh
        )

        self.trace.append(
            TraceEntry(
                step=self.current_step,
                role=role,
                text=context,
                recon_error=float(surprise_metrics.reconstruction_error),
                surprise_z=float(surprise_metrics.surprise_score),
                is_surprising=bool(surprise_metrics.is_surprising),
                critical_event=critical_event,
                did_write=bool(stored),
                write_id=write_id,
                retrieved_ids=retrieved_ids,
                retrieved_scores=retrieved_scores,
                did_use=bool(did_use),
            )
        )

        self.current_step += 1
        return {
            "surprise_metrics": surprise_metrics,
            "stored_in_memory": stored,
            "similar_memories": similar_memories,
            "memory_size": len(self.memory),
            "step": self.current_step,
        }

    def get_statistics(self) -> Dict[str, Any]:
        memory_stats = self.memory.get_statistics()
        surprise_stats = self.surprise_detector.get_statistics()
        return {
            "total_steps": self.current_step,
            "memory_stats": memory_stats,
            "surprise_stats": surprise_stats,
        }

    def get_trace_df(self):
        import pandas as pd

        return pd.DataFrame([asdict(x) for x in self.trace])
