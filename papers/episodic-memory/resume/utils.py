import json
from typing import Any, Dict, Optional

import numpy as np
import torch


def save_metrics(metrics: Dict[str, Any], output_path: str):
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            serializable_metrics[key] = value.cpu().tolist()
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"[OK] Metrics saved to {output_path}")


def compute_memory_efficiency(agent) -> Dict[str, float]:
    stats = agent.get_statistics()
    
    return {
        'utilization': stats['memory_stats']['utilization'],
        'surprise_rate': stats['surprise_stats']['surprise_rate'],
        'mean_surprise_in_memory': stats['memory_stats'].get('mean_surprise', 0),
        'compression_ratio': stats['total_steps'] / max(stats['memory_stats']['size'], 1)
    }


def generate_report(agent, output_path: Optional[str] = None) -> str:
    stats = agent.get_statistics()
    efficiency = compute_memory_efficiency(agent)
    
    report = []
    report.append("=" * 60)
    report.append("ReSuME Agent Performance Report")
    report.append("=" * 60)
    report.append("")
    
    report.append("PROCESSING STATISTICS:")
    report.append(f"  Total steps processed: {stats['total_steps']}")
    report.append(f"  States flagged as surprising: {stats['surprise_stats']['surprising_states']}")
    report.append(f"  Surprise rate: {stats['surprise_stats']['surprise_rate']:.1%}")
    report.append("")
    
    report.append("SURPRISE DETECTION:")
    report.append(f"  Mean reconstruction error: {stats['surprise_stats']['mean_error']:.4f}")
    report.append(f"  Std reconstruction error: {stats['surprise_stats']['std_error']:.4f}")
    report.append(f"  Threshold (z-score): {stats['surprise_stats']['threshold']:.2f}")
    report.append("")
    
    report.append("MEMORY STATISTICS:")
    report.append(f"  Current size: {stats['memory_stats']['size']}/{stats['memory_stats']['max_size']}")
    report.append(f"  Utilization: {stats['memory_stats']['utilization']:.1%}")
    report.append(f"  Total stored: {stats['memory_stats']['total_stored']}")
    report.append(f"  Total evicted: {stats['memory_stats']['total_evicted']}")
    if stats['memory_stats']['size'] > 0:
        report.append(f"  Mean surprise in memory: {stats['memory_stats']['mean_surprise']:.2f}")
        report.append(f"  Surprise range: [{stats['memory_stats']['min_surprise']:.2f}, "
                     f"{stats['memory_stats']['max_surprise']:.2f}]")
    report.append("")
    
    report.append("EFFICIENCY METRICS:")
    report.append(f"  Compression ratio: {efficiency['compression_ratio']:.1f}x")
    report.append(f"  Memory utilization: {efficiency['utilization']:.1%}")
    report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"[OK] Report saved to {output_path}")
    
    return report_text


class ActivationBuffer:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.activations = []
        self.metadata = []
    
    def add(self, activation: torch.Tensor, meta: Optional[Dict] = None):
        self.activations.append(activation.cpu())
        self.metadata.append(meta or {})
        
        if len(self.activations) > self.max_size:
            self.activations.pop(0)
            self.metadata.pop(0)
    
    def get_batch(self, batch_size: int):
        if len(self.activations) == 0:
            return None, None
        
        size = min(batch_size, len(self.activations))
        acts = torch.stack(self.activations[:size])
        metas = self.metadata[:size]
        
        return acts, metas
    
    def clear(self):
        self.activations.clear()
        self.metadata.clear()
    
    def __len__(self):
        return len(self.activations)
