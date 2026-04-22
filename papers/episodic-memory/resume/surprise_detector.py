import torch
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class SurpriseMetrics:
    reconstruction_error: float
    surprise_score: float
    is_surprising: bool
    normalized_error: float
    timestamp: Optional[int] = None


class SurpriseDetector:
    def __init__(
        self,
        sae,
        surprise_threshold: float = 2.0,
        use_adaptive_threshold: bool = True,
        window_size: int = 100,
        warmup_steps: int = 20,
        min_std: float = 1e-3,
        z_clip: float = 10.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.sae = sae.to(device)
        self.sae.eval()
        self.surprise_threshold = surprise_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.min_std = float(min_std)
        self.z_clip = float(z_clip)

        self.device = device
        
        self.error_history: List[float] = []
        self.running_mean: float = 0.0
        self.running_std: float = 1.0

        self.total_states = 0
        self.surprising_states = 0
        
    def compute_surprise(
        self,
        activations: torch.Tensor,
        update_stats: bool = True
    ) -> SurpriseMetrics:
        activations = activations.to(self.device).float()
        
        if activations.dim() == 3:
            raise ValueError("Passe um state pooled [hidden] ou [batch, hidden], não [batch, seq, hidden].")
        elif activations.dim() == 1:
            activations = activations.unsqueeze(0)


        expected = getattr(self.sae, "input_dim", None)
        if expected is not None and activations.size(-1) != expected:
            raise ValueError(
                f"Activation dim mismatch: got {activations.size(-1)}, "
                f"SAE expects {expected}. Ensure you are extracting the same layer/stream "
                f"used to train the SAE (e.g., RouteSAE layer 16)."
            )

            
        with torch.no_grad():
            reconstruction_error = self.sae.compute_reconstruction_error(activations)
            error_value = reconstruction_error.mean().item()

        if update_stats:
            self.error_history.append(error_value)
            if len(self.error_history) > self.window_size:
                self.error_history.pop(0)
            
            if len(self.error_history) > 1:
                self.running_mean = np.mean(self.error_history)
                self.running_std = max(float(np.std(self.error_history)), self.min_std)
            if len(self.error_history) > 2:
                self.running_mean = float(np.mean(self.error_history))
                self.running_std = float(np.std(self.error_history, ddof=1))
                self.running_std = max(self.running_std, self.min_std)
                
        z = (error_value - self.running_mean) / self.running_std

        if self.use_adaptive_threshold:
            if len(self.error_history) < max(2, self.warmup_steps):
                surprise_score = 0.0
            else:
                z = (error_value - self.running_mean) / self.running_std
                z = float(np.clip(z, -self.z_clip, self.z_clip))
                surprise_score = z
        else:
            surprise_score = float(error_value)


        is_surprising = surprise_score > self.surprise_threshold

        self.total_states += 1
        if is_surprising:
            self.surprising_states += 1
        
        return SurpriseMetrics(
            reconstruction_error=error_value,
            surprise_score=surprise_score,
            is_surprising=is_surprising,
            normalized_error=surprise_score,
            timestamp=self.total_states
        )
    
    def get_surprise_rate(self) -> float:
        if self.total_states == 0:
            return 0.0
        return self.surprising_states / self.total_states
    
    def reset_statistics(self):
        self.error_history.clear()
        self.running_mean = 0.0
        self.running_std = 1.0
        self.total_states = 0
        self.surprising_states = 0
    
    def get_statistics(self) -> Dict[str, float]:
        return {
            'mean_error': self.running_mean,
            'std_error': self.running_std,
            'total_states': self.total_states,
            'surprising_states': self.surprising_states,
            'surprise_rate': self.get_surprise_rate(),
            'threshold': self.surprise_threshold
        }
    
    def calibrate_threshold(self, target_rate: float = 0.1):
        if len(self.error_history) < 10:
            print("Not enough history to calibrate threshold")
            return

        errors = np.array(self.error_history)
        z_scores = (errors - self.running_mean) / self.running_std
        
        percentile = (1 - target_rate) * 100
        self.surprise_threshold = np.percentile(z_scores, percentile)
        
        print(f"Calibrated threshold to {self.surprise_threshold:.2f} "
              f"for target rate {target_rate:.1%}")
