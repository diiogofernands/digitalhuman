import json, torch
from typing import Optional
import torch.nn as nn
from huggingface_hub import hf_hub_download

try:
    from safetensors.torch import load_file as safetensors_load_file
except ImportError:
    safetensors_load_file = None


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coefficient: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coefficient = sparsity_coefficient
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed, _ = self.forward(x)
        return torch.norm(x - reconstructed, dim=-1, p=2)
    
    def compute_relative_reconstruction_error(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        reconstructed, _ = self.forward(x)
        num = (x - reconstructed).pow(2).sum(dim=-1)  
        den = x.pow(2).sum(dim=-1) + eps                 
        return num / den


def _load_state_dict_any(weights_path: str, device: str) -> dict:
    """
    load either .pt/.bin (torch) or .safetensors (safetensors)
    returns a state_dict (plain dict[str, Tensor])
    """
    if weights_path.endswith(".safetensors"):
        if safetensors_load_file is None:
            raise ImportError("safetensors is required to load .safetensors. Install with: pip install safetensors")
        return safetensors_load_file(weights_path)
    obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint format in {weights_path}: {type(obj)}")
    return obj


def load_local_sae(
    path: str,
    layer: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> SparseAutoencoder:
    from pathlib import Path
    
    base_path = Path(path)
    config_path = base_path / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)

    if layer is None:
        available_layers = [d.name.replace("layers.", "") for d in base_path.iterdir() 
                          if d.is_dir() and d.name.startswith("layers.")]
        if not available_layers:
            raise ValueError(f"No layers found in {base_path}")
        layer = int(available_layers[0])
        print(f"Auto-detected layer: {layer}")
    
    layer_path = base_path / f"layers.{layer}"
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer {layer} not found at {layer_path}")

    cfg_path = layer_path / "cfg.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            layer_cfg = json.load(f)
    else:
        layer_cfg = {}

    d_in = layer_cfg.get("d_in", config.get("input_dim", 2560))
    expansion = layer_cfg.get("expansion_factor", config.get("sae", {}).get("expansion_factor", 8))
    d_hidden = d_in * expansion
    
    print(f"Creating SAE: d_in={d_in}, d_hidden={d_hidden} (expansion={expansion})")
    
    sae = SparseAutoencoder(
        input_dim=d_in,
        hidden_dim=d_hidden,
        sparsity_coefficient=0.1,
    )
    
    weights_path = layer_path / "sae.safetensors"
    if not weights_path.exists():
        weights_path = layer_path / "sae.pt"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"No weights found in {layer_path}")
    
    print(f"Loading weights from {weights_path}")
    state_dict = _load_state_dict_any(str(weights_path), device=device)

    key_mapping = {
        'W_enc': 'encoder.weight',
        'b_enc': 'encoder.bias',
        'W_dec': 'decoder.weight',
        'b_dec': 'decoder.bias',
    }
    
    mapped_state_dict = {}
    for key, value in state_dict.items():
        mapped_key = key_mapping.get(key, key)
        mapped_state_dict[mapped_key] = value

    if "decoder.weight" in mapped_state_dict and "encoder.weight" in mapped_state_dict:
        enc_hidden_dim, enc_input_dim = mapped_state_dict["encoder.weight"].shape
        dec_w = mapped_state_dict["decoder.weight"]
        
        if tuple(dec_w.shape) == (enc_hidden_dim, enc_input_dim):
            mapped_state_dict["decoder.weight"] = dec_w.T

    if "encoder.weight" in mapped_state_dict:
        actual_hidden_dim, actual_input_dim = mapped_state_dict["encoder.weight"].shape
        if actual_input_dim != sae.input_dim or actual_hidden_dim != sae.hidden_dim:
            print(f"Adjusting SAE dimensions: {actual_input_dim} -> {actual_hidden_dim}")
            sae = SparseAutoencoder(
                input_dim=actual_input_dim,
                hidden_dim=actual_hidden_dim,
                sparsity_coefficient=sae.sparsity_coefficient,
            )
    
    missing, unexpected = sae.load_state_dict(mapped_state_dict, strict=False)
    
    sae.to(device)
    sae.eval()
    
    print(f"[OK] Loaded local SAE from {path}")
    print(f"  Layer: {layer}")
    print(f"  Input dim: {sae.input_dim}, Hidden dim: {sae.hidden_dim}")
    if missing:
        print(f"Missing keys: {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}")
    
    return sae


def load_pretrained_sae(
    model_name: str = "davidoneil/sae-per-layer-persona",
    layer: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    revision: str = "main",
    allow_random_fallback: bool = False,
) -> SparseAutoencoder:
    try:
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            revision=revision,
            repo_type="model",
        )
        with open(config_path, "r") as f:
            config = json.load(f)

        sae = SparseAutoencoder(
            input_dim=int(config.get("input_dim", 4096)),
            hidden_dim=int(config.get("hidden_dim", 16384)),
            sparsity_coefficient=float(config.get("sparsity_coefficient", 0.1)),
        )

        subfolder = None
        candidate_files = []

        if layer is not None:
            subfolder = f"layers.{layer}"
            candidate_files = ["sae.safetensors", "sae.pt", f"layer_{layer}.pt"]
        else:
            subfolder = None
            candidate_files = ["sae.safetensors", "sae.pt"]

        weights_path = None
        last_err = None

        for fname in candidate_files:
            try:
                weights_path = hf_hub_download(
                    repo_id=model_name,
                    filename=fname,
                    subfolder=subfolder,
                    revision=revision,
                    repo_type="model",
                )
                break
            except Exception as e:
                last_err = e
                weights_path = None

        if weights_path is None and layer is None:
            for default_layer in (16, 12, 20, 24):
                for fname in ("sae.safetensors", "sae.pt", f"layer_{default_layer}.pt"):
                    try:
                        weights_path = hf_hub_download(
                            repo_id=model_name,
                            filename=fname,
                            subfolder=f"layers.{default_layer}",
                            revision=revision,
                            repo_type="model",
                        )
                        layer = default_layer  
                        break
                    except Exception as e:
                        last_err = e
                        weights_path = None
                if weights_path is not None:
                    break

        if weights_path is None:
            raise FileNotFoundError(
                f"Could not find SAE weights in repo '{model_name}' (revision={revision}). Last error: {last_err}"
            )

        state_dict = _load_state_dict_any(weights_path, device=device)
        
        key_mapping = {
            'W_enc': 'encoder.weight',
            'b_enc': 'encoder.bias',
            'W_dec': 'decoder.weight',
            'b_dec': 'decoder.bias',
        }
        
        mapped_state_dict = {}
        for key, value in state_dict.items():
            mapped_key = key_mapping.get(key, key)
            mapped_state_dict[mapped_key] = value
        
        if "decoder.weight" in mapped_state_dict and "encoder.weight" in mapped_state_dict:
            enc_hidden_dim, enc_input_dim = mapped_state_dict["encoder.weight"].shape
            dec_w = mapped_state_dict["decoder.weight"]
            
            if tuple(dec_w.shape) == (enc_hidden_dim, enc_input_dim):
                mapped_state_dict["decoder.weight"] = dec_w.T

        if "encoder.weight" in mapped_state_dict:
            actual_hidden_dim, actual_input_dim = mapped_state_dict["encoder.weight"].shape
            if actual_input_dim != sae.input_dim or actual_hidden_dim != sae.hidden_dim:
                print(f"Dimension mismatch detected!")
                print(f"  Config: input_dim={sae.input_dim}, hidden_dim={sae.hidden_dim}")
                print(f"  Checkpoint: input_dim={actual_input_dim}, hidden_dim={actual_hidden_dim}")
                print(f"  Recreating SAE with checkpoint dimensions...")
                
                sae = SparseAutoencoder(
                    input_dim=actual_input_dim,
                    hidden_dim=actual_hidden_dim,
                    sparsity_coefficient=sae.sparsity_coefficient,
                )
        
        missing, unexpected = sae.load_state_dict(mapped_state_dict, strict=False)

        sae.to(device)
        sae.eval()

        print(f"Loaded SAE from {model_name} (revision={revision})")
        if layer is not None:
            print(f"  Layer: {layer}")
        print(f"  Weights: {weights_path}")
        print(f"  Input dim: {sae.input_dim}, Hidden dim: {sae.hidden_dim}")
        if missing:
            print(f"  Missing keys (first 10): {missing[:10]}")
        if unexpected:
            print(f"  Unexpected keys (first 10): {unexpected[:10]}")

        return sae

    except Exception as e:
        if not allow_random_fallback:
            raise RuntimeError(
                f"Could not load pre-trained SAE from '{model_name}' (revision={revision}, layer={layer}). "
                f"Original error: {e}"
            ) from e

        print(f"Could not load pre-trained SAE: {e}")
        print("  Initializing random SAE for demonstration purposes (allow_random_fallback=True)")

        sae = SparseAutoencoder(
            input_dim=4096,
            hidden_dim=16384,
            sparsity_coefficient=0.1,
        )
        sae.to(device)
        sae.eval()
        return sae
