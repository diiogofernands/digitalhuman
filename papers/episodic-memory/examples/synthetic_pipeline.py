import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resume import ReSuMEAgent
from resume.sae_loader import SparseAutoencoder, load_pretrained_sae
from resume.utils import generate_report, save_metrics

DEFAULT_CONFIG_PATH = Path(__file__).with_name("synthetic_config.yaml")


def load_llm(model_name: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading LLM: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()
    return model, tokenizer


def extract_llm_activations(
    model,
    tokenizer,
    texts: List[str],
    layer: int = -1,
    device: str = "cuda",
) -> torch.Tensor:
    all_activations = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            activation = outputs.hidden_states[layer][:, -1, :]
            all_activations.append(activation.cpu())

    return torch.cat(all_activations, dim=0)


def generate_llm_test_texts() -> Tuple[List[str], List[str]]:
    routine_texts = [
        "Hello, how are you today?",
        "The weather is nice outside.",
        "I'm going to the store later.",
        "What time is the meeting?",
        "Thanks for your help!",
        "See you tomorrow.",
        "I had a good day at work.",
        "Let's have lunch together.",
        "The book was interesting.",
        "I need to finish this project.",
    ] * 5

    critical_texts = [
        "URGENT: The system is experiencing critical errors and needs immediate attention!",
        "WARNING: Unauthorized access detected from unknown location.",
        "EMERGENCY: Patient requires immediate medical intervention.",
        "ALERT: Security breach in progress, take defensive measures now.",
        "CRITICAL: Database corruption detected, initiating backup protocols.",
        "Breaking: Major policy changes announced affecting all departments.",
        "Attention: Significant performance degradation detected in core systems.",
        "Priority: Legal compliance issue requires immediate resolution.",
        "Notice: Unexpected financial discrepancy requires investigation.",
        "Important: Strategic decision point reached, input needed urgently.",
    ]

    ood_texts = [
        "The purple elephant danced quantum mechanics on Tuesday.",
        "print('Hello') * inf while undefined behavior leaks into reality.txt",
        "Mixed-language chaos with symbols and disjoint vocabulary tokens.",
        "ERROR NULL POINTER EXCEPTION AT MEMORY ADDRESS 0x00000000",
        "asdfghjkl qwertyuiop keyboard smash detected 1234567890",
    ]

    texts = routine_texts + critical_texts + ood_texts
    labels = ["routine"] * len(routine_texts) + ["critical"] * len(critical_texts) + ["ood"] * len(ood_texts)

    indices = torch.randperm(len(texts)).tolist()
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    return texts, labels


def generate_test_states(config: Dict[str, Any]) -> Tuple[torch.Tensor, List[str]]:
    n_routine = int(config["experiment"].get("n_routine_states", 50))
    n_critical = int(config["experiment"].get("n_critical_states", 10))
    n_ood = int(config["experiment"].get("n_ood_states", 5))
    state_dim = int(config["experiment"].get("state_dim", 4096))

    routine = torch.randn(n_routine, state_dim) * 0.5
    critical = torch.randn(n_critical, state_dim) * 0.9 + 1.2
    ood = torch.randn(n_ood, state_dim) * 1.2 + 2.0

    states = torch.cat([routine, critical, ood], dim=0)
    labels = ["routine"] * n_routine + ["critical"] * n_critical + ["ood"] * n_ood

    perm = torch.randperm(states.size(0))
    states = states[perm]
    labels = [labels[idx] for idx in perm.tolist()]
    return states, labels


def run_experiment(config: Dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning example experiment:")
    print(f"  SAE: {config['sae']['model_name']}")
    print(f"  Surprise threshold: {config['surprise']['threshold']}")
    print(f"  Memory size: {config['memory']['max_size']}")

    torch.manual_seed(config["experiment"]["seed"])

    try:
        sae = load_pretrained_sae(
            model_name=config["sae"]["model_name"],
            layer=config["sae"].get("layer"),
            device=config["llm"]["device"],
        )
    except Exception:
        print("  Using random SAE (pre-trained not available)")
        sae = SparseAutoencoder(
            input_dim=config["sae"]["input_dim"],
            hidden_dim=config["sae"]["hidden_dim"],
            sparsity_coefficient=config["sae"]["sparsity_coefficient"],
        )

    agent = ReSuMEAgent(
        sae=sae,
        surprise_threshold=config["surprise"]["threshold"],
        memory_size=config["memory"]["max_size"],
        eviction_policy=config["memory"]["eviction_policy"],
        device=config["llm"]["device"],
    )

    use_real_llm = config.get("experiment", {}).get("use_real_llm", False)
    if use_real_llm:
        try:
            llm_model, llm_tokenizer = load_llm(
                config["llm"]["model_name"],
                config["llm"]["device"],
            )
            texts, labels = generate_llm_test_texts()
            states = extract_llm_activations(
                llm_model,
                llm_tokenizer,
                texts,
                layer=config["llm"]["layer_index"],
                device=config["llm"]["device"],
            )
            states = states.to(config["llm"]["device"])
            if states.shape[1] != sae.input_dim:
                print(f"WARNING: LLM activation dim ({states.shape[1]}) != SAE input dim ({sae.input_dim})")
        except Exception as exc:
            print(f"Error loading LLM: {exc}")
            use_real_llm = False

    if not use_real_llm:
        print("Generating synthetic test states...")
        config["experiment"]["state_dim"] = sae.input_dim
        states, labels = generate_test_states(config)

    print(f"Processing {len(states)} states...")
    results = []
    for idx, (state, label) in enumerate(zip(states, labels)):
        result = agent.process(state, context=f"{label} state {idx}")
        results.append(
            {
                "step": idx,
                "label": label,
                "surprise_score": result["surprise_metrics"].surprise_score,
                "is_surprising": result["surprise_metrics"].is_surprising,
                "stored": result["stored_in_memory"],
            }
        )

    metrics = compute_experiment_metrics(agent, results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_metrics(metrics, str(output_dir / f"metrics_{timestamp}.json"))
    generate_report(agent, str(output_dir / f"report_{timestamp}.txt"))
    with open(output_dir / f"config_{timestamp}.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(config, handle)

    print(f"\nResults saved to {output_dir}")
    return metrics


def compute_experiment_metrics(agent, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = agent.get_statistics()
    category_metrics = {}

    for category in ["routine", "critical", "ood"]:
        category_results = [row for row in results if row["label"] == category]
        if category_results:
            surprise_scores = [row["surprise_score"] for row in category_results]
            stored_count = sum(row["stored"] for row in category_results)
            category_metrics[category] = {
                "count": len(category_results),
                "mean_surprise": float(np.mean(surprise_scores)),
                "std_surprise": float(np.std(surprise_scores)),
                "min_surprise": float(np.min(surprise_scores)),
                "max_surprise": float(np.max(surprise_scores)),
                "stored_count": stored_count,
                "stored_rate": stored_count / len(category_results),
            }

    return {
        "agent_stats": stats,
        "category_metrics": category_metrics,
        "timestamp": datetime.now().isoformat(),
    }


def run_sweep(base_config_path: Path, output_dir: Path):
    with open(base_config_path, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    results = []
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        config = base_config.copy()
        config["surprise"]["threshold"] = threshold
        print(f"\nRunning with surprise_threshold={threshold}")
        metrics = run_experiment(config, output_dir / f"threshold_{threshold}")
        metrics["params"] = {"surprise_threshold": threshold}
        results.append(metrics)

    save_metrics({"results": results}, str(output_dir / "sweep_results.json"))
    print(f"\n[OK] Sweep complete! Results in {output_dir / 'sweep_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run the synthetic ReSuME example pipeline")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to example config file")
    parser.add_argument("--output-dir", type=str, default="results/example", help="Output directory")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        run_sweep(Path(args.config), output_dir)
        return

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_experiment(config, output_dir)


if __name__ == "__main__":
    main()
