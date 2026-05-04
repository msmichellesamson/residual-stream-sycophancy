#!/usr/bin/env python3
"""
run_probe_sweep.py

Main entry point for the layer-wise sycophancy probe sweep.

The core question: at which layer in the residual stream does Pythia
"decide" to be sycophantic? If we can find a clean linear boundary,
that tells us something about how early (or late) the sycophancy
signal is represented -- which has implications for where interventions
might actually work.

Inspired loosely by the geometry-of-truth work (Marks & Tegmark 2023)
and Anthropic's sycophancy evals -- but much smaller scope, just
trying to understand the representation structure.

Usage:
    python src/experiments/run_probe_sweep.py \
        --model pythia-1.4b \
        --activation-type resid_post \
        --n-samples 200

    python src/experiments/run_probe_sweep.py \
        --model pythia-410m \
        --activation-type resid_mid \
        --n-samples 100 \
        --seed 1337
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformer_lens import HookedTransformer

# make sure we can import from sibling dirs
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.sycophancy_dataset import SycophancyDataset, SUPPORTED_DATASETS
from src.data.completions import collect_completions
from src.probes.linear_probe import LinearProbe, ProbeResult
from src.probes.layer_sweep import LayerSweep, SweepResult

# ---------------------------------------------------------------------------
# model name -> TransformerLens name mapping
# Pythia is the main focus here because it has clean layer-by-layer
# residual stream access and it's small enough to run locally.
# also tried GPT-2 but the sycophancy signal is weaker / noisier there
# ---------------------------------------------------------------------------
MODEL_ALIASES = {
    "pythia-70m":   "EleutherAI/pythia-70m-deduped",
    "pythia-160m":  "EleutherAI/pythia-160m-deduped",
    "pythia-410m":  "EleutherAI/pythia-410m-deduped",
    "pythia-1.4b":  "EleutherAI/pythia-1.4b-deduped",
    "pythia-2.8b":  "EleutherAI/pythia-2.8b-deduped",
    "gpt2":         "gpt2",
    "gpt2-medium":  "gpt2-medium",
}

# activation types supported by TransformerLens hook points
# resid_post is the main one we care about -- the actual residual stream
# state after each transformer block. resid_mid is between attn and MLP.
ACTIVATION_TYPES = [
    "resid_post",   # after full transformer block (attn + MLP)
    "resid_mid",    # after attention, before MLP (interesting to split contribution)
    "resid_pre",    # before transformer block (same as resid_post of prev layer)
    "attn_out",     # just the attention contribution
    "mlp_out",      # just the MLP contribution
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe Pythia residual stream for sycophancy signal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pythia-410m",
        choices=list(MODEL_ALIASES.keys()),
        help="Model to probe",
    )
    parser.add_argument(
        "--activation-type",
        type=str,
        default="resid_post",
        choices=ACTIVATION_TYPES,
        help="Which hook point to extract activations from",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of sycophancy/non-sycophancy pairs to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="anthropic_hh_sycophancy",
        choices=SUPPORTED_DATASETS,
        help="Source dataset for sycophancy examples",
    )
    parser.add_argument(
        "--token-aggregation",
        type=str,
        default="last",
        choices=["last", "mean", "max"],
        help=(
            "How to aggregate over token positions. 'last' = final token "
            "(makes sense for decoder models -- this is what 'decides' the next token). "
            "'mean' is sometimes more stable but loses positional info."
        ),
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        default="logistic",
        choices=["logistic", "linear_svm", "mlp"],
        help="Probe architecture. Logistic is simplest / most interpretable.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for probe evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/mps/cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/raw",
        help="Where to save results",
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        default=False,
        help="Save raw activations to disk (large! only do this for small models)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Comma-separated layer indices to probe, e.g. '0,4,8,12'. "
            "Default: probe all layers."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for activation extraction",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Extra logging",
    )
    return parser.parse_args()


def detect_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon -- TransformerLens has decent MPS support now
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    print("WARNING: no GPU found, falling back to CPU -- this will be slow for 1.4B+")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device) -> HookedTransformer:
    """
    Load model via TransformerLens.

    NOTE: fold_ln=True folds layer norm params into adjacent weights,
    which makes residual stream analysis cleaner (the LN doesn't
    change the "direction" of the residual stream in the folded version).
    This is standard practice for mech interp work.
    """
    hf_name = MODEL_ALIASES[model_name]
    print(f"Loading {model_name} ({hf_name}) on {device}...")
    t0 = time.time()

    model = HookedTransformer.from_pretrained(
        hf_name,
        fold_ln=True,          # fold layer norms -- cleaner for linear probing
        center_writing_weights=True,   # zero-mean the residual stream contributions
        center_unembed=True,
        device=str(device),
    )
    model.eval()

    elapsed = time.time() - t0
    print(f"Loaded in {elapsed:.1f}s")
    print(f"  n_layers: {model.cfg.n_layers}")
    print(f"  d_model:  {model.cfg.d_model}")
    print(f"  n_heads:  {model.cfg.n_heads}")
    print(f"  d_mlp:    {model.cfg.d_mlp}")

    return model


def parse_layer_list(layers_str: str | None, n_layers: int) -> list[int]:
    """Parse --layers arg or default to all layers."""
    if layers_str is None:
        return list(range(n_layers))
    layers = [int(x.strip()) for x in layers_str.split(",")]
    # validate
    bad = [l for l in layers if l < 0 or l >= n_layers]
    if bad:
        raise ValueError(f"Layer indices out of range for {n_layers}-layer model: {bad}")
    return sorted(layers)


def build_output_path(args, model: HookedTransformer) -> Path:
    """
    Build a descriptive output directory so we can compare runs later.
    Format: results/raw/{model}_{activation_type}_{n_samples}_{timestamp}/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{args.activation_type}_{args.n_samples}s_{ts}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_run_metadata(out_dir: Path, args, model: HookedTransformer, device: torch.device):
    """Save everything needed to reproduce this run."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_hf_name": MODEL_ALIASES[args.model],
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "activation_type": args.activation_type,
        "n_samples": args.n_samples,
        "dataset": args.dataset,
        "token_aggregation": args.token_aggregation,
        "probe_type": args.probe_type,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "device": str(device),
        "layers_probed": args.layers,
        "batch_size": args.batch_size,
        # hardware info can be helpful for reproducibility context
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {out_dir / 'run_metadata.json'}")
    return meta


def save_sweep_results(out_dir: Path, sweep_result: SweepResult, args):
    """
    Save the full sweep results in a few formats:
    - JSON for programmatic access
    - CSV for easy plotting in pandas / R
    """
    # JSON
    results_dict = sweep_result.to_dict()
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # CSV (one row per layer)
    import csv
    csv_path = out_dir / "per_layer_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "mean_cv_accuracy",
            "std_cv_accuracy",
            "auc_roc",
            "probe_type",
            "activation_type",
            "model",
        ])
        for layer_result in sweep_result.layer_results:
            writer.writerow([
                layer_result.layer_idx,
                f"{layer_result.mean_cv_accuracy:.4f}",
                f"{layer_result.std_cv_accuracy:.4f}",
                f"{layer_result.auc_roc:.4f}" if layer_result.auc_roc is not None else "",
                args.probe_type,
                args.activation_type,
                args.model,
            ])
    print(f"Results saved: {out_dir / 'sweep_results.json'}, {csv_path}")


def print_sweep_summary(sweep_result: SweepResult, model: HookedTransformer):
    """
    Print a human-readable summary of the sweep results.
    I want to see at a glance:
    - Which layers have above-chance accuracy
    - Where accuracy peaks
    - Whether there's a clear "transition" layer
    """
    print("\n" + "=" * 60)
    print("LAYER SWEEP RESULTS")
    print("=" * 60)

    n_layers = model.cfg.n_layers
    results = sorted(sweep_result.layer_results, key=lambda r: r.layer_idx)

    # find the peak
    best = max(results, key=lambda r: r.mean_cv_accuracy)
    chance_level = 0.5  # binary classification

    print(f"\nLayer | CV Acc ± Std  | AUC   | Notes")
    print("-" * 55)

    prev_acc = None
    for r in results:
        acc = r.mean_cv_accuracy
        std = r.std_cv_accuracy
        auc = r.auc_roc

        # flag interesting layers
        notes = []
        if r.layer_idx == best.layer_idx:
            notes.append("<-- PEAK")
        if acc > chance_level + 0.15:
            notes.append("strong")
        elif acc > chance_level + 0.05:
            notes.append("above chance")
        if prev_acc is not None and acc - prev_acc > 0.05:
            notes.append("↑ jump")
        elif prev_acc is not None and acc - prev_acc < -0.05:
            notes.append("↓ drop")

        auc_str = f"{auc:.3f}" if auc is not None else "  N/A "
        print(
            f"  {r.layer_idx:3d} | {acc:.3f} ± {std:.3f}  | {auc_str} | {', '.join(notes)}"
        )
        prev_acc = acc

    print("-" * 55)
    print(f"\nBest layer: {best.layer_idx} / {n_layers - 1}")
    print(f"Best CV accuracy: {best.mean_cv_accuracy:.3f} ± {best.std_cv_accuracy:.3f}")
    if best.auc_roc is not None:
        print(f"Best AUC-ROC: {best.auc_roc:.3f}")

    # rough interpretation
    rel_depth = best.layer_idx / (n_layers - 1)
    print(f"\nRelative depth of peak: {rel_depth:.1%} through the network")
    if rel_depth < 0.33:
        print("Interpretation: sycophancy signal appears EARLY -- possibly in how")
        print("  the input is encoded rather than late reasoning.")
    elif rel_depth < 0.66:
        print("Interpretation: signal peaks in middle layers -- aligns with the")
        print("  'middle layers do most work' finding in many interp papers.")
    else:
        print("Interpretation: signal peaks LATE -- possibly a final-stage decision")
        print("  or output formatting behavior.")

    # check if there's a clean transition or it's gradual
    accs = [r.mean_cv_accuracy for r in results]
    acc_range = max(accs) - min(accs)
    print(f"\nAccuracy range across layers: {acc_range:.3f}")
    if acc_range < 0.05:
        print("NOTE: Very flat curve -- sycophancy signal may be distributed across")
        print("  all layers, OR the probe isn't finding a clear signal at any layer.")
    print("=" * 60)


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = detect_device(args.device)
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model = load_model(args.model, device)
    out_dir = build_output_path(args, model)
    meta = save_run_metadata(out_dir, args, model, device)

    # -----------------------------------------------------------------------
    # Load sycophancy dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset '{args.dataset}' (n={args.n_samples})...")
    dataset = SycophancyDataset(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"  Loaded {len(dataset)} examples")
    print(f"  Label distribution: {dataset.label_distribution()}")

    # sanity check -- if the dataset is heavily imbalanced the probe
    # results will be misleading; warn but don't stop
    pos_frac = dataset.label_distribution().get(1, 0) / len(dataset)
    if not (0.3 <= pos_frac <= 0.7):
        print(
            f"WARNING: Label imbalance (pos_frac={pos_frac:.2f}). "
            "Consider stratified sampling in SycophancyDataset."
        )

    # -----------------------------------------------------------------------
    # Collect activations
    # -----------------------------------------------------------------------
    layers_to_probe = parse_layer_list(args.layers, model.cfg.n_layers)
    print(f"\nWill probe {len(layers_to_probe)} layers: {layers_to_probe[:5]}{'...' if len(layers_to_probe) > 5 else ''}")
    print(f"Activation type: {args.activation_type}")
    print(f"Token aggregation: {args.token_aggregation}")

    print("\nCollecting activations (this is the slow part)...")
    t0 = time.time()

    # collect_completions handles batching + hook point extraction
    # returns dict: layer_idx -> (n_samples, d_model) tensor
    activations_by_layer = collect_completions(
        model=model,
        dataset=dataset,
        activation_type=args.activation_type,
        layers=layers_to_probe,
        token_aggregation=args.token_aggregation,
        batch_size=args.batch_size,
        device=device,
        verbose=args.verbose,
    )

    elapsed = time.time() - t0
    print(f"Activation collection done in {elapsed:.1f}s")

    # optionally save activations -- useful for offline analysis
    # but they're big (n_samples * d_model * n_layers * 4 bytes)
    if args.save_activations:
        act_path = out_dir / "activations.pt"
        # move to CPU before saving
        activations_cpu = {k: v.cpu() for k, v in activations_by_layer.items()}
        torch.save(activations_cpu, act_path)
        size_mb = act_path.stat().st_size / 1e6
        print(f"Activations saved to {act_path} ({size_mb:.1f} MB)")

    # -----------------------------------------------------------------------
    # Run probe sweep
    # -----------------------------------------------------------------------
    print(f"\nRunning {args.probe_type} probe sweep ({args.cv_folds}-fold CV)...")
    t0 = time.time()

    labels = dataset.get_labels()  # list[int], 1=sycophantic, 0=not

    sweep = LayerSweep(
        probe_type=args.probe_type,
        cv_folds=args.cv_folds,
        seed=args.seed,
        verbose=args.verbose,
    )

    sweep_result = sweep.run(
        activations_by_layer=activations_by_layer,
        labels=labels,
        layers=layers_to_probe,
    )

    elapsed = time.time() - t0
    print(f"Probe sweep done in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Save + display results
    # -----------------------------------------------------------------------
    save_sweep_results(out_dir, sweep_result, args)
    print_sweep_summary(sweep_result, model)

    # -----------------------------------------------------------------------
    # Generate plots
    # -----------------------------------------------------------------------
    print("\nGenerating plots...")
    try:
        from src.analysis.plot_sweep import plot_accuracy_by_layer, plot_probe_heatmap

        fig_path = out_dir / "accuracy_by_layer.png"
        plot_accuracy_by_layer(
            sweep_result=sweep_result,
            model_name=args.model,
            activation_type=args.activation_type,
            save_path=fig_path,
        )
        print(f"  Plot saved: {fig_path}")

        # NOTE: heatmap of probe weights is interesting but only really
        # meaningful for the peak layer -- doing all layers would be huge
        best_layer = max(
            sweep_result.layer_results, key=lambda r: r.mean_cv_accuracy
        ).layer_idx
        if hasattr(sweep_result, "probe_weights") and sweep_result.probe_weights:
            heatmap_path = out_dir / f"probe_weights_layer{best_layer}.png"
            plot_probe_heatmap(
                weights=sweep_result.probe_weights[best_layer],
                layer_idx=best_layer,
                save_path=heatmap_path,
            )
            print(f"  Weights heatmap saved: {heatmap_path}")

    except ImportError as e:
        # analysis module might not exist yet -- that's fine
        print(f"  Skipping plots (missing module: {e})")
    except Exception as e:
        # don't let plotting failure kill the run
        print(f"  Plot generation failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\nAll outputs in: {out_dir}")
    print("\nNext steps:")
    print("  - Check accuracy_by_layer.png for the accuracy curve shape")
    print("  - If there's a sharp transition, that's the interesting layer to dig into")
    print("  - Try --activation-type resid_mid to separate attn vs MLP contribution")
    print("  - Try --token-aggregation mean if 'last' is noisy")
    print("  - Compare pythia-410m vs pythia-1.4b -- does the peak layer shift?")

    return 0


if __name__ == "__main__":
    sys.exit(main())