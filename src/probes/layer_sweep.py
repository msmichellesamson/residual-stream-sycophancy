"""
Layer sweep: probe every layer of Pythia for sycophancy signal.

Motivation: We want to know *where* in the residual stream the model has
"decided" to be sycophantic -- i.e., at which layer a linear probe on the
residual stream (or attn-out / mlp-out) can best predict whether the
completion will be sycophantic.

The intuition (from Anthropic's sycophancy work and some of the sleeper
agents / representation engineering papers) is that if we can localise the
signal early, it suggests that sycophancy is a "planning" phenomenon rather
than a late-stage token-generation one.  If it only appears in the last few
layers, it might be more of an output-formatting thing.

This file just orchestrates the sweep.  The heavy lifting lives in
linear_probe.py.  Results get written to results/ as a CSV so we can
make plots without re-running everything.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from transformer_lens import HookedTransformer

# local imports -- relative because this is part of a small package
from src.data.sycophancy_dataset import SycophancyDataset
from src.data.completions import CompletionBatch
from src.probes.linear_probe import LinearProbe, ProbeResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Types / config
# ──────────────────────────────────────────────────────────────────────────────

ActivationType = Literal["residual", "attn_out", "mlp_out"]

# Map our friendly names to TransformerLens hook names.
# TransformerLens uses:
#   hook_resid_post   -> residual stream *after* the whole layer
#   hook_attn_out     -> attention output (before adding to residual)
#   hook_mlp_out      -> mlp output (before adding to residual)
#
# NOTE: "residual" here means the full residual stream post-layer,
# which is what most probing papers use.  attn_out and mlp_out are
# more surgical -- they let us ask "is this the attention or the MLP
# that's carrying the signal?"
_HOOK_TEMPLATES: dict[ActivationType, str] = {
    "residual": "blocks.{layer}.hook_resid_post",
    "attn_out": "blocks.{layer}.hook_attn_out",
    "mlp_out":  "blocks.{layer}.hook_mlp_out",
}


@dataclass
class SweepConfig:
    model_name: str = "EleutherAI/pythia-1.4b"
    activation_type: ActivationType = "residual"
    # which token position to probe -- "last" means the last token of the
    # prompt (before the completion).  Could also try "mean" over the prompt.
    probe_position: Literal["last", "mean"] = "last"
    # probe training
    probe_cv_folds: int = 5
    probe_max_iter: int = 1000
    probe_C: float = 1.0           # LogReg regularisation
    random_seed: int = 42
    # device -- will be overridden by detect_device() if not set
    device: Optional[str] = None
    # where to save results
    results_dir: Path = Path("results")
    # if True, re-run even if cached CSV exists
    force_rerun: bool = False
    # cap on number of dataset examples (useful for quick iteration)
    max_examples: Optional[int] = None


@dataclass
class LayerResult:
    layer: int
    hook_name: str
    probe_acc: float
    probe_auc: float          # AUROC -- sometimes more informative than acc
    chance_baseline: float    # majority-class accuracy
    n_train: int
    n_test: int
    probe_coef_norm: float    # L2 norm of the probe's weight vector
    # runtime metadata
    elapsed_sec: float = 0.0
    extra: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    logger.info("Using device: %s", dev)
    return dev


def _hook_name_for_layer(activation_type: ActivationType, layer: int) -> str:
    return _HOOK_TEMPLATES[activation_type].format(layer=layer)


def _extract_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,   # shape: (batch, seq_len)
    hook_name: str,
    probe_position: Literal["last", "mean"],
) -> np.ndarray:
    """
    Run a forward pass with a single hook and return activations at the
    requested position as a numpy array of shape (batch, d_model).

    We only cache the one hook we care about to keep memory usage down.
    On A100 this isn't a big deal, but on my laptop (MPS) it absolutely is.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_name,  # only materialise this activation
            return_type=None,        # don't need logits
        )

    # cache[hook_name] has shape (batch, seq_len, d_model)
    acts = cache[hook_name]  # type: ignore[index]

    if probe_position == "last":
        # last *non-padding* token.  We assume right-padded (TransformerLens
        # default).  If sequences are different lengths this needs attention.
        # TODO: handle variable-length sequences properly -- for now the
        # dataset gives us fixed-length truncated prompts so this is fine.
        acts = acts[:, -1, :]   # (batch, d_model)
    elif probe_position == "mean":
        acts = acts.mean(dim=1)  # (batch, d_model)
    else:
        raise ValueError(f"Unknown probe_position: {probe_position}")

    return acts.float().cpu().numpy()


def _chance_accuracy(labels: np.ndarray) -> float:
    """Majority-class baseline accuracy."""
    dc = DummyClassifier(strategy="most_frequent")
    dc.fit(labels.reshape(-1, 1), labels)
    return float(dc.score(labels.reshape(-1, 1), labels))


# ──────────────────────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_layer_sweep(
    dataset: SycophancyDataset,
    completions: CompletionBatch,
    cfg: SweepConfig | None = None,
) -> pd.DataFrame:
    """
    For each layer in the model, train a linear probe on the activations and
    record its cross-validated accuracy and AUROC.

    Returns a DataFrame with columns:
        layer, hook_name, probe_acc, probe_auc, chance_baseline,
        n_train, n_test, probe_coef_norm, elapsed_sec

    The DataFrame is also written to results/<model_shortname>_<act_type>.csv.

    Args:
        dataset:     Labelled sycophancy examples (prompt + label).
        completions: Pre-generated model completions corresponding to the
                     dataset examples.  We need these to decide which token
                     position to probe (the end of the prompt).
        cfg:         SweepConfig.  Uses defaults if None.
    """
    if cfg is None:
        cfg = SweepConfig()

    if cfg.device is None:
        cfg.device = detect_device()

    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # Friendly name for filenames / logging
    model_short = cfg.model_name.split("/")[-1]
    result_path = cfg.results_dir / f"{model_short}_{cfg.activation_type}.csv"
    meta_path   = cfg.results_dir / f"{model_short}_{cfg.activation_type}_meta.json"

    if result_path.exists() and not cfg.force_rerun:
        logger.info("Found cached results at %s, loading...", result_path)
        return pd.read_csv(result_path)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", cfg.model_name)
    t0 = time.perf_counter()
    model = HookedTransformer.from_pretrained(
        cfg.model_name,
        center_writing_weights=True,    # standard TransformerLens setup
        center_unembed=True,
        fold_ln=True,
        device=cfg.device,
    )
    model.eval()
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    logger.info(
        "Model loaded in %.1fs | layers=%d d_model=%d",
        time.perf_counter() - t0, n_layers, d_model,
    )

    # ── Prepare data ──────────────────────────────────────────────────────────
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    examples = dataset.get_labeled_examples()
    if cfg.max_examples is not None:
        examples = examples[: cfg.max_examples]
        logger.info("Capped to %d examples", len(examples))

    labels  = np.array([ex.label for ex in examples])  # 1=sycophantic, 0=honest
    prompts = [ex.prompt for ex in examples]

    logger.info(
        "Dataset: %d examples | sycophantic=%d honest=%d",
        len(labels), labels.sum(), (1 - labels).sum(),
    )

    chance = _chance_accuracy(labels)
    logger.info("Chance baseline (majority class): %.3f", chance)

    # Tokenise all prompts once -- doing this inside the layer loop would be
    # wasteful.  TransformerLens tokeniser returns a padded tensor.
    logger.info("Tokenising prompts...")
    tokens = model.to_tokens(prompts, prepend_bos=True)  # (N, seq_len)
    tokens = tokens.to(cfg.device)

    # ── Layer sweep ───────────────────────────────────────────────────────────
    layer_results: list[LayerResult] = []

    for layer_idx in range(n_layers):
        hook_name = _hook_name_for_layer(cfg.activation_type, layer_idx)
        t_layer = time.perf_counter()

        logger.info(
            "[layer %2d/%d] hook=%s",
            layer_idx, n_layers - 1, hook_name,
        )

        # Extract activations for all examples at this layer.
        # We process in batches to avoid OOM -- even small models can be
        # tight on MPS with a full dataset pass.
        batch_size = 32  # TODO: expose this in SweepConfig
        all_acts = []
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_tokens = tokens[start:end]
            acts = _extract_activations(
                model, batch_tokens, hook_name, cfg.probe_position
            )
            all_acts.append(acts)

        X = np.vstack(all_acts)  # (N, d_model)
        assert X.shape == (len(labels), d_model), (
            f"Unexpected activation shape: {X.shape}"
        )

        # Train and evaluate the linear probe.
        probe = LinearProbe(
            C=cfg.probe_C,
            max_iter=cfg.probe_max_iter,
            cv_folds=cfg.probe_cv_folds,
            random_seed=cfg.random_seed,
        )
        result: ProbeResult = probe.fit_evaluate(X, labels)

        elapsed = time.perf_counter() - t_layer

        lr = LayerResult(
            layer=layer_idx,
            hook_name=hook_name,
            probe_acc=result.cv_accuracy,
            probe_auc=result.cv_auroc,
            chance_baseline=chance,
            n_train=result.n_train,
            n_test=result.n_test,
            probe_coef_norm=result.coef_norm,
            elapsed_sec=elapsed,
        )
        layer_results.append(lr)

        logger.info(
            "  acc=%.3f  auc=%.3f  chance=%.3f  coef_norm=%.2f  (%.1fs)",
            lr.probe_acc, lr.probe_auc, lr.chance_baseline,
            lr.probe_coef_norm, lr.elapsed_sec,
        )

        # Opportunistically flush so we don't lose results if something crashes
        # mid-sweep.  Small overhead, big peace of mind.
        _save_partial(layer_results, result_path)

    # ── Wrap up ───────────────────────────────────────────────────────────────
    df = _results_to_dataframe(layer_results)
    df.to_csv(result_path, index=False)

    # Save config + summary stats as JSON for reproducibility
    summary = {
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(cfg).items()
        },
        "n_layers": n_layers,
        "d_model": d_model,
        "n_examples": len(labels),
        "best_layer": int(df.loc[df.probe_acc.idxmax(), "layer"]),
        "best_acc": float(df.probe_acc.max()),
        "best_auc": float(df.probe_auc.max()),
        "chance_baseline": float(chance),
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Sweep complete. Best layer: %d (acc=%.3f, auc=%.3f)",
        summary["best_layer"], summary["best_acc"], summary["best_auc"],
    )

    # Print a quick summary table to stdout -- useful when running in a terminal
    _print_summary_table(df)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Compare activation types (convenience wrapper)
# ──────────────────────────────────────────────────────────────────────────────

def run_multi_activation_sweep(
    dataset: SycophancyDataset,
    completions: CompletionBatch,
    base_cfg: SweepConfig | None = None,
    activation_types: list[ActivationType] | None = None,
) -> dict[ActivationType, pd.DataFrame]:
    """
    Run the layer sweep for multiple activation types and return a dict of
    DataFrames.  Useful for the notebook where we want to compare residual vs
    attn_out vs mlp_out on the same plot.

    NOTE: This reloads the model for each activation type, which is wasteful.
    A smarter version would cache all activations in one pass and then probe
    them separately.  For now this is fine because we're not doing this at scale
    and the model fits comfortably in memory.
    """
    if base_cfg is None:
        base_cfg = SweepConfig()
    if activation_types is None:
        activation_types = ["residual", "attn_out", "mlp_out"]

    results: dict[ActivationType, pd.DataFrame] = {}
    for act_type in activation_types:
        logger.info("=" * 60)
        logger.info("Running sweep for activation_type=%s", act_type)
        logger.info("=" * 60)

        # Don't mutate the base config
        import copy
        cfg = copy.deepcopy(base_cfg)
        cfg.activation_type = act_type

        df = run_layer_sweep(dataset, completions, cfg)
        results[act_type] = df

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def _results_to_dataframe(results: list[LayerResult]) -> pd.DataFrame:
    rows = [
        {
            "layer":            r.layer,
            "hook_name":        r.hook_name,
            "probe_acc":        r.probe_acc,
            "probe_auc":        r.probe_auc,
            "chance_baseline":  r.chance_baseline,
            "acc_above_chance": r.probe_acc - r.chance_baseline,
            "n_train":          r.n_train,
            "n_test":           r.n_test,
            "probe_coef_norm":  r.probe_coef_norm,
            "elapsed_sec":      r.elapsed_sec,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def _save_partial(results: list[LayerResult], path: Path) -> None:
    """Write what we have so far -- called after each layer."""
    df = _results_to_dataframe(results)
    df.to_csv(path.with_suffix(".partial.csv"), index=False)


def _print_summary_table(df: pd.DataFrame) -> None:
    """Quick ASCII summary -- not pretty but functional."""
    print("\n" + "=" * 60)
    print("Layer sweep results summary")
    print("=" * 60)
    print(f"{'Layer':>6}  {'Acc':>6}  {'AUC':>6}  {'+Chance':>8}  {'CoefNorm':>9}")
    print("-" * 60)
    for _, row in df.iterrows():
        marker = " <-- best" if row.probe_acc == df.probe_acc.max() else ""
        print(
            f"{int(row.layer):>6}  {row.probe_acc:.4f}  {row.probe_auc:.4f}"
            f"  {row.acc_above_chance:+.4f}  {row.probe_coef_norm:>9.2f}"
            f"{marker}"
        )
    print("=" * 60)
    print(
        f"Chance baseline: {df.chance_baseline.iloc[0]:.4f}  |  "
        f"Best acc: {df.probe_acc.max():.4f} at layer {df.probe_acc.idxmax()}\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point -- handy for quick runs without opening a notebook
# ──────────────────────────────────────────────────────────────────────────────

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Run a layer-wise sycophancy probe sweep on a Pythia model."
    )
    p.add_argument("--model",       default="EleutherAI/pythia-1.4b")
    p.add_argument(
        "--activation-type",
        choices=["residual", "attn_out", "mlp_out"],
        default="residual",
    )
    p.add_argument(
        "--probe-position",
        choices=["last", "mean"],
        default="last",
    )
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--results-dir",  default="results")
    p.add_argument("--force-rerun",  action="store_true")
    p.add_argument(
        "--all-activation-types",
        action="store_true",
        help="Run for residual, attn_out, mlp_out in sequence.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


if __name__ == "__main__":
    import sys
    from src.data.sycophancy_dataset import SycophancyDataset
    from src.data.completions import CompletionBatch

    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = SweepConfig(
        model_name=args.model,
        activation_type=args.activation_type,
        probe_position=args.probe_position,
        max_examples=args.max_examples,
        random_seed=args.seed,
        results_dir=Path(args.results_dir),
        force_rerun=args.force_rerun,
    )

    # Load dataset and completions.  These handle their own caching.
    dataset     = SycophancyDataset.load_default()
    completions = CompletionBatch.load_or_generate(dataset, model_name=cfg.model_name)

    if args.all_activation_types:
        dfs = run_multi_activation_sweep(dataset, completions, base_cfg=cfg)
        for act_type, df in dfs.items():
            print(f"\n── {act_type} ──")
            _print_summary_table(df)
    else:
        df = run_layer_sweep(dataset, completions, cfg)

    sys.exit(0)