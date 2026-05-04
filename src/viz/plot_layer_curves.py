"""
Produces two key diagnostic figures for the residual-stream sycophancy probe sweep:

  Figure 1 – Probe accuracy vs. layer, one curve per model size (1.4B, 410M, 70M)
              on a shared axis so you can eyeball where information 'crystallises'
              across scale.

  Figure 2 – For a single model (default: 1.4B), stacked bar showing how much of
              the probe signal comes from attn_out vs mlp_out at each layer.
              Helps answer "is this an attention thing or an MLP thing?"

Usage:
    python src/viz/plot_layer_curves.py \
        --results_dir results/probe_sweep \
        --out_dir results/figures \
        --breakdown_model pythia-1.4b

Expects JSON files like:
    results/probe_sweep/pythia-1.4b_layer_sweep.json
    results/probe_sweep/pythia-410m_layer_sweep.json
    results/probe_sweep/pythia-70m_layer_sweep.json

Each JSON is a list of dicts with at least:
    {"layer": int, "accuracy": float, "component": str}
where component is one of "resid_post", "attn_out", "mlp_out"

If the breakdown file doesn't exist yet, Figure 2 is skipped gracefully.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Agg backend so this works headless on a remote box
matplotlib.use("Agg")

# ── colour palette ────────────────────────────────────────────────────────────
# Tried tab10 first; these feel cleaner for a three-line plot.
MODEL_COLOURS = {
    "pythia-1.4b": "#1f77b4",  # mpl blue
    "pythia-410m": "#ff7f0e",  # mpl orange
    "pythia-70m":  "#2ca02c",  # mpl green
}

# Component colours for the stacked-bar breakdown
COMPONENT_COLOURS = {
    "attn_out": "#4e79a7",
    "mlp_out":  "#f28e2b",
    # resid_post is the sum – not plotted as its own bar
}

# Cosmetic constants – tweaked until they looked right on a 1280-wide figure
FIG1_SIZE  = (9, 5)
FIG2_SIZE  = (10, 5)
TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE  = 9
LEGEND_SIZE = 9
LINE_WIDTH  = 1.8
MARKER_SIZE = 5


# ── data loading ─────────────────────────────────────────────────────────────

def load_sweep_json(path: Path) -> list[dict]:
    """
    Load a layer-sweep result file.

    Each record should have at minimum:
        layer      : int
        accuracy   : float  (0–1)
        component  : str    ("resid_post" | "attn_out" | "mlp_out")

    Extra keys (e.g. AUC, std, sample counts) are ignored here but the sweep
    script keeps them for its own logging.
    """
    with open(path) as f:
        records = json.load(f)

    # Validate loosely – crash early if the format is wrong
    required = {"layer", "accuracy", "component"}
    for i, r in enumerate(records):
        missing = required - set(r.keys())
        if missing:
            raise ValueError(
                f"Record {i} in {path} is missing keys: {missing}\n"
                f"Got: {list(r.keys())}"
            )

    return records


def filter_component(records: list[dict], component: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull out (layers, accuracies) for a single residual component.
    Returns arrays sorted by layer index.
    """
    filtered = [r for r in records if r["component"] == component]
    if not filtered:
        raise ValueError(
            f"No records with component='{component}'. "
            f"Available: {set(r['component'] for r in records)}"
        )
    filtered.sort(key=lambda r: r["layer"])
    layers     = np.array([r["layer"]    for r in filtered])
    accuracies = np.array([r["accuracy"] for r in filtered])
    return layers, accuracies


def load_breakdown(records: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns {component: (layers, accuracies)} for attn_out + mlp_out.
    Used for Figure 2.
    """
    result = {}
    for comp in ("attn_out", "mlp_out"):
        try:
            result[comp] = filter_component(records, comp)
        except ValueError:
            warnings.warn(
                f"Component '{comp}' missing from breakdown data; "
                "Figure 2 will be incomplete."
            )
    return result


# ── Figure 1: accuracy vs layer, all model sizes ─────────────────────────────

def plot_accuracy_curves(
    model_data: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    chance_level: float = 0.5,
) -> None:
    """
    model_data: {model_name: (layers, accuracies)}
                layers are 0-indexed; accuracy is the resid_post probe.
    """
    fig, ax = plt.subplots(figsize=FIG1_SIZE)

    for model_name, (layers, accs) in sorted(model_data.items()):
        colour = MODEL_COLOURS.get(model_name, "#333333")

        # Smooth the curve slightly so layer-to-layer noise doesn't dominate
        # visually.  The raw values are still in the JSON.  Window=3 is tiny
        # enough that it doesn't obscure real inflection points.
        smoothed = _smooth(accs, window=3)

        ax.plot(
            layers, smoothed,
            color=colour,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=_pretty_model_name(model_name),
        )
        # Ghost line for raw (helps show how much smoothing is doing)
        ax.plot(
            layers, accs,
            color=colour,
            linewidth=0.5,
            alpha=0.25,
            linestyle="--",
        )

    # Chance-level reference
    ax.axhline(
        chance_level,
        color="#999999",
        linewidth=1.0,
        linestyle=":",
        label=f"Chance ({chance_level:.0%})",
    )

    # The ceiling line is useful: if probes never break ~0.75 it says something
    # about whether sycophancy is linearly decodable at all.
    ax.axhline(
        1.0,
        color="#cccccc",
        linewidth=0.5,
        linestyle="-",
    )

    ax.set_xlabel("Layer (residual stream after block)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Linear probe accuracy", fontsize=LABEL_SIZE)
    ax.set_title(
        "Where is sycophancy-relevant information in the residual stream?\n"
        "Linear probe accuracy vs. layer — Pythia family",
        fontsize=TITLE_SIZE,
        pad=10,
    )

    ax.set_ylim(0.35, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="lower right")

    _add_footnote(
        ax,
        "Shaded / dashed lines are raw per-layer accuracy; solid is window-3 smoothed.\n"
        "Probe trained on last-token activations; 80/20 train-test split, 5-fold CV.",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig1] saved → {out_path}")


# ── Figure 2: attn_out vs mlp_out contribution breakdown ─────────────────────

def plot_component_breakdown(
    breakdown: dict[str, tuple[np.ndarray, np.ndarray]],
    model_name: str,
    out_path: Path,
) -> None:
    """
    Stacked bar chart: for each layer, how much of the 'probe signal' comes
    from attn_out vs mlp_out?

    I'm using a simple approach: normalise each bar so attn+mlp = 1, then
    colour-code.  Absolute accuracy is shown as a line overlay on a secondary
    axis, so you can see if high-contributing layers also have high absolute
    accuracy.

    NOTE: this isn't a rigorous causal decomposition (that would need
    activation patching à la Elhage et al.).  It's just showing which
    component's activations are more linearly separable for sycophancy at
    each layer.  Caveat plastered on the plot.
    """
    if len(breakdown) < 2:
        warnings.warn("Not enough components for breakdown plot; skipping Figure 2.")
        return

    attn_layers, attn_accs = breakdown["attn_out"]
    mlp_layers,  mlp_accs  = breakdown["mlp_out"]

    # They should have the same layer indices – warn loudly if not
    if not np.array_equal(attn_layers, mlp_layers):
        warnings.warn(
            "attn_out and mlp_out have different layer indices. "
            "Aligning to intersection."
        )
        common = np.intersect1d(attn_layers, mlp_layers)
        attn_mask = np.isin(attn_layers, common)
        mlp_mask  = np.isin(mlp_layers, common)
        attn_layers, attn_accs = attn_layers[attn_mask], attn_accs[attn_mask]
        mlp_layers,  mlp_accs  = mlp_layers[mlp_mask],  mlp_accs[mlp_mask]

    layers = attn_layers  # shared after alignment
    n_layers = len(layers)
    bar_width = 0.65

    # Normalise to get fractions; guard against both being 0 (shouldn't happen)
    total = attn_accs + mlp_accs
    total = np.where(total == 0, 1.0, total)  # avoid div-by-zero
    attn_frac = attn_accs / total
    mlp_frac  = mlp_accs  / total

    fig, ax1 = plt.subplots(figsize=FIG2_SIZE)

    x = np.arange(n_layers)

    bars_attn = ax1.bar(
        x, attn_frac,
        width=bar_width,
        label="attn_out fraction",
        color=COMPONENT_COLOURS["attn_out"],
        alpha=0.85,
    )
    bars_mlp = ax1.bar(
        x, mlp_frac,
        width=bar_width,
        bottom=attn_frac,
        label="mlp_out fraction",
        color=COMPONENT_COLOURS["mlp_out"],
        alpha=0.85,
    )

    ax1.set_xlabel("Layer", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Relative probe accuracy (fraction)", fontsize=LABEL_SIZE)
    ax1.set_ylim(0, 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, fontsize=TICK_SIZE - 1)
    ax1.tick_params(axis="y", labelsize=TICK_SIZE)

    # Overlay absolute accuracy from attn_out as a sanity-check line
    ax2 = ax1.twinx()
    ax2.plot(
        x, attn_accs,
        color=COMPONENT_COLOURS["attn_out"],
        linestyle="--",
        linewidth=1.2,
        marker="s",
        markersize=4,
        alpha=0.6,
        label="attn_out accuracy (abs)",
    )
    ax2.plot(
        x, mlp_accs,
        color=COMPONENT_COLOURS["mlp_out"],
        linestyle="--",
        linewidth=1.2,
        marker="^",
        markersize=4,
        alpha=0.6,
        label="mlp_out accuracy (abs)",
    )
    ax2.set_ylabel("Absolute probe accuracy", fontsize=LABEL_SIZE, labelpad=8)
    ax2.set_ylim(0.0, 1.15)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax2.tick_params(labelsize=TICK_SIZE)

    ax1.set_title(
        f"Attention vs. MLP contribution to sycophancy probe — {_pretty_model_name(model_name)}\n"
        "(fraction of combined accuracy; dashed = absolute accuracy per component)",
        fontsize=TITLE_SIZE,
        pad=10,
    )

    # Merge legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1  + labels2,
        fontsize=LEGEND_SIZE,
        loc="upper left",
    )

    _add_footnote(
        ax1,
        "NOTE: fraction = attn_acc / (attn_acc + mlp_acc) per layer. "
        "Not a causal decomposition — just linear separability per component.",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig2] saved → {out_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Simple uniform moving average.  Edges use smaller windows (no padding).
    np.convolve with 'same' would introduce edge artefacts on short arrays.
    """
    if window <= 1 or len(arr) < window:
        return arr.copy()
    result = np.empty_like(arr, dtype=float)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        result[i] = arr[lo:hi].mean()
    return result


def _pretty_model_name(name: str) -> str:
    """
    "pythia-1.4b" → "Pythia-1.4B" etc.
    Small cosmetic thing but it looks better on the legend.
    """
    parts = name.split("-")
    # Capitalise first word, uppercase last (size suffix) if it looks like one
    out = []
    for i, p in enumerate(parts):
        if i == 0:
            out.append(p.capitalize())
        elif p[-1].isalpha() and any(c.isdigit() for c in p):
            # looks like "1.4b" or "70m"
            out.append(p.upper())
        else:
            out.append(p)
    return "-".join(out)


def _add_footnote(ax: plt.Axes, text: str) -> None:
    """Tiny grey footnote below the plot area."""
    ax.annotate(
        text,
        xy=(0, -0.14),
        xycoords="axes fraction",
        fontsize=7,
        color="#666666",
        va="top",
        ha="left",
        wrap=True,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate probe-accuracy figures from layer-sweep results."
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/probe_sweep"),
        help="Directory containing *_layer_sweep.json files.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/figures"),
        help="Where to save the PNGs.",
    )
    p.add_argument(
        "--breakdown_model",
        type=str,
        default="pythia-1.4b",
        help="Which model to use for the attn/mlp breakdown figure.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["pythia-1.4b", "pythia-410m", "pythia-70m"],
        help="Model names to include in Figure 1.",
    )
    p.add_argument(
        "--chance",
        type=float,
        default=0.5,
        help="Chance-level accuracy for the horizontal reference line.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Figure 1: accuracy vs. layer curves")
    print("=" * 60)

    model_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing_models = []

    for model_name in args.models:
        sweep_path = args.results_dir / f"{model_name}_layer_sweep.json"
        if not sweep_path.exists():
            warnings.warn(f"No sweep file found for {model_name} at {sweep_path}; skipping.")
            missing_models.append(model_name)
            continue

        records = load_sweep_json(sweep_path)
        try:
            layers, accs = filter_component(records, "resid_post")
        except ValueError as e:
            warnings.warn(f"Couldn't extract resid_post for {model_name}: {e}")
            missing_models.append(model_name)
            continue

        model_data[model_name] = (layers, accs)
        print(
            f"  {model_name}: {len(layers)} layers, "
            f"acc range [{accs.min():.3f}, {accs.max():.3f}], "
            f"peak at layer {layers[accs.argmax()]}"
        )

    if missing_models:
        print(f"\n  [warn] skipped {missing_models} — files not found")

    if model_data:
        fig1_path = args.out_dir / "probe_accuracy_by_layer.png"
        plot_accuracy_curves(model_data, fig1_path, chance_level=args.chance)
    else:
        print("[fig1] no data loaded — skipping Figure 1")

    # ── Figure 2 ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Figure 2: attn/mlp breakdown for {args.breakdown_model}")
    print("=" * 60)

    breakdown_path = args.results_dir / f"{args.breakdown_model}_layer_sweep.json"
    if not breakdown_path.exists():
        print(f"  [skip] {breakdown_path} not found")
    else:
        records   = load_sweep_json(breakdown_path)
        breakdown = load_breakdown(records)

        # Quick sanity print
        for comp, (lyr, acc) in breakdown.items():
            peak_layer = lyr[acc.argmax()]
            print(
                f"  {comp}: {len(lyr)} layers, "
                f"acc range [{acc.min():.3f}, {acc.max():.3f}], "
                f"peak at layer {peak_layer}"
            )

        fig2_path = args.out_dir / f"attn_mlp_breakdown_{args.breakdown_model}.png"
        plot_component_breakdown(breakdown, args.breakdown_model, fig2_path)

    print()
    print("Done.")


if __name__ == "__main__":
    main()