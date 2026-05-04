src/probes/linear_probe.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from dataclasses import dataclass, field
from typing import Optional
import warnings
import logging

logger = logging.getLogger(__name__)

# LogisticRegression with liblinear is fast enough for d_model <= 4096
# For bigger models (e.g. Pythia-12B) consider lbfgs with saga or just
# reduce dimensionality first with PCA. Haven't hit that wall yet.
DEFAULT_MAX_ITER = 1000
DEFAULT_C = 1.0  # regularization - 1.0 is fine for most layers


@dataclass
class ProbeResult:
    layer: int
    accuracy: float
    auc: float
    # coefficients shape: (d_model,) - useful for patching experiments later
    coef: np.ndarray
    intercept: float
    # keep fold-level scores so we can see variance, not just mean
    fold_accuracies: list[float] = field(default_factory=list)
    fold_aucs: list[float] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0
    # sometimes useful to know if a layer was trivially easy or hard
    chance_accuracy: float = 0.5

    def __repr__(self):
        return (
            f"ProbeResult(layer={self.layer}, "
            f"acc={self.accuracy:.3f}±{np.std(self.fold_accuracies):.3f}, "
            f"auc={self.auc:.3f}, "
            f"n={self.n_samples})"
        )

    def is_above_chance(self, threshold: float = 0.05) -> bool:
        # rough check - is accuracy more than `threshold` above chance?
        # not a stat test, just a quick filter for plotting
        return self.accuracy > (self.chance_accuracy + threshold)


def train_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    n_folds: int = 5,
    C: float = DEFAULT_C,
    max_iter: int = DEFAULT_MAX_ITER,
    scale_features: bool = True,
    random_seed: int = 42,
    verbose: bool = False,
) -> ProbeResult:
    """
    Train a logistic regression probe on activations from a single layer.

    activations: (n_samples, d_model) - residual stream at this layer
    labels: (n_samples,) - 1 = sycophantic completion, 0 = honest completion

    Returns ProbeResult with accuracy, AUC, and coefficient vector.

    Note: we use StratifiedKFold to handle class imbalance in the dataset.
    The sycophancy dataset I'm using (from Anthropic's sycophancy eval) is
    roughly 60/40, not terrible, but stratification is free so why not.
    """
    assert activations.ndim == 2, f"expected (n, d_model), got {activations.shape}"
    assert len(activations) == len(labels), "activations/labels length mismatch"

    n_samples, n_features = activations.shape
    n_pos = labels.sum()
    n_neg = n_samples - n_pos
    chance = max(n_pos, n_neg) / n_samples

    if verbose:
        logger.info(
            f"Layer {layer}: n={n_samples}, pos={n_pos}, neg={n_neg}, "
            f"chance={chance:.3f}, d_model={n_features}"
        )

    # StandardScaler matters a lot here - activations have very different
    # scales across layers (early layers tend to have smaller magnitude).
    # Without scaling, regularization is effectively layer-dependent which
    # would confound the layer comparison.
    scaler = StandardScaler() if scale_features else None

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_accuracies = []
    fold_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(activations, labels)):
        X_train, X_val = activations[train_idx], activations[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="liblinear",
            random_state=random_seed,
        )

        with warnings.catch_warnings():
            # liblinear occasionally warns about convergence on weird folds,
            # usually not a real problem at max_iter=1000. Log it though.
            warnings.simplefilter("error")
            try:
                clf.fit(X_train, y_train)
            except warnings.WarningMessage as e:
                logger.warning(f"Layer {layer}, fold {fold_idx}: convergence warning: {e}")
                clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        fold_accuracies.append(accuracy_score(y_val, y_pred))
        fold_aucs.append(roc_auc_score(y_val, y_prob))

    # Now fit on the full dataset to get the coefficient vector for
    # downstream patching / direction extraction experiments.
    # This is the "deploy" version of the probe - not used for eval.
    if scaler is not None:
        X_full = scaler.fit_transform(activations)
    else:
        X_full = activations

    clf_full = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="liblinear",
        random_state=random_seed,
    )
    clf_full.fit(X_full, labels)

    mean_acc = float(np.mean(fold_accuracies))
    mean_auc = float(np.mean(fold_aucs))

    result = ProbeResult(
        layer=layer,
        accuracy=mean_acc,
        auc=mean_auc,
        coef=clf_full.coef_[0].copy(),  # (d_model,)
        intercept=float(clf_full.intercept_[0]),
        fold_accuracies=fold_accuracies,
        fold_aucs=fold_aucs,
        n_samples=n_samples,
        n_features=n_features,
        chance_accuracy=float(chance),
    )

    if verbose:
        print(result)

    return result


def sweep_layers(
    all_activations: dict[int, np.ndarray],
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = DEFAULT_C,
    random_seed: int = 42,
    verbose: bool = True,
) -> list[ProbeResult]:
    """
    Run train_probe over all cached layers. all_activations is a dict
    mapping layer_index -> (n_samples, d_model) array.

    This is the main entry point for the layer-sweep experiments.
    """
    results = []
    layers = sorted(all_activations.keys())

    print(f"Sweeping {len(layers)} layers, n_samples={len(labels)}")

    for layer in layers:
        acts = all_activations[layer]
        result = train_probe(
            activations=acts,
            labels=labels,
            layer=layer,
            n_folds=n_folds,
            C=C,
            random_seed=random_seed,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            marker = "**" if result.is_above_chance(threshold=0.08) else "  "
            print(
                f"  {marker} layer {layer:3d}: "
                f"acc={result.accuracy:.3f} "
                f"(±{np.std(result.fold_accuracies):.3f}) "
                f"auc={result.auc:.3f}"
            )

    return results


def extract_probe_directions(results: list[ProbeResult]) -> dict[int, np.ndarray]:
    """
    Pull out the coefficient vectors from a sweep. These are the "sycophancy
    directions" per layer - I want to use these for activation patching later
    to see if zeroing out the direction suppresses sycophantic completions.

    NOTE: the coef vectors are in scaled space (StandardScaler was applied
    during training). For patching to work properly I need to re-scale them
    back to activation space - that's a TODO. For now, just using these
    vectors for analysis (dot products, cosine similarity across layers).
    """
    return {r.layer: r.coef for r in results}


def best_probe_layer(results: list[ProbeResult], metric: str = "accuracy") -> ProbeResult:
    """
    Find the layer with the best probe performance. Usually there's a clear
    peak somewhere in the middle-to-late layers.
    """
    assert metric in ("accuracy", "auc"), f"unknown metric: {metric}"
    return max(results, key=lambda r: getattr(r, metric))


# quick sanity check you can run directly
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Running sanity check with random data...")

    rng = np.random.default_rng(0)
    n, d = 200, 2048  # roughly Pythia-1.4B d_model

    # simulate a weak linear signal in one direction
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)

    labels = rng.integers(0, 2, size=n)
    activations = rng.standard_normal((n, d))
    # add a small signal along `direction` for positive class
    activations[labels == 1] += 0.5 * direction

    result = train_probe(activations, labels, layer=0, verbose=True)
    print(f"\nSanity check result: {result}")

    # Should be somewhere between 0.55-0.65 with this SNR
    assert result.accuracy > 0.52, f"probe failed sanity check: {result.accuracy}"
    print("Sanity check passed.")