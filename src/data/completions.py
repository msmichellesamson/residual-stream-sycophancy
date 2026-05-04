src/data/completions.py

import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm

from sycophancy_dataset import SycophancyDataset, PromptVariant


# where to dump everything
OUTPUT_DIR = Path("data/completions")
ACTIVATIONS_DIR = Path("data/activations")

# these are the models i'm testing across - starting small, scaling if results hold
TARGET_MODELS = [
    "pythia-1.4b",
    # "pythia-2.8b",   # uncomment if you have the vram
    # "gpt2",          # useful sanity check - known to be more sycophantic than pythia
]

# only care about the final input token's residual stream state
# the hypothesis is: by the time we decode the next token, the "decision" 
# about whether to validate or push back is already baked into the residual stream
# at some layer. let's see if a linear classifier can find it.
HOOK_NAME_TEMPLATE = "blocks.{layer}.hook_resid_post"


@dataclass
class CompletionRecord:
    prompt_id: str
    variant: str          # "honest" | "sycophantic" | "baseline"
    model_name: str
    prompt_text: str
    completion: str
    token_ids: list[int]
    label: int            # 0 = honest/baseline, 1 = sycophantic
    generation_time_s: float
    device: str


@dataclass
class ActivationSnapshot:
    prompt_id: str
    variant: str
    model_name: str
    label: int
    # shape: (n_layers, d_model) - residual stream at final token, each layer
    # saved as list-of-lists for json serializability (i'll convert to np later)
    residual_stream: list   # will be list[list[float]]
    n_layers: int
    d_model: int


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("No GPU found, using CPU - this will be slow for anything bigger than pythia-160m")
    return device


def load_model(model_name: str, device: str) -> HookedTransformer:
    print(f"\nLoading {model_name}...")
    # TransformerLens uses EleutherAI prefix for pythia models
    tl_name = f"EleutherAI/{model_name}" if "pythia" in model_name else model_name
    
    model = HookedTransformer.from_pretrained(
        tl_name,
        center_writing_weights=False,   # keep this off - we want raw activations
        center_unembed=False,
        fold_ln=False,                  # don't fold layernorm - want pre-norm residual
        device=device,
    )
    model.eval()
    
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  d_model: {model.cfg.d_model}")
    print(f"  n_heads: {model.cfg.n_heads}")
    return model


def run_greedy_decoding(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 80,
    device: str = "cpu",
) -> tuple[str, list[int]]:
    """
    Simple greedy decoding - not sampling, not beam search.
    For this task we want deterministic outputs so we can compare across runs.
    The sycophancy signal should be strong enough to show up in greedy decoding.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
    input_len = tokens.shape[1]
    
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)  # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1, :]
            next_token = next_token_logits.argmax(dim=-1).item()
            generated_ids.append(next_token)
            
            # stop on EOS or common end markers
            # NOTE: pythia doesn't have a clean EOS story, so also stop on newline-heavy outputs
            if next_token == model.tokenizer.eos_token_id:
                break
            
            next_token_tensor = torch.tensor([[next_token]], device=device)
            tokens = torch.cat([tokens, next_token_tensor], dim=1)
            
            # hard cap at 512 total tokens to avoid OOM on longer prompts
            if tokens.shape[1] > 512:
                break
    
    completion = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completion, generated_ids


def extract_residual_stream(
    model: HookedTransformer,
    prompt: str,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run a forward pass with hooks to collect residual stream at every layer,
    specifically at the *final input token position*.
    
    Returns: shape (n_layers, d_model)
    
    Why the final input token? The model's "decision" about what to generate next
    is encoded in the residual stream at that position. Earlier positions have
    already been processed and their contribution is folded in via attention.
    This is the standard approach for probing - see Geva et al. 2022 on the
    residual stream as a communication channel.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
    final_token_pos = tokens.shape[1] - 1
    
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    # collect hooks for all layers
    residual_states = {}
    
    def make_hook(layer_idx):
        def hook_fn(value: torch.Tensor, hook: HookPoint):
            # value shape: (batch, seq_len, d_model)
            # grab only the final position, detach from graph, move to cpu immediately
            # to avoid accumulating GPU memory across layers
            residual_states[layer_idx] = value[0, final_token_pos, :].detach().cpu().float().numpy()
        return hook_fn
    
    hook_list = [
        (HOOK_NAME_TEMPLATE.format(layer=i), make_hook(i))
        for i in range(n_layers)
    ]
    
    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            fwd_hooks=hook_list,
            return_type=None,   # don't need logits here
        )
    
    # stack into (n_layers, d_model)
    stacked = np.stack([residual_states[i] for i in range(n_layers)], axis=0)
    return stacked


def process_dataset(
    model: HookedTransformer,
    dataset: SycophancyDataset,
    model_name: str,
    device: str,
    max_new_tokens: int = 80,
) -> tuple[list[CompletionRecord], list[ActivationSnapshot]]:
    
    completions = []
    snapshots = []
    
    # NOTE: this loop is the slow part - each item needs two forward passes
    # (one with hooks for activations, one for generation). Could optimize
    # by caching the prompt's KV and only running the generation from there,
    # but let's get correctness first.
    
    items = dataset.get_all_items()
    print(f"\nProcessing {len(items)} prompt variants...")
    
    for item in tqdm(items, desc=f"  {model_name}"):
        prompt_text = item.prompt_text
        variant = item.variant
        label = item.label
        
        # --- generation pass ---
        t0 = time.time()
        try:
            completion, token_ids = run_greedy_decoding(
                model, prompt_text, max_new_tokens=max_new_tokens, device=device
            )
        except Exception as e:
            # TODO: handle this more gracefully - for now just skip and note it
            print(f"\n  WARNING: generation failed for {item.prompt_id}/{variant}: {e}")
            continue
        gen_time = time.time() - t0
        
        completions.append(CompletionRecord(
            prompt_id=item.prompt_id,
            variant=variant,
            model_name=model_name,
            prompt_text=prompt_text,
            completion=completion,
            token_ids=token_ids,
            label=label,
            generation_time_s=gen_time,
            device=device,
        ))
        
        # --- activation extraction pass ---
        try:
            resid = extract_residual_stream(model, prompt_text, device=device)
        except Exception as e:
            print(f"\n  WARNING: activation extraction failed for {item.prompt_id}/{variant}: {e}")
            continue
        
        snapshots.append(ActivationSnapshot(
            prompt_id=item.prompt_id,
            variant=variant,
            model_name=model_name,
            label=label,
            residual_stream=resid.tolist(),   # convert for json
            n_layers=resid.shape[0],
            d_model=resid.shape[1],
        ))
        
        # sanity print every 20 items so we can see it's working
        if len(completions) % 20 == 0:
            print(f"\n  [sample] {item.prompt_id} | variant={variant} | label={label}")
            print(f"  prompt: {prompt_text[:80]}...")
            print(f"  completion: {completion[:100]}...")
    
    return completions, snapshots


def save_completions(records: list[CompletionRecord], model_name: str):
    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "completions.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    
    print(f"  Saved {len(records)} completions to {out_path}")
    
    # also do a quick sanity check on label distribution
    labels = [r.label for r in records]
    n_syco = sum(labels)
    n_honest = len(labels) - n_syco
    print(f"  Label distribution: {n_honest} honest/baseline, {n_syco} sycophantic")


def save_activations(snapshots: list[ActivationSnapshot], model_name: str):
    out_dir = ACTIVATIONS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # save metadata as json (small)
    meta = [
        {
            "prompt_id": s.prompt_id,
            "variant": s.variant,
            "label": s.label,
            "n_layers": s.n_layers,
            "d_model": s.d_model,
        }
        for s in snapshots
    ]
    with open(out_dir / "activation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # save the actual arrays as a single npz - much faster to load than json
    # shape will be (n_samples, n_layers, d_model)
    arrays = np.array([np.array(s.residual_stream) for s in snapshots])
    labels = np.array([s.label for s in snapshots])
    
    np.savez_compressed(
        out_dir / "residual_streams.npz",
        activations=arrays,
        labels=labels,
    )
    
    print(f"  Saved activations: {arrays.shape} -> {out_dir / 'residual_streams.npz'}")
    print(f"  File size: {(out_dir / 'residual_streams.npz').stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Run completions + activation extraction")
    parser.add_argument("--models", nargs="+", default=TARGET_MODELS,
                        help="Model names to run (TransformerLens compatible)")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-path", type=str, default="data/sycophancy_prompts.json",
                        help="Path to the labeled prompt dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only first 5 items per model - sanity check")
    args = parser.parse_args()
    
    # reproducibility - though greedy decoding is already deterministic,
    # this matters for anything we do with the activations downstream
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = get_device()
    
    # load the labeled dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    dataset = SycophancyDataset(args.dataset_path)
    print(f"  {len(dataset)} unique prompts, {dataset.n_variants()} variants each")
    
    if args.dry_run:
        print("  DRY RUN: limiting to first 5 items")
        dataset = dataset.subset(5)
    
    all_start = time.time()
    
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        
        model = load_model(model_name, device)
        
        completions, snapshots = process_dataset(
            model=model,
            dataset=dataset,
            model_name=model_name,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        
        print(f"\nSaving results for {model_name}...")
        save_completions(completions, model_name)
        save_activations(snapshots, model_name)
        
        # free VRAM before loading the next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  Freed VRAM. Current usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    elapsed = time.time() - all_start
    print(f"\nDone. Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Completions in: {OUTPUT_DIR}")
    print(f"Activations in: {ACTIVATIONS_DIR}")
    
    # remind myself what to run next
    print("\nNext: run src/probing/train_probes.py to fit linear classifiers on these activations")


if __name__ == "__main__":
    main()