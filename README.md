# Tracking Sycophancy in the Residual Stream: A Layer-by-Layer Probe on Pythia

> At what layer does a transformer "decide" to agree with you instead of being correct?

## The question I'm exploring

Sycophancy in language models — the tendency to tell users what they want to hear rather than what's true — has been documented at the behavioral level for a while now. But the mechanistic story is murkier. [Perez et al. (2022)](https://arxiv.org/abs/2212.09251) showed that RLHF-trained models reliably shift their stated positions under social pressure, and [Anthropic's model card work](https://www.anthropic.com/research/sycophancy-to-subterfuge-investigating-reward-tampering-in-language-models) has made it a first-class safety concern. What I want to understand is: if sycophancy is a representational phenomenon, not just a behavioral one, can I find it in the residual stream? Specifically, is there a layer — or a transition between layers — where the model's internal state becomes predictably different between "I'll tell you the truth" and "I'll tell you what you want"?

Going in, I know that linear probes on residual stream activations have worked for other semantic distinctions (e.g., [Burns et al.'s contrast-consistent probing](https://arxiv.org/abs/2212.03827)), but it's not obvious sycophancy is represented linearly, or that it's localized enough to find with a 50-sample probe. I'm also using Pythia — a base model with no RLHF — which is a deliberate choice I'll explain below, and which makes the interpretation noisier.

## Why I care

The SRE part of my brain thinks about failure modes. In production systems, the worst failures aren't the loud crashes — they're the quiet ones where the system keeps running but is subtly doing the wrong thing. Sycophancy is that failure mode for AI: the model looks fine, it's fluent, it's confident, and it's drifting toward whatever the user wants to believe. That's not a safety failure in a dramatic sense; it's a reliability failure in a much more mundane and pervasive sense.

The interpretability angle matters to me because behavioral evals can only catch sycophancy when it surfaces. If the representation is already there — if by layer 10 the model has "decided" to capitulate — then you could, in principle, detect or intervene before the output ever lands. That's the thing worth chasing: can you build a monitor that lives inside the forward pass, not downstream of it?

## What's in here

**Data construction**
- `src/data/sycophancy_dataset.py` — builds the 80-pair stimulus set. Paired prompts: a neutral version asking for a factual judgment, and a socially-pressured variant where the user has signaled a preferred answer (e.g., "I really believe X, am I right?" framing). Three scenario types: factual disagreement, preference validation, and opinion reversal. Labels are 0 (neutral) / 1 (socially pressured).
- `src/data/completions.py` — runs greedy decoding on each prompt across the target model(s), and uses TransformerLens hooks to snapshot the residual stream activation on the final input token at every layer. Saves completions and activations separately so I don't have to re-run inference every time I tweak the probe.

**Probes and sweeps**
- `src/probes/linear_probe.py` — thin sklearn wrapper that trains a logistic regression on a layer's activations and returns accuracy and the coefficient vector. Nothing fancy; the point is to test whether the information is linearly decodable, not to build the best classifier.
- `src/probes/layer_sweep.py` — loops over all layers for a given model and activation type (residual stream / attention output / MLP output), runs the probe at each, and returns a tidy DataFrame. This is where most of the experiment time actually lives.
- `src/experiments/run_probe_sweep.py` — main entry point. Accepts `--model`, `--activation-type`, and `--n-samples` flags. Runs data collection and the full sweep, writes results to `results/raw/`.

**Visualization**
- `src/viz/plot_layer_curves.py` — two charts: (1) probe accuracy vs. layer across all three Pythia sizes on one figure, (2) attention-out vs. MLP-out breakdown to see which component is carrying the signal. Both save to `results/figures/`.

**Analysis and scratchpad**
- `notebooks/exploration.ipynb` — where I sanity-checked activation shapes, looked at individual completions to verify my labels weren't wrong, and ran one-off probes before committing to the full sweep. Messy in the expected ways.
- `results/findings.md` — the actual research note with charts embedded inline. This is the thing to read if you want to understand what the experiments showed rather than how the code works.

No Dockerfile, no deployment infrastructure — this is a local experiments repo, runs on a single GPU or CPU if the models are small enough.

## What I'm finding (so far)

- **The signal is there, but it's layer-dependent in a non-obvious way.** For `pythia-1b`, probe accuracy on residual stream activations is near chance (52-54%) in the early layers, rises through the middle, and peaks somewhere in the upper-middle third of the network. I expected either a monotonic rise or a late-layer spike; what I'm seeing is more of a plateau with a hump, which I don't have a clean story for yet.

- **MLP outputs seem to carry more of the signal than attention outputs at the layers where accuracy peaks.** This is tentative — the probe is small and the dataset is small — but the attention-out curves are noisier and peak later. I'm not sure whether that means something real about how sycophancy is computed, or whether it's a sample size artifact.

- **The base model (no RLHF) still shows decodable sycophancy signal.** I expected noise. What I'm seeing is that even without instruction tuning, the socially-pressured prompts leave a different residual stream fingerprint than the neutral ones. This might just mean the probe is picking up on surface-level prompt differences (presence of first-person hedging, sentence length, etc.) rather than anything like "intent to please." I haven't ruled that out.

- **Controlling for prompt length is harder than I expected.** The socially-pressured variants are systematically longer because the social framing takes words. I'm probing on the final input token, which might be doing different positional work across the two prompt types. I've been trying to construct matched-length pairs but it's awkward and I don't fully trust my matching yet.

- **80 pairs is probably not enough.** I can get probe accuracy numbers in the 60-70% range at peak layers, which feels meaningful but the confidence intervals are wide enough that I'm reluctant to make strong claims. The experiment is a signal-detection exercise at this point, not a measurement.

- **Pythia-160m is mostly noise; pythia-410m starts to show the pattern.** The very small model might just not have enough capacity to represent the distinction internally in a linear-probe-detectable way, or the 80-pair dataset is genuinely too small for that model size. I haven't been able to separate those explanations.

## What I'd do next

- **Extend to an RLHF-trained model.** The most important comparison I don't have is: does the same probe, trained on the same data, work better or worse on an instruction-tuned model? If the signal is stronger post-RLHF, that would support the hypothesis that training to be agreeable leaves interpretable residual stream traces. This would need a model where I can access activations (e.g., via TransformerLens's [Pythia-RLHF port](https://huggingface.co/EleutherAI) or a comparable open model), not just API access.

- **Use a richer sycophancy dataset.** The 80-pair hand-labeled set has too many confounders. The [Anthropic sycophancy evaluation suite](https://github.com/anthropics/evals) or the dataset from the [Sharma et al. paper](https://arxiv.org/abs/2310.13548) would let me run the same probe sweep with much higher confidence that I'm actually measuring sycophancy and not "socially pressured phrasing."

- **Try probing on intermediate tokens, not just the final one.** Probing the final input token is a choice I made for simplicity, but it's not obvious that the decision signal would peak there. Looking at the token right before the first output token, or averaging over the social-pressure tokens specifically, might find stronger or more interpretable signal.

- **Causal intervention to test the probe's relevance.** Finding that a probe can linearly decode a direction doesn't tell you that direction matters to the behavior. The next honest step is an activation patching or steering experiment: patch the "sycophantic direction" into a prompt that would otherwise produce a truthful output and see if the model actually capitulates. [Zou et al. (2023)](https://arxiv.org/abs/2310.01405) have a clean methodology for this kind of representation engineering.

- **Sparse probing to find attending heads.** The layer-sweep tells me where; it doesn't tell me what computational units are responsible. Running attention head attribution (e.g., through TransformerLens's [activation patching utilities](https://github.com/neelnanda-io/TransformerLens)) on the high-signal layers might narrow it to specific heads, which would be a more actionable finding.

## Status

The data pipeline, probe training, and layer sweep all run end-to-end. I've done full sweeps on `pythia-160m`, `pythia-410m`, and `pythia-1b`. The charts in `results/figures/` are real outputs from those runs. The findings in `results/findings.md` reflect what those runs actually showed.

What's still synthetic: the dataset itself. Eighty hand-constructed pairs is enough to explore the methodology but not enough to trust the quantitative claims at face value. The probe accuracy numbers I'm seeing are suggestive, not conclusive. The confound around prompt length and phrasing differences between neutral/pressured variants is real and unresolved. Treat this as a methodology exploration and a replication target, not a result.

## References

- [Perez et al., "Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models" (2022)](https://arxiv.org/abs/2212.09251) — behavioral characterization of sycophancy in RLHF models; primary motivation for this project
- [Sharma et al., "Towards Understanding Sycophancy in Language Models" (2023)](https://arxiv.org/abs/2310.13548) — structured evaluation of sycophancy across model families; the dataset I want to plug in next
- [Burns et al., "Discovering Latent Knowledge in Language Models Without Supervision" (2022)](https://arxiv.org/abs/2212.03827) — the contrast-consistent probing methodology that this work is most directly building on
- [Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (2023)](https://arxiv.org/abs/2310.01405) — the activation patching / steering approach I want to try as a follow-up
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — Neel Nanda's mechanistic interpretability library; the hook infrastructure that makes this entire approach tractable
- [EleutherAI Pythia model suite](https://github.com/EleutherAI/pythia) — Black et al. (2023); the model family I'm running experiments on