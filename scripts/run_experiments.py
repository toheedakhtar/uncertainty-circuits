# What we are looking for, 
# We aim to understand why and where the model expresses uncertainty or refusal when faced with a fictional or unanswerable question. 
# Specifically, we want to determine if this behavior arises from specific parts of the model’s internal computations like 
# layers or attention heads—and whether it is caused by distinct circuits or superposition of information. 

# Experiments to test this, 
# -------- Layerwise Residual Difference:
# Measures how much the model’s internal residual representations differ layer-by-layer between a real factual question and a fictional/uncertain question, 
# showing where uncertainty starts to emerge in the network.

# -------- Headwise Divergence Heatmap:
# Analyzes which attention heads in each layer contribute most to the difference in model processing between the real and fictional questions, 
# helping identify specific heads responsible for uncertainty or refusal behavior.

# -------- Ablation Curve on Top Heads:
# Gradually “turns off” (ablates) the most divergent heads to see how this affects the model’s probability assigned to uncertainty-related tokens, 
# revealing if those heads drive the model’s uncertainty or refusal responses.

import os
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

RESULTS_DIR = "./result_another_more_ucwords"
os.makedirs(RESULTS_DIR, exist_ok=True)

# MODEL_NAMES = ["gpt2-small", "opt-125m"]
MODEL_NAMES = ["qwen2.5-1.5b-instruct"]

# Device / dtype configuration (use half precision on GPU to save memory)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Runtime config -> device={DEVICE}, dtype={DTYPE}")

# Define your prompts (replace with your real prompts) 
prompt_real = "What is the freezing point of water at standard atmospheric pressure?"
prompt_fake = "What is the freezing point of water on Planet Xylon, where gravity is twice that of Earth?"

# === Helpers ===

def get_uncertainty_tokens(model):
    """Get uncertainty tokens for the model's tokenizer"""
    uncertainty_words = ["unknown", "unsure", "don", "'t", "know", "can't", "unable", "impossible", 
                        "doesn't", "exist", "fictional", "imaginary", "no", "such", "thing", "hypothetical", 
                        "made", "up", "never", "happened"]
    token_ids = []
    for w in uncertainty_words:
        try:
            toks = model.to_tokens(" " + w, prepend_bos=False)[0]
            token_ids.extend(toks.tolist())
        except:
            pass
    return list(set(token_ids))

def run_model_with_cache(prompt, model):
    # Cache only the activations we actually use to reduce memory/time
    # names_filter = lambda name: (
    #     "hook_resid_post" in name or
    #     "attn.hook_z" in name
    # )
    # logits, cache = model.run_with_cache(prompt, names_filter=names_filter)
    logits, cache = model.run_with_cache(prompt)

    return logits, cache

def residual_norm_diff(cache_real, cache_fake, model):
    diffs = []
    for layer in range(model.cfg.n_layers):
        # Get activations for this layer
        real_resid = cache_real[f"blocks.{layer}.hook_resid_post"][0]  # shape: seq, d_model
        fake_resid = cache_fake[f"blocks.{layer}.hook_resid_post"][0]  # shape: seq, d_model
        
        # Handle different sequence lengths by taking minimum
        min_len = min(real_resid.shape[0], fake_resid.shape[0])
        
        # Compare only up to minimum length
        diff = torch.norm(
            real_resid[:min_len, :] - fake_resid[:min_len, :]
        ).item()
        diffs.append(diff)
    return diffs

def plot_layerwise_diff(diffs, filename):
    plt.figure()
    plt.plot(range(len(diffs)), diffs, marker='o')
    plt.title("Layerwise residual difference: real vs fake prompt")
    plt.xlabel("Layer")
    plt.ylabel("Residual norm difference")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    plt.close()

def headwise_divergence(cache_real, cache_fake, model):
    # Calculate per head difference at attention output hooks
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_diffs = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        real_heads = cache_real[f"blocks.{layer}.attn.hook_z"][0]  # shape: seq, heads, d_head
        fake_heads = cache_fake[f"blocks.{layer}.attn.hook_z"][0]
        
        # Handle different sequence lengths
        min_len = min(real_heads.shape[0], fake_heads.shape[0])
        
        for head in range(n_heads):
            diff = torch.norm(real_heads[:min_len, head, :] - fake_heads[:min_len, head, :]).item()
            head_diffs[layer, head] = diff

    return head_diffs

def plot_headwise_heatmap(head_diffs, filename):
    plt.figure(figsize=(10,6))
    plt.imshow(head_diffs.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label="Head contribution to divergence")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Head-wise divergence (real vs fake prompt)")
    plt.savefig(filename)
    plt.show()
    plt.close()

def ablate_heads(heads_to_ablate, prompt, model):
    # Organize heads by layer
    from collections import defaultdict 
    heads_by_layer = defaultdict(list)
    for (layer, head) in heads_to_ablate:
        heads_by_layer[layer].append(head)

    # print(f"heads to ablate: {heads_by_layer}")
    
    # Create hooks for each layer
    hooks = []
    for layer, heads in heads_by_layer.items():
        def make_hook(heads_to_zero):
            def hook_fn(value, hook):
                value = value.clone()
                for h in heads_to_zero:
                    value[:, :, h, :] = 0  # ← Zero before projection (hook_z)
                return value
            return hook_fn
        
        # CHANGED: Use hook_z instead of hook_attn_out
        hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(heads)))
    
    logits = model.run_with_hooks(prompt, fwd_hooks=hooks)
    return logits[0, -1]

def uncertainty_mass(logits, uncertainty_tokens):
    probs = torch.softmax(logits, dim=-1)
    return probs[uncertainty_tokens].sum().item()

# def ablation_curve(prompt, head_diffs, filename, model, uncertainty_tokens):
#     # Sort heads by descending difference
#     flat_heads = sorted(
#         [((l,h), head_diffs[l,h].item()) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
#         key=lambda x: x[1], reverse=True
#     )
#     top_heads = [x[0] for x in flat_heads]

#     masses = []
#     steps = list(range(0, len(top_heads), 5))  # ablate 0,5,10,... heads for speed
#     if steps[-1] != len(top_heads):
#         steps.append(len(top_heads))

#     for k in steps:
#         heads_to_ablate = top_heads[:k]
#         logits = ablate_heads(heads_to_ablate, prompt, model)
#         mass = uncertainty_mass(logits, uncertainty_tokens)
#         masses.append(mass)

#     plt.figure()
#     plt.plot(steps, masses, marker='o')
#     plt.xlabel("Number of top heads ablated")
#     plt.ylabel("Uncertainty token probability mass")
#     plt.title("Effect of ablating top heads on uncertainty")
#     plt.grid(True)
#     plt.savefig(filename)
#     plt.close()

def ablation_curve_focused(prompt, head_diffs, filename, model, uncertainty_tokens):
    flat_heads = sorted(
        [((l,h), head_diffs[l,h].item()) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
        key=lambda x: x[1], reverse=True
    )
    top_heads = [x[0] for x in flat_heads[:50]]  # ← Only top 50 heads
    
    masses = []
    steps = list(range(0, len(top_heads), 1))  # Ablate 1 head at a time for precision

    for k in steps:
        heads_to_ablate = top_heads[:k]
        logits = ablate_heads(heads_to_ablate, prompt, model)
        mass = uncertainty_mass(logits, uncertainty_tokens)
        masses.append(mass)

    plt.figure()
    plt.plot(steps, masses, marker='o')
    plt.xlabel("Number of top 50 heads ablated")
    plt.ylabel("Uncertainty token probability mass")
    plt.title("Effect of ablating most divergent heads")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    plt.close()

# === Run experiments ===

for model_name in MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print(f"Loading {model_name}...")
        model = HookedTransformer.from_pretrained_no_processing(model_name, device=DEVICE, dtype=DTYPE)
        
        # Get uncertainty tokens for this model
        uncertainty_tokens = get_uncertainty_tokens(model)
        print(f"Uncertainty tokens: {uncertainty_tokens}")
        
        # Create model-specific results directory
        model_results_dir = os.path.join(RESULTS_DIR, model_name.replace("/", "_"))
        os.makedirs(model_results_dir, exist_ok=True)
        
        print("Running model on real prompt...")
        logits_real, cache_real = run_model_with_cache(prompt_real, model)

        print("Running model on fake prompt...")
        logits_fake, cache_fake = run_model_with_cache(prompt_fake, model)

        print("Computing layerwise residual differences...")
        layer_diffs = residual_norm_diff(cache_real, cache_fake, model)
        plot_layerwise_diff(layer_diffs, os.path.join(model_results_dir, "layerwise_diff.png"))

        print("Computing headwise divergence heatmap...")
        head_diffs = headwise_divergence(cache_real, cache_fake, model)
        plot_headwise_heatmap(head_diffs, os.path.join(model_results_dir, "headwise_divergence.png"))

        # print("Running ablation curve...")
        # # ablation_curve(prompt_fake, head_diffs, os.path.join(model_results_dir, "ablation_curve.png"), model, uncertainty_tokens)
        # ablation_curve_focused(prompt_fake, head_diffs, os.path.join(model_results_dir, "ablation_curve.png"), model, uncertainty_tokens)

        print(f"Results for {model_name} saved to {model_results_dir}")
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue

print(f"\n{'='*60}")
print(f"All experiments completed! Results saved to {RESULTS_DIR}")
print(f"{'='*60}")
