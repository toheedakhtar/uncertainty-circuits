import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import HookedTransformer


class HeadAttentionAnalyzer:
    """Analyze attention patterns of a specific head across multiple prompts."""

    def __init__(self, model_name, layer, head, results_dir="./results_head_attention"):
        self.model_name = model_name
        self.layer = layer
        self.head = head
        self.results_dir = results_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        os.makedirs(self.results_dir, exist_ok=True)
        self.out_dir = os.path.join(self.results_dir, model_name.replace("/", "_"))
        os.makedirs(self.out_dir, exist_ok=True)

        print(f"Loading {model_name}...")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device, dtype=self.dtype)
        print(f"model loaded.")

    def _get_attention_pattern(self, prompt):
        """Extract attention pattern for a single prompt."""
        tokens = self.model.to_tokens(prompt)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens)

        attn_pattern = cache[f"blocks.{self.layer}.attn.hook_pattern"]
        attn_pattern_lh = attn_pattern[0, self.head]
        token_str = self.model.to_str_tokens(tokens)

        return attn_pattern_lh, token_str

    def run(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]

        # Collect all attention patterns
        patterns = []
        for idx, prompt in enumerate(prompts):
            print(f"Computing attention for prompt {idx+1}: {prompt[:60]}...")
            attn, tokens = self._get_attention_pattern(prompt)
            patterns.append((attn, tokens, prompt))

        # Create subplot figure
        n_prompts = len(patterns)
        fig, axes = plt.subplots(1, n_prompts, figsize=(8 * n_prompts, 8))
        if n_prompts == 1:
            axes = [axes]

        fig.suptitle(f"Attention Patterns L{self.layer}H{self.head} - {self.model_name}",
                     fontsize=14, fontweight="bold")

        for idx, (attn, tokens, prompt) in enumerate(patterns):
            ax = axes[idx]
            im = ax.imshow(attn.cpu().numpy(), origin='lower', cmap='hot')
            ax.set_xlabel("Key Token Position")
            ax.set_ylabel("Query Token Position")
            ax.set_title(f"Prompt {idx+1}: {prompt[:50]}...", fontsize=10)
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            plt.colorbar(im, ax=ax, label="Attention Weight")

        plt.tight_layout()

        # Save combined plot
        fname = f"L{self.layer}H{self.head}_all_prompts.png"
        out_path = os.path.join(self.out_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved: {out_path}")

        plt.show()
        plt.close()


if __name__ == "__main__":
    analyzer = HeadAttentionAnalyzer(model_name="meta-llama/Llama-3.2-1B-Instruct", layer=11, head=3)
    prompt = ["What is the freezing point of water on Planet Xylon?", "What is freezing point of water on earth?"]
    analyzer.run(prompts=prompt)
    