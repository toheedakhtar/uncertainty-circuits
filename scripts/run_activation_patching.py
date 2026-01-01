import os
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


class ActivationPatcher:
    """Run activation patching to measure uncertainty mass shifts."""

    def __init__(
        self,
        model_name,
        prompt_real,
        prompt_fake,
        results_dir="./results_activation_patching",
        patch_type="resid",
        patch_targets=None,
    ):
        self.model_name = model_name
        self.prompt_real = prompt_real
        self.prompt_fake = prompt_fake
        self.results_dir = results_dir
        self.patch_type = patch_type  # "resid" or "head"
        self.patch_targets = patch_targets
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        os.makedirs(self.results_dir, exist_ok=True)
        self.out_dir = os.path.join(self.results_dir, model_name.replace("/", "_"))
        os.makedirs(self.out_dir, exist_ok=True)

        print(f"Loading {model_name} on {self.device} ({self.dtype})...")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device, dtype=self.dtype)
        self.uncertainty_tokens = self._get_uncertainty_tokens()
        print(f"Uncertainty vocab size: {len(self.uncertainty_tokens)}")

    def _get_uncertainty_tokens(self):
        words = [
            "unknown", "unsure", "don", "'t", "know", "can't", "unable", "impossible",
            "doesn't", "exist", "fictional", "imaginary", "no", "such", "thing",
            "hypothetical", "made", "up", "never", "happened",
        ]
        token_ids = []
        for w in words:
            try:
                toks = self.model.to_tokens(" " + w, prepend_bos=False)[0]
                token_ids.extend(toks.tolist())
            except Exception:
                continue
        return sorted(set(token_ids))

    @staticmethod
    def _uncertainty_mass(logits, token_ids):
        probs = torch.softmax(logits, dim=-1)
        return probs[token_ids].sum().item()

    def _collect_cache(self, prompt):
        tokens = self.model.to_tokens(prompt)
        if self.patch_type == "resid":
            names_filter = lambda n: "hook_resid_post" in n
        else:
            names_filter = lambda n: "attn.hook_z" in n
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens, names_filter=names_filter)
        return logits[0, -1], cache

    def _patch_layer(self, corrupted_tokens, clean_cache, layer_idx):
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        clean_resid = clean_cache[hook_name]

        def patch_fn(value, hook):
            value = value.clone()
            value[:, -1, :] = clean_resid[:, -1, :]
            return value

        with torch.no_grad():
            logits = self.model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, patch_fn)],
            )
        return logits[0, -1]

    def _patch_head(self, corrupted_tokens, clean_cache, layer_idx, head_idx):
        hook_name = f"blocks.{layer_idx}.attn.hook_z"
        clean_z = clean_cache[hook_name]

        def patch_fn(value, hook):
            value = value.clone()
            value[:, -1, head_idx, :] = clean_z[:, -1, head_idx, :]
            return value

        with torch.no_grad():
            logits = self.model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, patch_fn)],
            )
        return logits[0, -1]

    def run(self):
        logits_clean, cache_clean = self._collect_cache(self.prompt_real)
        logits_corr, _ = self._collect_cache(self.prompt_fake)

        mass_clean = self._uncertainty_mass(logits_clean, self.uncertainty_tokens)
        mass_corr = self._uncertainty_mass(logits_corr, self.uncertainty_tokens)
        print(f"Baseline (clean):  {mass_clean:.4e}")
        print(f"Baseline (corrupt): {mass_corr:.4e}")

        patched_mass = []
        pct_change = []

        corrupted_tokens = self.model.to_tokens(self.prompt_fake)

        if self.patch_type == "resid":
            layers = list(range(self.model.cfg.n_layers)) if self.patch_targets is None else self.patch_targets
            labels = [f"L{l}" for l in layers]
            for layer in layers:
                logits_patched = self._patch_layer(corrupted_tokens, cache_clean, layer)
                mass_patched = self._uncertainty_mass(logits_patched, self.uncertainty_tokens)
                patched_mass.append(mass_patched)
                delta = mass_patched - mass_corr
                pct = (delta / mass_corr * 100) if mass_corr != 0 else 0.0
                pct_change.append(pct)
                print(f"Layer {layer:02d}: patched mass={mass_patched:.4e}, pct change={pct:+.2f}%")
        else:
            if not self.patch_targets:
                raise ValueError("For head patching, provide patch_targets as list of (layer, head)")
            layers = list(range(len(self.patch_targets)))
            labels = [f"L{l}H{h}" for l, h in self.patch_targets]
            for (layer, head) in self.patch_targets:
                logits_patched = self._patch_head(corrupted_tokens, cache_clean, layer, head)
                mass_patched = self._uncertainty_mass(logits_patched, self.uncertainty_tokens)
                patched_mass.append(mass_patched)
                delta = mass_patched - mass_corr
                pct = (delta / mass_corr * 100) if mass_corr != 0 else 0.0
                pct_change.append(pct)
                print(f"L{layer}H{head}: patched mass={mass_patched:.4e}, pct change={pct:+.2f}%")

        self._plot(labels, patched_mass, pct_change, mass_corr)

        return {
            "mass_clean": mass_clean,
            "mass_corrupt": mass_corr,
            "patched_mass": patched_mass,
            "pct_change": pct_change,
            "out_dir": self.out_dir,
        }

    def _plot(self, labels, patched_mass, pct_change, mass_corr):
        x = list(range(len(labels)))
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Activation Patching: {self.model_name}", fontweight="bold")

        axes[0].plot(x, [mass_corr] * len(x), label="Baseline (corrupt)", color="red", linestyle="--")
        axes[0].plot(x, patched_mass, label="Patched (clean)", color="blue")
        axes[0].set_ylabel("Uncertainty mass")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].bar(x, pct_change, color=["green" if p < 0 else "orange" for p in pct_change])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_xlabel("Patch target")
        axes[1].set_ylabel("% change vs corrupt")
        axes[1].grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(self.out_dir, "activation_patching.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    PROMPT_REAL = "What is the freezing point of water at standard atmospheric pressure?"
    PROMPT_FAKE = "What is the freezing point of water on Planet Xylon, where gravity is twice that of Earth?"
    # Example 1: residual stream patching for all layers
    # patcher = ActivationPatcher(MODEL_NAME, PROMPT_REAL, PROMPT_FAKE, patch_type="resid")

    # Example 2: head-level patching for specific heads
    patch_targets = [(11, 10)]  # list of (layer, head)
    patcher = ActivationPatcher(
        MODEL_NAME,
        PROMPT_REAL,
        PROMPT_FAKE,
        patch_type="head",
        patch_targets=patch_targets,
    )
    patcher.run()
