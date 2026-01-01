# Targeted Ablation Experiment (Class-based)
# Tests specific heads to understand their causal role in uncertainty behavior

import os
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from collections import defaultdict

class HeadAblation:
    """Performs targeted head ablation experiments to measure causal effects on uncertainty tokens."""
    
    def __init__(self, model_name, prompt_real, prompt_fake, results_dir="./results_targeted_ablation"):
        """Initialize the ablation experiment.
        
        Args:
            model_name (str): Model identifier (e.g., 'gpt2-small', 'opt-125m')
            prompt_real (str): Factual prompt
            prompt_fake (str): Fictional/unanswerable prompt
            results_dir (str): Base directory for saving results
        """
        self.model_name = model_name
        self.prompt_real = prompt_real
        self.prompt_fake = prompt_fake
        self.results_dir = results_dir
        
        # Setup device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        os.makedirs(self.results_dir, exist_ok=True)
        self.model_dir = os.path.join(self.results_dir, model_name.replace("/", "_"))
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load model
        print(f"Loading {model_name}...")
        self.model = HookedTransformer.from_pretrained(
            model_name, device=self.device, dtype=self.dtype
        )
        
        # Get uncertainty tokens
        self.uncertainty_tokens = self._get_uncertainty_tokens()
        print(f"  Uncertainty tokens: {len(self.uncertainty_tokens)} found")
    
    def _get_uncertainty_tokens(self):
        """Get uncertainty tokens for the model's tokenizer."""
        uncertainty_words = [
            "unknown", "unsure", "don", "'t", "know", "can't", "unable", "impossible",
            "doesn't", "exist", "fictional", "imaginary", "no", "such", "thing",
            "hypothetical", "made", "up", "never", "happened"
        ]
        token_ids = []
        for w in uncertainty_words:
            try:
                toks = self.model.to_tokens(" " + w, prepend_bos=False)[0]
                token_ids.extend(toks.tolist())
            except Exception:
                pass
        return sorted(list(set(token_ids)))
    
    def _ablate_heads(self, heads_to_ablate, prompt):
        """Ablate specific heads and return logits."""
        heads_by_layer = defaultdict(list)
        for (layer, head) in heads_to_ablate:
            heads_by_layer[layer].append(head)
        
        hooks = []
        for layer, heads in heads_by_layer.items():
            def make_hook(heads_to_zero):
                def hook_fn(value, hook):
                    value = value.clone()
                    for h in heads_to_zero:
                        value[:, :, h, :] = 0
                    return value
                return hook_fn
            
            hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(heads)))
        
        logits = self.model.run_with_hooks(prompt, fwd_hooks=hooks)
        return logits[0, -1]
    
    def _uncertainty_mass(self, logits):
        """Compute probability mass on uncertainty tokens."""
        probs = torch.softmax(logits, dim=-1)
        return probs[self.uncertainty_tokens].sum().item()
    
    def _test_single_head(self, layer, head, prompt):
        """Test ablating a single head on a given prompt."""
        # Baseline (no ablation)
        logits_baseline = self.model(self.model.to_tokens(prompt))[0, -1]
        mass_baseline = self._uncertainty_mass(logits_baseline)
        
        # Ablate the head
        logits_ablated = self._ablate_heads([(layer, head)], prompt)
        mass_ablated = self._uncertainty_mass(logits_ablated)
        
        # Calculate change
        change = mass_ablated - mass_baseline
        percent_change = (change / mass_baseline * 100) if mass_baseline != 0 else 0
        
        return {
            "baseline": mass_baseline,
            "ablated": mass_ablated,
            "change": change,
            "percent_change": percent_change
        }

    
    def _plot_results(self, results_real, results_fake):
        """Generate comprehensive visualization."""
        sorted_heads = sorted(results_real.keys(), key=lambda x: (x[0], x[1]))
        short_labels = [f"L{l}H{h}" for l, h in sorted_heads]
        
        real_changes = [results_real[k]["change"] for k in sorted_heads]
        fake_changes = [results_fake[k]["change"] for k in sorted_heads]
        
        # FIXED: Consistent ΔΔ calculation (fake - real)
        # Negative ΔΔ means fake dropped more (causal circuit)
        delta_delta = [fake_changes[i] - real_changes[i] for i in range(len(sorted_heads))]
        
        real_baselines = [results_real[k]["baseline"] for k in sorted_heads]
        fake_baselines = [results_fake[k]["baseline"] for k in sorted_heads]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Causal Uncertainty Circuit Analysis - {self.model_name}", 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Absolute Change Comparison
        ax = axes[0, 0]
        x = range(len(sorted_heads))
        width = 0.35
        ax.bar([i - width/2 for i in x], real_changes, width, label='Real', 
               alpha=0.8, color='steelblue')
        ax.bar([i + width/2 for i in x], fake_changes, width, label='Fake', 
               alpha=0.8, color='coral')
        ax.set_ylabel('Change in Uncertainty Mass', fontweight='bold')
        ax.set_title('Absolute Change: Real vs Fake Prompts', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Differential Effect (ΔΔ)
        ax = axes[0, 1]
        # FIXED: Color coding (green for negative ΔΔ = causal circuit)
        colors = ['green' if dd < 0 else 'red' for dd in delta_delta]
        ax.bar(short_labels, delta_delta, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=1.5)
        ax.set_ylabel('ΔΔ (Fake Change - Real Change)', fontweight='bold')
        ax.set_title('Differential Effect on Uncertainty', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(short_labels, rotation=45, ha='right')
        
        # FIXED: Updated annotation with correct interpretation
        ax.text(0.02, 0.98, 
                'ΔΔ = (fake change − real change)\n'
                'Negative ΔΔ = stronger uncertainty drop on fake prompts (causal circuit)\n'
                'Positive ΔΔ = stronger uncertainty drop on real prompts\n\n'
                'Change < 0 ⇒ ablation increases uncertainty (head suppresses it)\n'
                'Change > 0 ⇒ ablation decreases uncertainty (head generates it)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Plot 3: Baseline Comparison
        ax = axes[1, 0]
        x = range(len(sorted_heads))
        width = 0.35
        ax.bar([i - width/2 for i in x], real_baselines, width, label='Real Baseline', 
               alpha=0.8, color='lightblue')
        ax.bar([i + width/2 for i in x], fake_baselines, width, label='Fake Baseline', 
               alpha=0.8, color='lightcoral')
        ax.set_ylabel('Uncertainty Mass', fontweight='bold')
        ax.set_title('Baseline Uncertainty: Real vs Fake', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary Table
        ax = axes[1, 1]
        ax.axis('off')
        table_data = [['Head', 'Prompt', 'Baseline', 'Ablated', 'Change', '% Chg', 'ΔΔ']]
        
        for i, (layer, head) in enumerate(sorted_heads):
            r_real = results_real[(layer, head)]
            r_fake = results_fake[(layer, head)]
            
            table_data.append([
                f'L{layer}H{head}', 'Real',
                f"{r_real['baseline']:.2e}",
                f"{r_real['ablated']:.2e}",
                f"{r_real['change']:+.2e}",
                f"{r_real['percent_change']:+.1f}%", ''
            ])
            table_data.append([
                '', 'Fake',
                f"{r_fake['baseline']:.2e}",
                f"{r_fake['ablated']:.2e}",
                f"{r_fake['change']:+.2e}",
                f"{r_fake['percent_change']:+.1f}%",
                f"{delta_delta[i]:+.2e}"
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.12, 0.10, 0.14, 0.14, 0.14, 0.12, 0.14])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(table_data)):
            pair_idx = (i - 1) // 2
            color = '#f0f0f0' if pair_idx % 2 == 0 else 'white'
            for j in range(7):
                table[(i, j)].set_facecolor(color)
                # FIXED: Green for negative ΔΔ (causal circuit)
                if j == 6 and table_data[i][6]:
                    val = float(table_data[i][6])
                    table[(i, j)].set_facecolor('#90EE90' if val < 0 else '#FFB6C6')
        
        plt.tight_layout()
        out_path = os.path.join(self.model_dir, "single_head_effects.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
    
    def _print_summary(self, results_real, results_fake):
        """Print summary table to console."""
        print(f"\n{'='*90}")
        print(f"SUMMARY - {self.model_name}")
        print(f"{'='*90}")
        print(f"{'Head':<10} {'Prompt':<8} {'Baseline':<12} {'Ablated':<12} {'Change':<12} {'% Chg':<10} {'ΔΔ':<12}")
        print(f"{'-'*90}")
        
        sorted_heads = sorted(results_real.keys(), key=lambda x: (x[0], x[1]))
        for layer, head in sorted_heads:
            r_real = results_real[(layer, head)]
            r_fake = results_fake[(layer, head)]
            # FIXED: Consistent calculation (fake - real)
            delta_delta = r_fake['change'] - r_real['change']
            
            print(f"L{layer}H{head:<7} Real     {r_real['baseline']:<12.2e} {r_real['ablated']:<12.2e} "
                  f"{r_real['change']:+12.2e} {r_real['percent_change']:+10.1f}%")
            print(f"{'':10} Fake     {r_fake['baseline']:<12.2e} {r_fake['ablated']:<12.2e} "
                  f"{r_fake['change']:+12.2e} {r_fake['percent_change']:+10.1f}% {delta_delta:+12.2e}")
            print()
        
        print(f"{'='*90}")
        print("Note: Negative ΔΔ indicates stronger uncertainty reduction on fake prompts")
        print("(evidence of causal uncertainty circuit)")
        print(f"{'='*90}\n")
    
    def run(self, target_heads):
        """Run the ablation experiment on specified heads.
        
        Args:
            target_heads (list): List of (layer, head) tuples to test
        
        Returns:
            dict: Results for both prompts
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {self.model_name}")
        print(f"Testing {len(target_heads)} heads: {target_heads}")
        print(f"{'='*60}")
        
        results_real = {}
        results_fake = {}
        
        for layer, head in target_heads:
            print(f"\n  L{layer}H{head}:")
            
            # Test on real prompt
            print(f"    Real prompt...", end=" ")
            result_real = self._test_single_head(layer, head, self.prompt_real)
            results_real[(layer, head)] = result_real
            print(f"✓ (Δ={result_real['change']:+.2e})")
            
            # Test on fake prompt
            print(f"    Fake prompt...", end=" ")
            result_fake = self._test_single_head(layer, head, self.prompt_fake)
            results_fake[(layer, head)] = result_fake
            print(f"✓ (Δ={result_fake['change']:+.2e})")
            
            # FIXED: Consistent differential effect calculation
            delta_delta = result_fake['change'] - result_real['change']
            # FIXED: Correct marker logic (negative ΔΔ = causal circuit)
            marker = "✓" if delta_delta < 0 else "✗"
            print(f"    ΔΔ: {delta_delta:+.2e} {marker}")
        
        # Generate plots and summaries
        print(f"\nGenerating visualizations...")
        self._plot_results(results_real, results_fake)
        self._print_summary(results_real, results_fake)
        
        # Cleanup
        del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results_real, results_fake


# === Example Usage ===
if __name__ == "__main__":
    
    # Configuration
    MODEL_CONFIG = {
        # heuristics from logit lens, peak delta
        "meta-llama/Llama-3.2-1B-Instruct": [
            (11, 0),
            (11, 3),
            (14,15),
            (14,22)
        ]
    }
    
    # Prompts
    prompt_real = "What is the freezing point of water at standard atmospheric pressure?"
    prompt_fake = "What is the freezing point of water on Planet Xylon, where gravity is twice that of Earth?"
    
    # Run experiments
    for model_name, target_heads in MODEL_CONFIG.items():
        try:
            exp = HeadAblation(model_name, prompt_real, prompt_fake)
            exp.run(target_heads)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("All experiments completed!")