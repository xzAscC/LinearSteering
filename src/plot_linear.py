
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

def get_model_name_for_path(model_name: str) -> str:
    return model_name.split("/")[-1]

def plot_linearity():
    # Configuration
    # Order: Qwen (Left), Gemma (Right)
    models_order = [
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
        ("google/gemma-2-2b", "gemma-2-2b")
    ]
    models_config = {
        "Qwen/Qwen3-1.7B": {"max_layers": 28, "files_name": "Qwen3-1.7B"},
        "google/gemma-2-2b": {"max_layers": 26, "files_name": "gemma-2-2b"}
    }
    
    concepts = ["evil", "optimistic", "refusal", "sycophantic", "language_en_fr_paired"]
    concept_renames = {
        "language_en_fr_paired": "translation"
    }
    
    # Style settings
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24, # Increased title size
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
    })
    
    # Define colors for concepts
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    concept_colors = {c: colors_list[i % len(colors_list)] for i, c in enumerate(concepts)}
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 1 row, 2 cols figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8)) # Wider figure for side-by-side
    
    legend_elements = [] # Collect legend items only once
    
    def load_linearity_scores(model_name, concept, vector_type, is_remove):
        """
        Robustly load linearity scores handling two storage formats:
        1. File naming: 'linearity_{c}_{t}.pt' vs 'linearity_{c}_{t}_remove.pt'
        2. Data structure: results[layer] = {'mean_score': ...} vs results[layer] = score
        """
        # Determine path
        suffix = "_remove" if is_remove else ""
        path = f"assets/linear/{model_name}/linearity_{concept}_{vector_type}{suffix}.pt"
        
        # Fallback for "remove" if specialized file doesn't exist (maybe in old format file?)
        # But for now, we assume the file structure we know.
        if not os.path.exists(path):
            # Try without suffix if it was remove but file missing? 
            # No, that would be standard data. Risk of mixing.
            return None, None

        try:
            data = torch.load(path)
            if "results" not in data:
                return None, None
                
            results = data["results"]
            layers = sorted([k for k in results.keys() if isinstance(k, (int, float, str)) and str(k).isdigit()])
            layers = [int(l) for l in layers]
            layers.sort()
            
            if not layers:
                return None, None
                
            scores = []
            for l in layers:
                val = results[l]
                if isinstance(val, dict):
                     scores.append(val.get("mean_score", 0.0))
                elif isinstance(val, (float, int)):
                     scores.append(float(val))
                elif isinstance(val, torch.Tensor):
                     scores.append(val.item())
                else:
                     scores.append(0.0)
            
            return layers, scores
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, None

    for idx, (model_full, model_name) in enumerate(models_order):
        ax = axes[idx]
        config = models_config.get(model_full)
        if not config: 
            print(f"Config not found for {model_full}")
            continue
            
        max_layers = config["max_layers"]
        has_data = False
        
        for concept in concepts:
            display_concept = concept_renames.get(concept, concept)
            color = concept_colors[concept]
            
            # 1. Standard (No Remove)
            # Random
            layers, scores = load_linearity_scores(model_name, concept, "random", is_remove=False)
            if layers:
                x_axis = [l / (max_layers - 1) * 100 for l in layers]
                ax.plot(x_axis, scores, color=color, marker='o', linestyle='--', 
                         linewidth=1.5, markersize=8, alpha=0.7)
                has_data = True
                
            # Concept
            layers, scores = load_linearity_scores(model_name, concept, "concept", is_remove=False)
            if layers:
                x_axis = [l / (max_layers - 1) * 100 for l in layers]
                ax.plot(x_axis, scores, color=color, marker='*', linestyle='-', 
                         linewidth=2, markersize=12, alpha=0.9)
                has_data = True

            # 2. Remove (Concept Removed)
            # Random
            layers, scores = load_linearity_scores(model_name, concept, "random", is_remove=True)
            if layers:
                x_axis = [l / (max_layers - 1) * 100 for l in layers]
                # Plot Remove w/ Color but Hollow markers and dotted line
                ax.plot(x_axis, scores, color=color, marker='o', linestyle=':', 
                         linewidth=1.5, markersize=8, alpha=0.6,
                         markerfacecolor='white', markeredgewidth=1.5)
                has_data = True
            
            # Concept
            layers, scores = load_linearity_scores(model_name, concept, "concept", is_remove=True)
            if layers:
                x_axis = [l / (max_layers - 1) * 100 for l in layers]
                # Plot Remove w/ Color but Hollow markers and dash-dot line
                ax.plot(x_axis, scores, color=color, marker='*', linestyle='-.', 
                         linewidth=2, markersize=12, alpha=0.6,
                         markerfacecolor='white', markeredgewidth=1.5)
                has_data = True

        if not has_data:
            print(f"No data found for {model_name}, but continuing to keep plot structure.")
            
        ax.set_xlabel("Layer Depth (%)", fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Linearity Score (Var Explained)", fontweight='bold')
        ax.set_title(f"{model_name}", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(-0.05, 1.05)

    # Build Legend (Shared)
    # 1. Concepts
    for c in sorted(concepts):
        display_c = concept_renames.get(c, c)
        legend_elements.append(Line2D([0], [0], color=concept_colors[c], lw=3, label=display_c))
        
    # 2. Types
    # Separator or just append
    legend_elements.append(Line2D([0], [0], color='w', label=' ', alpha=0)) # Spacer
    
    # Legend for Styles
    # Standard (Filled)
    legend_elements.append(Line2D([0], [0], color='black', marker='*', linestyle='-', label='Concept (Standard)', markersize=12))
    legend_elements.append(Line2D([0], [0], color='black', marker='o', linestyle='--', label='Random (Standard)', markersize=8))
    
    # Remove (Hollow)
    legend_elements.append(Line2D([0], [0], color='black', marker='*', linestyle='-.', 
                                  label='Concept (Removed)', markersize=12, markerfacecolor='white', markeredgewidth=1.5))
    legend_elements.append(Line2D([0], [0], color='black', marker='o', linestyle=':', 
                                  label='Random (Removed)', markersize=8, markerfacecolor='white', markeredgewidth=1.5))
    
    # Create legend below the figure
    # Using fig.legend instead of ax.legend
    # Adjust layout to make room
    fig.subplots_adjust(bottom=0.25)
    
    leg = fig.legend(handles=legend_elements, loc='lower center', 
                     bbox_to_anchor=(0.5, 0.02), # Position at bottom center
                     ncol=5, # Multiple columns to spread it out
                     frameon=True, framealpha=0.9,
                     borderaxespad=0.5,
                     handletextpad=0.5,
                     columnspacing=1.0)
                     
    plt.setp(leg.get_texts(), fontweight='bold')
    
    # Save combined plot
    save_path = os.path.join(output_dir, "linearity_combined.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_linearity()
