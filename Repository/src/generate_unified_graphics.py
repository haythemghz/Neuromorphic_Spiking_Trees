import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for publication
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

def plot_pareto_frontiers(df, output_path='unified_pareto.png'):
    """Plot Accuracy vs Latency for all datasets and models."""
    plt.figure(figsize=(12, 7))
    # Aggregate over folds
    agg_df = df.groupby(['Dataset', 'Model', 'Lambda']).agg({
        'Accuracy': ['mean', 'std'],
        'Latency': ['mean', 'std']
    }).reset_index()
    agg_df.columns = ['Dataset', 'Model', 'Lambda', 'Acc_Mean', 'Acc_Std', 'Lat_Mean', 'Lat_Std']

    sns.scatterplot(data=agg_df, x='Lat_Mean', y='Acc_Mean', hue='Dataset', style='Model', s=100)
    
    # Add error bars (optional, can be messy)
    # plt.errorbar(agg_df['Lat_Mean'], agg_df['Acc_Mean'], xerr=agg_df['Lat_Std'], yerr=agg_df['Acc_Std'], fmt='none', alpha=0.3)

    plt.title("Accuracy-Latency Pareto Frontiers (Unified Benchmark)")
    plt.xlabel("Mean Inference Latency (Time Steps)")
    plt.ylabel("Mean Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {output_path}")

def plot_lambda_impact_dual(df, output_path='unified_lambda_impact.png'):
    """Dual-axis plots for key datasets showing Accuracy/Latency vs Lambda."""
    datasets = df['Dataset'].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(int(np.ceil(n_ds/2)), 2, figsize=(15, 5 * np.ceil(n_ds/2)))
    axes = axes.flatten()

    for i, ds in enumerate(datasets):
        ax1 = axes[i]
        subset = df[(df['Dataset'] == ds) & (df['Model'] == 'NST')]
        subset = subset.groupby('Lambda').mean(numeric_only=True).reset_index()
        
        color = 'tab:blue'
        ax1.set_xlabel('Regularization Lambda')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(subset['Lambda'], subset['Accuracy'], marker='o', color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title(f"Lambda Sensitivity: {ds}")

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Latency', color=color)
        ax2.plot(subset['Lambda'], subset['Latency'], marker='s', linestyle='--', color=color, label='Latency')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {output_path}")

def plot_structural_parsimony(df, output_path='unified_depth_trend.png'):
    """Plot Tree Depth vs Lambda to prove interpretability via sparsity."""
    plt.figure(figsize=(10, 6))
    agg_df = df[df['Model'] == 'NST'].groupby(['Dataset', 'Lambda'])['Depth'].mean().reset_index()
    sns.lineplot(data=agg_df, x='Lambda', y='Depth', hue='Dataset', marker='o')
    plt.title("Structural Parsimony: Tree Depth vs Regularization")
    plt.xlabel(r"$\lambda$ (Latency Penalty)")
    plt.ylabel("Mean Tree Depth")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {output_path}")

def plot_ensemble_gain(df, output_path='unified_ensemble_gain.png'):
    """Compare NST vs NSF Accuracy (Best Lambda)."""
    best_results = []
    for ds in df['Dataset'].unique():
        for mod in ['NST', 'NSF']:
            subset = df[(df['Dataset'] == ds) & (df['Model'] == mod)]
            if not subset.empty:
                # Find best lambda on average
                best_lam_idx = subset.groupby('Lambda')['Accuracy'].mean().idxmax()
                best_acc = subset[subset['Lambda'] == best_lam_idx]['Accuracy'].mean()
                best_results.append({'Dataset': ds, 'Model': mod, 'Best Accuracy': best_acc})
    
    res_df = pd.DataFrame(best_results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x='Dataset', y='Best Accuracy', hue='Model')
    plt.title("Ensemble Performance Gain (Forest vs Single Tree)")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {output_path}")

def main():
    if not os.path.exists('unified_results.csv'):
        print("unified_results.csv not found. Run the benchmark first.")
        return
    
    df = pd.read_csv('unified_results.csv')
    
    plot_pareto_frontiers(df)
    plot_lambda_impact_dual(df)
    plot_structural_parsimony(df)
    plot_ensemble_gain(df)

if __name__ == "__main__":
    main()
