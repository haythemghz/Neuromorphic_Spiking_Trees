import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os

def run_stats():
    if not os.path.exists('unified_results.csv'):
        print("unified_results.csv not found.")
        return
    
    df = pd.read_csv('unified_results.csv')
    
    # We want to compare NSF (best lambda) against RF and CART per dataset
    datasets = df['Dataset'].unique()
    
    stats_summary = []
    
    for ds in datasets:
        ds_df = df[df['Dataset'] == ds]
        
        # Get accuracy arrays across folds
        # NSF (Best Average Lambda)
        nsf_agg_all = ds_df[ds_df['Model'] == 'NSF']
        nsf_agg = nsf_agg_all.groupby('Lambda')['Accuracy'].mean()
        best_lam = nsf_agg.idxmax()
        nsf_accs = ds_df[(ds_df['Model'] == 'NSF') & (ds_df['Lambda'] == best_lam)].sort_values('Fold')['Accuracy'].values
        
        # RF
        rf_accs = ds_df[ds_df['Model'] == 'RF'].sort_values('Fold')['Accuracy'].values
        
        # CART
        cart_accs = ds_df[ds_df['Model'] == 'CART'].sort_values('Fold')['Accuracy'].values
        
        # Wilcoxon tests
        try:
            # Check if there's any difference
            if np.array_equal(nsf_accs, rf_accs):
                stat_rf, p_rf = 0, 1.0
            else:
                stat_rf, p_rf = wilcoxon(nsf_accs, rf_accs)
        except Exception:
            stat_rf, p_rf = np.nan, np.nan
            
        try:
            if np.array_equal(nsf_accs, cart_accs):
                stat_cart, p_cart = 0, 1.0
            else:
                stat_cart, p_cart = wilcoxon(nsf_accs, cart_accs)
        except Exception:
            stat_cart, p_cart = np.nan, np.nan
            
        stats_summary.append({
            'Dataset': ds,
            'NSF_Mean': np.mean(nsf_accs),
            'RF_Mean': np.mean(rf_accs),
            'CART_Mean': np.mean(cart_accs),
            'NSF_vs_RF_p': p_rf,
            'NSF_vs_CART_p': p_cart
        })
        
    summary_df = pd.DataFrame(stats_summary)
    summary_df.to_csv('statistical_validation.csv', index=False)
    print("Statistical summary saved to statistical_validation.csv")
    print(summary_df)

if __name__ == "__main__":
    run_stats()
