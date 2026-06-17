import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from hp_model import hp_formula, fit_hp

def compute_af_and_tests(df):
    SD  = df['SD'].values
    
    # Fit HP on resultant
    res = fit_hp(SD, df['PPV'].values, 'Resultant')
    hp_pred = hp_formula(SD, res['K'], res['alpha'])
    
    df = df.copy()
    df['AF_V'] = df['V'] / hp_pred
    df['AF_L'] = df['L'] / hp_pred
    df['AF_T'] = df['T'] / hp_pred
    df['hp_pred_resultant'] = hp_pred
    
    print("\n=== AMPLIFICATION FACTOR STATS ===")
    for af in ['AF_V', 'AF_L', 'AF_T']:
        print(f"\n{af}:")
        print(f"  Mean  = {df[af].mean():.4f}")
        print(f"  Std   = {df[af].std():.4f}")
        print(f"  Min   = {df[af].min():.4f}")
        print(f"  Max   = {df[af].max():.4f}")
    
    print("\n=== SPEARMAN CORRELATION: AF vs Scaled Distance ===")
    spearman_results = {}
    for af in ['AF_V', 'AF_L', 'AF_T']:
        rho, p = spearmanr(df['SD'], df[af])
        spearman_results[af] = {'rho': round(rho, 4), 'p': round(p, 6)}
        sig = "*** SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {af}: rho={rho:.4f}, p={p:.6f}  {sig}")
    
    print("\n=== SPEARMAN CORRELATION: Component Ratios vs Scaled Distance ===")
    ratio_spearman = {}
    for ratio in ['ratio_VL', 'ratio_VT', 'ratio_LT']:
        rho, p = spearmanr(df['SD'], df[ratio])
        ratio_spearman[ratio] = {'rho': round(rho, 4), 'p': round(p, 6)}
        sig = "*** SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {ratio}: rho={rho:.4f}, p={p:.6f}  {sig}")
    
    print("\n=== KRUSKAL-WALLIS TEST: V vs L vs T ===")
    stat, p_kw = kruskal(df['V'], df['L'], df['T'])
    print(f"  H-stat={stat:.4f}, p={p_kw:.6f}")
    if p_kw < 0.05:
        print("  SIGNIFICANT: V, L, T come from different distributions")
        print("  Running pairwise Mann-Whitney U tests:")
        for pair, (a, b) in [('V vs L', ('V','L')), 
                               ('V vs T', ('V','T')), 
                               ('L vs T', ('L','T'))]:
            u, p_mw = mannwhitneyu(df[a], df[b], alternative='two-sided')
            sig = "significant" if p_mw < 0.05 else "not significant"
            print(f"    {pair}: U={u:.1f}, p={p_mw:.6f}  ({sig})")
    else:
        print("  NOT significant at 0.05 level")
        print("  NOTE: Anisotropy confirmed by slope differences, not distribution means")
    
    return df, spearman_results, ratio_spearman

def plot_amplification_factors(df, spearman_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    af_cols = ['AF_V', 'AF_L', 'AF_T']
    names   = ['Vertical', 'Longitudinal', 'Transverse']
    colors  = ['#2196F3', '#4CAF50', '#FF5722']
    
    for ax, af, name, color in zip(axes, af_cols, names, colors):
        ax.scatter(df['SD'], df[af], alpha=0.55, s=28, color=color, zorder=3)
        
        # Trend line
        z = np.polyfit(df['SD'], df[af], 1)
        p = np.poly1d(z)
        SD_s = np.sort(df['SD'].values)
        ax.plot(SD_s, p(SD_s), 'k--', lw=1.5, label='Linear trend')
        
        # Mean line
        ax.axhline(df[af].mean(), color='red', lw=1.5, 
                   linestyle=':', label=f'Mean={df[af].mean():.3f}')
        
        rho = spearman_results[af]['rho']
        pv  = spearman_results[af]['p']
        sig_txt = f"ρ={rho:.3f}, p={pv:.4f}"
        ax.set_title(f'AF — {name}\n{sig_txt}', fontsize=11)
        ax.set_xlabel('Scaled Distance (m/kg^0.5)')
        ax.set_ylabel(f'Amplification Factor')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Directional Amplification Factors vs Scaled Distance', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/figures/fig8_amplification_factors.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved fig8")

def plot_af_by_sd_bins(df):
    """AF summary table by SD bins — becomes a practical correction table"""
    df = df.copy()
    df['SD_bin'] = pd.cut(df['SD'], bins=5, 
                          labels=['Very Close\n(<14)', 'Close\n(14-21)', 
                                  'Medium\n(21-28)', 'Far\n(28-35)', 'Very Far\n(>35)'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    af_cols = ['AF_V', 'AF_L', 'AF_T']
    names   = ['Vertical', 'Longitudinal', 'Transverse']
    colors  = ['#2196F3', '#4CAF50', '#FF5722']
    
    for ax, af, name, color in zip(axes, af_cols, names, colors):
        grouped = df.groupby('SD_bin', observed=True)[af].agg(['mean','std'])
        ax.bar(range(len(grouped)), grouped['mean'], 
               yerr=grouped['std'], color=color, alpha=0.7,
               capsize=4, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, fontsize=8)
        ax.set_xlabel('Scaled Distance Bin')
        ax.set_ylabel('Mean AF (±std)')
        ax.set_title(f'{name} AF by Distance Bin')
        ax.axhline(df[af].mean(), color='red', lw=1, linestyle=':', label='Overall mean')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    
    plt.suptitle('Amplification Factor by Scaled Distance Bin\n(Practical Correction Table)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/figures/fig9_af_bins.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved fig9")
    
    # Print the actual correction table
    print("\n=== PRACTICAL CORRECTION TABLE ===")
    for af, name in zip(af_cols, names):
        grouped = df.groupby('SD_bin', observed=True)[af].agg(['mean','std','count'])
        print(f"\n{name}:")
        print(grouped.round(4).to_string())

if __name__ == "__main__":
    df = pd.read_csv("data/processed.csv")
    df, spearman_res, ratio_spearman = compute_af_and_tests(df)
    plot_amplification_factors(df, spearman_res)
    plot_af_by_sd_bins(df)
    df.to_csv("data/processed_with_af.csv", index=False)