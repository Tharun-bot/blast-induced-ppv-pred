# Run this as a script or notebook: eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

df = pd.read_csv("data/processed.csv")

# ── Figure 1: Distribution of all 4 targets ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
targets = [('PPV', 'Resultant PPV'), ('V', 'Vertical PPV'),
           ('L', 'Longitudinal PPV'), ('T', 'Transverse PPV')]
colors = ['#333333', '#2196F3', '#4CAF50', '#FF5722']

for ax, (col, name), color in zip(axes.flatten(), targets, colors):
    ax.hist(df[col], bins=25, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(df[col].mean(), color='red', lw=1.5, linestyle='--', 
               label=f'Mean={df[col].mean():.1f}')
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('PPV (mm/s)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)

plt.suptitle('Distribution of PPV Components', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/fig1_distributions.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved fig1")

# ── Figure 2: Log-log scatter — SD vs each component ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (col, name), color in zip(axes.flatten(), targets, colors):
    normal = df[df['high_divergence'] == 0]
    anom   = df[df['high_divergence'] == 1]
    
    ax.scatter(normal['SD'], normal[col], alpha=0.5, s=25, color=color, label='Normal')
    if len(anom) > 0:
        ax.scatter(anom['SD'], anom[col], alpha=0.9, s=60, color='red', 
                   marker='*', label='High divergence (top 10%)', zorder=5)
    
    # Log-log regression line
    logSD  = np.log10(df['SD'])
    logPPV = np.log10(df[col])
    slope, intercept = np.polyfit(logSD, logPPV, 1)
    SD_range = np.linspace(df['SD'].min(), df['SD'].max(), 200)
    ax.plot(SD_range, 10**(intercept) * SD_range**slope, 
            'k--', lw=1.5, label=f'slope={slope:.3f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Scaled Distance (m/kg^0.5)')
    ax.set_ylabel('PPV (mm/s)')
    ax.set_title(f'{name} — log-log slope: {slope:.3f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('PPV vs Scaled Distance (Log-Log Scale)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/fig2_loglog_scatter.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved fig2")

# Print slopes — THIS IS CRITICAL TO NOTE
print("\n=== LOG-LOG SLOPES (RQ3 answer preview) ===")
for col, name in targets:
    slope, intercept = np.polyfit(np.log10(df['SD']), np.log10(df[col]), 1)
    print(f"{name:25s}: slope = {slope:.4f}")

# ── Figure 3: Component ratios vs Scaled Distance ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ratio_cols = [('ratio_VL','V/L'), ('ratio_VT','V/T'), ('ratio_LT','L/T')]

for ax, (col, name) in zip(axes, ratio_cols):
    ax.scatter(df['SD'], df[col], alpha=0.5, s=25, color='#555555')
    ax.axhline(1.0, color='red', lw=1.5, linestyle='--', label='Isotropy (=1)')
    
    # Trend line
    z = np.polyfit(df['SD'], df[col], 1)
    p = np.poly1d(z)
    SD_sorted = np.sort(df['SD'].values)
    ax.plot(SD_sorted, p(SD_sorted), 'b-', lw=1.5, label='Trend')
    
    ax.set_xlabel('Scaled Distance (m/kg^0.5)')
    ax.set_ylabel(f'Ratio {name}')
    ax.set_title(f'Component Ratio {name} vs Scaled Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/figures/fig3_ratios.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved fig3")

# ── Figure 4: Correlation heatmap ────────────────────────────────────────────
feat_cols = ['Distance', 'Charge', 'SD', 'logD', 'logW', 'logSD', 
             'W2_D', 'sqrtW', 'PPV', 'V', 'L', 'T']
corr = df[feat_cols].corr()

plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, annot_kws={'size': 8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/fig4_correlation.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved fig4")

# ── Figure 5: Boxplot of the 3 components side by side ───────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bp_data = [df['V'], df['L'], df['T']]
bp = ax.boxplot(bp_data, labels=['Vertical', 'Longitudinal', 'Transverse'],
                patch_artist=True, notch=True)
colors_bp = ['#2196F3', '#4CAF50', '#FF5722']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('PPV (mm/s)')
ax.set_title('Distribution Comparison: V, L, T Components')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("outputs/figures/fig5_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved fig5")