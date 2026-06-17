import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

TARGETS      = ['PPV', 'V', 'L', 'T']
TARGET_NAMES = {
    'PPV': 'Resultant', 'V': 'Vertical',
    'L': 'Longitudinal', 'T': 'Transverse'
}
FEATURES = ['Distance', 'Charge', 'SD', 'logD', 'logW', 'logSD', 'W2_D', 'sqrtW']
FEATURE_LABELS = ['Distance', 'Charge', 'SD', 'log(D)', 'log(W)', 'log(SD)', 'W²/D', '√W']

def run_shap_for_target(target: str):
    data     = joblib.load(f"outputs/models/{target}_all_models.pkl")
    xgb_model = data['XGBoost']
    X_test    = data['X_test']
    X_train   = data['X_train']
    
    explainer   = shap.TreeExplainer(xgb_model, X_train)
    shap_values = explainer.shap_values(X_test)
    
    # Mean |SHAP| per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    
    imp_df = pd.DataFrame({
        'Feature': FEATURE_LABELS,
        'Feature_key': FEATURES,
        'Mean_Abs_SHAP': mean_abs
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    imp_df['Rank'] = range(1, len(FEATURES)+1)
    
    return shap_values, explainer, imp_df, X_test

def plot_beeswarm(shap_values, X_test, target, name):
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=FEATURE_LABELS,
        show=False, max_display=8,
        plot_type='dot',
        color_bar_label='Feature value'
    )
    plt.title(f'SHAP Beeswarm — {name} PPV', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/fig_shap_beeswarm_{target}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved beeswarm for {name}")

def plot_bar(imp_df, name, target):
    plt.figure(figsize=(8, 5))
    colors_bar = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(imp_df)))[::-1]
    plt.barh(imp_df['Feature'][::-1], 
             imp_df['Mean_Abs_SHAP'][::-1],
             color=colors_bar, edgecolor='white', alpha=0.85)
    plt.xlabel('Mean |SHAP value| (mm/s)', fontsize=11)
    plt.title(f'SHAP Feature Importance — {name} PPV', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"outputs/figures/fig_shap_bar_{target}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved bar for {name}")

def plot_dependence(shap_values, X_test, target, name):
    """Dependence plot for top 2 features"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Top 2 features are always logSD (#0) and one of logW/Distance
    for ax_idx, feat_idx in enumerate([5, 4]):  # logSD=5, logW=4
        feat_name = FEATURE_LABELS[feat_idx]
        ax = axes[ax_idx]
        
        x_vals = X_test[:, feat_idx]
        s_vals = shap_values[:, feat_idx]
        
        # Color by logW for context
        color_vals = X_test[:, 4]  # logW
        sc = ax.scatter(x_vals, s_vals, c=color_vals, 
                        cmap='RdYlBu_r', alpha=0.7, s=35)
        plt.colorbar(sc, ax=ax, label='log(W)')
        ax.axhline(0, color='black', lw=1, linestyle='--')
        ax.set_xlabel(f'{feat_name}', fontsize=11)
        ax.set_ylabel('SHAP value (impact on PPV)', fontsize=11)
        ax.set_title(f'{name}: SHAP dependence — {feat_name}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'SHAP Dependence Plots — {name} PPV', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"outputs/figures/fig_shap_dependence_{target}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved dependence for {name}")

def build_cross_component_table(all_importance: dict) -> pd.DataFrame:
    """8 features × 4 components: rank and mean |SHAP| per cell"""
    rows = []
    for feat_label in FEATURE_LABELS:
        row = {'Feature': feat_label}
        for target, imp_df in all_importance.items():
            name = TARGET_NAMES[target]
            feat_row = imp_df[imp_df['Feature'] == feat_label]
            if len(feat_row) > 0:
                row[f'{name}_rank'] = int(feat_row['Rank'].values[0])
                row[f'{name}_shap'] = round(feat_row['Mean_Abs_SHAP'].values[0], 4)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Sort by resultant rank
    df = df.sort_values('Resultant_rank')
    return df

def interpret_shap_table(shap_table: pd.DataFrame):
    """Print physical interpretation of rank differences"""
    print("\n=== SHAP CROSS-COMPONENT ANALYSIS ===")
    print(shap_table[['Feature', 'Resultant_rank', 'Vertical_rank', 
                       'Longitudinal_rank', 'Transverse_rank']].to_string(index=False))
    
    print("\n=== RANK DIFFERENCES (note these for paper discussion) ===")
    for _, row in shap_table.iterrows():
        feat = row['Feature']
        ranks = {
            'V': row['Vertical_rank'],
            'L': row['Longitudinal_rank'],
            'T': row['Transverse_rank']
        }
        min_comp = min(ranks, key=ranks.get)
        max_comp = max(ranks, key=ranks.get)
        diff = ranks[max_comp] - ranks[min_comp]
        if diff >= 2:
            print(f"  {feat:12s}: ranks differ by {diff} "
                  f"(most important for {min_comp}, least for {max_comp})")

if __name__ == "__main__":
    all_shap_values = {}
    all_importance  = {}
    
    for target in TARGETS:
        name = TARGET_NAMES[target]
        print(f"\n{'='*40}")
        print(f"SHAP Analysis: {name}")
        print('='*40)
        
        shap_vals, explainer, imp_df, X_test = run_shap_for_target(target)
        all_shap_values[target] = shap_vals
        all_importance[target]  = imp_df
        
        print(f"\n  Feature importance ranking:")
        print(imp_df[['Feature', 'Mean_Abs_SHAP', 'Rank']].to_string(index=False))
        
        plot_beeswarm(shap_vals, X_test, target, name)
        plot_bar(imp_df, name, target)
        plot_dependence(shap_vals, X_test, target, name)
    
    # Cross-component table
    shap_table = build_cross_component_table(all_importance)
    shap_table.to_csv("outputs/tables/table5_shap_rankings.csv", index=False)
    
    interpret_shap_table(shap_table)
    
    # Figure: Side-by-side importance comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    colors_comp = ['#333333', '#2196F3', '#4CAF50', '#FF5722']
    
    for ax, target, color in zip(axes, TARGETS, colors_comp):
        imp = all_importance[target]
        name = TARGET_NAMES[target]
        ax.barh(imp['Feature'][::-1], imp['Mean_Abs_SHAP'][::-1],
                color=color, alpha=0.75, edgecolor='white')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean |SHAP|')
        if target == 'PPV':
            ax.set_ylabel('Feature')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('SHAP Feature Importance — All Components', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/figures/fig_shap_comparison_all.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved shap comparison figure")
    
    print("\n\nTable 5 (SHAP rankings) saved.")
    print("Check outputs/tables/table5_shap_rankings.csv")