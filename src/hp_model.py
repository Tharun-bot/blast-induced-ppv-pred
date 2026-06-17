import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def hp_formula(SD, K, alpha):
    return K * SD**(-alpha)

def fit_hp(SD: np.ndarray, ppv: np.ndarray, label: str):
    try:
        popt, pcov = curve_fit(
            hp_formula, SD, ppv,
            p0=[1000, 1.5],
            bounds=([1, 0.3], [100000, 3.5]),
            maxfev=20000
        )
    except RuntimeError:
        # Try different initial guess if first fails
        popt, pcov = curve_fit(
            hp_formula, SD, ppv,
            p0=[500, 1.0],
            bounds=([1, 0.3], [100000, 3.5]),
            maxfev=20000
        )
    
    K_fit, alpha_fit = popt
    perr = np.sqrt(np.diag(pcov))  # parameter std errors
    pred = hp_formula(SD, K_fit, alpha_fit)
    
    r2   = r2_score(ppv, pred)
    rmse = np.sqrt(mean_squared_error(ppv, pred))
    mae  = mean_absolute_error(ppv, pred)
    
    # Residuals
    residuals = ppv - pred
    
    print(f"\n{'='*40}")
    print(f"Component: {label}")
    print(f"  K     = {K_fit:.4f}  (±{perr[0]:.4f})")
    print(f"  alpha = {alpha_fit:.4f}  (±{perr[1]:.4f})")
    print(f"  R²    = {r2:.4f}")
    print(f"  RMSE  = {rmse:.4f} mm/s")
    print(f"  MAE   = {mae:.4f} mm/s")
    print(f"  Residual mean = {residuals.mean():.4f}")
    print(f"  Residual std  = {residuals.std():.4f}")
    
    return {
        'Component': label,
        'K': round(K_fit, 4),
        'alpha': round(alpha_fit, 4),
        'K_std': round(perr[0], 4),
        'alpha_std': round(perr[1], 4),
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'predictions': pred,
        'residuals': residuals
    }

def log_linear_slopes(df):
    print("\n=== LOG-LOG ATTENUATION SLOPES ===")
    results = {}
    for col, name in [('PPV','Resultant'),('V','Vertical'),('L','Longitudinal'),('T','Transverse')]:
        logSD  = np.log10(df['SD'])
        logPPV = np.log10(df[col])
        slope, intercept, r, p, se = stats.linregress(logSD, logPPV)
        results[name] = {
            'slope': round(slope, 4),
            'intercept': round(intercept, 4),
            'R2': round(r**2, 4),
            'p_value': round(p, 8),
            'std_error': round(se, 6)
        }
        print(f"{name:15s}: slope={slope:.4f}, R²={r**2:.4f}, p={p:.2e}")
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/processed.csv")
    SD = df['SD'].values
    
    components = [
        ('PPV', 'Resultant'),
        ('V',   'Vertical'),
        ('L',   'Longitudinal'),
        ('T',   'Transverse')
    ]
    
    hp_results = {}
    for col, name in components:
        hp_results[name] = fit_hp(SD, df[col].values, name)
    
    # Save Table 1
    table1_rows = []
    for name, res in hp_results.items():
        table1_rows.append({
            'Component': name,
            'K': res['K'],
            'K_std': res['K_std'],
            'alpha': res['alpha'],
            'alpha_std': res['alpha_std'],
            'R2': res['R2'],
            'RMSE': res['RMSE'],
            'MAE': res['MAE']
        })
    
    table1 = pd.DataFrame(table1_rows)
    table1.to_csv("outputs/tables/table1_hp_results.csv", index=False)
    print("\n\nTABLE 1 — HP Results:")
    print(table1.to_string(index=False))
    
    # Log-log slopes
    slopes = log_linear_slopes(df)
    slopes_df = pd.DataFrame(slopes).T
    slopes_df.to_csv("outputs/tables/table1b_slopes.csv")
    
    # Figure: 4-panel HP fits
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    colors_hp = ['#333333', '#2196F3', '#4CAF50', '#FF5722']
    SD_range   = np.linspace(df['SD'].min(), df['SD'].max(), 300)
    
    for idx, ((col, name), color) in enumerate(zip(components, colors_hp)):
        ax  = fig.add_subplot(gs[idx//2, idx%2])
        res = hp_results[name]
        
        ax.scatter(df['SD'], df[col], alpha=0.5, s=25, color=color, label='Measured')
        ax.plot(SD_range, hp_formula(SD_range, res['K'], res['alpha']),
                'k-', lw=2, label=f"HP fit\nK={res['K']}, α={res['alpha']}")
        
        # Residual band
        pred_vals = hp_formula(df['SD'].values, res['K'], res['alpha'])
        upper = pred_vals + res['RMSE']
        lower = pred_vals - res['RMSE']
        SD_order = df['SD'].argsort()
        ax.fill_between(df['SD'].values[SD_order], 
                        lower[SD_order], upper[SD_order],
                        alpha=0.15, color='gray', label='±RMSE band')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Scaled Distance (m/kg^0.5)', fontsize=10)
        ax.set_ylabel('PPV (mm/s)', fontsize=10)
        ax.set_title(f'{name}  |  R²={res["R2"]}  RMSE={res["RMSE"]} mm/s', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Holmberg-Persson Formula Fit — All Components', fontsize=14, fontweight='bold')
    plt.savefig("outputs/figures/fig6_hp_fits.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved fig6")
    
    # Figure: Residual analysis
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for idx, ((col, name), color) in enumerate(zip(components, colors_hp)):
        res  = hp_results[name]
        pred = res['predictions']
        resid = res['residuals']
        
        # Residual vs fitted
        axes[0, idx].scatter(pred, resid, alpha=0.5, s=20, color=color)
        axes[0, idx].axhline(0, color='red', lw=1.5)
        axes[0, idx].set_xlabel('Fitted PPV')
        axes[0, idx].set_ylabel('Residual')
        axes[0, idx].set_title(f'{name}\nResid vs Fitted')
        axes[0, idx].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy.stats import probplot
        probplot(resid, plot=axes[1, idx])
        axes[1, idx].set_title(f'{name} — Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig("outputs/figures/fig7_hp_residuals.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved fig7")