import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor  # Uncomment when GPU build is ready
from sklearn.svm import SVR
import matplotlib.pyplot as plt

TARGETS      = ['PPV', 'V', 'L', 'T']
TARGET_NAMES = {
    'PPV': 'Resultant', 'V': 'Vertical',
    'L': 'Longitudinal', 'T': 'Transverse'
}


def build_and_train_stacker(target: str):
    data = joblib.load(f"outputs/models/{target}_all_models.pkl")

    xgb_params = dict(data['XGBoost_params'])
    svr_params  = data['SVR_params']

    # ── Base learners ─────────────────────────────────────────────────────────
    # When LightGBM is ready, add back:
    # lgb_params = dict(data['LightGBM_params'])
    # ('lgb', LGBMRegressor(**lgb_params)),
    base_learners = [
        ('xgb', XGBRegressor(**xgb_params)),
        ('svr', SVR(**svr_params)),
    ]

    stacker = StackingRegressor(
        estimators=base_learners,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        passthrough=False,
        n_jobs=-1
    )

    X_tr = data['X_train']
    X_te = data['X_test']
    y_tr = data['y_train']
    y_te = data['y_test']

    print(f"  Training stacker for {TARGET_NAMES[target]}...")
    stacker.fit(X_tr, y_tr)
    pred = stacker.predict(X_te)

    r2   = r2_score(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae  = mean_absolute_error(y_te, pred)

    print(f"  Stacker: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    joblib.dump({
        'stacker':     stacker,
        'X_test':      X_te,
        'y_test':      y_te,
        'predictions': pred
    }, f"outputs/models/{target}_stacker.pkl")

    return {
        'R2':          round(r2, 4),
        'RMSE':        round(rmse, 4),
        'MAE':         round(mae, 4),
        'predictions': pred,
        'y_test':      y_te
    }


def reconstruction_test():
    print("\n=== RECONSTRUCTION TEST ===")

    component_data = {}
    for t in ['V', 'L', 'T']:
        component_data[t] = joblib.load(f"outputs/models/{t}_stacker.pkl")

    assert np.allclose(component_data['V']['X_test'],
                       component_data['L']['X_test']), \
        "Test sets don't align — check random_state"
    assert np.allclose(component_data['V']['X_test'],
                       component_data['T']['X_test']), \
        "Test sets don't align — check random_state"
    print("  Test set alignment verified ✓")

    V_pred = component_data['V']['predictions']
    L_pred = component_data['L']['predictions']
    T_pred = component_data['T']['predictions']
    V_true = component_data['V']['y_test']
    L_true = component_data['L']['y_test']
    T_true = component_data['T']['y_test']

    reconstructed_pred = np.sqrt(V_pred**2 + L_pred**2 + T_pred**2)
    reconstructed_true = np.sqrt(V_true**2 + L_true**2 + T_true**2)

    recon_r2   = r2_score(reconstructed_true, reconstructed_pred)
    recon_rmse = np.sqrt(mean_squared_error(reconstructed_true, reconstructed_pred))
    recon_mae  = mean_absolute_error(reconstructed_true, reconstructed_pred)

    direct_data = joblib.load("outputs/models/PPV_stacker.pkl")
    direct_pred = direct_data['predictions']
    direct_true = direct_data['y_test']

    direct_r2   = r2_score(direct_true, direct_pred)
    direct_rmse = np.sqrt(mean_squared_error(direct_true, direct_pred))
    direct_mae  = mean_absolute_error(direct_true, direct_pred)

    improvement = ((direct_rmse - recon_rmse) / direct_rmse) * 100

    print(f"\n  Direct resultant stacker:  R²={direct_r2:.4f}  RMSE={direct_rmse:.4f}  MAE={direct_mae:.4f}")
    print(f"  Reconstructed from V+L+T:  R²={recon_r2:.4f}  RMSE={recon_rmse:.4f}  MAE={recon_mae:.4f}")
    print(f"  RMSE improvement: {improvement:.2f}%")

    if improvement > 0:
        print("  RESULT: Component decomposition IMPROVES resultant prediction ✓")
    else:
        print("  RESULT: Direct prediction is marginally better.")
        print("  Still valid — component-wise modelling is justified by SHAP findings.")

    return {
        'direct_R2':      round(direct_r2, 4),
        'direct_RMSE':    round(direct_rmse, 4),
        'recon_R2':       round(recon_r2, 4),
        'recon_RMSE':     round(recon_rmse, 4),
        'improvement_pct':round(improvement, 2)
    }


def plot_predicted_vs_actual(stacker_results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['#333333', '#2196F3', '#4CAF50', '#FF5722']

    for ax, (target, res), color in zip(axes.flatten(),
                                         stacker_results.items(), colors):
        y_true  = res['y_test']
        y_pred  = res['predictions']
        name    = TARGET_NAMES[target]
        max_val = max(y_true.max(), y_pred.max()) * 1.05

        ax.scatter(y_true, y_pred, alpha=0.6, s=30, color=color, zorder=3)
        ax.plot([0, max_val], [0, max_val], 'k--', lw=1.5, label='Perfect fit')
        ax.plot([0, max_val], [0, max_val*1.1], 'r:', lw=1, alpha=0.6, label='+10%')
        ax.plot([0, max_val], [0, max_val*0.9], 'r:', lw=1, alpha=0.6, label='-10%')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel('Measured PPV (mm/s)')
        ax.set_ylabel('Predicted PPV (mm/s)')
        ax.set_title(f'{name}\nR²={res["R2"]}  RMSE={res["RMSE"]} mm/s')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Stacking Ensemble: Predicted vs Measured PPV',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/figures/fig10_predicted_vs_actual.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved fig10")


def plot_model_comparison(all_results: dict, stacker_results: dict,
                           hp_results_path: str):
    hp_df   = pd.read_csv(hp_results_path)
    hp_r2   = dict(zip(hp_df['Component'], hp_df['R2']))
    hp_rmse = dict(zip(hp_df['Component'], hp_df['RMSE']))

    components  = ['Resultant', 'Vertical', 'Longitudinal', 'Transverse']
    target_map  = {'Resultant': 'PPV', 'Vertical': 'V',
                   'Longitudinal': 'L', 'Transverse': 'T'}

    # LightGBM removed from model_names for now
    # Add 'LightGBM' back between XGBoost and SVR when ready
    model_names = ['HP Formula', 'XGBoost', 'SVR', 'RandomForest', 'Stacker']
    colors_bar  = ['#9E9E9E', '#2196F3', '#FF9800', '#9C27B0', '#F44336']

    x     = np.arange(len(components))
    width = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, metric in enumerate(['R2', 'RMSE']):
        ax = axes[ax_idx]
        for m_idx, model in enumerate(model_names):
            vals = []
            for comp in components:
                t = target_map[comp]
                if model == 'HP Formula':
                    vals.append(hp_r2[comp] if metric == 'R2'
                                else hp_rmse[comp])
                elif model == 'Stacker':
                    vals.append(stacker_results[t][metric])
                else:
                    vals.append(all_results[t][model][metric])

            offset = (m_idx - len(model_names)/2) * width + width/2
            ax.bar(x + offset, vals, width, label=model,
                   color=colors_bar[m_idx], alpha=0.8)

        ax.set_xlabel('PPV Component')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=15)
        ax.set_title(f'Model Comparison — {metric}')
        ax.set_ylabel(metric)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        if metric == 'R2':
            ax.set_ylim(0.5, 1.0)

    plt.suptitle('All Models vs HP Formula Baseline', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/figures/fig11_model_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved fig11")


if __name__ == "__main__":
    stacker_results = {}

    for target in TARGETS:
        print(f"\n{'='*45}")
        stacker_results[target] = build_and_train_stacker(target)

    recon_results = reconstruction_test()
    plot_predicted_vs_actual(stacker_results)

    table2 = pd.read_csv("outputs/tables/table2_ml_results.csv")
    all_results_rebuilt = {}
    for target in TARGETS:
        name = TARGET_NAMES[target]
        all_results_rebuilt[target] = {}
        # LightGBM removed from model list here too
        for model in ['XGBoost', 'SVR', 'RandomForest']:
            row = table2[(table2['Target'] == name) & (table2['Model'] == model)]
            if len(row) > 0:
                all_results_rebuilt[target][model] = {
                    'R2':   row['R2'].values[0],
                    'RMSE': row['RMSE'].values[0]
                }

    plot_model_comparison(
        all_results_rebuilt, stacker_results,
        "outputs/tables/table1_hp_results.csv"
    )

    rows = []
    for target, res in stacker_results.items():
        rows.append({
            'Target': TARGET_NAMES[target],
            'R2':     res['R2'],
            'RMSE':   res['RMSE'],
            'MAE':    res['MAE']
        })
    pd.DataFrame(rows).to_csv("outputs/tables/table3_stacker_results.csv", index=False)
    pd.DataFrame([recon_results]).to_csv(
        "outputs/tables/table4_reconstruction_test.csv", index=False)

    print("\nAll stacker tables saved.")