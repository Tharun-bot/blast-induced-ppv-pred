# learning_curve.py

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, KFold
from xgboost import XGBRegressor

TARGETS      = ['PPV', 'V', 'L', 'T']
TARGET_NAMES = {'PPV': 'Resultant', 'V': 'Vertical', 
                'L': 'Longitudinal', 'T': 'Transverse'}

def plot_learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = ['#333333', '#2196F3', '#4CAF50', '#FF5722']
    
    for ax, target, color in zip(axes.flatten(), TARGETS, colors):
        data      = joblib.load(f"outputs/models/{target}_all_models.pkl")
        xgb_model = data['XGBoost']
        xgb_params = data['XGBoost_params']
        
        # Reconstruct model with best params (unfitted)
        model = XGBRegressor(**xgb_params)
        
        # Use full scaled data
        X_all = np.vstack([data['X_train'], data['X_test']])
        y_all = np.concatenate([data['y_train'], data['y_test']])
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_all, y_all,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        train_rmse = -train_scores.mean(axis=1)
        val_rmse   = -val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)
        
        ax.plot(train_sizes, train_rmse, 'o-', color=color, 
                lw=2, label='Train RMSE', alpha=0.8)
        ax.plot(train_sizes, val_rmse, 's--', color=color, 
                lw=2, label='Val RMSE', alpha=0.8)
        ax.fill_between(train_sizes, 
                        val_rmse - val_std, 
                        val_rmse + val_std,
                        alpha=0.15, color=color)
        
        # Gap at full training size
        gap = val_rmse[-1] - train_rmse[-1]
        converging = val_rmse[-1] < val_rmse[-2]
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('RMSE (mm/s)')
        ax.set_title(
            f'{TARGET_NAMES[target]}\n'
            f'Val RMSE@full={val_rmse[-1]:.2f}, '
            f'gap={gap:.2f}, '
            f'{"converging ✓" if converging else "not converged"}'
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        print(f"\n{TARGET_NAMES[target]}:")
        print(f"  Train RMSE at full data: {train_rmse[-1]:.4f}")
        print(f"  Val RMSE at full data:   {val_rmse[-1]:.4f}")
        print(f"  Gap (overfit indicator): {gap:.4f}")
        print(f"  Val RMSE still decreasing at 100% data: {converging}")
    
    plt.suptitle('Learning Curves — XGBoost per PPV Component', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/figures/fig_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved learning curves")

if __name__ == "__main__":
    plot_learning_curves()