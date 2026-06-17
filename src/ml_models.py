import numpy as np
import pandas as pd
import optuna
import joblib
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor  # Uncomment when GPU build is ready

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = ['Distance', 'Charge', 'SD', 'logD', 'logW', 'logSD', 'W2_D', 'sqrtW']
TARGETS  = ['PPV', 'V', 'L', 'T']
TARGET_NAMES = {
    'PPV': 'Resultant', 'V': 'Vertical',
    'L': 'Longitudinal', 'T': 'Transverse'
}

# ── GPU check ─────────────────────────────────────────────────────────────────
def check_gpu():
    xgb_gpu = False
    try:
        test = XGBRegressor(device='cuda', tree_method='hist',
                            n_estimators=5, verbosity=0)
        test.fit(np.random.rand(20, 3), np.random.rand(20))
        xgb_gpu = True
        print("  XGBoost GPU: AVAILABLE ✓")
    except Exception as e:
        print(f"  XGBoost GPU: NOT available ({e}) — falling back to CPU")

    # ── LightGBM GPU check (skipped for now) ─────────────────────────────────
    # lgb_gpu = False
    # try:
    #     from lightgbm import LGBMRegressor
    #     test = LGBMRegressor(device='gpu', n_estimators=5, verbose=-1)
    #     test.fit(np.random.rand(20, 3), np.random.rand(20))
    #     lgb_gpu = True
    #     print("  LightGBM GPU: AVAILABLE ✓")
    # except Exception as e:
    #     print(f"  LightGBM GPU: NOT available ({e}) — falling back to CPU")
    # return xgb_gpu, lgb_gpu

    lgb_gpu = False  # Remove this line when uncommenting above
    return xgb_gpu, lgb_gpu


def evaluate(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    if label:
        print(f"  {label:15s}: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {'R2': round(r2,4), 'RMSE': round(rmse,4), 'MAE': round(mae,4)}


def cv_rmse(model, X, y, n_splits=5):
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = -cross_val_score(model, X, y, cv=kf,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return round(scores.mean(), 4), round(scores.std(), 4)


def get_split(df, target_col):
    X = df[FEATURES].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        stratify=df['sd_quartile'].values,
        random_state=42
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ── XGBoost Optuna (GPU) ──────────────────────────────────────────────────────
def tune_xgboost(X_train, y_train, n_trials=100, use_gpu=True):
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0, 5),
            'random_state': 42,
            'verbosity': 0,
        }
        if use_gpu:
            params['device']      = 'cuda'
            params['tree_method'] = 'hist'
        else:
            params['n_jobs'] = -1

        model  = XGBRegressor(**params)
        kf     = KFold(n_splits=5, shuffle=True, random_state=42)
        n_jobs = 1 if use_gpu else -1
        scores = -cross_val_score(model, X_train, y_train, cv=kf,
                                   scoring='neg_root_mean_squared_error',
                                   n_jobs=n_jobs)
        return scores.mean()

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    if use_gpu:
        best.update({'random_state': 42, 'verbosity': 0,
                     'device': 'cuda', 'tree_method': 'hist'})
    else:
        best.update({'random_state': 42, 'verbosity': 0, 'n_jobs': -1})

    model = XGBRegressor(**best)
    model.fit(X_train, y_train)
    return model, best, study


# ── LightGBM Optuna (skipped — uncomment when GPU build is ready) ─────────────
# def tune_lightgbm(X_train, y_train, n_trials=100, use_gpu=True):
#     from lightgbm import LGBMRegressor
#     def objective(trial):
#         params = {
#             'n_estimators':      trial.suggest_int('n_estimators', 100, 1000),
#             'max_depth':         trial.suggest_int('max_depth', 3, 12),
#             'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
#             'num_leaves':        trial.suggest_int('num_leaves', 15, 255),
#             'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
#             'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
#             'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
#             'random_state': 42,
#             'verbose': -1,
#         }
#         if use_gpu:
#             params['device'] = 'gpu'
#         else:
#             params['n_jobs'] = -1
#         model  = LGBMRegressor(**params)
#         kf     = KFold(n_splits=5, shuffle=True, random_state=42)
#         n_jobs = 1 if use_gpu else -1
#         scores = -cross_val_score(model, X_train, y_train, cv=kf,
#                                    scoring='neg_root_mean_squared_error',
#                                    n_jobs=n_jobs)
#         return scores.mean()
#     study = optuna.create_study(direction='minimize',
#                                  sampler=optuna.samplers.TPESampler(seed=42))
#     study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
#     best = study.best_params
#     if use_gpu:
#         best.update({'random_state': 42, 'verbose': -1, 'device': 'gpu'})
#     else:
#         best.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1})
#     model = LGBMRegressor(**best)
#     model.fit(X_train, y_train)
#     return model, best, study


# ── SVR GridSearch (CPU) ──────────────────────────────────────────────────────
def tune_svr(X_train, y_train):
    param_grid = {
        'C':       [0.1, 1, 10, 100, 500, 1000],
        'gamma':   ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.5, 1.0]
    }
    gs = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5,
                      scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    print(f"  SVR best params: {gs.best_params_}")
    return gs.best_estimator_, gs.best_params_


# ── Random Forest (CPU) ───────────────────────────────────────────────────────
def train_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


# ── Master loop ───────────────────────────────────────────────────────────────
def train_all(df, n_trials=100):
    print("="*55)
    print("Checking GPU availability...")
    xgb_gpu, lgb_gpu = check_gpu()
    print("="*55)

    all_results = {}

    for target in TARGETS:
        name = TARGET_NAMES[target]
        print(f"\n{'#'*55}")
        print(f"  TARGET: {name} ({target})")
        print(f"{'#'*55}")

        X_tr, X_te, y_tr, y_te, scaler = get_split(df, target)

        # ── XGBoost ──────────────────────────────────────────────────────────
        print(f"\n[1/3] Tuning XGBoost ({n_trials} trials, GPU={xgb_gpu})...")
        xgb_m, xgb_p, xgb_study = tune_xgboost(X_tr, y_tr, n_trials, use_gpu=xgb_gpu)
        xgb_cv_m, xgb_cv_s = cv_rmse(xgb_m, X_tr, y_tr)
        xgb_eval = evaluate(y_te, xgb_m.predict(X_te), "XGBoost-test")
        xgb_eval['CV_mean'] = xgb_cv_m
        xgb_eval['CV_std']  = xgb_cv_s

        # ── LightGBM (skipped) ────────────────────────────────────────────────
        # Uncomment this entire block when LightGBM GPU build is ready
        # Also change [1/3] → [1/4] and [2/3] → [2/4] etc above and below
        # print(f"\n[2/4] Tuning LightGBM ({n_trials} trials, GPU={lgb_gpu})...")
        # lgb_m, lgb_p, lgb_study = tune_lightgbm(X_tr, y_tr, n_trials, use_gpu=lgb_gpu)
        # lgb_cv_m, lgb_cv_s = cv_rmse(lgb_m, X_tr, y_tr)
        # lgb_eval = evaluate(y_te, lgb_m.predict(X_te), "LightGBM-test")
        # lgb_eval['CV_mean'] = lgb_cv_m
        # lgb_eval['CV_std']  = lgb_cv_s

        # ── SVR ───────────────────────────────────────────────────────────────
        print(f"\n[2/3] Tuning SVR (CPU)...")
        svr_m, svr_p = tune_svr(X_tr, y_tr)
        svr_cv_m, svr_cv_s = cv_rmse(svr_m, X_tr, y_tr)
        svr_eval = evaluate(y_te, svr_m.predict(X_te), "SVR-test")
        svr_eval['CV_mean'] = svr_cv_m
        svr_eval['CV_std']  = svr_cv_s

        # ── Random Forest ─────────────────────────────────────────────────────
        print(f"\n[3/3] Training Random Forest (CPU)...")
        rf_m = train_rf(X_tr, y_tr)
        rf_cv_m, rf_cv_s = cv_rmse(rf_m, X_tr, y_tr)
        rf_eval = evaluate(y_te, rf_m.predict(X_te), "RF-test")
        rf_eval['CV_mean'] = rf_cv_m
        rf_eval['CV_std']  = rf_cv_s

        # ── Best model ────────────────────────────────────────────────────────
        # Add ('LightGBM', lgb_eval) back to this list when uncommented
        best_name = min(
            [('XGBoost', xgb_eval),
             ('SVR', svr_eval),
             ('RF', rf_eval)],
            key=lambda x: x[1]['RMSE']
        )[0]
        print(f"\n  Best individual model: {best_name}")

        all_results[target] = {
            'XGBoost':      xgb_eval,
            # 'LightGBM':   lgb_eval,  # Uncomment when ready
            'SVR':          svr_eval,
            'RandomForest': rf_eval
        }

        joblib.dump({
            'XGBoost':        xgb_m,
            'XGBoost_params': xgb_p,
            # 'LightGBM':        lgb_m,   # Uncomment when ready
            # 'LightGBM_params': lgb_p,   # Uncomment when ready
            'SVR':            svr_m,
            'SVR_params':     svr_p,
            'RandomForest':   rf_m,
            'scaler':         scaler,
            'X_train':        X_tr,
            'X_test':         X_te,
            'y_train':        y_tr,
            'y_test':         y_te
        }, f"outputs/models/{target}_all_models.pkl")

        print(f"  Saved: outputs/models/{target}_all_models.pkl")

    # ── Table 2 ───────────────────────────────────────────────────────────────
    rows = []
    for target, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            rows.append({
                'Target':       TARGET_NAMES[target],
                'Model':        model_name,
                'R2':           metrics['R2'],
                'RMSE':         metrics['RMSE'],
                'MAE':          metrics['MAE'],
                'CV_RMSE_mean': metrics['CV_mean'],
                'CV_RMSE_std':  metrics['CV_std']
            })

    table2 = pd.DataFrame(rows)
    table2.to_csv("outputs/tables/table2_ml_results.csv", index=False)

    print("\n\n=== TABLE 2 — ML RESULTS ===")
    print(table2.to_string(index=False))

    return all_results


if __name__ == "__main__":
    df = pd.read_csv("data/processed.csv")
    results = train_all(df, n_trials=100)