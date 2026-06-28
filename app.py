# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blast PPV Predictor",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for target in ['PPV', 'V', 'L', 'T']:
        models[target] = joblib.load(f"outputs/models/{target}_all_models.pkl")
        models[f"{target}_stacker"] = joblib.load(f"outputs/models/{target}_stacker.pkl")
    return models

@st.cache_resource
def load_data():
    return pd.read_csv("data/processed.csv")

models = load_models()
df     = load_data()

FEATURES = ['Distance', 'Charge', 'SD', 'logD', 'logW', 'logSD', 'W2_D', 'sqrtW']
FEATURE_LABELS = ['Distance', 'Charge', 'SD', 'log(D)', 'log(W)', 'log(SD)', 'W²/D', '√W']

def engineer_input(distance, charge):
    sd    = distance / np.sqrt(charge)
    logD  = np.log10(distance)
    logW  = np.log10(charge)
    logSD = np.log10(sd)
    W2_D  = charge**2 / distance
    sqrtW = np.sqrt(charge)
    return np.array([[distance, charge, sd, logD, logW, logSD, W2_D, sqrtW]])

def predict_all(distance, charge):
    X_raw = engineer_input(distance, charge)
    results = {}
    for target in ['PPV', 'V', 'L', 'T']:
        scaler  = models[target]['scaler']
        X_sc    = scaler.transform(X_raw)
        stacker = models[f"{target}_stacker"]['stacker']
        results[target] = float(stacker.predict(X_sc)[0])
    reconstructed = np.sqrt(
        results['V']**2 + results['L']**2 + results['T']**2
    )
    results['reconstructed'] = reconstructed
    return results

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["PPV Predictor", "Results Dashboard", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("200 blast monitoring records")
st.sidebar.markdown("Open-pit mine, India")
st.sidebar.markdown("SD range: 6.97–43.47 m/kg^0.5")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PPV PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
if page == "PPV Predictor":
    st.title("💥 Blast PPV Directional Predictor")
    st.markdown(
        "Predict **Vertical, Longitudinal, and Transverse** Peak Particle Velocity "
        "components from blast parameters using a stacking ensemble (XGBoost + SVR + Ridge)."
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Blast Parameters")

        distance = st.slider(
            "Distance from blast to sensor (m)",
            min_value=300, max_value=1250,
            value=600, step=10
        )
        charge = st.slider(
            "Charge per delay (kg)",
            min_value=650, max_value=2950,
            value=1500, step=50
        )

        sd = distance / np.sqrt(charge)
        st.metric("Scaled Distance (m/kg^0.5)", f"{sd:.2f}")

        # Warn if outside training range
        if sd < 6.97 or sd > 43.47:
            st.warning(
                f"SD={sd:.2f} is outside training range (6.97–43.47). "
                "Predictions may be unreliable."
            )

        predict_btn = st.button("Predict PPV", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            with st.spinner("Running stacking ensemble..."):
                results = predict_all(distance, charge)

            st.subheader("Predicted PPV Components")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Vertical",     f"{results['V']:.2f} mm/s")
            m2.metric("Longitudinal", f"{results['L']:.2f} mm/s")
            m3.metric("Transverse",   f"{results['T']:.2f} mm/s")
            m4.metric("Reconstructed Resultant", f"{results['reconstructed']:.2f} mm/s",
                      help="√(V²+L²+T²) from component predictions")

            # Direct resultant
            direct = results['PPV']
            st.caption(f"Direct resultant stacker prediction: {direct:.2f} mm/s")

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 3.5))
            components = ['Vertical', 'Longitudinal', 'Transverse', 'Reconstructed\nResultant', 'Direct\nResultant']
            values     = [results['V'], results['L'], results['T'],
                          results['reconstructed'], results['PPV']]
            colors     = ['#2196F3', '#4CAF50', '#FF5722', '#333333', '#9E9E9E']
            bars = ax.bar(components, values, color=colors, alpha=0.85, edgecolor='white')
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)
            ax.set_ylabel('PPV (mm/s)')
            ax.set_title('Predicted PPV by Component')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(values) * 1.25)
            st.pyplot(fig)
            plt.close()

            # Dominant component
            dominant = max(['V', 'L', 'T'], key=lambda x: results[x])
            names = {'V': 'Vertical', 'L': 'Longitudinal', 'T': 'Transverse'}
            st.info(f"**Dominant component:** {names[dominant]} ({results[dominant]:.2f} mm/s)")

            # Amplification factors
            st.subheader("Directional Amplification Factors")
            st.caption("AF = predicted component / predicted resultant — fraction of energy per direction")
            af_cols = st.columns(3)
            for col, (key, name) in zip(af_cols, [('V','Vertical'),('L','Longitudinal'),('T','Transverse')]):
                af = results[key] / results['PPV']
                col.metric(f"AF_{name[0]}", f"{af:.3f}")

            # HP formula comparison
            st.subheader("Comparison: HP Formula vs ML Stacker")
            K_resultant = 2950.92
            alpha       = 1.3686
            hp_pred     = K_resultant * sd**(-alpha)

            comp_df = pd.DataFrame({
                'Method':    ['HP Formula', 'ML Stacker (Direct)', 'ML Stacker (Reconstructed)'],
                'R²':        [0.750, 0.796, 0.817],
                'Predicted PPV (mm/s)': [round(hp_pred, 2), round(results['PPV'], 2),
                                          round(results['reconstructed'], 2)]
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        else:
            st.info("Set blast parameters on the left and click **Predict PPV**.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Results Dashboard":
    st.title("📊 Research Results Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs([
        "HP Baseline", "ML Performance", "SHAP Analysis", "Amplification Factors"
    ])

    # ── Tab 1: HP Baseline ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Holmberg-Persson Formula — Per Component")
        st.markdown(
            "The HP formula PPV = K × SD⁻ᵅ was fitted independently to all four targets. "
            "Attenuation exponent α is statistically indistinguishable across V, L, T "
            "(range 1.3686–1.3738, 95% CI ±0.059–0.060), confirming the HP formula "
            "cannot differentiate directional behaviour."
        )

        hp_df = pd.DataFrame({
            'Component':   ['Resultant', 'Vertical', 'Longitudinal', 'Transverse'],
            'K':           [2950.92, 1725.05, 1731.14, 1706.13],
            'α':           [1.3686, 1.3716, 1.3738, 1.3692],
            'R²':          [0.750, 0.755, 0.755, 0.752],
            'RMSE (mm/s)': [24.31, 13.93, 13.95, 13.98],
            'MAE (mm/s)':  [17.29, 9.87, 9.91, 9.94]
        })
        st.dataframe(hp_df, use_container_width=True, hide_index=True)

        # HP fit scatter
        st.subheader("HP Fit vs Measured (Log-Log)")
        from scipy.optimize import curve_fit
        def hp_formula(SD, K, alpha):
            return K * SD**(-alpha)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        targets  = [('PPV','Resultant','#333333'), ('V','Vertical','#2196F3'),
                    ('L','Longitudinal','#4CAF50'), ('T','Transverse','#FF5722')]
        hp_params = {
            'PPV': (2950.92, 1.3686), 'V': (1725.05, 1.3716),
            'L': (1731.14, 1.3738),   'T': (1706.13, 1.3692)
        }
        SD_range = np.linspace(df['SD'].min(), df['SD'].max(), 300)

        for ax, (col, name, color) in zip(axes.flatten(), targets):
            K, alpha = hp_params[col]
            ax.scatter(df['SD'], df[col], alpha=0.5, s=20, color=color, label='Measured')
            ax.plot(SD_range, hp_formula(SD_range, K, alpha),
                    'k-', lw=2, label=f'HP: K={K}, α={alpha}')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('Scaled Distance (m/kg^0.5)')
            ax.set_ylabel('PPV (mm/s)')
            r2 = hp_params[col]
            ax.set_title(f'{name} | R²={hp_params[col][0]:.0f}')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Fix titles
        r2_vals = {'PPV': 0.750, 'V': 0.755, 'L': 0.755, 'T': 0.752}
        for ax, (col, name, color) in zip(axes.flatten(), targets):
            K, alpha = hp_params[col]
            ax.set_title(f'{name} | R²={r2_vals[col]}  RMSE={hp_df[hp_df["Component"]==name]["RMSE (mm/s)"].values[0]} mm/s')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: ML Performance ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Individual Model Results")

        ml_df = pd.DataFrame({
            'Component':        ['Resultant','Resultant','Resultant',
                                 'Vertical','Longitudinal','Transverse'],
            'Model':            ['XGBoost','SVR','Random Forest','SVR','SVR','SVR'],
            'R²':               [0.747, 0.801, 0.753, 0.818, 0.820, 0.818],
            'RMSE (mm/s)':      [22.74, 20.17, 22.47, 11.29, 11.24, 11.27],
            'CV RMSE (mean±std)':['25.87±2.64','25.31±1.58','27.97±2.74',
                                  '14.52±1.06','14.50±1.07','14.62±1.01']
        })
        st.dataframe(ml_df, use_container_width=True, hide_index=True)

        st.subheader("Stacking Ensemble vs HP Formula")
        stacker_df = pd.DataFrame({
            'Component':    ['Resultant', 'Vertical', 'Longitudinal', 'Transverse'],
            'HP R²':        [0.750, 0.755, 0.755, 0.752],
            'Stacker R²':   [0.796, 0.815, 0.817, 0.819],
            'HP RMSE':      [24.31, 13.93, 13.95, 13.98],
            'Stacker RMSE': [20.43, 11.41, 11.33, 11.24],
            'RMSE Reduction': ['−16.0%', '−18.1%', '−18.8%', '−19.6%']
        })
        st.dataframe(stacker_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        comps   = ['Resultant', 'Vertical', 'Longitudinal', 'Transverse']
        x       = np.arange(len(comps))
        width   = 0.35

        # R² comparison
        hp_r2      = [0.750, 0.755, 0.755, 0.752]
        stacker_r2 = [0.796, 0.815, 0.817, 0.819]
        axes[0].bar(x - width/2, hp_r2, width, label='HP Formula',
                    color='#9E9E9E', alpha=0.8)
        axes[0].bar(x + width/2, stacker_r2, width, label='ML Stacker',
                    color='#2196F3', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(comps, rotation=10)
        axes[0].set_ylabel('R²')
        axes[0].set_title('R² Comparison: HP vs ML Stacker')
        axes[0].legend()
        axes[0].set_ylim(0.6, 0.9)
        axes[0].grid(True, alpha=0.3, axis='y')

        # RMSE comparison
        hp_rmse      = [24.31, 13.93, 13.95, 13.98]
        stacker_rmse = [20.43, 11.41, 11.33, 11.24]
        axes[1].bar(x - width/2, hp_rmse, width, label='HP Formula',
                    color='#9E9E9E', alpha=0.8)
        axes[1].bar(x + width/2, stacker_rmse, width, label='ML Stacker',
                    color='#F44336', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(comps, rotation=10)
        axes[1].set_ylabel('RMSE (mm/s)')
        axes[1].set_title('RMSE Comparison: HP vs ML Stacker')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Reconstruction test
        st.subheader("Reconstruction Test")
        st.markdown(
            "Predicting V, L, T separately and reconstructing √(V²+L²+T²) "
            "**outperforms** direct resultant prediction."
        )
        recon_df = pd.DataFrame({
            'Method':    ['Direct Resultant Stacker', 'Reconstructed from V+L+T'],
            'R²':        [0.796, 0.817],
            'RMSE (mm/s)': [20.43, 19.59],
            'Improvement': ['—', '−4.14% RMSE']
        })
        st.dataframe(recon_df, use_container_width=True, hide_index=True)
        st.success("Component-wise decomposition improves resultant accuracy by 4.14% RMSE.")

    # ── Tab 3: SHAP ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader("SHAP Feature Importance — All Components")
        st.markdown(
            "SHAP TreeExplainer applied to XGBoost model per component. "
            "Key finding: **log(SD) and log(D) rank higher for Transverse** than V or L, "
            "consistent with surface wave propagation characteristics."
        )

        shap_df = pd.DataFrame({
            'Feature':          ['SD','Distance','W²/D','log(SD)','Charge','log(D)','log(W)','√W'],
            'Resultant Rank':   [1, 2, 3, 4, 5, 6, 7, 8],
            'Resultant SHAP':   [24.85, 8.66, 3.06, 2.78, 1.99, 1.23, 0.21, 0.00],
            'Vertical Rank':    [1, 2, 4, 3, 5, 6, 7, 8],
            'Vertical SHAP':    [14.04, 5.04, 1.59, 1.65, 1.06, 0.69, 0.14, 0.00],
            'Longitudinal Rank':[1, 2, 3, 4, 5, 6, 7, 8],
            'Longitudinal SHAP':[14.51, 4.99, 1.71, 1.56, 1.06, 0.64, 0.20, 0.00],
            'Transverse Rank':  [1, 2, 5, 3, 6, 4, 7, 8],
            'Transverse SHAP':  [10.63, 4.55, 1.45, 3.90, 1.12, 1.56, 0.27, 0.09],
        })
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        # SHAP bar chart
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        shap_data = [
            ('Resultant',   [24.85,8.66,3.06,2.78,1.99,1.23,0.21,0.00], '#333333'),
            ('Vertical',    [14.04,5.04,1.59,1.65,1.06,0.69,0.14,0.00], '#2196F3'),
            ('Longitudinal',[14.51,4.99,1.71,1.56,1.06,0.64,0.20,0.00], '#4CAF50'),
            ('Transverse',  [10.63,4.55,1.45,3.90,1.12,1.56,0.27,0.09], '#FF5722'),
        ]
        feat_labels = ['SD','Distance','W²/D','log(SD)','Charge','log(D)','log(W)','√W']

        for ax, (name, shap_vals, color) in zip(axes, shap_data):
            sorted_idx = np.argsort(shap_vals)
            ax.barh([feat_labels[i] for i in sorted_idx],
                    [shap_vals[i] for i in sorted_idx],
                    color=color, alpha=0.8, edgecolor='white')
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Mean |SHAP|')
            ax.grid(True, alpha=0.3, axis='x')

        plt.suptitle('SHAP Feature Importance — All PPV Components',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "**Key insight:** log(SD) SHAP value for Transverse (3.90) is 2.4× "
            "that of Vertical (1.65), indicating transverse motion has a more complex "
            "logarithmic distance dependence — consistent with surface wave propagation."
        )

    # ── Tab 4: Amplification Factors ──────────────────────────────────────────
    with tab4:
        st.subheader("Directional Amplification Factors")
        st.markdown(
            "AF = measured component / HP-predicted resultant. "
            "Mean AF ≈ 0.577 ≈ 1/√3, consistent with near-isotropic energy partitioning."
        )

        af_summary = pd.DataFrame({
            'Component':    ['Vertical', 'Longitudinal', 'Transverse'],
            'Mean AF':      [0.573, 0.571, 0.571],
            'Std':          [0.244, 0.245, 0.245],
            'Spearman ρ':   [-0.156, -0.141, -0.137],
            'p-value':      [0.027, 0.046, 0.054],
            'Significant':  ['Yes (p<0.05)', 'Yes (p<0.05)', 'Marginal (p=0.054)']
        })
        st.dataframe(af_summary, use_container_width=True, hide_index=True)

        # Correction table
        st.subheader("Practical Correction Table (AF by Scaled Distance Bin)")
        correction_df = pd.DataFrame({
            'SD Bin':           ['Very Close (<14)', 'Close (14–21)',
                                 'Medium (21–28)', 'Far (28–35)', 'Very Far (>35)*'],
            'n':                [57, 54, 58, 29, 2],
            'AF_V (mean±std)':  ['0.589±0.177', '0.570±0.226',
                                 '0.558±0.268', '0.572±0.340', '0.677±0.000'],
            'AF_L (mean±std)':  ['0.589±0.177', '0.567±0.227',
                                 '0.556±0.269', '0.571±0.340', '0.676±0.012'],
            'AF_T (mean±std)':  ['0.587±0.177', '0.567±0.227',
                                 '0.555±0.269', '0.570±0.340', '0.682±0.012'],
        })
        st.dataframe(correction_df, use_container_width=True, hide_index=True)
        st.caption("*Very Far bin contains only 2 records — use with caution.")

        # Ratio analysis
        st.subheader("Component Ratio Analysis")
        ratio_df = pd.DataFrame({
            'Ratio':        ['V/L vs SD', 'L/T vs SD', 'V/T vs SD'],
            'Spearman ρ':   [+0.202, -0.175, +0.058],
            'p-value':      [0.004, 0.013, 0.419],
            'Significant':  ['Yes', 'Yes', 'No'],
            'Interpretation': [
                'Vertical becomes stronger relative to L at distance',
                'Longitudinal decays faster relative to T at distance',
                'No significant trend'
            ]
        })
        st.dataframe(ratio_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Directional Anisotropy in Blast-Induced Ground Vibration

    This application presents results from a research project investigating
    directional anisotropy in blast-induced ground vibration using ensemble
    machine learning.

    ### Research Summary
    Unlike existing literature which predicts a single resultant PPV,
    this work independently predicts all three directional components —
    **Vertical (V), Longitudinal (L), and Transverse (T)** — using a
    stacking ensemble with SHAP interpretability.

    ### Key Contributions
    1. HP formula fitted per component — α indistinguishable across V/L/T
    2. Ensemble ML (XGBoost + SVR + Ridge) reduces RMSE by **16–20%** vs HP
    3. SHAP reveals log(SD) and log(D) elevated for Transverse — surface wave signature
    4. Site-specific amplification factor correction table per direction
    5. Component reconstruction √(V²+L²+T²) improves resultant RMSE by **4.14%**

    ### Models Used
    - **XGBoost** — tuned with Optuna (100 trials, 5-fold CV, CUDA GPU)
    - **SVR (RBF kernel)** — tuned with GridSearchCV
    - **Stacking ensemble** — Ridge regression meta-learner

    ### Dataset
    200 field blast monitoring records, open-pit mine, India.
    Distance: 300–1250 m | Charge: 650–2950 kg | SD: 6.97–43.47 m/kg^0.5

    ### Authors
    MR Rashmi, KB Tharun Krishna
    Department of Mining Engineering, NIT Karnataka

    ### Repository
    [github.com/Tharun-bot/blast-induced-ppv-pred](https://github.com/Tharun-bot/blast-induced-ppv-pred)
    """)

    st.markdown("---")
    st.markdown("**Target Journal:** Rock Mechanics and Rock Engineering (Springer, IF 6.2)")
    st.markdown("**Status:** Manuscript in preparation")