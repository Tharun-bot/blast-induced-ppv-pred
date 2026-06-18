import pandas as pd
import numpy as np

def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Rename columns to short names
    df.columns = [
        'id', 'Distance', 'Charge', 'SD', 
        'PPV', 'V', 'L', 'T'
    ]
    
    assert df.isnull().sum().sum() == 0, "Nulls found"
    
    # Engineered features
    df['logD']   = np.log10(df['Distance'])
    df['logW']   = np.log10(df['Charge'])
    df['logSD']  = np.log10(df['SD'])
    df['W2_D']   = df['Charge']**2 / df['Distance']
    df['sqrtW']  = np.sqrt(df['Charge'])
    
    # Component ratios
    df['ratio_VL'] = df['V'] / df['L']
    df['ratio_VT'] = df['V'] / df['T']
    df['ratio_LT'] = df['L'] / df['T']
    
    # Dominant component per row
    df['dominant'] = df[['V', 'L', 'T']].idxmax(axis=1)
    
    # Directional divergence: how much V/L/T disagree with EACH OTHER
    # (not with the resultant — a component is ~42% below the resultant by
    # geometry alone for near-isotropic data, so comparing to PPV was always
    # going to fire on every row regardless of real divergence)
    comp = df[['V', 'L', 'T']]
    df['divergence_pct'] = (comp.max(axis=1) - comp.min(axis=1)) / comp.mean(axis=1)
    # Data-driven flag: top decile of observed divergence (empirical, not assumed)
    threshold = df['divergence_pct'].quantile(0.90)
    df['high_divergence'] = (df['divergence_pct'] > threshold).astype(int)
    
    # SD quartile for stratified split later
    df['sd_quartile'] = pd.qcut(df['SD'], q=4, labels=False)
    
    return df

if __name__ == "__main__":
    df = load_and_engineer("data/raw/blast_data.csv")
    df.to_csv("data/processed.csv", index=False)
    
    print("Shape:", df.shape)
    print("\nHigh-divergence rows (top decile):", df['high_divergence'].sum())
    print("Divergence threshold used:", round(df['divergence_pct'].quantile(0.90)*100, 2), "%")
    print("\nDominant component counts:")
    print(df['dominant'].value_counts())
    print("\nComponent ratio stats:")
    print(df[['ratio_VL', 'ratio_VT', 'ratio_LT']].describe().round(4))
    print("\nFeature matrix preview:")
    print(df[['Distance','Charge','SD','logD','logW','logSD','W2_D','sqrtW']].describe().round(3))