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
    
    # Anomaly flag: any component deviates >30% from resultant
    df['anomalous'] = (
        (abs(df['V'] - df['PPV']) / df['PPV'] > 0.09) |
        (abs(df['L'] - df['PPV']) / df['PPV'] > 0.09) |
        (abs(df['T'] - df['PPV']) / df['PPV'] > 0.09)
    ).astype(int)
    
    # SD quartile for stratified split later
    df['sd_quartile'] = pd.qcut(df['SD'], q=4, labels=False)
    
    return df

if __name__ == "__main__":
    df = load_and_engineer("data/raw/blast_data.csv")
    df.to_csv("data/processed.csv", index=False)
    
    print("Shape:", df.shape)
    print("\nAnomalous rows:", df['anomalous'].sum())
    print("\nDominant component counts:")
    print(df['dominant'].value_counts())
    print("\nComponent ratio stats:")
    print(df[['ratio_VL', 'ratio_VT', 'ratio_LT']].describe().round(4))
    print("\nFeature matrix preview:")
    print(df[['Distance','Charge','SD','logD','logW','logSD','W2_D','sqrtW']].describe().round(3))