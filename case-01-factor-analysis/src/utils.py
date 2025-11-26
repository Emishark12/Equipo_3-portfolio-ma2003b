import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def generate_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    plt.close()

def generate_scree_plot(df):
    # Preprocessing
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    fa = FactorAnalyzer(n_factors=10, rotation=None)
    fa.fit(df_scaled)
    ev, v = fa.get_eigenvalues()
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(ev)+1), ev, marker='o')
    plt.title('Scree Plot (Kaiser Criterion)')
    plt.xlabel('Factor')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
    return ev

def generate_factor_loadings_plot(df, n_factors):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(df_scaled)
    loadings = pd.DataFrame(fa.loadings_, index=df.columns)
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0)
    plt.title(f'Factor Loadings ({n_factors} Factors)')
    plt.tight_layout()
    plt.show()  
    plt.close()

def format_factor_loadings(df, threshold=0.4):
    """
    Formats the factor loadings dataframe with colors.
    Values below the threshold are hidden.
    Significant values are highlighted in green.
    """
    df_presentation = df.copy()
    # Mask values below threshold
    df_presentation[np.abs(df_presentation) < threshold] = np.nan

    def highlight_high(val):
        if pd.isna(val):
            return '' 
        return 'background-color: #8fbc8f' # DarkSeaGreen
    
    # Use map if available (pandas >= 2.1.0), otherwise applymap
    styler = df_presentation.style
    if hasattr(styler, 'map'):
        styler = styler.map(highlight_high)
    else:
        styler = styler.applymap(highlight_high)
        
    return styler.format(precision=3, na_rep="")

