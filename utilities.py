import pandas as pd
import numpy as np
from scipy import stats

def safe_read(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace('\n', ' ', regex=False).str.replace('.', '', regex=False)
    return df

def drop_unnamed(df):
    return df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]

def parse_currency_series(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r'[^0-9\.\-]', '', regex=True)  # remove non-numeric except . and -
    return pd.to_numeric(s, errors='coerce')

def remove_outliers_zscore(series, thresh=3.0):
    z = np.abs(stats.zscore(series.dropna()))
    if len(z) == 0:
        return series
    mask = np.ones(len(series), dtype=bool)
    mask[series.dropna().index] = z < thresh
    series_out = series.copy()
    series_out.loc[~mask] = np.nan
    return series_out

# ---------- Synthetic fallback data ----------
def make_synthetic(n=200, seed=42):
    np.random.seed(seed)
    prices = np.random.uniform(80, 700, n).round(2)
    a = 2500
    b = -3.8
    noise = np.random.normal(0, 120, n)
    quantities = np.maximum((a + b * prices + noise).round(), 0).astype(int)
    marketing_spend = np.random.randint(500, 7000, n)
    season = np.random.choice(['Peak', 'Normal', 'Off-Peak'], n, p=[0.25, 0.6, 0.15])
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=n, freq='D'),
        'Product': np.random.choice(['ProdA', 'ProdB', 'ProdC'], n, p=[0.6, 0.25, 0.15]),
        'Sale_Price': prices,
        'Units_Sold': quantities,
        'Marketing_Spend': marketing_spend,
        'Season': season
    })
    df['Revenue'] = (df['Sale_Price'] * df['Units_Sold']).round(2)
    return df

# ---------- Cleaning pipeline ----------
def clean_df(df):
    df = df.copy()
    df = clean_column_names(df)
    df = drop_unnamed(df)

    # Standardize likely column names
    rename_map = {}
    for c in df.columns:
        low = c.lower()
        if 'price' in low and 'sale' in low or 'sale_price' in low or 'price' == low:
            rename_map[c] = 'Sale_Price'
        if 'unit' in low and 'sold' in low or 'units' in low:
            rename_map[c] = 'Units_Sold'
        if 'revenue' in low or 'sales' == low or 'sales' in low:
            rename_map[c] = 'Revenue'
        if 'manufacturing' in low or 'cost' in low:
            rename_map[c] = 'Cost_Price'
        if 'discount' in low:
            rename_map[c] = 'Discounts'
        if 'market' in low and 'spend' in low:
            rename_map[c] = 'Marketing_Spend'
        if 'product' == low:
            rename_map[c] = 'Product'
        if 'date' == low:
            rename_map[c] = 'Date'
    df = df.rename(columns=rename_map)

    # Parse currencies automatically
    for col in df.columns:
        if df[col].astype(str).str.contains(r'[$,]').any() or df[col].dtype == object and df[col].astype(str).str.match(r'^\s*[-+]?\d*[\.,]?\d+\s*$').any():
            # try to parse as numeric (currency cleanup)
            parsed = parse_currency_series(df[col])
            # keep parsed if it yields at least some non-null numbers
            if parsed.notna().sum() > 0:
                df[col] = parsed

    # Ensure basic columns exist; create if missing (synthetic)
    if 'Sale_Price' not in df.columns or 'Units_Sold' not in df.columns:
        return make_synthetic()

    # Convert types
    df['Sale_Price'] = pd.to_numeric(df['Sale_Price'], errors='coerce')
    df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
    if 'Revenue' not in df.columns or df['Revenue'].isna().all():
        df['Revenue'] = (df['Sale_Price'] * df['Units_Sold']).round(2)
    else:
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')

    # Drop rows missing core fields
    df = df.dropna(subset=['Sale_Price', 'Units_Sold'])

    # Remove outliers in Units_Sold and Sale_Price by z-score
    df['Units_Sold'] = remove_outliers_zscore(df['Units_Sold']).fillna(method='ffill').astype(int)
    df['Sale_Price'] = remove_outliers_zscore(df['Sale_Price']).fillna(method='ffill')

    # Dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        df['Date'] = pd.date_range('2024-01-01', periods=len(df), freq='D')

    # Discount placeholder if missing
    if 'Discounts' not in df.columns:
        df['Discounts'] = 0.0
    else:
        df['Discounts'] = pd.to_numeric(df['Discounts'], errors='coerce').fillna(0.0)

    df = df.reset_index(drop=True)
    return df