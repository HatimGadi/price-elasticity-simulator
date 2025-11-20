import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

def add_features(df):
    df = df.copy()

    # 1. Create Discounts column if missing
    if 'Discounts' not in df.columns:
        df['Discounts'] = 0.0  # default no discount

    # 2. Discount percentage
    df['Discount_Pct'] = np.where(
        df['Sale_Price'] > 0,
        df['Discounts'] / df['Sale_Price'],
        0
    )

    # 3. Competitor price (synthetic unless you have real data)
    df['Comp_Price'] = df['Sale_Price'] * np.random.uniform(0.95, 1.1, len(df))
    df['Price_vs_Comp'] = df['Sale_Price'] - df['Comp_Price']

    # 4. Season index (create default if missing)
    if 'Season' not in df.columns:
        df['Season'] = 'Normal'

    season_map = {'Peak': 1.15, 'Normal': 1.0, 'Off-Peak': 0.9}
    df['Seasonal_Index'] = df['Season'].map(season_map).fillna(1.0)

    return df


# ---------- Modeling ----------
def fit_linear_model(df, feature='Sale_Price', target='Units_Sold'):
    X = df[[feature]].values.reshape(-1, 1)
    y = df[target].values
    m = LinearRegression()
    m.fit(X, y)
    return m

def fit_loglog_model(df, feature='Sale_Price', target='Units_Sold'):
    # log-log: log(Q) = a + b*log(P)
    df2 = df[(df[feature] > 0) & (df[target] > 0)].copy()
    X = np.log(df2[[feature]].values).reshape(-1, 1)
    y = np.log(df2[target].values)
    m = LinearRegression()
    m.fit(X, y)
    return m, df2

# Elasticity from linear model (point elasticity)
def elasticity_point_linear(model, price, quantity):
    b = model.coef_[0]
    return b * (price / quantity)

# Elasticity from log-log model (coefficient = elasticity)
def elasticity_loglog(model):
    return float(model.coef_[0])

# ---------- Simulator core ----------
def simulate_price_linear(model, price, cost_per_unit=0.0):
    qty = model.intercept_ + model.coef_[0] * price
    qty = max(qty, 0)
    revenue = qty * price
    profit = (price - cost_per_unit) * qty
    elasticity = elasticity_point_linear(model, price, qty) if qty>0 else 0.0
    return {'Price': price, 'Pred_Qty': qty, 'Revenue': revenue, 'Profit': profit, 'Elasticity': elasticity}

def find_optimal_price_revenue(model, price_min=1, price_max=2000):
    # brute force search (coarse)
    prices = np.linspace(price_min, price_max, 1000)
    sims = [simulate_price_linear(model, p) for p in prices]
    best = max(sims, key=lambda x: x['Revenue'])
    return best