# app.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from utilities import clean_df, safe_read, make_synthetic
from FeatureEngineering import add_features, fit_linear_model, fit_loglog_model, find_optimal_price_revenue, elasticity_loglog, simulate_price_linear, elasticity_point_linear
from engine import price_recommendation_engine

st.set_page_config(layout="wide", page_title="Price Elasticity Simulator")

st.title("Price Elasticity Simulator — Full Pipeline")

# Load data 
uploaded = st.file_uploader("Upload sales CSV (optional)", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)
else:
    raw = safe_read('/mnt/data/Sales_v1.csv')
df = clean_df(raw) if raw is not None else make_synthetic()
df = add_features(df)

# product selector 
products = df['Product'].unique() if 'Product' in df.columns else np.array(['All'])
selected_product = st.sidebar.selectbox("Select Product", options=np.append(['All'], products))

if selected_product == 'All':
    working_df = df.copy()
else:
    working_df = df[df['Product'] == selected_product].copy()

st.sidebar.markdown("### Simulator inputs")
default_price = float(working_df['Sale_Price'].median())
price_input = st.sidebar.slider("Simulate Price", min_value=float(max(1, working_df['Sale_Price'].min()*0.5)),
                                max_value=float(working_df['Sale_Price'].max()*1.8),
                                value=default_price, step=1.0)
cost_input = st.sidebar.number_input("Cost per unit (for profit calc)", value=0.0, step=1.0)
marketing_change = st.sidebar.slider("Marketing spend change (%)", -100, 200, 0)
season_override = st.sidebar.selectbox("Simulate season", options=['No change', 'Peak', 'Normal', 'Off-Peak'])

# Visualize raw data
st.subheader("Data Snapshot")
st.dataframe(working_df.head(50))

st.subheader("Price vs Units Sold")
fig, ax = plt.subplots()
ax.scatter(working_df['Sale_Price'], working_df['Units_Sold'], alpha=0.6)
ax.set_xlabel("Sale Price")
ax.set_ylabel("Units Sold")
ax.set_title("Price vs Units Sold")
st.pyplot(fig)

# Fit models
lin_model = fit_linear_model(working_df)
log_model, log_df2 = None, None
try:
    log_model, log_df2 = fit_loglog_model(working_df)
except Exception:
    log_model = None

st.subheader("Model Coefficients")
st.write("Linear model intercept (a):", float(lin_model.intercept_))
st.write("Linear model price coefficient (b):", float(lin_model.coef_[0]))
if log_model is not None:
    st.write("Log-Log model elasticity (approx):", float(log_model.coef_[0]))
else:
    st.write("Log-Log model: insufficient positive data for fit")

# Elasticity distribution
working_df['Elasticity_Point'] = working_df.apply(lambda r: elasticity_point_linear(lin_model, r['Sale_Price'], r['Units_Sold']) if r['Units_Sold']>0 else np.nan, axis=1)
st.subheader("Elasticity (sample)")
st.write(working_df['Elasticity_Point'].describe())

# Simulate a price
sim = simulate_price_linear(lin_model, price_input, cost_per_unit=cost_input)

sim_multiplier = 1.0
if marketing_change != 0:
    sim_multiplier *= (1 + marketing_change / 100.0)
if season_override != 'No change':
    season_map = {'Peak': 1.15, 'Normal': 1.0, 'Off-Peak': 0.9}
    sim_multiplier *= season_map.get(season_override, 1.0)
sim['Pred_Qty_Adjusted'] = sim['Pred_Qty'] * sim_multiplier
sim['Revenue_Adjusted'] = sim['Pred_Qty_Adjusted'] * sim['Price']
sim['Profit_Adjusted'] = (sim['Price'] - cost_input) * sim['Pred_Qty_Adjusted']

st.subheader("Simulation Result")
st.metric("Predicted Units (base)", int(sim['Pred_Qty']))
st.metric("Predicted Units (adjusted)", int(sim['Pred_Qty_Adjusted']))
st.metric("Revenue (adjusted)", f"{sim['Revenue_Adjusted']:.2f}")
st.metric("Profit (adjusted)", f"{sim['Profit_Adjusted']:.2f}")
st.write("Elasticity (point, linear):", f"{sim['Elasticity']:.3f}")

# Revenue sensitivity table
st.subheader("Revenue Sensitivity (Price Sweep)")
prices_sweep = np.linspace(max(1, working_df['Sale_Price'].min()*0.5), working_df['Sale_Price'].max()*1.8, 100)
sweep = [simulate_price_linear(lin_model, p, cost_per_unit=cost_input) for p in prices_sweep]
sweep_df = pd.DataFrame(sweep)
sweep_df['Revenue_Adjusted'] = sweep_df['Revenue'] * (1 + marketing_change/100.0)
st.dataframe(sweep_df.sort_values('Revenue_Adjusted', ascending=False).head(10))

# Optimal price 
best = find_optimal_price_revenue(lin_model,
                                  price_min=float(max(1, working_df['Sale_Price'].min()*0.5)),
                                  price_max=float(working_df['Sale_Price'].max()*1.8))
st.subheader("Optimal Price for Max Revenue (linear model)")
st.write(best)

st.subheader("Smart Pricing Recommendation Engine")
recommend = price_recommendation_engine(working_df, lin_model, price_input, cost_input)
st.write(recommend)

# Demand curve plot + fitted line
st.subheader("Demand Curve and Fitted Lines")
fig2, ax2 = plt.subplots()
ax2.scatter(working_df['Sale_Price'], working_df['Units_Sold'], alpha=0.4, label='data')
ps = np.linspace(working_df['Sale_Price'].min(), working_df['Sale_Price'].max(), 100)
pred_q_lin = lin_model.intercept_ + lin_model.coef_[0] * ps
ax2.plot(ps, pred_q_lin, label='Linear fit')
if log_model is not None:
    log_pred = np.exp(log_model.intercept_ + log_model.coef_[0] * np.log(ps))
    ax2.plot(ps, log_pred, label='Log-Log fit')
ax2.set_xlabel("Price")
ax2.set_ylabel("Units Sold")
ax2.legend()
st.pyplot(fig2)

# Executive summary panel
st.subheader("Executive Summary (auto)")
summary_lines = []
avg_elast = working_df['Elasticity_Point'].mean()
summary_lines.append(f"Sample size: {len(working_df)}")
summary_lines.append(f"Average elasticity (point, linear): {avg_elast:.3f}")
summary_lines.append(f"Most elastic product (by avg abs elasticity): N/A (single product view)")
summary_lines.append(f"Best revenue price (linear model): Price = {best['Price']:.2f} with revenue {best['Revenue']:.2f}")
st.write("\n".join(summary_lines))


#  Export cleaned dataset
st.subheader("Download cleaned dataset")
st.write("Use the button below to download the cleaned dataset for further analysis.")
st.download_button("Download CSV", data=working_df.to_csv(index=False).encode('utf-8'), file_name='cleaned_sales.csv', mime='text/csv')
