# 📘 Price Elasticity Simulator

A complete business decision support system that analyzes how pricing affects product demand, revenue, profit, and market behavior. This project helps organizations make smarter pricing decisions using data, analytics, and simulation.

---

## 📊 Overview

This tool allows you to:

- Analyze historical pricing vs units sold
- Calculate **price elasticity of demand**
- Build **demand curves**
- Predict impact of price changes
- Simulate scenarios such as:
  - “If price increases by 8%, revenue falls by X%”
  - “If price drops by 5%, units increase by Y”
- Recommend **optimal pricing**
- Generate AI-like actionable insights
- Display everything in an **interactive Streamlit dashboard**

---

## ✨ Key Features

### 1. **Data Cleaning & Preparation**
- Removes missing or inconsistent entries  
- Cleans currency formatting  
- Handles outliers  
- Adds competitor price simulation  
- Adds seasonal demand factors

### 2. **Feature Engineering**
Automatically adds:
- Discount percentage  
- Competitor pricing & price gap  
- Seasonal index  
- Marketing spend impact  

### 3. **Price–Demand Modeling**
Models included:
- Linear Regression  
- Log-Log Elasticity Model  
- (Extendable to Polynomial Curve Model)

Elasticity formula:
```bash
Elasticity = (% Change in Quantity) / (% Change in Price)
```
Elasticity interpretation:
- `< -1` → **Elastic** (price sensitive)  
- `-1 to 0` → **Inelastic** (safe to increase price)

### 4. **Elasticity Simulator**
Inputs:
- New price  
- Cost price  
- Marketing change  
- Season impact  

Outputs:
- Predicted quantity  
- Revenue  
- Profit  
- Elasticity  
- Margin and risk impact  

### 5. **Smart Pricing Recommendation Engine**
Provides business decisions such as:
- Increase or decrease price  
- Expected revenue changes  
- Competitor comparison  
- Seasonal recommendations  
- Optimal revenue-generating price  

### 6. **Interactive Dashboard**
Includes:
- Demand curve  
- Price vs units sold chart  
- Elasticity distribution  
- Revenue sensitivity table  
- Optimal pricing recommendation  
- AI-style insights panel  

---


---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/price-elasticity-simulator
cd price-elasticity-simulator
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Run the Streamlit Dashboard
```bash
streamlit run app/app.py
```
---


---
## 🧠 Models Used
**Linear Regression**

Base demand model used to generate the demand curve.

**Log-Log Model**

Coefficient directly gives elasticity.

**Rule-Based AI Recommendation Engine**

Turns elasticity + season + competitor data into business advice.

## 📊 Dashboard Highlights

- Demand curve visualization

- Price vs units sold chart

- Elasticity distribution

- Revenue sensitivity matrix

- Optimal price finder

- Business recommendation panel

## 📈 Example Insights

- “Demand is elastic at the current price. A 10% drop may increase units by 18–22%.”

- “Competitor pricing is lower. Reducing price can increase competitiveness.”

- “Peak season ahead—recommended small price increase of 4–6%.”

- “Maximum revenue occurs at ₹298. Current price ₹350 is too high.”

## 🚀 Future Enhancements

- Multi-product cross elasticity

- ARIMA / Prophet forecasting

- Profit-based optimization model

- SQL database integration

- Power BI dashboard version
