from FeatureEngineering import simulate_price_linear, find_optimal_price_revenue

def price_recommendation_engine(df, linear_model, current_price, cost_price=0):

    curr = simulate_price_linear(linear_model, current_price, cost_per_unit=cost_price)

    best = find_optimal_price_revenue(linear_model,
                                      price_min=float(df['Sale_Price'].min()),
                                      price_max=float(df['Sale_Price'].max())*1.5)
    best_price = best['Price']
    best_revenue = best['Revenue']

    #  Elasticity check
    elasticity_now = curr['Elasticity']

    if elasticity_now < -1:
        elasticity_label = "Elastic (Price sensitive)"
        action = "Reducing price may increase revenue"
    elif elasticity_now > -0.2:
        elasticity_label = "Highly Inelastic (Safe to increase price)"
        action = "You can safely increase the price"
    else:
        elasticity_label = "Inelastic (Some sensitivity)"
        action = "Small price changes won't affect sales much"

    #  Competitor comparison
    avg_comp_price = df['Comp_Price'].mean()
    if current_price > avg_comp_price:
        comp_msg = "You are priced ABOVE competitor average"
    else:
        comp_msg = "You are priced BELOW competitor average"

    #  Season impact
    if 'Seasonal_Index' in df.columns:
        season_avg = df['Seasonal_Index'].mean()
        if season_avg > 1.05:
            season_msg = "Demand is naturally high (Peak season)"
        elif season_avg < 0.95:
            season_msg = "Demand is low (Off-Peak). Consider promotions."
        else:
            season_msg = "Normal season"
    else:
        season_msg = "No season data available"

    #  Full recommendation
    return {
        "Current_Price": current_price,
        "Current_Elasticity": elasticity_now,
        "Elasticity_Interpretation": elasticity_label,
        "Action_Suggestion": action,
        "Best_Price_to_Set": round(best_price, 2),
        "Expected_Revenue_at_Best_Price": round(best_revenue, 2),
        "Competitor_Position": comp_msg,
        "Seasonal_Impact": season_msg
    }
