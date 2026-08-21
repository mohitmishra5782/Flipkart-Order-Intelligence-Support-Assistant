import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 6000

categories = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
cat_probs = [0.32, 0.22, 0.18, 0.18, 0.10]
payment_methods = ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"]
pay_probs = [0.42, 0.24, 0.24, 0.10]

product_category = rng.choice(categories, size=N, p=cat_probs)
payment_method = rng.choice(payment_methods, size=N, p=pay_probs)

base_price = {
    "Apparel": (400, 2200), "Electronics": (1200, 45000), "Home": (300, 8000),
    "Footwear": (500, 4500), "Beauty": (150, 2500),
}
price_inr = np.round(np.array([rng.uniform(*base_price[c]) for c in product_category]), 0)

discount_pct = np.clip(rng.normal(22, 15, N), 0, 75)
customer_tenure_days = np.clip(rng.exponential(380, N), 1, 2500).round(0)
num_previous_orders = np.clip((customer_tenure_days / 45) + rng.normal(0, 2, N), 0, None).round(0)
base_return_rate = np.clip(rng.beta(1.5, 9, N), 0, 1)
num_previous_returns = np.round(base_return_rate * num_previous_orders).clip(0, num_previous_orders)

delivery_distance_km = np.clip(rng.gamma(3, 90, N), 2, 2200).round(1)
delivery_days = np.clip(rng.normal(4.5, 2.2, N), 1, 21).round(0)
is_weekend_order = rng.integers(0, 2, N)

rating_given = rng.integers(1, 6, N).astype(float)
missing_mask = rng.random(N) < np.where(payment_method == "COD", 0.22, 0.06)
rating_given[missing_mask] = np.nan

fit_risk_cat = np.isin(product_category, ["Apparel", "Footwear"]).astype(float)
prev_return_ratio = np.where(num_previous_orders > 0,
                              num_previous_returns / np.maximum(num_previous_orders, 1), 0)

z = (-2.2 + 1.9 * prev_return_ratio + 0.55 * fit_risk_cat
     + 0.014 * (discount_pct - 20) / 10 + 0.9 * (payment_method == "COD").astype(float)
     + 0.10 * (delivery_days - 4.5) / 2 + 0.30 * (price_inr / base_price["Electronics"][1])
     + 0.05 * is_weekend_order - 0.15 * np.tanh(customer_tenure_days / 500))
prob_return = 1 / (1 + np.exp(-z))
returned = (rng.random(N) < prob_return).astype(int)

df = pd.DataFrame({
    "order_id": np.arange(1, N + 1), "product_category": product_category,
    "price_inr": price_inr, "discount_pct": np.round(discount_pct, 1),
    "payment_method": payment_method, "customer_tenure_days": customer_tenure_days.astype(int),
    "num_previous_orders": num_previous_orders.astype(int),
    "num_previous_returns": num_previous_returns.astype(int),
    "delivery_distance_km": delivery_distance_km, "delivery_days": delivery_days.astype(int),
    "is_weekend_order": is_weekend_order, "rating_given": rating_given, "returned": returned,
})
df.to_csv("orders_dataset.csv", index=False)
print("Rows:", len(df), "| Return rate:", round(df["returned"].mean(), 4))
