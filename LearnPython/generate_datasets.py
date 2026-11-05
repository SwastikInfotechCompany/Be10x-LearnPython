"""
ShopEase Dataset Generator
Generates realistic industrial datasets for:
1. Customers (Demographics, Region, Signup)
2. Products (Categories: Electronics, Sports, Fashion, Home, UnitPrice, UnitCost)
3. Orders (Deliberately messy real-world data with missing discounts, varied date formats)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_shopease_data(output_dir="data", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Customers Dataset
    cities_regions = [
        ("New York", "North America"),
        ("San Francisco", "North America"),
        ("Chicago", "North America"),
        ("London", "Europe"),
        ("Berlin", "Europe"),
        ("Paris", "Europe"),
        ("Tokyo", "Asia-Pacific"),
        ("Singapore", "Asia-Pacific"),
        ("Sydney", "Asia-Pacific"),
        ("Toronto", "North America"),
        ("Mumbai", "Asia-Pacific"),
        ("Sao Paulo", "Latin America")
    ]
    
    first_names = ["James", "Emma", "Liam", "Olivia", "Noah", "Ava", "William", "Sophia", 
                   "Mason", "Isabella", "Ethan", "Mia", "Alexander", "Charlotte", "Daniel", 
                   "Amelia", "Henry", "Harper", "Lucas", "Evelyn", "Aarav", "Priya", "Kenji", "Elena"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                  "Davis", "Rodriguez", "Martinez", "Hernandez", "Wilson", "Anderson", "Taylor", 
                  "Thomas", "Patel", "Tanaka", "Mueller", "Dubois", "Silva"]

    num_customers = 600
    customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
    customer_data = []

    start_date = datetime(2023, 1, 1)
    for i, cid in enumerate(customer_ids):
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        city, region = random.choice(cities_regions)
        signup = start_date + timedelta(days=random.randint(0, 700))
        age_group = random.choice(["18-25", "26-35", "36-50", "51-65", "65+"])
        # Deliberately leave a few emails/cities missing to test data hygiene
        email = f"{fname.lower()}.{lname.lower()}{random.randint(1,99)}@example.com" if random.random() > 0.03 else None
        
        customer_data.append({
            "CustomerID": cid,
            "CustomerName": f"{fname} {lname}",
            "Email": email,
            "City": city,
            "Region": region,
            "SignupDate": signup.strftime("%Y-%m-%d"),
            "AgeGroup": age_group
        })

    df_customers = pd.DataFrame(customer_data)
    df_customers.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    print(f"[OK] Generated {len(df_customers)} customers -> {output_dir}/customers.csv")

    # 2. Products Dataset
    products = [
        # Electronics
        ("PROD-E101", "Ultra Pro Laptop 15-inch", "Electronics", 1299.99, 850.00),
        ("PROD-E102", "Noise-Cancelling Headphones", "Electronics", 249.99, 120.00),
        ("PROD-E103", "4K OLED Smart TV 55-inch", "Electronics", 899.99, 580.00),
        ("PROD-E104", "Wireless Ergonomic Mouse", "Electronics", 49.99, 18.00),
        ("PROD-E105", "Mechanical Gaming Keyboard", "Electronics", 129.99, 60.00),
        ("PROD-E106", "Smartwatch Series 5", "Electronics", 349.99, 190.00),
        
        # Sports & Outdoors
        ("PROD-S201", "Carbon Fiber Road Bike", "Sports", 1499.00, 890.00),
        ("PROD-S202", "Pro Running Shoes Elite", "Sports", 160.00, 65.00),
        ("PROD-S203", "Adjustable Dumbbell Set", "Sports", 299.99, 140.00),
        ("PROD-S204", "Yoga Mat High Density", "Sports", 39.99, 12.00),
        ("PROD-S205", "Waterproof Camping Tent", "Sports", 219.99, 105.00),
        ("PROD-S206", "Thermal Hydration Flask", "Sports", 29.99, 8.50),

        # Fashion & Apparel
        ("PROD-F301", "Italian Leather Jacket", "Fashion", 399.00, 180.00),
        ("PROD-F302", "Slim Fit Denim Jeans", "Fashion", 79.99, 28.00),
        ("PROD-F303", "Merino Wool Winter Sweater", "Fashion", 119.99, 45.00),
        ("PROD-F304", "Casual Breathable Sneakers", "Fashion", 89.99, 32.00),
        ("PROD-F305", "Polarized Designer Sunglasses", "Fashion", 149.99, 40.00),

        # Home & Kitchen
        ("PROD-H401", "Espresso Machine Pro", "Home & Kitchen", 599.99, 340.00),
        ("PROD-H402", "Cast Iron Dutch Oven 6qt", "Home & Kitchen", 119.99, 52.00),
        ("PROD-H403", "Smart Robotic Vacuum Cleaner", "Home & Kitchen", 399.99, 210.00),
        ("PROD-H404", "Chef's Japanese Steel Knife Set", "Home & Kitchen", 189.99, 75.00),
        ("PROD-H405", "Air Purifier HEPA Filtration", "Home & Kitchen", 149.99, 70.00),

        # Books & Stationery
        ("PROD-B501", "Executive Hardbound Notebook Set", "Books & Stationery", 24.99, 7.00),
        ("PROD-B502", "Data Science & AI Handbook", "Books & Stationery", 54.99, 22.00),
        ("PROD-B503", "Luxury Fountain Pen Brass", "Books & Stationery", 69.99, 25.00)
    ]

    df_products = pd.DataFrame(products, columns=["ProductID", "ProductName", "Category", "UnitPrice", "UnitCost"])
    df_products.to_csv(os.path.join(output_dir, "products.csv"), index=False)
    print(f"[OK] Generated {len(df_products)} products -> {output_dir}/products.csv")

    # 3. Orders Dataset (with intentional industrial data hygiene issues)
    num_orders = 10000
    order_records = []
    
    start_order_date = datetime(2025, 1, 1)
    end_order_date = datetime(2025, 12, 31)
    date_range_days = (end_order_date - start_order_date).days

    # Customer loyalty distribution: some high-value repeat customers (Pareto distribution)
    customer_weights = np.random.pareto(a=1.5, size=num_customers)
    customer_weights /= customer_weights.sum()

    product_pids = [p[0] for p in products]

    for order_idx in range(1, num_orders + 1):
        order_id = f"ORD-{20250000 + order_idx}"
        chosen_cust = np.random.choice(customer_ids, p=customer_weights)
        chosen_prod = random.choice(product_pids)
        
        # Chronological distribution: simulate retail seasonality (higher in Q4 holiday season)
        day_offset = random.randint(0, date_range_days)
        # 30% higher chance for Q4 (days 273 - 365)
        if random.random() < 0.35:
            day_offset = random.randint(270, date_range_days)
            
        order_date_dt = start_order_date + timedelta(days=day_offset)
        
        # Real-world dirty date formats
        fmt_rand = random.random()
        if fmt_rand < 0.70:
            order_date_str = order_date_dt.strftime("%Y-%m-%d")
        elif fmt_rand < 0.85:
            order_date_str = order_date_dt.strftime("%m/%d/%Y")
        else:
            order_date_str = order_date_dt.strftime("%d-%b-%Y") # e.g. 15-Nov-2025

        qty = np.random.choice([1, 2, 3, 4, 5, 8, 10], p=[0.55, 0.25, 0.10, 0.05, 0.03, 0.01, 0.01])
        
        # Missing discount percentage: 22% of orders have NULL discount (in retail: no promo applied = 0.0)
        # Promos range from 0%, 5%, 10%, 15%, 20%, 25%
        if random.random() < 0.22:
            discount_pct = np.nan # Deliberate null for AI Data Hygiene audit!
        else:
            discount_pct = round(random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]), 2)
            
        # Payment Method
        payment_method = random.choice(["Credit Card", "PayPal", "Apple Pay", "Wire Transfer", "Debit Card"])
        
        order_records.append({
            "OrderID": order_id,
            "CustomerID": chosen_cust,
            "ProductID": chosen_prod,
            "OrderDate": order_date_str,
            "Quantity": qty,
            "DiscountPct": discount_pct,
            "PaymentMethod": payment_method
        })

    df_orders = pd.DataFrame(order_records)
    df_orders.to_csv(os.path.join(output_dir, "orders.csv"), index=False)
    print(f"[OK] Generated {len(df_orders)} orders -> {output_dir}/orders.csv")
    print(f"[Hygiene Alert] Missing DiscountPct count in raw orders: {df_orders['DiscountPct'].isna().sum()} rows ({df_orders['DiscountPct'].isna().mean():.1%})")

if __name__ == "__main__":
    generate_shopease_data()
