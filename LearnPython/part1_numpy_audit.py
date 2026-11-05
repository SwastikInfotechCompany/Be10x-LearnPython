"""
Part 1: Numerical Architecture & High-Speed Computation (NumPy)
Industrial Implementation & Performance Audit

Key Modules:
1. Computational Efficiency Audit (1,000,000-row salary dataset: List vs NumPy 10% bonus vectorization)
2. Array Architecture & Reshaping (Linear inventory list -> 2D Multi-Warehouse Matrix)
3. Statistical Foundations (Multi-axis aggregation: mean, max, sum across Warehouses and Product lines)
"""

import time
import sys
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def run_part1_audit(num_employees=1_000_000, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("PART 1: NUMERICAL ARCHITECTURE & COMPUTATIONAL EFFICIENCY AUDIT")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. COMPUTATIONAL EFFICIENCY AUDIT: Lists vs NumPy
    # -------------------------------------------------------------
    print(f"\n[Audit 1/3] Benchmarking 10% Salary Bonus on {num_employees:,} employees...")
    
    # Generate identical baseline salary data
    np.random.seed(42)
    raw_salaries_np = np.random.uniform(45_000, 160_000, size=num_employees).astype(np.float64)
    raw_salaries_list = raw_salaries_np.tolist()

    # Time Python List computation
    start_list = time.perf_counter()
    bonus_list = [salary * 1.10 for salary in raw_salaries_list]
    end_list = time.perf_counter()
    list_time_sec = end_list - start_list
    list_time_ms = list_time_sec * 1000

    # Time NumPy Vectorized computation
    start_np = time.perf_counter()
    bonus_np = raw_salaries_np * 1.10
    end_np = time.perf_counter()
    np_time_sec = end_np - start_np
    np_time_ms = np_time_sec * 1000

    speedup_factor = list_time_sec / np_time_sec if np_time_sec > 0 else float("inf")
    time_saved_pct = ((list_time_sec - np_time_sec) / list_time_sec) * 100

    # Memory Footprint Comparison
    # Python list: base list object + pointer per item (8 bytes each) + PyFloatObject overhead (~24 bytes each)
    list_memory_mb = (sys.getsizeof(raw_salaries_list) + (num_employees * 24)) / (1024 * 1024)
    np_memory_mb = raw_salaries_np.nbytes / (1024 * 1024)

    print(f" -> Standard Python List Execution Time: {list_time_ms:.2f} ms ({list_time_sec:.4f} s)")
    print(f" -> NumPy Vectorized Execution Time:     {np_time_ms:.2f} ms ({np_time_sec:.4f} s)")
    print(f" -> Speedup Factor:                      {speedup_factor:.1f}x FASTER")
    print(f" -> Time Efficiency Gap:                 {time_saved_pct:.2f}% reduction in execution time")
    print(f" -> Memory Footprint: List: {list_memory_mb:.2f} MB vs NumPy: {np_memory_mb:.2f} MB ({list_memory_mb/np_memory_mb:.1f}x memory efficiency)")

    # -------------------------------------------------------------
    # 2. ARRAY ARCHITECTURE & RESHAPING: Warehouse Inventory
    # -------------------------------------------------------------
    print("\n[Audit 2/3] Array Architecture & Reshaping: Warehouse Inventory Tracking...")
    warehouses = ["WH-North-US", "WH-East-US", "WH-West-US", "WH-Central-EU", "WH-APAC-HQ"]
    product_categories = [
        "Laptops", "Smartphones", "Smartwatches", "4K-TVs", 
        "Road-Bikes", "Running-Shoes", "Gym-Weights", "Camp-Tents",
        "Leather-Jackets", "Denim-Jeans", "Espresso-Machines", "Robotic-Vacuums"
    ]
    
    total_elements = len(warehouses) * len(product_categories) # 5 * 12 = 60
    
    # Simulate a raw 1D linear stream from automated barcode scanners
    raw_linear_inventory = np.random.randint(15, 450, size=total_elements, dtype=np.int32)
    
    # Reshape 1D -> 2D Matrix (5 Warehouses x 12 Categories)
    warehouse_matrix = raw_linear_inventory.reshape(len(warehouses), len(product_categories))

    architecture_audit = {
        "1D_Raw_Shape": list(raw_linear_inventory.shape),
        "1D_NDIM": int(raw_linear_inventory.ndim),
        "2D_Reshaped_Shape": list(warehouse_matrix.shape),
        "2D_NDIM": int(warehouse_matrix.ndim),
        "Total_Elements_Size": int(warehouse_matrix.size),
        "Data_Type": str(warehouse_matrix.dtype),
        "Item_Size_Bytes": int(warehouse_matrix.itemsize),
        "Total_Buffer_Nbytes": int(warehouse_matrix.nbytes),
        "Strides": list(warehouse_matrix.strides),
        "Is_C_Contiguous": bool(warehouse_matrix.flags['C_CONTIGUOUS'])
    }
    
    print(f" -> 1D Linear Stream Shape: {raw_linear_inventory.shape} (ndim={raw_linear_inventory.ndim})")
    print(f" -> Reshaped 2D Matrix:    {warehouse_matrix.shape} (ndim={warehouse_matrix.ndim})")
    print(f" -> Memory Strides:         {warehouse_matrix.strides} (Row step: {warehouse_matrix.strides[0]} bytes)")
    print(f" -> Contiguity Status:      C-Contiguous = {warehouse_matrix.flags['C_CONTIGUOUS']}")

    # -------------------------------------------------------------
    # 3. STATISTICAL FOUNDATIONS ACROSS AXES
    # -------------------------------------------------------------
    print("\n[Audit 3/3] Statistical Aggregation across Multidimensional Axes...")
    # Axis 0: Column-wise aggregation (across all Warehouses for each Product Category)
    stock_per_category_sum = np.sum(warehouse_matrix, axis=0)
    stock_per_category_avg = np.mean(warehouse_matrix, axis=0)
    stock_per_category_max = np.max(warehouse_matrix, axis=0)

    # Axis 1: Row-wise aggregation (across all Products for each Regional Warehouse)
    stock_per_warehouse_sum = np.sum(warehouse_matrix, axis=1)
    stock_per_warehouse_avg = np.mean(warehouse_matrix, axis=1)
    stock_per_warehouse_max = np.max(warehouse_matrix, axis=1)

    global_total_units = int(np.sum(warehouse_matrix))
    global_avg_units = float(np.mean(warehouse_matrix))
    global_std_units = float(np.std(warehouse_matrix))

    busiest_warehouse_idx = int(np.argmax(stock_per_warehouse_sum))
    busiest_warehouse_name = warehouses[busiest_warehouse_idx]
    highest_stock_cat_idx = int(np.argmax(stock_per_category_sum))
    highest_stock_cat_name = product_categories[highest_stock_cat_idx]

    print(f" -> Global Inventory Total Units: {global_total_units:,}")
    print(f" -> Global Mean Stock per Cell:   {global_avg_units:.1f} +/- {global_std_units:.1f}")
    print(f" -> Warehouse with Max Stock:     {busiest_warehouse_name} ({stock_per_warehouse_sum[busiest_warehouse_idx]:,} units)")
    print(f" -> Category with Max Stock:      {highest_stock_cat_name} ({stock_per_category_sum[highest_stock_cat_idx]:,} units)")

    # -------------------------------------------------------------
    # GENERATE VISUAL AUDIT CHART
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Execution Time Bar Chart (Log Scale & Linear Annotations)
    ax1 = axes[0]
    bars = ax1.bar(['Python List Comprehension', 'NumPy Vectorized Array'], 
                   [list_time_ms, np_time_ms], 
                   color=['#ef4444', '#10b981'], width=0.5, edgecolor='#334155', linewidth=1.5)
    ax1.set_ylabel('Execution Time (Milliseconds)', fontsize=11, fontweight='bold')
    ax1.set_title(f'1,000,000 Row Salary Bonus Calculation\nSpeedup: {speedup_factor:.1f}x Faster with NumPy', 
                  fontsize=12, fontweight='bold', pad=12)
    
    # Value annotations on bars
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f} ms',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    ax1.text(0.5, 0.65, f'Time-Efficiency Gap:\n{time_saved_pct:.1f}% Execution Time Saved', 
             transform=ax1.transAxes, ha='center',
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#94a3b8", lw=1.5),
             fontsize=11, color='#0f172a', fontweight='bold')

    # Chart 2: Warehouse Inventory Distribution by Region (Axis 1)
    ax2 = axes[1]
    y_pos = np.arange(len(warehouses))
    ax2.barh(y_pos, stock_per_warehouse_sum, color='#3b82f6', edgecolor='#1e3a8a', linewidth=1.2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(warehouses, fontsize=10, fontweight='normal')
    ax2.invert_yaxis()  # top-down
    ax2.set_xlabel('Total Units in Stock (Across 12 Categories)', fontsize=11, fontweight='bold')
    ax2.set_title('Regional Warehouse Inventory Audit\n(2D Matrix Axis 1 Sum)', fontsize=12, fontweight='bold', pad=12)

    for i, v in enumerate(stock_per_warehouse_sum):
        ax2.text(v + 15, i, f'{v:,} units', va='center', fontweight='bold', fontsize=10, color='#1e293b')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "part1_benchmark_chart.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\n[OK] Saved Benchmark Visualization -> {chart_path}")

    # Build Deliverable Payload
    report_data = {
        "benchmark": {
            "num_employees": num_employees,
            "list_time_ms": round(list_time_ms, 2),
            "np_time_ms": round(np_time_ms, 2),
            "speedup_factor": round(speedup_factor, 1),
            "time_saved_pct": round(time_saved_pct, 2),
            "list_memory_mb": round(list_memory_mb, 2),
            "np_memory_mb": round(np_memory_mb, 2)
        },
        "architecture_audit": architecture_audit,
        "warehouse_summary": {
            "warehouses": warehouses,
            "categories": product_categories,
            "stock_matrix": warehouse_matrix.tolist(),
            "stock_per_warehouse_sum": stock_per_warehouse_sum.tolist(),
            "stock_per_warehouse_avg": [round(float(x), 1) for x in stock_per_warehouse_avg],
            "stock_per_warehouse_max": [int(x) for x in stock_per_warehouse_max],
            "stock_per_category_sum": stock_per_category_sum.tolist(),
            "stock_per_category_avg": [round(float(x), 1) for x in stock_per_category_avg],
            "stock_per_category_max": [int(x) for x in stock_per_category_max],
            "global_total_units": global_total_units,
            "global_avg_units": round(global_avg_units, 1),
            "busiest_warehouse": busiest_warehouse_name,
            "highest_stock_category": highest_stock_cat_name
        }
    }

    report_json_path = os.path.join(output_dir, "part1_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[OK] Saved Performance Audit Report -> {report_json_path}")
    return report_data

if __name__ == "__main__":
    run_part1_audit()
