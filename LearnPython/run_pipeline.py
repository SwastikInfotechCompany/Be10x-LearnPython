"""
End-to-End Orchestrator Pipeline
Executes:
1. Dataset Generation (ShopEase Orders, Products, Customers)
2. Part 1: Numerical Architecture & High-Speed Computation (NumPy)
3. Part 2: Strategic Data Engineering & AI-Driven Insights (Pandas)
"""

import sys
import time
from generate_datasets import generate_shopease_data
from part1_numpy_audit import run_part1_audit
from part2_pandas_shopease import run_part2_pipeline

def main():
    print("=" * 80)
    print("   AI-AUGMENTED DATA ANALYTICS & PYTHON AUTOMATION PIPELINE")
    print("   Industrial Problem Statement Solution - End-to-End Execution")
    print("=" * 80)
    start_total = time.perf_counter()

    # Step 1: Generate Datasets
    print("\n>>> STEP 1: Generating Realistic ShopEase Datasets...")
    generate_shopease_data(output_dir="data", seed=42)

    # Step 2: Part 1 NumPy Audit
    print("\n>>> STEP 2: Executing Part 1 (NumPy Computational Efficiency Audit)...")
    part1_res = run_part1_audit(num_employees=1_000_000, output_dir="reports")

    # Step 3: Part 2 Pandas Pipeline
    print("\n>>> STEP 3: Executing Part 2 (Pandas Relational & Chronological Engineering)...")
    part2_res = run_part2_pipeline(data_dir="data", output_dir="reports")

    total_duration = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_duration:.2f} SECONDS!")
    print("All deliverables generated in 'reports/' and 'data/' directories.")
    print("=" * 80)

if __name__ == "__main__":
    main()
