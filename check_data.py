"""
check_data.py
PRG4206 Machine Learning Assignment: Explainable Predictive Maintenance
Script to inspect and validate the AI4I 2020 dataset in data/.
Generates a comprehensive summary saved to results/dataset_summary.txt.
"""

import sys
from pathlib import Path
import pandas as pd


def check_dataset():
    # Resolve project root relative to script location
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_file = results_dir / "dataset_summary.txt"

    print("=" * 70)
    print("PRG4206 Predictive Maintenance - Dataset Verification & Inspection")
    print("=" * 70)

    # 1. Verify data directory and locate CSV
    if not data_dir.exists():
        print(f"[ERROR] Data directory does not exist at: {data_dir}")
        sys.exit(1)

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV file found in: {data_dir}")
        sys.exit(1)

    # Prefer ai4i2020.csv if present
    target_csv = None
    for f in csv_files:
        if f.name.lower() == "ai4i2020.csv":
            target_csv = f
            break
    if target_csv is None:
        target_csv = csv_files[0]

    filename = target_csv.name
    file_size_bytes = target_csv.stat().st_size
    print(f"Dataset File Found: {target_csv.name}")
    print(f"Path: {target_csv}")
    print(f"File Size: {file_size_bytes:,} bytes ({file_size_bytes / 1024:.2f} KB)")

    # 2. Load CSV
    try:
        df = pd.read_csv(target_csv)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)

    rows, cols = df.shape
    print(f"\nDimensions: {rows} rows, {cols} columns")

    # 3. Column inspection
    columns = list(df.columns)
    dtypes = df.dtypes
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / rows) * 100
    total_missing = missing_counts.sum()
    duplicate_rows = df.duplicated().sum()

    # 4. Target variable verification
    target_col = "Machine failure"
    target_exists = target_col in df.columns
    if not target_exists:
        print(f"[ERROR] Target column '{target_col}' not found in dataset columns: {columns}")
        validation_succeeded = False
        target_counts = None
        target_dist_str = "TARGET COLUMN MISSING"
    else:
        target_counts = df[target_col].value_counts()
        target_pct = df[target_col].value_counts(normalize=True) * 100
        target_dist_str = "\n".join(
            [f"  Class {cls}: {count:,} ({target_pct[cls]:.2f}%)" for cls, count in target_counts.items()]
        )

    # Check failure breakdown modes if present
    failure_mode_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    present_failure_modes = [col for col in failure_mode_cols if col in df.columns]

    # 5. Validation Check
    validation_issues = []
    if rows == 0 or cols == 0:
        validation_issues.append("Dataset is empty.")
    if not target_exists:
        validation_issues.append(f"Target column '{target_col}' not found.")
    if total_missing > 0:
        validation_issues.append(f"Found {total_missing} missing values.")

    validation_succeeded = len(validation_issues) == 0

    # 6. Format summary report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("AI4I 2020 PREDICTIVE MAINTENANCE DATASET SUMMARY")
    report_lines.append("PRG4206 Machine Learning Assignment")
    report_lines.append("=" * 70)
    report_lines.append(f"File Name: {filename}")
    report_lines.append(f"File Path: {target_csv.resolve()}")
    report_lines.append(f"File Size: {file_size_bytes:,} bytes ({file_size_bytes / 1024:.2f} KB)")
    report_lines.append(f"Dataset Dimensions: {rows} rows x {cols} columns")
    report_lines.append(f"Duplicate Rows: {duplicate_rows}")
    report_lines.append(f"Total Missing Values: {total_missing}")
    report_lines.append(f"Validation Status: {'SUCCESS' if validation_succeeded else 'FAILED'}")
    if validation_issues:
        report_lines.append(f"Validation Issues: {'; '.join(validation_issues)}")
    report_lines.append("-" * 70)
    report_lines.append("COLUMNS, DATA TYPES & MISSING VALUES:")
    report_lines.append("-" * 70)
    report_lines.append(f"{'#':<4} {'Column Name':<32} {'Dtype':<12} {'Missing':<10} {'Missing %':<10}")
    report_lines.append("-" * 70)
    for idx, col in enumerate(columns, start=1):
        report_lines.append(
            f"{idx:<4} {col:<32} {str(dtypes[col]):<12} {missing_counts[col]:<10} {missing_pct[col]:<10.2f}%"
        )

    report_lines.append("-" * 70)
    report_lines.append(f"TARGET DISTRIBUTION ('{target_col}'):")
    report_lines.append("-" * 70)
    if target_exists:
        report_lines.append(target_dist_str)
        neg_count = target_counts.get(0, 0)
        pos_count = target_counts.get(1, 0)
        imbalance_ratio = (neg_count / pos_count) if pos_count > 0 else float("inf")
        report_lines.append(f"  Class Imbalance Ratio (0:1): {imbalance_ratio:.2f} : 1")
    else:
        report_lines.append("  Target column not found.")

    if present_failure_modes:
        report_lines.append("-" * 70)
        report_lines.append("SPECIFIC FAILURE MODE BREAKDOWN:")
        report_lines.append("-" * 70)
        for fmode in present_failure_modes:
            fcount = df[fmode].sum()
            fpct = (fcount / rows) * 100
            report_lines.append(f"  {fmode:<6}: {fcount:,} occurrences ({fpct:.2f}%)")

    report_lines.append("-" * 70)
    report_lines.append("NUMERICAL FEATURES DESCRIPTIVE STATISTICS:")
    report_lines.append("-" * 70)
    desc = df.describe().T
    report_lines.append(desc.to_string())

    if "Type" in df.columns:
        report_lines.append("-" * 70)
        report_lines.append("CATEGORICAL FEATURE 'Type' VALUE COUNTS:")
        report_lines.append("-" * 70)
        type_counts = df["Type"].value_counts()
        type_pct = df["Type"].value_counts(normalize=True) * 100
        for t_val, t_count in type_counts.items():
            report_lines.append(f"  Type '{t_val}': {t_count:,} ({type_pct[t_val]:.2f}%)")

    report_lines.append("=" * 70)

    summary_content = "\n".join(report_lines) + "\n"

    # Write summary file
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_content)

    print(f"\n[SUCCESS] Summary report saved to: {summary_file}")
    print("\n--- Summary Highlights ---")
    print(f"Dataset Shape: {rows} rows x {cols} columns")
    print(f"Missing Values: {total_missing}")
    print(f"Target Distribution:\n{target_dist_str}")
    print(f"Validation Succeeded: {validation_succeeded}")
    print("--------------------------")

    if not validation_succeeded:
        sys.exit(1)


if __name__ == "__main__":
    check_dataset()
