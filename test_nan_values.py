import pandas as pd
import numpy as np
from Backtest_Vectorized import compute_backtest_vectorized as compute_backtest_vectorized_original
from Backtest_Vectorized_Class import compute_backtest_vectorized as compute_backtest_vectorized_class

def test_nan_values():
    # Create sample positions DataFrame
    positions = pd.DataFrame({
        'ES=F': [0, 0, 1, 1, 0],
        'NQ=F': [0, 1, 0, 0, 1],
        'EURUSD=X': [0, 0, 0, 0, 0]
    }, index=pd.date_range('2023-01-01', periods=5))

    # Create sample settings dictionary
    settings = {
        'mults': {'ES=F': 50, 'NQ=F': 20, 'EURUSD=X': 1},
        'startcash': 10000,
        'exposition_lim': 0.8,
        'commission': 5.0,
        'buy_at_market': False,
        'qstats': False,
        'add_days': 0
    }

    # Create sample data_dict
    data_dict = {}
    for ticker in positions.columns:
        data_dict[ticker] = {
            'Open': pd.Series([100, 101, 102, 103, 104], index=positions.index),
            'High': pd.Series([105, 106, 107, 108, 109], index=positions.index),
            'Low': pd.Series([95, 96, 97, 98, 99], index=positions.index),
            'Close': pd.Series([102, 103, 104, 105, 106], index=positions.index)
        }

    # Call both implementations
    print("Running original implementation...")
    bt_log_dict_original, log_history_original = compute_backtest_vectorized_original(positions, settings, data_dict)

    print("Running class implementation...")
    bt_log_dict_class, log_history_class = compute_backtest_vectorized_class(positions, settings, data_dict)

    # Check for NaN values in both log_history DataFrames
    print("\nChecking for NaN values in original implementation:")
    nan_count_original = log_history_original.isna().sum().sum()
    print(f"Total NaN values: {nan_count_original}")

    if nan_count_original > 0:
        print("Columns with NaN values:")
        for col in log_history_original.columns:
            nan_count = log_history_original[col].isna().sum()
            if nan_count > 0:
                print(f"  {col}: {nan_count} NaN values")

    print("\nChecking for NaN values in class implementation:")
    nan_count_class = log_history_class.isna().sum().sum()
    print(f"Total NaN values: {nan_count_class}")

    if nan_count_class > 0:
        print("Columns with NaN values:")
        for col in log_history_class.columns:
            nan_count = log_history_class[col].isna().sum()
            if nan_count > 0:
                print(f"  {col}: {nan_count} NaN values")

    # Compare the number of NaN values
    print(f"\nDifference in NaN values: {nan_count_class - nan_count_original}")

    # Print column names of both DataFrames
    print("\nColumns in original implementation:")
    print(log_history_original.columns.tolist())
    print("\nColumns in class implementation:")
    print(log_history_class.columns.tolist())

    # Check the shape of both DataFrames
    print(f"\nShape of original implementation: {log_history_original.shape}")
    print(f"Shape of class implementation: {log_history_class.shape}")

    # Find columns that are in one DataFrame but not the other
    original_cols = set(log_history_original.columns)
    class_cols = set(log_history_class.columns)
    print("\nColumns in original but not in class:")
    print(original_cols - class_cols)
    print("\nColumns in class but not in original:")
    print(class_cols - original_cols)

    # Check if there are columns with NaN values in class implementation but not in original
    print("\nColumns with NaN values in class implementation but not in original:")
    for col in log_history_class.columns:
        if col in log_history_original.columns:
            nan_count_class_col = log_history_class[col].isna().sum()
            nan_count_original_col = log_history_original[col].isna().sum()
            if nan_count_class_col > nan_count_original_col:
                print(f"  {col}: {nan_count_class_col - nan_count_original_col} more NaN values in class implementation")

    # Print a sample of rows with NaN values from both DataFrames
    print("\nSample rows with NaN values from original implementation:")
    sample_rows_original = log_history_original[log_history_original.isna().any(axis=1)].head(3)
    print(sample_rows_original)

    print("\nSample rows with NaN values from class implementation:")
    sample_rows_class = log_history_class[log_history_class.isna().any(axis=1)].head(3)
    print(sample_rows_class)

    return log_history_original, log_history_class

if __name__ == "__main__":
    log_history_original, log_history_class = test_nan_values()
