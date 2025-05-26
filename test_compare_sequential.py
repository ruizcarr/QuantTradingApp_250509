import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Backtest_Vectorized_Class import compute_backtest_vectorized
from Backtest_Sequential_Class import compute_backtest_sequential

def test_compare_sequential():
    """
    Test script to compare the vectorized and sequential implementations of the backtest.
    """
    print("Creating test data...")

    # Create sample positions DataFrame
    positions = pd.DataFrame({
        'ES=F': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 5,
        'NQ=F': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 5,
        'EURUSD=X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 5
    }, index=pd.date_range('2023-01-01', periods=50))

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
            'Open': pd.Series(np.random.rand(50) * 100 + 100, index=positions.index),
            'High': pd.Series(np.random.rand(50) * 10 + 110, index=positions.index),
            'Low': pd.Series(np.random.rand(50) * 10 + 90, index=positions.index),
            'Close': pd.Series(np.random.rand(50) * 10 + 100, index=positions.index)
        }

    # Ensure EURUSD=X has reasonable values
    data_dict['EURUSD=X'] = {
        'Open': pd.Series(np.random.rand(50) * 0.1 + 1.0, index=positions.index),
        'High': pd.Series(np.random.rand(50) * 0.1 + 1.05, index=positions.index),
        'Low': pd.Series(np.random.rand(50) * 0.1 + 0.95, index=positions.index),
        'Close': pd.Series(np.random.rand(50) * 0.1 + 1.0, index=positions.index)
    }

    # Run vectorized implementation
    print("\nRunning vectorized implementation...")
    start_time = time.time()
    bt_log_dict_vectorized, log_history_vectorized = compute_backtest_vectorized(positions, settings, data_dict)
    vectorized_time = time.time() - start_time
    print(f"Vectorized implementation time: {vectorized_time:.4f} seconds")

    # Run sequential implementation
    print("\nRunning sequential implementation...")
    start_time = time.time()
    bt_log_dict_sequential, log_history_sequential = compute_backtest_sequential(positions, settings, data_dict)
    sequential_time = time.time() - start_time
    print(f"Sequential implementation time: {sequential_time:.4f} seconds")

    # Calculate performance difference
    speedup = vectorized_time / sequential_time
    if speedup > 1:
        print(f"\nVectorized implementation is {speedup:.2f}x faster than sequential")
    else:
        print(f"\nSequential implementation is {1/speedup:.2f}x faster than vectorized")

    # Compare results
    print("\nComparing portfolio values:")
    portfolio_value_vectorized = bt_log_dict_vectorized['portfolio_value']
    portfolio_value_sequential = bt_log_dict_sequential['portfolio_value']

    # Calculate difference
    diff = portfolio_value_vectorized - portfolio_value_sequential
    max_diff = diff.abs().max()

    print(f"Maximum absolute difference in portfolio value: {max_diff}")

    # Compare positions
    print("\nComparing positions:")
    pos_vectorized = bt_log_dict_vectorized['pos']
    pos_sequential = bt_log_dict_sequential['pos']

    # Check if positions are equal
    positions_equal = pos_vectorized.equals(pos_sequential)
    print(f"Positions are equal: {positions_equal}")

    if not positions_equal:
        print("\nDifferences in positions:")

        # Print the first few rows of both position DataFrames
        print("\nFirst few rows of vectorized positions:")
        print(pos_vectorized.head())

        print("\nFirst few rows of sequential positions:")
        print(pos_sequential.head())

        # Calculate and print differences
        diff = pos_vectorized - pos_sequential

        # Check if there are any non-zero differences
        if diff.any().any():
            print("\nNon-zero differences found:")

            # Count the number of differences
            diff_count = 0
            for col in diff.columns:
                # Get rows with non-zero differences for this column
                rows = diff.index[diff[col] != 0]

                if len(rows) > 0:
                    diff_count += len(rows)
                    print(f"Column {col}: {len(rows)} differences")

                    # Print the first 5 differences for each column
                    for i, row in enumerate(rows[:5]):
                        print(f"  Row {row}: Vectorized={pos_vectorized.loc[row, col]} Sequential={pos_sequential.loc[row, col]}")

                    if len(rows) > 5:
                        print(f"  ... and {len(rows) - 5} more differences")

            print(f"\nTotal differences: {diff_count} out of {pos_vectorized.size} positions ({diff_count/pos_vectorized.size*100:.2f}%)")

            # Check if the differences are small
            abs_diff = diff.abs()
            max_diff = abs_diff.max().max()
            mean_diff = abs_diff.mean().mean()
            print(f"Maximum absolute difference: {max_diff}")
            print(f"Mean absolute difference: {mean_diff}")
        else:
            print("\nNo non-zero differences found. The positions are numerically equal but not identical objects.")

    # Compare portfolio values
    print("\nComparing portfolio values:")
    portfolio_equal = portfolio_value_vectorized.equals(portfolio_value_sequential)
    print(f"Portfolio values are equal: {portfolio_equal}")

    if not portfolio_equal:
        print("\nDifferences in portfolio values:")
        diff = portfolio_value_vectorized - portfolio_value_sequential
        if diff.any():
            rows = diff.index[diff != 0]
            for row in rows:
                print(f"  Row {row}: Vectorized={portfolio_value_vectorized.loc[row]} Sequential={portfolio_value_sequential.loc[row]}")

    # Return results
    return {
        'vectorized_time': vectorized_time,
        'sequential_time': sequential_time,
        'speedup': speedup,
        'positions_equal': positions_equal,
        'portfolio_equal': portfolio_equal,
        'max_portfolio_diff': max_diff
    }

if __name__ == "__main__":
    results = test_compare_sequential()
    print("\nTest completed.")
