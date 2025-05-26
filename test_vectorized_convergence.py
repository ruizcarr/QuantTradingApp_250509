import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from Backtest_Vectorized_Class import BacktestVectorized, BacktestSettings

def test_vectorized_convergence():
    """Test the performance of the vectorized convergence implementation."""
    # Create sample data
    n_rows = 100
    n_cols = 5

    # Create sample positions DataFrame
    positions = pd.DataFrame(
        np.random.randint(0, 2, size=(n_rows, n_cols)),
        index=pd.date_range('2023-01-01', periods=n_rows),
        columns=[f'Asset_{i}' for i in range(n_cols)]
    )

    # Create sample asset price DataFrame
    asset_price = pd.DataFrame(
        np.random.rand(n_rows, n_cols) * 100 + 100,
        index=positions.index,
        columns=positions.columns
    )

    # Create other required DataFrames
    weights_div_asset_price = pd.DataFrame(
        np.random.rand(n_rows, n_cols) * 0.1,
        index=positions.index,
        columns=positions.columns
    )

    opens = pd.DataFrame(
        np.random.rand(n_rows, n_cols) * 100 + 100,
        index=positions.index,
        columns=positions.columns
    )

    highs = opens + np.random.rand(n_rows, n_cols) * 10
    lows = opens - np.random.rand(n_rows, n_cols) * 10
    closes = opens + np.random.rand(n_rows, n_cols) * 5 - 2.5

    # Create other required inputs
    mults = np.ones(n_cols)
    portfolio_value_usd = pd.Series(10000, index=positions.index)
    weights = positions.copy()
    buy_trigger = pd.DataFrame(True, index=positions.index, columns=positions.columns)
    sell_trigger = pd.DataFrame(True, index=positions.index, columns=positions.columns)
    sell_stop_price = lows.copy()
    buy_stop_price = highs.copy()
    exchange_rate = pd.Series(1.0, index=positions.index)
    startcash_usd = 10000.0
    startcash = 10000.0
    exposition_lim = 0.8
    pos = pd.DataFrame(0, index=positions.index, columns=positions.columns)

    # Create backtest instance
    backtest_settings = BacktestSettings()
    backtest = BacktestVectorized(backtest_settings)

    # Create a copy of the original method for comparison
    original_method = backtest.compute_backtest_until_convergence

    # Create a modified version of the method with the original loop implementation
    def original_loop_implementation(
            self,
            weights_div_asset_price, asset_price, opens, highs, lows, closes, mults,
            portfolio_value_usd, weights, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
            exchange_rate, startcash_usd, startcash, exposition_lim, pos, max_iterations=200
    ):
        """Original loop implementation for comparison."""
        i = 0
        bt_log_dict = {}

        # Use numpy arrays for faster comparison
        prev_pos_values = None

        while i < max_iterations:
            # Call compute_backtest with all required arguments
            pos, portfolio_value_usd, bt_log_dict = self.compute_backtest(
                weights_div_asset_price, asset_price, opens, highs, lows, closes, mults,
                portfolio_value_usd, weights, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
                exchange_rate, startcash_usd, startcash, exposition_lim, pos
            )

            # Check if positions have changed using numpy for faster comparison
            pos_values = pos.values
            if prev_pos_values is not None and np.array_equal(pos_values, prev_pos_values):
                break

            prev_pos_values = pos_values.copy()
            i += 1

        bt_log_dict['n_iter'] = i
        return pos, portfolio_value_usd, bt_log_dict

    # Temporarily replace the method with the original implementation
    backtest.original_loop_implementation = original_loop_implementation.__get__(backtest, BacktestVectorized)

    # Run the original implementation and measure time
    start_time = time.time()
    pos_original, portfolio_original, log_dict_original = backtest.original_loop_implementation(
        weights_div_asset_price, asset_price, opens, highs, lows, closes, mults,
        portfolio_value_usd, weights, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
        exchange_rate, startcash_usd, startcash, exposition_lim, pos
    )
    original_time = time.time() - start_time

    # Run the vectorized implementation and measure time
    start_time = time.time()
    pos_vectorized, portfolio_vectorized, log_dict_vectorized = backtest.compute_backtest_until_convergence(
        weights_div_asset_price, asset_price, opens, highs, lows, closes, mults,
        portfolio_value_usd, weights, buy_trigger, sell_trigger, sell_stop_price, buy_stop_price,
        exchange_rate, startcash_usd, startcash, exposition_lim, pos
    )
    vectorized_time = time.time() - start_time

    # Compare results
    positions_equal = pos_original.equals(pos_vectorized)
    portfolio_diff = (portfolio_original - portfolio_vectorized).abs().max()
    iterations_original = log_dict_original.get('n_iter', 0)
    iterations_vectorized = log_dict_vectorized.get('n_iter', 0)

    # Print results
    print(f"Original implementation time: {original_time:.6f} seconds")
    print(f"Vectorized implementation time: {vectorized_time:.6f} seconds")
    print(f"Speedup: {original_time / vectorized_time:.2f}x")
    print(f"Positions equal: {positions_equal}")
    print(f"Maximum portfolio difference: {portfolio_diff}")
    print(f"Iterations (original): {iterations_original}")
    print(f"Iterations (vectorized): {iterations_vectorized}")

    # Debug position differences if they're not equal
    if not positions_equal:
        print("\nDebugging position differences:")
        # Check data types
        print(f"Original positions dtype: {pos_original.dtypes}")
        print(f"Vectorized positions dtype: {pos_vectorized.dtypes}")

        # Check if values are close but not exactly equal (floating point precision issues)
        if np.allclose(pos_original.values, pos_vectorized.values):
            print("Values are numerically close but not exactly equal (floating point precision issue)")

        # Find where the differences are
        diff = pos_original != pos_vectorized
        if diff.any().any():
            print("\nDifferences found at:")
            for col in diff.columns[diff.any()]:
                rows = diff.index[diff[col]]
                print(f"Column {col}:")
                for row in rows:
                    print(f"  Row {row}: Original={pos_original.loc[row, col]} Vectorized={pos_vectorized.loc[row, col]}")

    # Disable plotting for this environment
    # Plot performance comparison
    # plt.figure(figsize=(10, 6))
    # plt.bar(['Original', 'Vectorized'], [original_time, vectorized_time])
    # plt.title('Performance Comparison')
    # plt.ylabel('Time (seconds)')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # Add speedup text
    # plt.text(0.5, 0.5, f'Speedup: {original_time / vectorized_time:.2f}x',
    #          horizontalalignment='center',
    #          verticalalignment='center',
    #          transform=plt.gca().transAxes,
    #          fontsize=14,
    #          bbox=dict(facecolor='white', alpha=0.8))

    # plt.tight_layout()
    # plt.show()

    return {
        'original_time': original_time,
        'vectorized_time': vectorized_time,
        'speedup': original_time / vectorized_time,
        'positions_equal': positions_equal,
        'portfolio_diff': portfolio_diff,
        'iterations_original': iterations_original,
        'iterations_vectorized': iterations_vectorized
    }

if __name__ == "__main__":
    test_vectorized_convergence()
