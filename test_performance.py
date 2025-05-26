import time
import pandas as pd
import numpy as np
from Backtest_Vectorized import compute_backtest_vectorized as compute_backtest_vectorized_original
from Backtest_Vectorized_Class import compute_backtest_vectorized as compute_backtest_vectorized_class

def test_performance():
    # Create sample positions DataFrame (larger than the test case)
    positions = pd.DataFrame({
        'ES=F': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 10,
        'NQ=F': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 10,
        'EURUSD=X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 10
    }, index=pd.date_range('2023-01-01', periods=100))

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
            'Open': pd.Series(np.random.rand(100) * 100 + 100, index=positions.index),
            'High': pd.Series(np.random.rand(100) * 10 + 110, index=positions.index),
            'Low': pd.Series(np.random.rand(100) * 10 + 90, index=positions.index),
            'Close': pd.Series(np.random.rand(100) * 10 + 100, index=positions.index)
        }

    # Ensure EURUSD=X has reasonable values
    data_dict['EURUSD=X'] = {
        'Open': pd.Series(np.random.rand(100) * 0.1 + 1.0, index=positions.index),
        'High': pd.Series(np.random.rand(100) * 0.1 + 1.05, index=positions.index),
        'Low': pd.Series(np.random.rand(100) * 0.1 + 0.95, index=positions.index),
        'Close': pd.Series(np.random.rand(100) * 0.1 + 1.0, index=positions.index)
    }

    # Run original implementation
    print("Running original implementation...")
    start_time = time.time()
    bt_log_dict_original, log_history_original = compute_backtest_vectorized_original(positions, settings, data_dict)
    original_time = time.time() - start_time
    print(f"Original implementation time: {original_time:.4f} seconds")

    # Run class implementation
    print("\nRunning class implementation...")
    start_time = time.time()
    bt_log_dict_class, log_history_class = compute_backtest_vectorized_class(positions, settings, data_dict)
    class_time = time.time() - start_time
    print(f"Class implementation time: {class_time:.4f} seconds")

    # Calculate improvement
    improvement = (original_time - class_time) / original_time * 100
    print(f"\nPerformance improvement: {improvement:.2f}%")

    # Compare results
    print("\nComparing portfolio values:")
    portfolio_value_original = bt_log_dict_original['portfolio_value']
    portfolio_value_class = bt_log_dict_class['portfolio_value']
    
    # Calculate difference
    diff = portfolio_value_original - portfolio_value_class
    max_diff = diff.abs().max()
    
    print(f"Maximum absolute difference in portfolio value: {max_diff}")
    
    # Compare positions
    print("\nComparing positions:")
    pos_original = bt_log_dict_original['pos']
    pos_class = bt_log_dict_class['pos']
    
    # Check if positions are equal
    positions_equal = pos_original.equals(pos_class)
    print(f"Positions are equal: {positions_equal}")

if __name__ == "__main__":
    test_performance()