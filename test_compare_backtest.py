import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Backtest_Vectorized import compute_backtest_vectorized as compute_backtest_vectorized_original
from Backtest_Vectorized_Class import compute_backtest_vectorized as compute_backtest_vectorized_class

# Create a simple test case
def test_compare_backtest():
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
    try:
        print("Running original implementation...")
        bt_log_dict_original, log_history_original = compute_backtest_vectorized_original(positions, settings, data_dict)
        
        print("Running class implementation...")
        bt_log_dict_class, log_history_class = compute_backtest_vectorized_class(positions, settings, data_dict)
        
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
        
        if not positions_equal:
            print("Differences in positions:")
            print(pos_original - pos_class)
        
        # Plot results for visual comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        portfolio_value_original.plot(label='Original')
        portfolio_value_class.plot(label='Class')
        plt.title('Portfolio Value Comparison')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        diff.plot()
        plt.title('Difference in Portfolio Value')
        plt.tight_layout()
        plt.show()
        
        return max_diff, positions_equal
        
    except Exception as e:
        import traceback
        print(f"Test failed with error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    test_compare_backtest()