import pandas as pd
import numpy as np
from Backtest_Vectorized_Class import compute_backtest_vectorized

# Create a simple test case
def test_compute_backtest_vectorized():
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

    # Call the function
    try:
        bt_log_dict, log_history = compute_backtest_vectorized(positions, settings, data_dict)
        print("Test passed! No errors occurred.")
        return True
    except Exception as e:
        import traceback
        print(f"Test failed with error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_compute_backtest_vectorized()
