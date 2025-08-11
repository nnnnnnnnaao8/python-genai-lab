
import pandas as pd
from src.etl import aggregate

def test_median_mode():
    df = pd.DataFrame({"category": ["A"]*3, "value": [1, 100, 2]})
    out_mean = aggregate(df, mode="mean")
    out_median = aggregate(df, mode="median")
    mean_val = float(out_mean[out_mean["category"]=="A"]["agg_value"].iloc[0])
    median_val = float(out_median[out_median["category"]=="A"]["agg_value"].iloc[0])
    assert mean_val != median_val and median_val == 2.0

def test_aggregate_median():
    import pandas as pd
    from src.etl import aggregate
    df = pd.DataFrame({"category": ["A","A","A"], "value": [1, 100, 2]})
    out = aggregate(df, mode="median")
    val = float(out[out["category"]=="A"]["agg_value"].iloc[0])
    assert val == 2.0