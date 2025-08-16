import pandas as pd

df = pd.read_csv("evaluation/eval_results_raw.csv")

agg = df.groupby(["scenario", "method"]).agg(
    mean_wait=("system_mean_waiting_time", "mean"),
    std_wait=("system_mean_waiting_time", "std"),
    mean_speed=("system_mean_speed", "mean"),
    std_speed=("system_mean_speed", "std"),
    total_wait=("system_total_waiting_time", "mean"),
    arrived=("system_total_arrived", "mean"),
    departed=("system_total_departed", "mean"),
    backlogged=("system_total_backlogged", "mean"),
    teleported=("system_total_teleported", "mean"),
).reset_index()

agg.to_csv("evaluation/eval_results_agg.csv", index=False)
