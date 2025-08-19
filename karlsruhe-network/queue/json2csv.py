import json
import pandas as pd

with open("evaluation/eval_results.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv("evaluation/eval_results_raw.csv", index=False)
