import json
import pandas as pd
import os

# Pfade
json_path = "evaluation/eval_results.json"
raw_csv_path = "evaluation/eval_results_raw.csv"
agg_csv_path = "evaluation/eval_results_agg.csv"

# -----------------------------
# Schritt 1: JSON -> Raw CSV
# -----------------------------
print(f"Lese JSON-Datei: {json_path}")
with open(json_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv(raw_csv_path, index=False)
print(f"Raw CSV geschrieben: {raw_csv_path}")

# -----------------------------
# Schritt 2: Aggregation
# -----------------------------
# numerische Spalten automatisch finden (alles außer scenario, method, episode)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["episode"]]  # episode nicht mitteln

# Aggregationsdict
agg_dict = {}
for col in numeric_cols:
    agg_dict[f"{col}_mean"] = (col, "mean")
    agg_dict[f"{col}_std"] = (col, "std")

# Gruppieren nach Szenario + Methode
agg = df.groupby(["scenario", "method"]).agg(**agg_dict).reset_index()

# Gesamte Aggregation speichern
agg.to_csv(agg_csv_path, index=False)
print(f"Aggregierte Datei geschrieben: {agg_csv_path}")
print("Zeilen:", len(agg))

# -----------------------------
# Schritt 3: Pro-Methode CSVs
# -----------------------------
for method, df_method in agg.groupby("method"):
    safe_name = method.replace(" ", "_").replace("/", "_")
    out_path = f"evaluation/{safe_name}.csv"
    df_method.to_csv(out_path, index=False)
    print(f"Datei für Methode '{method}' geschrieben: {out_path}")
