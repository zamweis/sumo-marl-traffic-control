import pandas as pd
import os

# Raw CSV einlesen
df = pd.read_csv("evaluation/eval_results_raw.csv")

# numerische Spalten automatisch finden (alles außer scenario, method, episode)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["episode"]]  # episode nicht mitteln

# Aggregation: Mittelwert + Standardabweichung
agg_dict = {}
for col in numeric_cols:
    agg_dict[f"{col}_mean"] = (col, "mean")
    agg_dict[f"{col}_std"] = (col, "std")

# Gruppieren nach Szenario + Methode
agg = df.groupby(["scenario", "method"]).agg(**agg_dict).reset_index()

# Gesamte Aggregation speichern
agg_path = "evaluation/eval_results_agg.csv"
agg.to_csv(agg_path, index=False)
print(f"Aggregierte Datei geschrieben: {agg_path}")
print("Zeilen:", len(agg))

# eine Datei pro Methode schreiben
for method, df_method in agg.groupby("method"):
    safe_name = method.replace(" ", "_").replace("/", "_")
    out_path = f"evaluation/{safe_name}.csv"
    df_method.to_csv(out_path, index=False)
    print(f"Datei für Methode '{method}' geschrieben: {out_path}")
