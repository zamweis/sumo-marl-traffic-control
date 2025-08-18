import pandas as pd

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

# Speichern
agg.to_csv("evaluation/eval_results_agg.csv", index=False)

print("Aggregierte Datei geschrieben: evaluation/eval_results_agg.csv")
print("Zeilen:", len(agg))
print("Spalten:", agg.columns.tolist()[:10], "...")
