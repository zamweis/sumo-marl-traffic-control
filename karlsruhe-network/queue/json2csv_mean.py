import pandas as pd

# Raw CSV einlesen
df = pd.read_csv("evaluation/eval_results_raw.csv")

# numerische Spalten automatisch finden
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Aggregation bauen: Mittelwert + Standardabweichung für alle numerischen Spalten
agg_dict = {}
for col in numeric_cols:
    agg_dict[f"{col}_mean"] = (col, "mean")
    agg_dict[f"{col}_std"] = (col, "std")

# Gruppieren nach Szenario, Methode, Run (damit jedes Modell einzeln bleibt)
agg = df.groupby(["scenario", "method", "run_dir"]).agg(**agg_dict).reset_index()

# Speichern
agg.to_csv("evaluation/eval_results_agg.csv", index=False)

print("✅ Aggregierte Datei geschrieben: evaluation/eval_results_agg.csv")
print("Spalten:", agg.columns.tolist()[:10], "...")
