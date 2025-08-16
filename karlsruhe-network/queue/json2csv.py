import json
import pandas as pd

# Pfad zu deinen Evaluationsergebnissen
with open("evaluation/eval_results.json", "r") as f:
    data = json.load(f)

# In DataFrame konvertieren
df = pd.DataFrame(data)

# CSV speichern
df.to_csv("evaluation/eval_results.csv", index=False)

print("Fertig! CSV gespeichert unter evaluation/eval_results.csv")
