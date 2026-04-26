import os
import fastf1
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Starting model training...")

os.makedirs("data/cache", exist_ok=True)
os.makedirs("model", exist_ok=True)

fastf1.Cache.enable_cache("data/cache")

df = pd.read_csv("data/race_results.csv")
df = df.dropna(subset=["GridPosition", "Position", "Abbreviation", "TeamName", "Race"])
df = df[df["GridPosition"] > 0]
df["Position"] = df["Position"].astype(int)

le_driver  = LabelEncoder()
le_team    = LabelEncoder()
le_circuit = LabelEncoder()

df["DriverEncoded"]  = le_driver.fit_transform(df["Abbreviation"])
df["TeamEncoded"]    = le_team.fit_transform(df["TeamName"])
df["CircuitEncoded"] = le_circuit.fit_transform(df["Race"])

X = df[["GridPosition", "DriverEncoded", "TeamEncoded", "CircuitEncoded"]]
y = df["Position"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

with open("model/model.pkl", "wb") as f:    pickle.dump(model, f)
with open("model/le_driver.pkl", "wb") as f:  pickle.dump(le_driver, f)
with open("model/le_team.pkl", "wb") as f:    pickle.dump(le_team, f)
with open("model/le_circuit.pkl", "wb") as f: pickle.dump(le_circuit, f)

# Save stats CSVs
circuit_stats = df.groupby(["Race", "Abbreviation"])["Position"].mean().reset_index()
circuit_stats.columns = ["Race", "Driver", "AvgPosition"]
circuit_stats.to_csv("data/circuit_stats.csv", index=False)

driver_stats = df.groupby("Abbreviation")["Position"].mean().reset_index()
driver_stats.columns = ["Driver", "AvgPosition"]
driver_stats = driver_stats.sort_values("AvgPosition")
driver_stats.to_csv("data/driver_stats.csv", index=False)

print("Training complete!")