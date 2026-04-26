from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import subprocess, sys, os
if not os.path.exists("model/model.pkl"):
    print("Model not found, training now...")
    subprocess.run([sys.executable, "train.py"], check=True)
app = Flask(__name__, 
            template_folder="app/templates",
            static_folder="app/static")
app.jinja_env.globals.update(enumerate=enumerate)
with open("model/model.pkl", "rb") as f:   model = pickle.load(f)
with open("model/le_driver.pkl", "rb") as f:  le_driver = pickle.load(f)
with open("model/le_team.pkl", "rb") as f:    le_team = pickle.load(f)
with open("model/le_circuit.pkl", "rb") as f: le_circuit = pickle.load(f)

DRIVERS  = list(le_driver.classes_)
TEAMS    = list(le_team.classes_)
CIRCUITS = list(le_circuit.classes_)

driver_stats  = pd.read_csv("data/driver_stats.csv")
circuit_stats = pd.read_csv("data/circuit_stats.csv")

def predict_driver(driver, team, grid, circuit):
    driver_enc  = le_driver.transform([driver])[0]
    team_enc    = le_team.transform([team])[0]
    circuit_enc = le_circuit.transform([circuit])[0]
    proba       = model.predict_proba([[grid, driver_enc, team_enc, circuit_enc]])[0]
    classes     = model.classes_
    top3_idx    = np.argsort(proba)[::-1][:3]
    top3        = [(int(classes[i]), round(proba[i] * 100, 1)) for i in top3_idx]
    predicted   = top3[0][0]
    return predicted, top3

@app.route("/", methods=["GET", "POST"])
def index():
    result1 = None
    result2 = None
    error   = None
    form    = {}
    mode    = None

    # Season stats — leaderboard
    stats_data = driver_stats.to_dict(orient="records")

    # Circuit stats — top 5 drivers at selected circuit
    circuit_leaders = {}
    for circuit in CIRCUITS:
        top = circuit_stats[circuit_stats["Race"] == circuit].sort_values("AvgPosition").head(5)
        circuit_leaders[circuit] = top.to_dict(orient="records")

    if request.method == "POST":
        mode = request.form.get("mode", "single")
        try:
            form = {
                "driver1":  request.form["driver1"],
                "team1":    request.form["team1"],
                "grid1":    int(request.form["grid1"]),
                "circuit":  request.form["circuit"],
            }
            pred1, top3_1 = predict_driver(form["driver1"], form["team1"],
                                           form["grid1"], form["circuit"])
            result1 = {"driver": form["driver1"], "grid": form["grid1"],
                       "predicted": pred1, "top3": top3_1, "circuit": form["circuit"]}

            if mode == "h2h":
                form["driver2"] = request.form["driver2"]
                form["team2"]   = request.form["team2"]
                form["grid2"]   = int(request.form["grid2"])
                pred2, top3_2   = predict_driver(form["driver2"], form["team2"],
                                                 form["grid2"], form["circuit"])
                result2 = {"driver": form["driver2"], "grid": form["grid2"],
                           "predicted": pred2, "top3": top3_2}

        except Exception as e:
            error = str(e)

    return render_template("index.html",
                           drivers=DRIVERS, teams=TEAMS, circuits=CIRCUITS,
                           result1=result1, result2=result2,
                           error=error, form=form, mode=mode,
                           stats_data=stats_data,
                           circuit_leaders=circuit_leaders)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)